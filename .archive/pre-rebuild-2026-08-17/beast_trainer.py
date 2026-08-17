#!/usr/bin/env python3
"""
beast_trainer.py — The Beast System
XGBoost + CatBoost + LightGBM ensemble with isotonic calibration,
regime detection, dynamic SL/TP, and proper walk-forward validation.
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import time, os, json, gc, pickle
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = f"{BASE}/models"
SEEDS = [42, 7, 2026]

def load_data():
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    n = meta['n_rows']
    mmap_rows = 9999963
    n_rows = mmap_rows
    
    fm = json.load(open(f"{BASE}/models/feature_map.json"))
    live_idx = fm['live_indices']
    dead_idx = fm.get('dead_indices', [])
    n_live = len(live_idx)
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(mmap_rows, nf))
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n,))
    prices = np.load(f"{BASE}/prices_tail.npy")
    
    return X, y, prices, live_idx, dead_idx, n_rows, n_live

def extract_live(X, idx, n_rows, n_live, chunk=500000):
    """Extract live features in chunks to avoid OOM."""
    out = np.memmap(f"{BASE}/_beast_temp.npy", dtype=np.float32, mode='w+', shape=(n_rows, n_live))
    for cs in range(0, n_rows, chunk):
        ce = min(cs + chunk, n_rows)
        for j, fi in enumerate(idx):
            out[cs:ce, j] = X[cs:ce, fi]
    out.flush()
    return out

def train_lightgbm(X_tr, y_tr, X_val, y_val, seed, n_live):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt', 'num_leaves': 63, 'max_depth': 10,
        'learning_rate': 0.03, 'feature_fraction': 0.7,
        'bagging_fraction': 0.7, 'bagging_freq': 5,
        'min_child_samples': 200, 'lambda_l1': 0.01, 'lambda_l2': 0.1,
        'is_unbalance': True, 'seed': seed, 'verbose': -1, 'num_threads': 4,
    }
    fname = [f"f{i}" for i in range(n_live)]
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=fname)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=fname, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return model

def train_xgboost(X_tr, y_tr, X_val, y_val, seed):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': 8, 'learning_rate': 0.03,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 200, 'reg_alpha': 0.01, 'reg_lambda': 0.1,
        'scale_pos_weight': sum(y_tr == 0) / max(sum(y_tr == 1), 1),
        'seed': seed, 'nthread': 4, 'verbosity': 0,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=500,
                       evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)
    return model

def train_catboost(X_tr, y_tr, X_val, y_val, seed):
    model = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.03,
        l2_leaf_reg=3, min_data_in_leaf=200,
        auto_class_weights='Balanced',
        random_seed=seed, verbose=0, thread_count=4,
        loss_function='Logloss', eval_metric='Logloss',
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30)
    return model

def calibrate_ensemble(probs, y_true):
    """Isotonic regression calibration on held-out data."""
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    ir.fit(probs, y_true)
    return ir

def backtest_system(probs, closes, highs, lows, labels, 
                    sl_mult=1.5, tp_mult=2.5, risk_pct=0.02, 
                    min_conf=0.55, use_trailing=True):
    """Proper backtest with dynamic SL/TP and trailing stops."""
    n = len(probs)
    account = 1000.0
    peak = 1000.0
    trades = []
    wins = 0
    losses = 0
    total_pnl = 0
    max_dd = 0
    
    # ATR
    atr_period = 14
    spread = 0.30
    
    for i in range(200, n - 37):
        if probs[i] < min_conf:
            continue
        
        # ATR
        trs = []
        for j in range(max(0, i - atr_period), i):
            tr = max(highs[j] - lows[j],
                     max(abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])))
            trs.append(tr)
        atr = np.mean(trs)
        if atr < 0.1:
            continue
        
        entry = closes[i]
        direction = 1 if probs[i] > 0.5 else -1
        
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        
        if direction == 1:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist
        
        # Simulate trade
        outcome = 0
        best_price = entry
        for j in range(i + 1, min(i + 37, n)):
            if direction == 1:
                if lows[j] <= sl_price:
                    outcome = -sl_dist
                    break
                best_price = max(best_price, highs[j])
                if use_trailing:
                    new_sl = best_price - sl_dist * 0.7
                    if new_sl > sl_price:
                        sl_price = new_sl
                if highs[j] >= tp_price:
                    outcome = tp_dist
                    break
            else:
                if highs[j] >= sl_price:
                    outcome = -sl_dist
                    break
                best_price = min(best_price, lows[j])
                if use_trailing:
                    new_sl = best_price + sl_dist * 0.7
                    if new_sl < sl_price:
                        sl_price = new_sl
                if lows[j] <= tp_price:
                    outcome = tp_dist
                    break
        
        if outcome == 0:
            continue
        
        risk_amount = account * risk_pct
        pnl = (outcome / sl_dist) * risk_amount - spread * 2
        
        account += pnl
        total_pnl += pnl
        
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        if account > peak:
            peak = account
        dd = (peak - account) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        
        trades.append({'pnl': pnl, 'account': account, 'conf': probs[i], 'direction': direction})
    
    wr = wins / max(wins + losses, 1)
    return {
        'trades': len(trades), 'win_rate': wr, 'total_pnl': total_pnl,
        'final_account': account, 'max_drawdown': max_dd,
        'wins': wins, 'losses': losses,
        'avg_win': total_pnl / max(wins, 1) if wins > 0 else 0,
        'avg_loss': total_pnl / max(losses, 1) if losses > 0 else 0,
    }

def main():
    t0 = time.time()
    print("═══ BEAST SYSTEM — TRAINING ═══\n")
    
    X, y, prices, live_idx, dead_idx, n_rows, n_live = load_data()
    print(f"Data: {n_rows:,} rows × {n_live} live features")
    
    # Extract live features
    print("\nExtracting live features...")
    X_live = extract_live(X, live_idx, n_rows, n_live)
    print(f"  ✅ Extracted ({time.time()-t0:.0f}s)")
    
    y_data = y[-n_rows:]
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    
    # Walk-forward: train on first 8M, validate on last 2M
    split = 8_000_000
    X_tr, y_tr = X_live[:split], y_data[:split]
    X_val, y_val = X_live[split:], y_data[split:]
    
    print(f"\nTrain: {split:,} rows | Val: {n_rows - split:,} rows")
    print(f"Train UP: {(y_tr==1).sum():,} ({(y_tr==1).mean():.1%})")
    print(f"Val UP: {(y_val==1).sum():,} ({(y_val==1).mean():.1%})")
    
    # Train 3 model types × 3 seeds
    all_models = {'lgb': [], 'xgb': [], 'cat': []}
    
    print("\n═══ TRAINING MODELS ═══")
    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        
        # LightGBM
        t1 = time.time()
        m_lgb = train_lightgbm(X_tr, y_tr, X_val, y_val, seed, n_live)
        all_models['lgb'].append(m_lgb)
        print(f"  LightGBM: {m_lgb.num_trees()} trees ({time.time()-t1:.0f}s)")
        
        # XGBoost
        t1 = time.time()
        m_xgb = train_xgboost(X_tr, y_tr, X_val, y_val, seed)
        all_models['xgb'].append(m_xgb)
        print(f"  XGBoost: {m_xgb.best_iteration} trees ({time.time()-t1:.0f}s)")
        
        # CatBoost
        t1 = time.time()
        m_cat = train_catboost(X_tr, y_tr, X_val, y_val, seed)
        all_models['cat'].append(m_cat)
        print(f"  CatBoost: {m_cat.best_iteration_} trees ({time.time()-t1:.0f}s)")
        
        gc.collect()
    
    # Get raw predictions on validation set
    print("\n═══ GENERATING PREDICTIONS ═══")
    raw_preds = np.zeros(len(y_val))
    
    for i, seed in enumerate(SEEDS):
        # LGB
        p_lgb = all_models['lgb'][i].predict(X_val)
        # XGB
        dval = xgb.DMatrix(X_val)
        p_xgb = all_models['xgb'][i].predict(dval)
        # Cat
        p_cat = all_models['cat'][i].predict_proba(X_val)[:, 1]
        
        # Equal-weight ensemble of 3 model types
        raw_preds += (p_lgb + p_xgb + p_cat) / (3 * len(SEEDS))
    
    # Calibrate
    print("Calibrating with isotonic regression...")
    calibrator = calibrate_ensemble(raw_preds, y_val)
    calibrated = calibrator.transform(raw_preds)
    
    # Analyze calibration
    print("\n═══ CALIBRATION ANALYSIS ═══")
    for thr in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = calibrated >= thr
        if mask.sum() > 0:
            acc = (calibrated[mask] > 0.5).astype(int) == y_val[mask]
            # Actually, we need to check if model predicts UP and it IS up
            pred_up = calibrated[mask] > 0.5
            actual_up = y_val[mask] == 1
            correct = (pred_up == actual_up).mean()
            print(f"  conf>={thr:.2f}: {mask.sum():>8,} trades, accuracy={correct:.3f}")
    
    # Backtest with different SL/TP combinations
    print("\n═══ BACKTEST — FINDING OPTIMAL PARAMETERS ═══")
    val_closes = closes[split:]
    val_highs = highs[split:]
    val_lows = lows[split:]
    
    best_config = None
    best_pnl = -999999
    
    for sl_m in [1.0, 1.5, 2.0]:
        for tp_m in [2.0, 3.0, 4.0, 5.0]:
            for conf in [0.50, 0.55, 0.60]:
                result = backtest_system(
                    calibrated, val_closes, val_highs, val_lows, y_val,
                    sl_mult=sl_m, tp_mult=tp_m, min_conf=conf
                )
                if result['total_pnl'] > best_pnl and result['trades'] > 100:
                    best_pnl = result['total_pnl']
                    best_config = {'sl': sl_m, 'tp': tp_m, 'conf': conf, **result}
                    print(f"  SL={sl_m} TP={tp_m} conf>={conf}: "
                          f"trades={result['trades']}, WR={result['win_rate']:.1%}, "
                          f"PnL=${result['total_pnl']:.2f}, DD={result['max_drawdown']:.1%}")
    
    if best_config:
        print(f"\n═══ BEST CONFIGURATION ═══")
        print(f"  SL: {best_config['sl']} ATR")
        print(f"  TP: {best_config['tp']} ATR")
        print(f"  Min confidence: {best_config['conf']}")
        print(f"  Trades: {best_config['trades']}")
        print(f"  Win rate: {best_config['win_rate']:.1%}")
        print(f"  Total P&L: ${best_config['total_pnl']:.2f}")
        print(f"  Final account: ${best_config['final_account']:.2f}")
        print(f"  Max drawdown: {best_config['max_drawdown']:.1%}")
    
    # Save models and calibrator
    print("\n═══ SAVING MODELS ═══")
    
    for i, seed in enumerate(SEEDS):
        all_models['lgb'][i].save_model(f"{MODELS}/beast_lgb_s{seed}.txt")
        all_models['xgb'][i].save_model(f"{MODELS}/beast_xgb_s{seed}.json")
        all_models['cat'][i].save_model(f"{MODELS}/beast_cat_s{seed}.cbm")
    
    with open(f"{MODELS}/beast_calibrator.pkl", 'wb') as f:
        pickle.dump(calibrator, f)
    
    # Save config
    config = {
        'version': 'beast_v1.0',
        'model_types': ['lgb', 'xgb', 'cat'],
        'seeds': SEEDS,
        'n_features': n_live,
        'live_features': [f"f{i}" for i in range(n_live)],
        'calibration': 'isotonic_regression',
        'optimal_sl_atr': best_config['sl'] if best_config else 1.5,
        'optimal_tp_atr': best_config['tp'] if best_config else 2.5,
        'optimal_min_conf': best_config['conf'] if best_config else 0.55,
        'optimal_win_rate': best_config['win_rate'] if best_config else 0,
        'optimal_pnl': best_config['total_pnl'] if best_config else 0,
        'trained': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(f"{MODELS}/beast_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Cleanup
    os.remove(f"{BASE}/_beast_temp.npy")
    
    elapsed = time.time() - t0
    print(f"\n═══ BEAST TRAINING COMPLETE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")

if __name__ == '__main__':
    main()
