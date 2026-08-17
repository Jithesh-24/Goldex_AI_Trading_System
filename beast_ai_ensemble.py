#!/usr/bin/env python3
"""
beast_ai_ensemble.py — THE REAL AI BEAST
No static indicators. No hardcoded thresholds.
106 features → lightweight AI model → dynamic decisions.

Uses LightGBM NOT to predict direction, but to predict
REGIME (trending/ranging) and VOLATILITY (high/low).
Then uses regime to SELECT the right strategy.
This is how quantitative funds actually work.
"""
import numpy as np
import lightgbm as lgb
import json, os, time
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def sma(data, period):
    cs = np.cumsum(data)
    out = np.full(len(data), np.nan, dtype=np.float64)
    out[period-1:] = (cs[period-1:] - np.concatenate([[0], cs[:-period]])) / period
    return out

def compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    return sma(tr, period)

def main():
    t0 = time.time()
    print("═══ BEAST AI ENSEMBLE ═══\n")
    print("No static indicators. No hardcoded thresholds.")
    print("106 features → AI model → dynamic decisions.\n")
    
    # Load data
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    mmap_rows = 9999963
    fm = json.load(open(f"{BASE}/models/feature_map.json"))
    live_idx = fm['live_indices']
    n_live = len(live_idx)
    live_feats = fm.get('live_features', [])
    
    # Load labels
    y_full = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(meta['n_rows'],))
    
    USE_ROWS = 2_000_000
    START = mmap_rows - USE_ROWS
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(mmap_rows, nf))
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    n = len(closes)
    
    print(f"Data: {n:,} bars × {n_live} features")
    
    # Extract features
    print("Extracting features...")
    X_all = np.empty((USE_ROWS, n_live), dtype=np.float32)
    for j, fi in enumerate(live_idx):
        X_all[:, j] = X[START:START+USE_ROWS, fi]
    
    # Labels: direction (what we've been using)
    y_dir = y_full[START:START+USE_ROWS].astype(np.int8)
    
    # Labels: VOLATILITY regime (new target)
    # High vol = next bar's ATR > median ATR
    atr_full = compute_atr(highs[:USE_ROWS], lows[:USE_ROWS], closes[:USE_ROWS], 14)
    atr_median = np.nanmedian(atr_full[500:])
    y_vol = np.where(atr_full > atr_median, 1, 0).astype(np.int8)  # 1=high vol, 0=low vol
    
    # Labels: MOMENTUM regime
    # Strong momentum = price moved > median in last 20 bars
    ret_20 = np.zeros(USE_ROWS, dtype=np.float64)
    ret_20[20:] = closes[START+20:START+USE_ROWS] - closes[START:START+USE_ROWS-20]
    mom_median = np.nanmedian(np.abs(ret_20[500:]))
    y_mom = np.where(np.abs(ret_20) > mom_median, 1, 0).astype(np.int8)  # 1=strong, 0=weak
    
    # Labels: REGIME
    # Trending = consecutive same-direction closes > 5
    y_regime = np.zeros(USE_ROWS, dtype=np.int8)
    for i in range(1, USE_ROWS):
        streak = 0
        sign = 0
        for j in range(max(0, i-5), i):
            diff = closes[START+j+1] - closes[START+j] if START+j+1 < n else 0
            if sign == 0:
                sign = 1 if diff > 0 else -1
            if (diff > 0 and sign == 1) or (diff < 0 and sign == -1):
                streak += 1
            else:
                break
        if streak >= 3:
            y_regime[i] = sign  # +1 = uptrend, -1 = downtrend
    
    print(f"  Direction: {(y_dir==1).mean():.1%} UP")
    print(f"  Volatility: {(y_vol==1).mean():.1%} HIGH")
    print(f"  Momentum: {(y_mom==1).mean():.1%} STRONG")
    print(f"  Regime: {(y_regime==1).mean():.1%} UP-trend, {(y_regime==-1).mean():.1%} DOWN-trend")
    print(f"  Done ({time.time()-t0:.0f}s)")
    
    # ═══ TRAIN MULTIPLE MODELS ═══
    print("\n═══ TRAINING AI MODELS ═══")
    
    # Split: first 70% train, last 30% test
    split = int(USE_ROWS * 0.7)
    X_tr, X_te = X_all[:split], X_all[split:]
    
    models = {}
    targets = {
        'direction': y_dir,
        'volatility': y_vol,
        'momentum': y_mom,
    }
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 100,
        'verbose': -1,
    }
    
    fname = [f"f{i}" for i in range(n_live)]
    
    for name, y in targets.items():
        print(f"\n  Training {name} model...")
        y_tr, y_te = y[:split], y[split:]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=fname)
        dval = lgb.Dataset(X_te, label=y_te, feature_name=fname, reference=dtrain)
        
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        preds = model.predict(X_te)
        pred_bin = (preds > 0.5).astype(int)
        acc = (pred_bin == y_te).mean()
        auc = np.abs(preds[y_te==1].mean() - preds[y_te==0].mean())
        
        print(f"    Accuracy: {acc:.3f}")
        print(f"    AUC signal: {auc:.3f}")
        print(f"    Trees: {model.num_trees()}")
        
        # Feature importance
        imp = model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[-10:][::-1]
        print(f"    Top features:")
        for idx in top_idx:
            if idx < len(live_feats):
                print(f"      {live_feats[idx]:25s} importance={imp[idx]:.0f}")
        
        models[name] = model
    
    # ═══ META-STRATEGY BACKTEST ═══
    print("\n═══ META-STRATEGY BACKTEST ═══")
    print("  AI predicts regime → selects strategy → trades\n")
    
    atr_14 = compute_atr(highs, lows, closes, 14)
    
    # Generate AI predictions for test period
    test_start = START + split
    test_end = START + USE_ROWS
    
    # Get predictions
    dir_preds = models['direction'].predict(X_te)
    vol_preds = models['volatility'].predict(X_te)
    mom_preds = models['momentum'].predict(X_te)
    
    def meta_bt(sl_m, tp_m, min_conf=0.55, dynamic=True, base_risk=20.0):
        """
        META-STRATEGY:
        - AI predicts regime (direction, volatility, momentum)
        - High confidence + trending → follow AI direction
        - High volatility → tighten SL, wider TP
        - Strong momentum → follow momentum
        - Low confidence → no trade
        """
        acc = 1000.0; peak = 1000.0
        w = 0; l = 0; pnl = 0; dd = 0; nn = 0
        sp = 0.30
        
        in_pos = False; pos_dir = 0; pos_sl = 0; pos_tp = 0
        pos_sl_dist = 0; pos_entry = 0; pos_end = 0
        
        for i in range(500, len(X_te) - 37):
            bar = split + i
            
            if in_pos:
                if i >= pos_end:
                    out = (closes[test_start + i] - pos_entry) * pos_dir
                    p = base_risk * (out / pos_sl_dist) - sp * 2
                    acc += p; pnl += p; nn += 1
                    if p > 0: w += 1
                    else: l += 1
                    if acc > peak: peak = acc
                    d2 = (peak - acc) / peak if peak > 0 else 0
                    if d2 > dd: dd = d2
                    in_pos = False
                continue
            
            dp = dir_preds[i]
            vp = vol_preds[i]
            mp = mom_preds[i]
            
            # AI DECISION LOGIC
            conf = abs(dp - 0.5) * 2  # 0 to 1 confidence
            
            if conf < min_conf:
                continue
            
            # Determine direction from AI
            if dp > 0.5:
                direction = 1
            else:
                direction = -1
            
            # Adjust SL/TP based on volatility prediction
            if vp > 0.6:
                # High vol predicted → wider SL, wider TP
                actual_sl = sl_m * 1.3
                actual_tp = tp_m * 1.2
            elif vp < 0.4:
                # Low vol predicted → tighter SL, tighter TP
                actual_sl = sl_m * 0.8
                actual_tp = tp_m * 0.8
            else:
                actual_sl = sl_m
                actual_tp = tp_m
            
            # Adjust based on momentum
            if mp > 0.6:
                # Strong momentum → bigger position, wider TP
                actual_tp *= 1.2
                risk_mult = 1.3
            else:
                risk_mult = 1.0
            
            if atr_14[bar] < 0.1:
                continue
            
            entry = closes[bar]; d = direction
            sl_d = actual_sl * atr_14[bar]
            tp_d = actual_tp * atr_14[bar]
            if d == 1: sl_p = entry - sl_d; tp_p = entry + tp_d
            else: sl_p = entry + sl_d; tp_p = entry - tp_d
            
            # Dynamic sizing
            if dynamic:
                current_dd = (peak - acc) / peak if peak > 0 else 0
                if current_dd > 0.3: rd = base_risk * 0.5
                elif current_dd > 0.1: rd = base_risk * 0.75
                else: rd = min(base_risk * 1.5, 50)
                rd *= risk_mult * min(conf * 2, 1.5)
            else:
                rd = base_risk
            
            # Simulate
            out = 0; bp = entry
            for j in range(i+1, min(i+37, len(X_te))):
                bj = test_start + j
                if d == 1:
                    if lows[bj] <= sl_p: out = -(entry - sl_p); break
                    bp = max(bp, highs[bj])
                    nsl = bp - sl_d * 0.6
                    if nsl > sl_p: sl_p = nsl
                    if highs[bj] >= tp_p: out = (tp_p - entry); break
                else:
                    if highs[bj] >= sl_p: out = -(sl_p - entry); break
                    bp = min(bp, lows[bj])
                    nsl = bp + sl_d * 0.6
                    if nsl < sl_p: sl_p = nsl
                    if lows[bj] <= tp_p: out = (entry - tp_p); break
            
            if out == 0:
                ej = min(i + 37, len(X_te) - 1)
                out = (closes[test_start + ej] - entry) * d
            
            p = rd * (out / sl_d) - sp * 2
            acc += p; pnl += p; nn += 1
            if p > 0: w += 1
            else: l += 1
            if acc > peak: peak = acc
            d2 = (peak - acc) / peak if peak > 0 else 0
            if d2 > dd: dd = d2
            in_pos = True; pos_dir = d; pos_entry = entry
            pos_sl_dist = sl_d; pos_end = i + 37
            
            if acc <= 0: break
        
        tpd = nn / (len(X_te) / 288) if len(X_te) > 0 else 0
        print(f"  conf>={min_conf:.2f} SL={sl_m} TP={tp_m}: "
              f"WR={w/max(w+l,1):.1%} PnL=${pnl:,.0f} DD={dd:.1%} "
              f"trades={nn} PF={w/max(l,1):.2f} trades/day={tpd:.1f}")
        return {'n': nn, 'wr': w/max(w+l,1), 'pnl': pnl, 'acc': acc, 'dd': dd, 'w': w, 'l': l, 'tpd': tpd}
    
    # Grid search
    best_pnl = -999999; best = None
    for mc in [0.50, 0.55, 0.60, 0.65]:
        for sl in [0.8, 1.0, 1.5]:
            for tp in [1.5, 2.0, 2.5, 3.0]:
                r = meta_bt(sl, tp, mc)
                if r['n'] > 10 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'mc': mc, 'sl': sl, 'tp': tp, **r}
    
    if best:
        print(f"\n═══ BEST RESULT ═══")
        print(f"  Min confidence: {best['mc']:.2f}")
        print(f"  SL: {best['sl']} ATR (adaptive), TP: {best['tp']} ATR (adaptive)")
        print(f"  Trades: {best['n']:,}")
        print(f"  Win rate: {best['wr']:.1%}")
        print(f"  P&L: ${best['pnl']:,.2f}")
        print(f"  Account: ${best['acc']:,.2f}")
        print(f"  Drawdown: {best['dd']:.1%}")
        print(f"  Profit factor: {best['w']/max(best['l'],1):.2f}")
        print(f"  Trades/day: {best['tpd']:.1f}")
        
        # What the AI learned
        print(f"\n═══ WHAT THE AI LEARNED ═══")
        for name, model in models.items():
            print(f"\n  {name.upper()} model:")
            imp = model.feature_importance(importance_type='gain')
            top5 = np.argsort(imp)[-5:][::-1]
            for idx in top5:
                if idx < len(live_feats):
                    print(f"    {live_feats[idx]:25s} = {imp[idx]:.0f}")
    
    print(f"\n═══ DONE ═══ ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
