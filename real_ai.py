#!/usr/bin/env python3
"""
real_ai.py — THE REAL AI SYSTEM
Predicts: "will next bars have a profitable move?" (not direction)
Label: 1 if max(UP,DOWN) move in next N bars > threshold, else 0
Trains on 85 quant features with LightGBM.
"""
import numpy as np, lightgbm as lgb, json, os, time, gc
BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    t0 = time.time()
    print("═══ REAL AI: OPPORTUNITY DETECTION ═══\n")
    
    # Load prices
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:,3].astype(np.float64)
    highs = prices[:,1].astype(np.float64)
    lows = prices[:,2].astype(np.float64)
    n = len(closes)
    
    # Load features
    meta_f = json.load(open(f"{BASE}/quant_features_meta.json"))
    NAMES = meta_f['feature_names']
    n_feat = meta_f['n_features']
    X = np.memmap(f"{BASE}/quant_features_116.npy", dtype=np.float32, mode='r', shape=(n, n_feat))
    
    print(f"Data: {n:,} bars × {n_feat} features")
    
    # ═══ BUILD NEW LABELS ═══
    # "Is there a profitable move in the next 10 bars?"
    # Look at max favorable excursion (MFE) in next 10 bars
    # If MFE > 0.3 ATR → label=1 (opportunity), else label=0
    
    print("Building opportunity labels...")
    
    # ATR
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros(n)
    atr[1:] = tr
    for i in range(1, min(15, n)):
        atr[i] = np.mean(tr[:i+1])
    for i in range(15, n):
        atr[i] = np.mean(tr[i-14:i+1])
    
    LOOK = 10  # bars ahead
    THRESHOLD = 0.3  # ATR units
    
    y = np.zeros(n, dtype=np.int8)
    for i in range(n - LOOK):
        future_closes = closes[i+1:i+1+LOOK]
        future_highs = highs[i+1:i+1+LOOK]
        future_lows = lows[i+1:i+1+LOOK]
        curr = closes[i]
        a = atr[i]
        
        if a < 0.01:
            continue
        
        # Max favorable excursion (best entry from current price)
        max_up = np.max(future_highs - curr) / a
        max_down = np.max(curr - future_lows) / a
        
        # Opportunity = either direction has a big move
        if max(max_up, max_down) >= THRESHOLD:
            y[i] = 1
    
    n_opportunities = y.sum()
    print(f"  Labels: {n_opportunities:,} opportunities / {n:,} total ({n_opportunities/n*100:.1f}%)")
    
    # NaN fill
    print("Filling NaN...")
    for ci in range(0, n, 500_000):
        end = min(ci+500_000, n)
        chunk = X[ci:end]
        m = np.isnan(chunk)
        if m.any(): chunk[m] = 0.0
    gc.collect()
    
    # ═══ TRAIN ═══
    print("\n═══ TRAINING ═══\n")
    
    split = int(n * 0.7)
    train_start = max(0, split - 3_000_000)
    X_tr = X[train_start:split]; X_te = X[split:]
    y_tr = y[train_start:split]; y_te = y[split:]
    
    print(f"Train: {len(y_tr):,} ({y_tr.mean()*100:.1f}% opportunities)")
    print(f"Test:  {len(y_te):,} ({y_te.mean()*100:.1f}% opportunities)")
    
    params = {
        'objective': 'binary', 'metric': 'auc',
        'num_leaves': 63, 'max_depth': 10,
        'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'min_child_samples': 50, 'verbose': -1,
        'is_unbalance': True,
    }
    
    for seed in [42, 7, 2026]:
        print(f"\n  Seed {seed}...")
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=NAMES)
        dval = lgb.Dataset(X_te, label=y_te, feature_name=NAMES, reference=dtrain)
        
        model = lgb.train(params, dtrain, num_boost_round=1000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        preds = model.predict(X_te)
        
        # AUC
        sorted_idx = np.argsort(preds)
        sorted_y = y_te[sorted_idx]
        n_pos = max(int(y_te.sum()), 1)
        n_neg = max(len(y_te) - n_pos, 1)
        tpr = np.cumsum(sorted_y[::-1])[:n_pos] / n_pos
        fpr = np.cumsum(1-sorted_y[::-1])[:n_pos] / n_neg
        auc = float(np.trapezoid(tpr, fpr)) if len(tpr) > 1 else 0.5
        
        # Accuracy at different thresholds
        for thr in [0.3, 0.4, 0.5, 0.6]:
            pred_bin = (preds > thr).astype(int)
            acc = (pred_bin == y_te).mean()
            n_trades = pred_bin.sum()
            print(f"    thr={thr:.1f}: acc={acc:.4f} trades={n_trades:,} ({n_trades/len(y_te)*100:.1f}%)")
        
        # Precision at top predictions
        top_k = [1000, 5000, 10000, 50000]
        for k in top_k:
            if k < len(preds):
                top_idx = np.argsort(preds)[-k:]
                precision = y_te[top_idx].mean()
                print(f"    Top {k:>6,}: precision={precision:.4f}")
        
        print(f"    AUC: {auc:.4f} | Trees: {model.num_trees()}")
        
        imp = model.feature_importance(importance_type='gain')
        top10 = np.argsort(imp)[-10:][::-1]
        print(f"    Top features:")
        for idx in top10:
            print(f"      {NAMES[idx]:25s} gain={imp[idx]:.0f}")
        
        model.save_model(f"{BASE}/models/real_ai_s{seed}.txt")
    
    ensemble = {
        'models': ['real_ai_s42', 'real_ai_s7', 'real_ai_s2026'],
        'base_tf': 'm5', 'n_features': n_feat,
        'feature_names': NAMES, 'source': 'real_ai',
        'label': 'opportunity_detection',
        'look_ahead': LOOK, 'threshold_atr': THRESHOLD,
    }
    with open(f"{BASE}/models/real_ai_ensemble.json", 'w') as f:
        json.dump(ensemble, f, indent=2)
    
    total = time.time() - t0
    print(f"\n═══ DONE: {total:.0f}s ({total/60:.1f}min) ═══")

if __name__ == '__main__':
    main()
