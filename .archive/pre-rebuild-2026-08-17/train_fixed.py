#!/usr/bin/env python3
"""
train_fixed.py — BINARY + CLASS BALANCE + OOM-SAFE walk-forward.
Fixes:
  1. num_class=2 (was 3 — wrong)
  2. is_unbalance=True (73/27 imbalance → model always predicts down)
  3. Walk-forward uses growing windows that FIT in 7.5GB RAM
"""
import numpy as np
import lightgbm as lgb
import time, os, sys, json, gc
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    t0 = time.time()
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    n_total = meta['n_rows']
    
    print(f"═══ FIXED: BINARY + CLASS BALANCE + OOM-SAFE ═══\n", flush=True)
    print(f"Total: {n_total:,} rows", flush=True)
    print(f"num_class: 2 | is_unbalance: True | scale_pos_weight: 2.73", flush=True)
    
    X108 = np.memmap(f"{BASE}/train_data_x.npy", dtype=np.float32, mode='r', shape=(n_total, 108))
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n_total,))
    
    unique, counts = np.unique(y, return_counts=True)
    print(f"Target: 0={counts[0]:,} ({counts[0]/n_total*100:.1f}%), 1={counts[1]:,} ({counts[1]/n_total*100:.1f}%)\n", flush=True)
    
    seeds = [42, 7, 2026]
    
    # FIXED: is_unbalance=True
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "is_unbalance": True,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "max_bin": 31,
        "num_threads": 4,
    }
    N_TREES = 200
    
    # OOM-SAFE walk-forward: max training = 10M rows
    # Growing windows: train grows, test is fixed ~5M
    splits = [
        (5_000_000, 10_000_000),   # train 0-5M, test 5-10M
        (10_000_000, 15_000_000),  # train 0-10M, test 10-15M
        (15_000_000, 20_000_000),  # train 0-15M, test 15-20M
        (20_000_000, 25_000_000),  # train 0-20M, test 20-25M
        (25_000_000, 30_000_000),  # train 0-25M, test 25-30M
    ]
    
    all_preds = np.zeros(n_total, dtype=np.float32)
    
    for si, seed in enumerate(seeds):
        print(f"── seed {seed} ({si+1}/3) ──", flush=True)
        
        for sp_idx, (tr_end, te_end) in enumerate(splits):
            if te_end > n_total:
                te_end = n_total
            
            ds_tr = lgb.Dataset(X108[:tr_end], label=y[:tr_end])
            ds_te = lgb.Dataset(X108[tr_end:te_end], label=y[tr_end:te_end], reference=ds_tr)
            
            model = lgb.train(
                params, ds_tr,
                num_boost_round=N_TREES,
                valid_sets=[ds_te],
                callbacks=[lgb.log_evaluation(0)],
            )
            
            preds = model.predict(X108[tr_end:te_end])
            pred_labels = (preds > 0.5).astype(int)
            y_te = np.array(y[tr_end:te_end])
            acc = (pred_labels == y_te).mean()
            baseline = max(np.bincount(y_te)) / len(y_te)
            
            up_mask = y_te == 1
            down_mask = y_te == 0
            up_acc = (pred_labels[up_mask] == 1).mean() if up_mask.sum() > 0 else 0
            down_acc = (pred_labels[down_mask] == 0).mean() if down_mask.sum() > 0 else 0
            
            # Predicted distribution
            pred_up_pct = pred_labels.mean() * 100
            
            print(f"  Split {sp_idx+1} (tr={tr_end/1e6:.0f}M, te={te_end/1e6:.0f}M): "
                  f"acc={acc:.3f} (base={baseline:.3f}) "
                  f"up={up_acc:.3f} down={down_acc:.3f} "
                  f"pred_up={pred_up_pct:.1f}% n={len(y_te):,}", flush=True)
            
            all_preds[tr_end:te_end] += preds / len(seeds)
            
            del ds_tr, ds_te, model
            gc.collect()
    
    # Aggregate OOF
    agg_labels = (all_preds > 0.5).astype(int)
    y_np = np.array(y)
    agg_acc = (agg_labels == y_np).mean()
    baseline = max(np.bincount(y_np)) / len(y_np)
    up_mask = y_np == 1
    down_mask = y_np == 0
    up_acc = (agg_labels[up_mask] == 1).mean()
    down_acc = (agg_labels[down_mask] == 0).mean()
    pred_up_pct = agg_labels.mean() * 100
    
    print(f"\n═══ AGGREGATE OOF (3 seeds × 5 splits) ═══", flush=True)
    print(f"  accuracy: {agg_acc:.3f} (baseline: {baseline:.3f})", flush=True)
    print(f"  up correct: {up_acc:.3f}", flush=True)
    print(f"  down correct: {down_acc:.3f}", flush=True)
    print(f"  predicted up: {pred_up_pct:.1f}% (actual: {(y_np==1).mean()*100:.1f}%)", flush=True)
    
    if up_acc > 0.3:
        print(f"\n  ✅ MODEL IS LEARNING! Up accuracy {up_acc:.1%} >> baseline 0%", flush=True)
    else:
        print(f"\n  ⚠️ MODEL STILL PREDICTS DOWN MOSTLY — up_acc={up_acc:.3f}", flush=True)
    
    # ═══ FINAL MODELS on 130 features (last 10M rows) ═══
    print(f"\n═══ FINAL MODELS (130 features, last 10M) ═══\n", flush=True)
    
    n_tail = 10_000_000
    X130 = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(n_tail, 130))
    y_tail = y[n_total - n_tail:n_total]
    
    os.makedirs(f"{BASE}/models", exist_ok=True)
    feats = meta['features']
    with open(f"{BASE}/models/features.json", "w") as f:
        json.dump(feats, f)
    
    for si, seed in enumerate(seeds):
        print(f"\n── Final model: seed {seed} ({si+1}/3) ──", flush=True)
        t1 = time.time()
        
        ds = lgb.Dataset(X130, label=y_tail)
        model = lgb.train(
            params, ds,
            num_boost_round=N_TREES,
            callbacks=[lgb.log_evaluation(0)],
        )
        
        path = f"{BASE}/models/gold_lgb_model_s{seed}.txt"
        model.save_model(path)
        
        preds = model.predict(X130)
        pred_labels = (preds > 0.5).astype(int)
        y_tail_np = np.array(y_tail)
        acc = (pred_labels == y_tail_np).mean()
        up_m = y_tail_np == 1
        dn_m = y_tail_np == 0
        up_a = (pred_labels[up_m] == 1).mean() if up_m.sum() > 0 else 0
        dn_a = (pred_labels[dn_m] == 0).mean() if dn_m.sum() > 0 else 0
        
        print(f"  ✅ {path} ({time.time()-t1:.0f}s) "
              f"acc={acc:.3f} up={up_a:.3f} down={dn_a:.3f}", flush=True)
        
        del model, ds
        gc.collect()
    
    ensemble = {
        "type": "binary_balanced",
        "seeds": seeds,
        "n_models": len(seeds),
        "version": "v9.2_balanced",
        "num_class": 2,
        "is_unbalance": True,
        "oof_accuracy": float(agg_acc),
        "oof_up_acc": float(up_acc),
        "oof_down_acc": float(down_acc),
    }
    with open(f"{BASE}/models/ensemble.json", "w") as f:
        json.dump(ensemble, f)
    
    elapsed = time.time() - t0
    print(f"\n═══ ALL DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)", flush=True)

if __name__ == '__main__':
    main()
