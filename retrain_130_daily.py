#!/usr/bin/env python3
"""
retrain_130_daily.py — Daily 130-feature retrain (runs AFTER EOD loop).

The EOD loop handles:
  - Data refresh (merge_seed, build_m5_matrix, merge_live_outcomes)
  - Direction model, calibration, specialists (108 features)

This script handles:
  - Retrain MAIN model on 130 features (108 base + 22 Renaissance)
  - Uses the SAME mmap files (train_data_x_130.npy, train_data_y.npy)
  - Overwrites gold_lgb_model_s*.txt with new 130-feature models

Usage:
  python retrain_130_daily.py

Dependencies:
  - train_data_x_130.npy must exist (created by add_renaissance_to_mmap.py)
  - train_data_y.npy must exist (created by csv_to_mmap.py)
  - features.json must list 130 features
"""
import numpy as np
import lightgbm as lgb
import time, os, sys, json, gc
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = f"{BASE}/models"

def main():
    t0 = time.time()
    print("═══ DAILY 130-FEATURE RETRAIN ═══", flush=True)
    
    # Load metadata
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    n_total = meta['n_rows']
    n_features = meta.get('n_features', 108)
    
    # Check for 130-feature mmap
    mmap_130 = f"{BASE}/train_data_x_130.npy"
    if not os.path.exists(mmap_130):
        print(f"❌ {mmap_130} not found — run add_renaissance_to_mmap.py first")
        sys.exit(1)
    
    mmap_130_size = os.path.getsize(mmap_130)
    n_rows_130 = mmap_130_size // (130 * 4)  # float32
    print(f"130-feature mmap: {n_rows_130:,} rows × 130 features", flush=True)
    
    # Use last 10M rows (OOM-safe on i5/7.5GB)
    n_tail = min(10_000_000, n_rows_130)
    
    # Load features
    feats = json.load(open(f"{MODEL_DIR}/features.json"))
    assert len(feats) == 130, f"Expected 130 features, got {len(feats)}"
    
    # Load data (memmap — no copy)
    print(f"Loading mmap...", flush=True)
    X = np.memmap(mmap_130, dtype=np.float32, mode='r', 
                  shape=(n_rows_130, 130))
    X_tail = X[n_rows_130 - n_tail:n_rows_130]
    
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r',
                  shape=(n_total,))
    y_tail = y[n_total - n_tail:n_total].astype(np.int32)
    
    print(f"Loaded: {n_tail:,} rows × 130 features", flush=True)
    print(f"Target: 0={np.sum(y_tail==0):,}, 1={np.sum(y_tail==1):,}", flush=True)
    gc.collect()
    
    # LightGBM params (binary, balanced, OOM-safe)
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.08,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_bin": 31,
        "num_threads": 4,
        "verbosity": -1,
        "is_unbalance": True,
    }
    
    seeds = [42, 7, 2026]
    models = []
    
    for i, seed in enumerate(seeds):
        print(f"\n── seed {seed} ({i+1}/{len(seeds)}) ──", flush=True)
        params["seed"] = seed
        
        ds = lgb.Dataset(X_tail, label=y_tail)
        
        t1 = time.time()
        model = lgb.train(
            params, ds,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)]
        )
        t2 = time.time()
        
        # Evaluate
        preds = model.predict(X_tail)
        pred_labels = (preds > 0.5).astype(int)
        acc = np.mean(pred_labels == y_tail)
        
        up_mask = y_tail == 1
        down_mask = y_tail == 0
        up_acc = np.mean(pred_labels[up_mask] == 1) if up_mask.any() else 0
        down_acc = np.mean(pred_labels[down_mask] == 0) if down_mask.any() else 0
        pred_up = np.mean(pred_labels)
        
        print(f"  acc={acc:.3f} up={up_acc:.3f} down={down_acc:.3f} pred_up={pred_up*100:.1f}% "
              f"({t2-t1:.0f}s)", flush=True)
        
        # Save
        path = f"{MODEL_DIR}/gold_lgb_model_s{seed}.txt"
        model.save_model(path)
        sz = os.path.getsize(path) / 1024
        print(f"  ✅ Saved: {path} ({sz:.0f}KB)", flush=True)
        
        models.append(model)
        del ds
        gc.collect()
    
    # Update ensemble.json
    ensemble = {
        "type": "binary_balanced",
        "version": "v9.3_daily_retrain",
        "seeds": seeds,
        "num_class": 2,
        "is_unbalance": True,
        "n_features": 130,
        "trained_at": time.time(),
        "n_rows": n_tail,
    }
    with open(f"{MODEL_DIR}/ensemble.json", "w") as f:
        json.dump(ensemble, f, indent=2)
    
    # Ensemble accuracy
    avg_preds = np.mean([m.predict(X_tail) for m in models], axis=0)
    avg_labels = (avg_preds > 0.5).astype(int)
    ens_acc = np.mean(avg_labels == y_tail)
    ens_up = np.mean(avg_labels[up_mask] == 1) if up_mask.any() else 0
    ens_down = np.mean(avg_labels[down_mask] == 0) if down_mask.any() else 0
    
    del X, X_tail, y, y_tail, models
    gc.collect()
    
    total = time.time() - t0
    print(f"\n═══ DONE ═══ ({total:.0f}s = {total/60:.1f}min)", flush=True)
    print(f"Ensemble: acc={ens_acc:.3f} up={ens_up:.3f} down={ens_down:.3f}", flush=True)
    print(f"Models: {len(seeds)} seeds × 130 features × 200 trees", flush=True)

if __name__ == "__main__":
    main()
