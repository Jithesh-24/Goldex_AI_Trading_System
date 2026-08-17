#!/usr/bin/env python3
"""train_final_130.py — Train 3 final models on 130 features (108 base + 22 Renaissance)."""
import numpy as np
import lightgbm as lgb
import time, os, sys, json, gc

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    t0 = time.time()
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    feats = meta['features']
    nf = len(feats)
    n_tail = meta.get('tail_n_rows', 10_000_000)
    
    print(f"═══ TRAINING 3 FINAL MODELS (130 features) ═══\n", flush=True)
    print(f"Features: {nf} ({nf-22} base + 22 Renaissance)", flush=True)
    
    # Load 130-feature mmap
    mmap_path = f"{BASE}/train_data_x_130.npy"
    print(f"Loading mmap: {mmap_path}", flush=True)
    X = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=(n_tail, nf))
    print(f"Loaded: {n_tail:,} rows × {nf} features ({time.time()-t0:.1f}s)", flush=True)
    
    # Load labels
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(meta['n_rows'],))
    y = y[meta['n_rows'] - n_tail:meta['n_rows']].astype(np.int32)
    
    print(f"Target balance: {np.bincount(y).tolist()}", flush=True)
    
    # Save features
    with open(f"{BASE}/models/features.json", "w") as f:
        json.dump(feats, f)
    
    os.makedirs(f"{BASE}/models", exist_ok=True)
    
    seeds = [42, 7, 2026]
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
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
    
    for si, seed in enumerate(seeds):
        print(f"\n── Final model: seed {seed} ({si+1}/3) ──", flush=True)
        t1 = time.time()
        
        ds = lgb.Dataset(X, label=y)
        model = lgb.train(
            params, ds,
            num_boost_round=N_TREES,
            callbacks=[lgb.log_evaluation(0)],
        )
        
        path = f"{BASE}/models/gold_lgb_model_s{seed}.txt"
        model.save_model(path)
        
        elapsed = time.time() - t1
        print(f"  ✅ Saved: {path} ({elapsed:.0f}s)", flush=True)
        
        # Free memory
        del model, ds
        gc.collect()
    
    # Save ensemble
    ensemble = {"type": "placement", "seeds": seeds, "n_models": len(seeds), "version": "v9.0_renaissance_130"}
    with open(f"{BASE}/models/ensemble.json", "w") as f:
        json.dump(ensemble, f)
    
    elapsed = time.time() - t0
    print(f"\n═══ ALL 3 MODELS SAVED ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)", flush=True)
    for s in seeds:
        sz = os.path.getsize(f"{BASE}/models/gold_lgb_model_s{s}.txt") / 1024
        print(f"  seed {s}: {sz:.0f}KB", flush=True)

if __name__ == '__main__':
    main()
