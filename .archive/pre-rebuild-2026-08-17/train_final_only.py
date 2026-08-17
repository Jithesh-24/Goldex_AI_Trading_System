#!/usr/bin/env python3
"""train_final_only.py — Train 3 final models on LAST 10M rows (OOM-safe)."""
import numpy as np
import lightgbm as lgb
import time, os, sys, gc, json

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = f"{BASE}/models"
SEEDS = [42, 7, 2026]
TAIL = 10_000_000  # last 10M rows (~3 years) — recency-weighted

sys.path.insert(0, BASE)

def lgb_params(seed):
    return {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.08, "num_leaves": 31, "max_depth": 6,
        "min_child_samples": 100, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 3,
        "verbose": -1, "num_threads": 4, "seed": seed,
    }

def main():
    print("═══ TRAINING 3 FINAL MODELS (last 10M rows) ═══\n", flush=True)
    t0 = time.time()
    
    # Load mmap
    print("Loading mmap...", flush=True)
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    n = meta['n_rows']
    nf = meta['n_features']
    feats = meta['features']
    X_full = np.memmap(f"{BASE}/train_data_x.npy", dtype=np.float32, mode='r', shape=(n, nf))
    y_full = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n,))
    times_full = np.memmap(f"{BASE}/train_data_t.npy", dtype='datetime64[s]', mode='r', shape=(n,))
    
    # Take last 10M rows
    start = max(0, n - TAIL)
    X = np.array(X_full[start:n])  # copy to RAM for training
    y = np.array(y_full[start:n])
    times = np.array(times_full[start:n])
    del X_full, y_full, times_full; gc.collect()
    
    # Recency weights
    ts = times.astype(np.int64)
    w = np.exp(-(ts.max() - ts) / (120 * 86400))
    w = w / w.mean()
    
    print(f"Loaded: {len(X):,} rows × {len(feats)} features (from row {start:,})", flush=True)
    print(f"RAM: {X.nbytes/1024**3:.1f}GB + y + w = ~{X.nbytes/1024**3 + 0.1:.1f}GB\n", flush=True)
    
    # Train 3 final models
    for i, s in enumerate(SEEDS):
        print(f"── Final model: seed {s} ({i+1}/3) ──", flush=True)
        t1 = time.time()
        
        final = lgb.train(lgb_params(s),
                          lgb.Dataset(X, label=y, weight=w, free_raw_data=True,
                                      params={"max_bin": 31, "num_threads": 4}),
                          num_boost_round=200)
        
        name = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        tmp = name + ".tmp"
        final.save_model(tmp)
        os.replace(tmp, name)
        
        elapsed = time.time() - t1
        print(f"  ✅ Saved: {name} ({elapsed:.0f}s)", flush=True)
        del final; gc.collect()
    
    # Save features.json
    with open(f"{MODEL_DIR}/features.json", "w") as f:
        json.dump(feats, f)
    
    # Save ensemble.json
    with open(f"{MODEL_DIR}/ensemble.json", "w") as f:
        json.dump({"type": "placement", "seeds": SEEDS,
                   "n_models": len(SEEDS), "version": "v9.0_renaissance",
                   "trained_rows": len(X), "trained_from": str(times[0]),
                   "trained_to": str(times[-1])}, f)
    
    print(f"\n═══ ALL 3 MODELS SAVED ═══ ({time.time()-t0:.0f}s total)", flush=True)
    for s in SEEDS:
        sz = os.path.getsize(f"{MODEL_DIR}/gold_lgb_model_s{s}.txt") / 1024
        print(f"  seed {s}: {sz:.0f} KB", flush=True)

if __name__ == '__main__':
    main()
