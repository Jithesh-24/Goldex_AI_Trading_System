#!/usr/bin/env python3
"""train_final_binary.py — Train 3 final models (130feat, binary, balanced) with DIFFERENT seeds."""
import numpy as np
import lightgbm as lgb
import time, os, json, gc
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    t0 = time.time()
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    n_total = meta['n_rows']
    feats = meta['features']
    nf = len(feats)
    
    print(f"═══ FINAL MODELS: 130 feat, binary, balanced ═══\n", flush=True)
    
    # Find actual rows in 130-feature mmap
    fsize = os.path.getsize(f"{BASE}/train_data_x_130.npy")
    n_tail = fsize // (4 * nf)  # float32 = 4 bytes
    print(f"130-feature mmap: {n_tail:,} rows × {nf} features", flush=True)
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(n_tail, nf))
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n_total,))
    y_tail = y[n_total - n_tail:n_total].astype(np.int32)
    
    unique, counts = np.unique(y_tail, return_counts=True)
    print(f"Target: 0={counts[0]:,} ({counts[0]/n_tail*100:.1f}%), 1={counts[1]:,} ({counts[1]/n_tail*100:.1f}%)\n", flush=True)
    
    os.makedirs(f"{BASE}/models", exist_ok=True)
    with open(f"{BASE}/models/features.json", "w") as f:
        json.dump(feats, f)
    
    seeds = [42, 7, 2026]
    
    for si, seed in enumerate(seeds):
        print(f"── seed {seed} ({si+1}/3) ──", flush=True)
        t1 = time.time()
        
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
            "seed": seed,  # Different seed per model!
        }
        
        ds = lgb.Dataset(X, label=y_tail)
        model = lgb.train(
            params, ds,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)],
        )
        
        path = f"{BASE}/models/gold_lgb_model_s{seed}.txt"
        model.save_model(path)
        
        preds = model.predict(X)
        pred_labels = (preds > 0.5).astype(int)
        y_np = np.array(y_tail)
        acc = (pred_labels == y_np).mean()
        up_m = y_np == 1
        dn_m = y_np == 0
        up_a = (pred_labels[up_m] == 1).mean() if up_m.sum() > 0 else 0
        dn_a = (pred_labels[dn_m] == 0).mean() if dn_m.sum() > 0 else 0
        pred_up = pred_labels.mean() * 100
        
        print(f"  ✅ {path} ({time.time()-t1:.0f}s) "
              f"acc={acc:.3f} up={up_a:.3f} down={dn_a:.3f} pred_up={pred_up:.1f}%", flush=True)
        
        del model, ds
        gc.collect()
    
    ensemble = {
        "type": "binary_balanced",
        "version": "v9.2_fixed",
        "seeds": seeds,
        "num_class": 2,
        "is_unbalance": True,
        "n_features": nf,
    }
    with open(f"{BASE}/models/ensemble.json", "w") as f:
        json.dump(ensemble, f)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)", flush=True)

if __name__ == '__main__':
    main()
