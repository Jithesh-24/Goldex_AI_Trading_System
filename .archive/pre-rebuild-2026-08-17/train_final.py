#!/usr/bin/env python3
"""
train_final.py — OOM-safe, zero-feature-aware training.
Uses memmap directly (no copies), removes dead features, proper regularization.
"""
import numpy as np
import lightgbm as lgb
import time, os, json, gc
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = f"{BASE}/models"
SEEDS = [42, 7, 2026]
N_TRAIN = 10_000_000

def load_data():
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    n = meta['n_rows']
    feats = meta['features']
    
    # Find DEAD features (all zeros in last 100K rows)
    X_check = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(9999963, nf))
    dead = []
    for i in range(nf):
        col = X_check[:100000, i]
        if np.all(col == 0) or col.std() < 0.0001:
            dead.append(i)
    del X_check
    
    # Live features = non-dead
    live_idx = [i for i in range(nf) if i not in dead]
    live_feats = [feats[i] for i in live_idx]
    
    print(f"Total features: {nf}")
    print(f"Dead features: {len(dead)} → removed")
    print(f"Live features: {len(live_idx)} → used")
    print(f"Dead: {[feats[i] for i in dead]}")
    
    mmap_rows = 9999963  # actual mmap size
    start = max(0, mmap_rows - N_TRAIN)
    n_rows = mmap_rows - start  # always matches mmap
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(mmap_rows, nf))
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n,))
    
    return X, y, live_idx, live_feats, n_rows, len(live_idx)

def main():
    t0 = time.time()
    print("═══ TRAIN FINAL (OOM-safe, dead-feature-aware) ═══\n")
    
    X_full, y_full, live_idx, live_feats, n_rows, n_live = load_data()
    start = max(0, 9999963 - N_TRAIN)
    
    print(f"\nData: {n_rows:,} rows × {n_live} live features")
    print(f"Target: 0={int((y_full[start:start+n_rows]==0).sum()):,} ({(y_full[start:start+n_rows]==0).mean():.1%})")
    print(f"        1={int((y_full[start:start+n_rows]==1).sum()):,} ({(y_full[start:start+n_rows]==1).mean():.1%})\n")
    
    split_size = n_rows // 4
    
    # Extract live features into a NEW small memmap (OOM-safe)
    live_file = f"{BASE}/train_data_live.npy"
    print(f"Extracting {n_live} live features to temp file...", flush=True)
    X_live = np.memmap(live_file, dtype=np.float32, mode='w+', shape=(n_rows, n_live))
    
    CHUNK = 100_000
    for chunk_start in range(0, n_rows, CHUNK):
        end = min(chunk_start + CHUNK, n_rows)
        # Copy from full matrix (live features only)
        actual_len = end - chunk_start
        for j, fi in enumerate(live_idx):
            X_live[chunk_start:end, j] = X_full[start + chunk_start:start + chunk_start + actual_len, fi]
        if (chunk_start // CHUNK) % 10 == 0:
            print(f"  {chunk_start:,}/{n_rows:,}", flush=True)
    
    X_live.flush()
    del X_full
    gc.collect()
    print(f"  ✅ Live features extracted ({time.time()-t0:.0f}s)\n")
    
    fname = [f"f{i}" for i in range(n_live)]
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'is_unbalance': True,
        'verbose': -1,
        'num_threads': 4,
    }
    
    # Walk-forward
    print("═══ WALK-FORWARD (4 splits × 3 seeds) ═══")
    all_results = []
    
    for seed in SEEDS:
        print(f"\n── seed {seed} ──")
        split_accs = []
        
        for s in range(4):
            tr_end = (s + 1) * split_size
            te_end = min(tr_end + split_size, n_rows)
            if te_end <= tr_end:
                break
            
            # Use memmap slices — LightGBM can read from memmap directly
            dtrain = lgb.Dataset(X_live[:tr_end], label=y_full[start:start+tr_end], feature_name=fname)
            dval = lgb.Dataset(X_live[tr_end:te_end], label=y_full[start+tr_end:start+te_end], feature_name=fname, reference=dtrain)
            
            model = lgb.train(
                {**params, 'seed': seed},
                dtrain,
                num_boost_round=200,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            
            preds = model.predict(X_live[tr_end:te_end])
            y_val = y_full[start+tr_end:start+te_end]
            pred_bin = (preds > 0.5).astype(int)
            acc = (pred_bin == y_val).mean()
            up_mask = y_val == 1
            down_mask = y_val == 0
            up_acc = (pred_bin[up_mask] == y_val[up_mask]).mean() if up_mask.sum() > 0 else 0
            down_acc = (pred_bin[down_mask] == y_val[down_mask]).mean() if down_mask.sum() > 0 else 0
            pred_up = pred_bin.mean()
            
            print(f"  Split {s+1}: acc={acc:.3f} up={up_acc:.3f} down={down_acc:.3f} pred_up={pred_up:.1%} trees={model.num_trees()}")
            split_accs.append(acc)
            
            del dtrain, dval, model
            gc.collect()
        
        avg_acc = np.mean(split_accs)
        print(f"  Average acc: {avg_acc:.3f}")
        all_results.append(avg_acc)
    
    # Aggregate
    print(f"\n═══ OOF AGGREGATE ═══")
    print(f"  Accuracy: {np.mean(all_results):.3f}")
    
    # Train final models
    print(f"\n═══ TRAINING FINAL MODELS ═══")
    for seed in SEEDS:
        t1 = time.time()
        print(f"\n── seed {seed} ──")
        
        dtrain = lgb.Dataset(X_live[:n_rows], label=y_full[start:start+n_rows], feature_name=fname)
        
        model = lgb.train(
            {**params, 'seed': seed},
            dtrain,
            num_boost_round=200,
        )
        
        path = f"{MODELS}/gold_lgb_model_s{seed}.txt"
        model.save_model(path)
        
        preds = model.predict(X_live[:n_rows])
        y_all = y_full[start:start+n_rows]
        pred_bin = (preds > 0.5).astype(int)
        acc = (pred_bin == y_all).mean()
        up_mask = y_all == 1
        down_mask = y_all == 0
        up_acc = (pred_bin[up_mask] == y_all[up_mask]).mean() if up_mask.sum() > 0 else 0
        down_acc = (pred_bin[down_mask] == y_all[down_mask]).mean() if down_mask.sum() > 0 else 0
        pred_up = pred_bin.mean()
        
        elapsed = time.time() - t1
        print(f"  ✅ {path} ({elapsed:.0f}s) acc={acc:.3f} up={up_acc:.3f} down={down_acc:.3f} pred_up={pred_up:.1%}")
        
        del dtrain, model
        gc.collect()
    
    # Save ensemble
    ensemble = {
        'type': 'binary_balanced',
        'version': 'v10.1_clean',
        'seeds': SEEDS,
        'num_class': 2,
        'is_unbalance': True,
        'n_features': n_live,
        'n_features_total': len(feats) if 'feats' in dir() else 125,
        'base_tf': 'm5',
        'models': [f'gold_lgb_model_s{s}.txt' for s in SEEDS],
        'live_features': live_feats,
        'dead_features_removed': len(dead) if 'dead' in dir() else 0,
        'regularization': {'num_leaves': 31, 'max_depth': 8, 'lambda_l1': 0.1, 'lambda_l2': 1.0},
        'oof_accuracy': float(np.mean(all_results)),
        'trained': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(f"{MODELS}/ensemble.json", 'w') as f:
        json.dump(ensemble, f, indent=2)
    
    # Also save feature map for engine
    feature_map = {'live_features': live_feats, 'live_indices': live_idx, 'dead_indices': dead if 'dead' in dir() else []}
    with open(f"{BASE}/models/feature_map.json", 'w') as f:
        json.dump(feature_map, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")
    print(f"Ensemble: {MODELS}/ensemble.json")
    print(f"Feature map: {MODELS}/feature_map.json")
    
    # Cleanup temp file
    os.remove(live_file)

if __name__ == '__main__':
    main()
