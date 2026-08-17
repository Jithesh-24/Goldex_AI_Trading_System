#!/usr/bin/env python3
"""
train_proper.py — Retrain with correct Renaissance features + proper regularization.
Addresses: overfitting, dead features, class imbalance, feature drift.
"""
import numpy as np
import lightgbm as lgb
import time, os, json, gc
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = f"{BASE}/models"

SEEDS = [42, 7, 2026]
N_TRAIN = 10_000_000  # last 10M rows (RAM safe)
N_SPLITS = 4

def load_data():
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    n = meta['n_rows']
    nf = meta['n_features']
    feats = meta['features']
    
    start = max(0, n - N_TRAIN)
    n_rows = min(N_TRAIN, n)
    
    nf = meta['n_features']  # 125
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(9999963, nf))
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n,))
    
    # Use last 10M rows
    X = X[-n_rows:]
    y = y[start:start + n_rows]
    
    return X, y, feats, n_rows

def walk_forward(X, y, feats, seed):
    """4-split walk-forward with proper regularization."""
    n = len(y)
    split_size = n // N_SPLITS
    
    oof_preds = np.zeros(n, dtype=np.float32)
    oof_labels = np.zeros(n, dtype=np.int8)
    oof_mask = np.zeros(n, dtype=bool)
    
    for s in range(N_SPLITS):
        tr_end = (s + 1) * split_size
        te_end = min(tr_end + split_size, n)
        
        if te_end <= tr_end:
            break
        
        X_tr = np.array(X[:tr_end])
        y_tr = y[:tr_end]
        X_te = np.array(X[tr_end:te_end])
        y_te = y[tr_end:te_end]
        
        # LightGBM with PROPER REGULARIZATION
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,          # REDUCED from 63
            'max_depth': 8,            # ADDED max_depth limit
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_samples': 100,  # ADDED min child
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,          # L1 regularization
            'reg_lambda': 1.0,         # L2 regularization
            'is_unbalance': True,
            'seed': seed,
            'verbose': -1,
            'n_jobs': 4,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        preds = model.predict_proba(X_te)[:, 1]
        oof_preds[tr_end:te_end] = preds
        oof_labels[tr_end:te_end] = y_te
        oof_mask[tr_end:te_end] = True
        
        # Stats
        pred_binary = (preds > 0.5).astype(int)
        acc = (pred_binary == y_te).mean()
        up_mask = y_te == 1
        down_mask = y_te == 0
        up_acc = (pred_binary[up_mask] == y_te[up_mask]).mean() if up_mask.sum() > 0 else 0
        down_acc = (pred_binary[down_mask] == y_te[down_mask]).mean() if down_mask.sum() > 0 else 0
        pred_up_pct = pred_binary.mean()
        
        print(f"  Split {s+1}: acc={acc:.3f} up={up_acc:.3f} down={down_acc:.3f} pred_up={pred_up_pct:.1%}")
        
        del X_tr, y_tr, X_te, y_te
        gc.collect()
    
    # Aggregate OOF
    mask = oof_mask
    pred_binary = (oof_preds[mask] > 0.5).astype(int)
    labels = oof_labels[mask]
    
    acc = (pred_binary == labels).mean()
    up_mask = labels == 1
    down_mask = labels == 0
    up_acc = (pred_binary[up_mask] == labels[up_mask]).mean() if up_mask.sum() > 0 else 0
    down_acc = (pred_binary[down_mask] == labels[down_mask]).mean() if down_mask.sum() > 0 else 0
    pred_up_pct = pred_binary.mean()
    actual_up_pct = labels.mean()
    
    return {
        'seed': seed,
        'accuracy': float(acc),
        'up_accuracy': float(up_acc),
        'down_accuracy': float(down_acc),
        'predicted_up_pct': float(pred_up_pct),
        'actual_up_pct': float(actual_up_pct),
        'n_samples': int(mask.sum()),
    }

def train_final(X, y, seed):
    """Train final model on last 10M rows with proper regularization."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'is_unbalance': True,
        'seed': seed,
        'verbose': -1,
        'n_jobs': 4,
    }
    
    X_arr = np.array(X)
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_arr, y)
    
    # Save
    path = f"{MODELS}/gold_lgb_model_s{seed}.txt"
    model.booster_.save_model(path)
    
    # Stats
    preds = model.predict_proba(X_arr)[:, 1]
    pred_binary = (preds > 0.5).astype(int)
    acc = (pred_binary == y).mean()
    up_mask = y == 1
    down_mask = y == 0
    up_acc = (pred_binary[up_mask] == y[up_mask]).mean() if up_mask.sum() > 0 else 0
    down_acc = (pred_binary[down_mask] == y[down_mask]).mean() if down_mask.sum() > 0 else 0
    pred_up_pct = pred_binary.mean()
    
    del X_arr
    gc.collect()
    
    return path, {
        'accuracy': float(acc),
        'up_accuracy': float(up_acc),
        'down_accuracy': float(down_acc),
        'predicted_up_pct': float(pred_up_pct),
        'n_trees': model.n_estimators,
        'num_leaves': 31,
        'max_depth': 8,
    }

def main():
    t0 = time.time()
    print("═══ TRAIN PROPER (130 feat, binary, REGULARIZED) ═══\n")
    
    X, y, feats, n_rows = load_data()
    print(f"Data: {n_rows:,} rows × {len(feats)} features")
    print(f"Target: 0={int((y==0).sum()):,} ({(y==0).mean():.1%}), 1={int((y==1).sum()):,} ({(y==1).mean():.1%})\n")
    
    # Walk-forward validation
    print("═══ WALK-FORWARD (4 splits × 3 seeds) ═══")
    all_results = []
    for seed in SEEDS:
        print(f"\n── seed {seed} ──")
        result = walk_forward(X, y, feats, seed)
        all_results.append(result)
        print(f"  OOF: acc={result['accuracy']:.3f} up={result['up_accuracy']:.3f} down={result['down_accuracy']:.3f} pred_up={result['predicted_up_pct']:.1%}")
    
    # Aggregate
    agg = {
        'accuracy': np.mean([r['accuracy'] for r in all_results]),
        'up_accuracy': np.mean([r['up_accuracy'] for r in all_results]),
        'down_accuracy': np.mean([r['down_accuracy'] for r in all_results]),
        'predicted_up_pct': np.mean([r['predicted_up_pct'] for r in all_results]),
        'actual_up_pct': all_results[0]['actual_up_pct'],
    }
    
    print(f"\n═══ AGGREGATE OOF ═══")
    print(f"  accuracy: {agg['accuracy']:.3f}")
    print(f"  up accuracy: {agg['up_accuracy']:.3f}")
    print(f"  down accuracy: {agg['down_accuracy']:.3f}")
    print(f"  predicted up: {agg['predicted_up_pct']:.1%} (actual: {agg['actual_up_pct']:.1%})")
    
    # Train final models
    print(f"\n═══ TRAINING FINAL MODELS ═══")
    final_stats = []
    for seed in SEEDS:
        t1 = time.time()
        print(f"\n── seed {seed} ──")
        path, stats = train_final(X, y, seed)
        elapsed = time.time() - t1
        print(f"  ✅ {path} ({elapsed:.0f}s) acc={stats['accuracy']:.3f} up={stats['up_accuracy']:.3f}")
        final_stats.append(stats)
    
    # Save ensemble config
    ensemble = {
        'type': 'binary_balanced',
        'version': 'v10.0_regularized',
        'seeds': SEEDS,
        'num_class': 2,
        'is_unbalance': True,
        'n_features': 125,
        'base_tf': 'm5',
        'models': [f'gold_lgb_model_s{s}.txt' for s in SEEDS],
        'regularization': {
            'num_leaves': 31,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 100,
        },
        'oof_results': agg,
        'trained': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(f"{MODELS}/ensemble.json", 'w') as f:
        json.dump(ensemble, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")
    print(f"Ensemble: {MODELS}/ensemble.json")
    print(f"Models: 3 seeds × 200 trees × 130 features")

if __name__ == '__main__':
    main()
