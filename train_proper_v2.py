#!/usr/bin/env python3
"""
train_proper_v2.py — Native LightGBM (no sklearn needed).
125 features, binary, is_unbalance=True, proper regularization.
"""
import numpy as np
import lightgbm as lgb
import time, os, json, gc
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = f"{BASE}/models"
SEEDS = [42, 7, 2026]
N_SPLITS = 4

def load_data():
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    n = meta['n_rows']
    start = max(0, n - 10_000_000)
    n_rows = min(10_000_000, n)
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(9999963, nf))
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n,))
    return X[-n_rows:], y[start:start+n_rows], meta['features'], n_rows, nf

def train_lgb(X_tr, y_tr, X_val, y_val, seed, n_features):
    """Train LightGBM with native API + proper regularization."""
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
        'seed': seed,
        'verbose': -1,
        'num_threads': 4,
    }
    
    fname = [f"f{i}" for i in range(n_features)]
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=fname)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=fname, reference=dtrain)
    
    model = lgb.train(
        params, dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    return model

def eval_model(model, X, y, n_features):
    """Evaluate model."""
    preds = model.predict(X)
    pred_binary = (preds > 0.5).astype(int)
    acc = (pred_binary == y).mean()
    up_mask = y == 1
    down_mask = y == 0
    up_acc = (pred_binary[up_mask] == y[up_mask]).mean() if up_mask.sum() > 0 else 0
    down_acc = (pred_binary[down_mask] == y[down_mask]).mean() if down_mask.sum() > 0 else 0
    pred_up_pct = pred_binary.mean()
    return acc, up_acc, down_acc, pred_up_pct

def main():
    t0 = time.time()
    print("═══ TRAIN PROPER V2 (125 feat, native LGB, REGULARIZED) ═══\n")
    
    X, y, feats, n_rows, nf = load_data()
    print(f"Data: {n_rows:,} rows × {nf} features")
    print(f"Target: 0={int((y==0).sum()):,} ({(y==0).mean():.1%}), 1={int((y==1).sum()):,} ({(y==1).mean():.1%})\n")
    
    # Walk-forward validation
    print("═══ WALK-FORWARD (4 splits × 3 seeds) ═══")
    split_size = n_rows // N_SPLITS
    
    all_results = []
    for seed in SEEDS:
        print(f"\n── seed {seed} ──")
        oof_accs, oof_ups, oof_downs, oof_pups = [], [], [], []
        
        for s in range(N_SPLITS):
            tr_end = (s + 1) * split_size
            te_end = min(tr_end + split_size, n_rows)
            if te_end <= tr_end: break
            
            X_tr = np.array(X[:tr_end])
            y_tr = y[:tr_end]
            X_val = np.array(X[tr_end:te_end])
            y_val = y[tr_end:te_end]
            
            model = train_lgb(X_tr, y_tr, X_val, y_val, seed, nf)
            acc, up_acc, down_acc, pred_up = eval_model(model, X_val, y_val, nf)
            
            print(f"  Split {s+1}: acc={acc:.3f} up={up_acc:.3f} down={down_acc:.3f} pred_up={pred_up:.1%}")
            oof_accs.append(acc); oof_ups.append(up_acc); oof_downs.append(down_acc); oof_pups.append(pred_up)
            
            del X_tr, y_tr, X_val, y_val, model
            gc.collect()
        
        avg = {'seed': seed, 'accuracy': np.mean(oof_accs), 'up_accuracy': np.mean(oof_ups),
               'down_accuracy': np.mean(oof_downs), 'predicted_up_pct': np.mean(oof_pups),
               'actual_up_pct': float(y.mean())}
        all_results.append(avg)
        print(f"  Average: acc={avg['accuracy']:.3f} up={avg['up_accuracy']:.3f} down={avg['down_accuracy']:.3f}")
    
    # Aggregate
    agg_acc = np.mean([r['accuracy'] for r in all_results])
    agg_up = np.mean([r['up_accuracy'] for r in all_results])
    agg_down = np.mean([r['down_accuracy'] for r in all_results])
    agg_pup = np.mean([r['predicted_up_pct'] for r in all_results])
    agg_act = all_results[0]['actual_up_pct']
    
    print(f"\n═══ AGGREGATE OOF ═══")
    print(f"  accuracy: {agg_acc:.3f}")
    print(f"  up accuracy: {agg_up:.3f}")
    print(f"  down accuracy: {agg_down:.3f}")
    print(f"  predicted up: {agg_pup:.1%} (actual: {agg_act:.1%})")
    
    # Train final models
    print(f"\n═══ TRAINING FINAL MODELS ═══")
    for seed in SEEDS:
        t1 = time.time()
        print(f"\n── seed {seed} ──")
        X_arr = np.array(X)
        model = train_lgb(X_arr, y, X_arr, y, seed, nf)  # train on all data
        path = f"{MODELS}/gold_lgb_model_s{seed}.txt"
        model.save_model(path)
        acc, up_acc, down_acc, pred_up = eval_model(model, X_arr, y, nf)
        elapsed = time.time() - t1
        print(f"  ✅ {path} ({elapsed:.0f}s) acc={acc:.3f} up={up_acc:.3f} down={down_acc:.3f}")
        del X_arr, model
        gc.collect()
    
    # Save ensemble
    ensemble = {
        'type': 'binary_balanced',
        'version': 'v10.0_regularized',
        'seeds': SEEDS,
        'num_class': 2,
        'is_unbalance': True,
        'n_features': nf,
        'base_tf': 'm5',
        'models': [f'gold_lgb_model_s{s}.txt' for s in SEEDS],
        'regularization': {'num_leaves': 31, 'max_depth': 8, 'lambda_l1': 0.1, 'lambda_l2': 1.0, 'min_child_samples': 100},
        'oof': {'accuracy': agg_acc, 'up_accuracy': agg_up, 'down_accuracy': agg_down, 'predicted_up_pct': agg_pup},
        'trained': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(f"{MODELS}/ensemble.json", 'w') as f:
        json.dump(ensemble, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")
    print(f"Ensemble: {MODELS}/ensemble.json")

if __name__ == '__main__':
    main()
