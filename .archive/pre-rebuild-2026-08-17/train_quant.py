#!/usr/bin/env python3
"""train_quant.py — Train LightGBM on existing quant features. No feature computation."""
import numpy as np, lightgbm as lgb, json, os, time, gc
BASE = os.path.dirname(os.path.abspath(__file__))

t0 = time.time()
print("═══ TRAINING ON QUANT FEATURES ═══\n")

meta = json.load(open(f"{BASE}/quant_features_meta.json"))
NAMES = meta['feature_names']
n_feat = meta['n_features']
print(f"Features: {n_feat} | Names: {NAMES[:5]}...{NAMES[-5:]}")

X = np.memmap(f"{BASE}/quant_features_116.npy", dtype=np.float32, mode='r', shape=(meta['n_rows'], n_feat))
n = meta['n_rows']

y_full = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(json.load(open(f"{BASE}/train_data_meta.json"))['n_rows'],))
y = y_full[:n].copy()

print(f"Data: {n:,} rows × {n_feat} features")

# NaN fill in chunks
for ci in range(0, n, 500_000):
    end = min(ci+500_000, n)
    chunk = X[ci:end]
    m = np.isnan(chunk)
    if m.any(): chunk[m] = 0.0
gc.collect()

# Split: last 30% test, train on 2M before that
split = int(n * 0.7)
train_start = max(0, split - 2_000_000)
X_tr = X[train_start:split]; X_te = X[split:]
y_tr = y[train_start:split]; y_te = y[split:]
print(f"Train: {len(y_tr):,} | Test: {len(y_te):,}")

params = {
    'objective':'binary', 'metric':'auc',
    'num_leaves':31, 'max_depth':8, 'learning_rate':0.05,
    'feature_fraction':0.8, 'bagging_fraction':0.8, 'bagging_freq':5,
    'reg_alpha':0.1, 'reg_lambda':1.0, 'min_child_samples':100, 'verbose':-1,
}

results = []
for seed in [42, 7, 2026]:
    print(f"\n  Seed {seed}...")
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=NAMES)
    dval = lgb.Dataset(X_te, label=y_te, feature_name=NAMES, reference=dtrain)
    
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    
    preds = model.predict(X_te)
    pred_bin = (preds > 0.5).astype(int)
    acc = (pred_bin == y_te).mean()
    pred_up_rate = pred_bin.mean()
    actual_up_rate = y_te.mean()
    
    # Check if model learns anything
    # AUC via simple computation
    sorted_idx = np.argsort(preds)
    sorted_y = y_te[sorted_idx]
    n_pos = y_te.sum(); n_neg = len(y_te) - n_pos
    tpr = np.cumsum(sorted_y[::-1])[:n_pos] / max(n_pos, 1)
    fpr = np.cumsum(1-sorted_y[::-1])[:n_pos] / max(n_neg, 1)
    auc = np.trapezoid(tpr, fpr) if len(tpr) > 1 else 0.5
    
    print(f"    Acc: {acc:.4f} | AUC: {auc:.4f}")
    print(f"    Pred UP: {pred_up_rate:.3f} | Actual UP: {actual_up_rate:.3f}")
    print(f"    Trees: {model.num_trees()}")
    
    imp = model.feature_importance(importance_type='gain')
    top10 = np.argsort(imp)[-10:][::-1]
    print(f"    Top 10 features:")
    for idx in top10:
        print(f"      {NAMES[idx]:25s} gain={imp[idx]:.0f}")
    
    model.save_model(f"{BASE}/models/quant_lgb_s{seed}.txt")
    results.append({'seed': seed, 'acc': float(acc), 'auc': float(auc), 'trees': model.num_trees()})

ensemble = {
    'models': ['quant_lgb_s42', 'quant_lgb_s7', 'quant_lgb_s2026'],
    'base_tf': 'm5', 'n_features': n_feat,
    'feature_names': NAMES, 'source': 'quant_116',
}
with open(f"{BASE}/models/quant_ensemble.json", 'w') as f:
    json.dump(ensemble, f, indent=2)

total = time.time() - t0
print(f"\n═══ DONE: {total:.0f}s ({total/60:.1f}min) ═══")
for r in results:
    print(f"  Seed {r['seed']}: acc={r['acc']:.4f} auc={r['auc']:.4f} trees={r['trees']}")
