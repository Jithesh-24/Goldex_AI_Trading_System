"""
Phase 3B Part 2 steps 6/8 -- information test (mutual information) + full
118-feature walk-forward OOF (CatBoost) for native importance, permutation
importance, and fold-to-fold stability. Feeds research/v3_feature_selection.py.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.v3_importance_mi
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from research.audit_edge import oof_run, build_meta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
DS = os.path.join(OUT, "v3_dataset")


def load_dataset():
    with open(os.path.join(DS, "columns.json")) as f:
        cols = json.load(f)
    X = pd.DataFrame(np.load(os.path.join(DS, "X_v3.npy")), columns=cols["all_cols"])
    y_bin = pd.Series(np.load(os.path.join(DS, "y_bin.npy")))
    t0 = pd.Series(np.load(os.path.join(DS, "t0.npy")))
    t1 = pd.Series(np.load(os.path.join(DS, "t1.npy")))
    t0_nz = np.load(os.path.join(DS, "t0_nz.npy"))
    close = np.load(os.path.join(DS, "close.npy"))
    high = np.load(os.path.join(DS, "high.npy"))
    low = np.load(os.path.join(DS, "low.npy"))
    vol_tb = np.load(os.path.join(DS, "vol_tb.npy"))
    return X, y_bin, t0, t1, t0_nz, close, high, low, vol_tb, cols


def main():
    t_start = time.time()
    X, y_bin, t0, t1, t0_nz, close, high, low, vol_tb, cols = load_dataset()
    print(f"loaded {X.shape}")

    print("\n== full-118-feature primary OOF (CatBoost) ==")
    prim = oof_run(X, y_bin, t0, t1, tag="v3-primary-full", want_importance=True)
    print(f"primary mean acc: {np.mean([f['acc'] for f in prim['fold_metrics']]):.4f}")

    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    print("\n== full-118(+assumed_side)-feature meta OOF (CatBoost) ==")
    meta = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag="v3-meta-full", want_importance=True)
    print(f"meta mean acc: {np.mean([f['acc'] for f in meta['fold_metrics']]):.4f}")

    # ---- native CatBoost importance: mean + fold-to-fold CV (stability) ----
    def importance_stats(importances_list, cols_list):
        if not importances_list:
            return {}
        df_imp = pd.concat([pd.Series(d) for d in importances_list], axis=1)
        mean_imp = df_imp.mean(axis=1)
        std_imp = df_imp.std(axis=1)
        cv_imp = (std_imp / mean_imp.clip(lower=1e-9)).clip(upper=100)
        return {c: {"mean_importance": float(mean_imp.get(c, 0.0)),
                     "std_importance": float(std_imp.get(c, 0.0)),
                     "cv_importance": float(cv_imp.get(c, np.nan))}
                for c in cols_list}

    primary_importance = importance_stats(prim["importances"], list(X.columns))
    meta_importance = importance_stats(meta["importances"], list(X_meta.columns))

    # ---- mutual information (descriptive ranking only, not used inside OOF folds;
    # subsampled to 60k rows -- MI is a ranking aid, not a precision estimate, and
    # the kNN-based estimator is too slow at 300k rows x 118 features) ----
    rng = np.random.RandomState(42)
    print("\n== mutual information (primary target, 60k-row subsample) ==")
    sub_idx = rng.choice(len(X), size=min(60_000, len(X)), replace=False)
    Xf = X.iloc[sub_idx].fillna(X.median(numeric_only=True))
    mi_primary = mutual_info_classif(Xf, y_bin.iloc[sub_idx], discrete_features=False, random_state=42, n_neighbors=3)
    mi_primary_d = dict(zip(X.columns, mi_primary.tolist()))

    print("== mutual information (meta target, 60k-row subsample) ==")
    sub_idx_m = rng.choice(len(X_meta), size=min(60_000, len(X_meta)), replace=False)
    Xmf = X_meta.iloc[sub_idx_m].fillna(X_meta.median(numeric_only=True))
    mi_meta = mutual_info_classif(Xmf, y_meta.iloc[sub_idx_m], discrete_features=False, random_state=42, n_neighbors=3)
    mi_meta_d = dict(zip(X_meta.columns, mi_meta.tolist()))

    result = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_fold_metrics": prim["fold_metrics"], "meta_fold_metrics": meta["fold_metrics"],
        "primary_importance": primary_importance, "meta_importance": meta_importance,
        "mi_primary": mi_primary_d, "mi_meta": mi_meta_d,
        "columns": cols,
    }
    out_path = os.path.join(OUT, "v3_importance_mi.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("\n== top 20 candidate features by meta MI ==")
    cand_mi = {k: v for k, v in mi_meta_d.items() if k in cols["cand_cols"]}
    for k, v in sorted(cand_mi.items(), key=lambda kv: -kv[1])[:20]:
        pi = meta_importance.get(k, {})
        print(f"  {k}: MI={v:.4f} catboost_imp={pi.get('mean_importance', 0):.3f} "
              f"cv={pi.get('cv_importance', float('nan')):.2f}")
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
