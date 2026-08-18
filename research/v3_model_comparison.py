"""
Phase 3B Part 9 -- model architecture comparison (CatBoost vs LightGBM vs
XGBoost) on the SAME survivor feature set (26 base + 17 candidates), same
purged walk-forward folds, same VAL_FRACTION early-stopping split. Compares
OOF discrimination, calibration, fold stability, training cost, inference
cost. Meta-stage only (that's where the deployed system's actual precision
lives, per Phase 1A finding #4) -- primary is a weak, near-50/50 stage
regardless of architecture (already established by every prior run this
session), so re-litigating primary across 3 libraries would not be
informative.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.v3_model_comparison
"""
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from learning.cv import PurgedWalkForwardCV
from learning.train import CATBOOST_KW, N_SPLITS, EMBARGO_BARS, VAL_FRACTION
from research.audit_edge import oof_run, build_meta, manual_log_loss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
DS = os.path.join(OUT, "v3_dataset")


def make_model(kind):
    if kind == "catboost":
        return CatBoostClassifier(**CATBOOST_KW)
    if kind == "lightgbm":
        return LGBMClassifier(max_depth=4, n_estimators=2000, learning_rate=0.02,
                               reg_lambda=15, random_state=42, verbosity=-1, n_jobs=-1)
    if kind == "xgboost":
        return XGBClassifier(max_depth=4, n_estimators=2000, learning_rate=0.02,
                              reg_lambda=15, random_state=42, verbosity=0, n_jobs=-1,
                              eval_metric="logloss", early_stopping_rounds=100)
    raise ValueError(kind)


def fit_predict(kind, X, y, train_pos, test_pos):
    cut = int(len(train_pos) * (1 - VAL_FRACTION))
    tr, va = train_pos[:cut], train_pos[cut:]
    model = make_model(kind)
    t0 = time.time()
    if kind == "catboost":
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
    elif kind == "lightgbm":
        import lightgbm as lgb
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    else:  # xgboost
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
    fit_time = time.time() - t0
    t0 = time.time()
    proba = model.predict_proba(X.iloc[test_pos])[:, 1]
    infer_time = time.time() - t0
    return proba, fit_time, infer_time, model


def run_model_oof(kind, X, y_bin, t0, t1):
    cv = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_bars=EMBARGO_BARS)
    n = len(X)
    oof_proba = np.full(n, np.nan)
    fold_metrics = []
    for fold, (train_pos, test_pos) in enumerate(cv.split(t0.to_numpy(), t1.to_numpy())):
        proba, fit_t, infer_t, model = fit_predict(kind, X, y_bin, train_pos, test_pos)
        pred = (proba >= 0.5).astype(np.int64)
        y_true = y_bin.iloc[test_pos].to_numpy()
        acc = float((pred == y_true).mean())
        ll = manual_log_loss(y_true, proba)
        oof_proba[test_pos] = proba
        fold_metrics.append({"fold": fold, "acc": acc, "logloss": ll,
                              "fit_time_s": fit_t, "infer_time_s": infer_t,
                              "n_test": len(test_pos)})
        print(f"  [{kind}] fold {fold}: acc={acc:.4f} logloss={ll:.4f} "
              f"fit={fit_t:.1f}s infer={infer_t*1000:.1f}ms")
    has_oof = np.isfinite(oof_proba)
    return oof_proba, has_oof, fold_metrics


def calibration_slope(proba, y):
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    a, b = 0.0, 1.0
    for _ in range(50):
        z = a + b * logit_p
        pr = 1 / (1 + np.exp(-z))
        w = np.clip(pr * (1 - pr), 1e-6, None)
        ga = np.sum(y - pr); gb = np.sum((y - pr) * logit_p)
        haa = -np.sum(w); hbb = -np.sum(w * logit_p ** 2); hab = -np.sum(w * logit_p)
        det = haa * hbb - hab ** 2
        if abs(det) < 1e-12:
            break
        da = (ga * hbb - gb * hab) / det; db = (gb * haa - ga * hab) / det
        a -= da; b -= db
    return float(a), float(b)


def main():
    t_start = time.time()
    with open(os.path.join(DS, "columns.json")) as f:
        cols = json.load(f)
    with open(os.path.join(OUT, "v3_feature_survivors.json")) as f:
        surv = json.load(f)
    use_cols = cols["base_cols"] + surv["survivors"]

    X_full = pd.DataFrame(np.load(os.path.join(DS, "X_v3.npy")), columns=cols["all_cols"])
    X = X_full[use_cols]
    y_bin = pd.Series(np.load(os.path.join(DS, "y_bin.npy")))
    t0 = pd.Series(np.load(os.path.join(DS, "t0.npy")))
    t1 = pd.Series(np.load(os.path.join(DS, "t1.npy")))
    t0_nz = np.load(os.path.join(DS, "t0_nz.npy"))
    close = np.load(os.path.join(DS, "close.npy"))
    high = np.load(os.path.join(DS, "high.npy"))
    low = np.load(os.path.join(DS, "low.npy"))
    vol_tb = np.load(os.path.join(DS, "vol_tb.npy"))

    print(f"== primary OOF (CatBoost only, reused as the shared side/meta-target generator) ==")
    prim = oof_run(X, y_bin, t0, t1, tag="modelcmp-primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    result = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "feature_cols": use_cols + ["assumed_side"], "models": {}}
    for kind in ["catboost", "lightgbm", "xgboost"]:
        print(f"\n== meta OOF: {kind} ==")
        oof_proba, has_oof_m, fold_metrics = run_model_oof(kind, X_meta, y_meta, t0_meta, t1_meta)
        p = oof_proba[has_oof_m]; y = y_meta.to_numpy()[has_oof_m]
        a, b = calibration_slope(p, y)
        mean_acc = float(np.mean([f["acc"] for f in fold_metrics]))
        mean_ll = float(np.mean([f["logloss"] for f in fold_metrics]))
        acc_std = float(np.std([f["acc"] for f in fold_metrics]))
        total_fit = float(np.sum([f["fit_time_s"] for f in fold_metrics]))
        mean_infer_ms = float(np.mean([f["infer_time_s"] for f in fold_metrics])) * 1000
        result["models"][kind] = {
            "fold_metrics": fold_metrics, "mean_acc": mean_acc, "mean_logloss": mean_ll,
            "fold_acc_std": acc_std, "calibration_intercept": a, "calibration_slope": b,
            "total_fit_time_s": total_fit, "mean_infer_time_ms": mean_infer_ms,
            "brier": float(np.mean((p - y) ** 2)),
        }
        print(f"  {kind}: mean_acc={mean_acc:.4f} fold_std={acc_std:.4f} logloss={mean_ll:.4f} "
              f"calib_slope={b:.3f} total_fit={total_fit:.1f}s infer/fold={mean_infer_ms:.1f}ms")

    out_path = os.path.join(OUT, "v3_model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
