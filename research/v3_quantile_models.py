"""
Phase 3B Part 10 -- distributional model research. Trains CatBoost quantile
regression (loss_function="Quantile:alpha=q") for MAE_R and MFE_R at
q in {0.5, 0.75, 0.9}, walk-forward OOF, using the survivor feature set +
assumed_side. Compares empirical coverage (fraction of actual outcomes at
or below the predicted quantile -- should equal q if well-calibrated)
against the SIMPLE baseline of Part 3's per-vol_state empirical quantile
(same q, but a single constant per state instead of a conditional
prediction). The question this answers: does a conditional model improve
coverage accuracy beyond simple state-conditioning, or is the extra
model complexity not earning its keep?

Does NOT deploy anything -- MAE/MFE quantile prediction stays research-only,
no SL/TP change.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.v3_quantile_models
"""
import json
import os
import time

import numba
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from learning.cv import PurgedWalkForwardCV
from learning.train import N_SPLITS, EMBARGO_BARS, VAL_FRACTION
from research.audit_edge import oof_run, build_meta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
DS = os.path.join(OUT, "v3_dataset")
QUANTILES = [0.5, 0.75, 0.9]


@numba.njit(cache=True)
def _mae_mfe_core(close, high, low, t0_idx, t1_idx, side, vol_at_t0):
    n = len(t0_idx)
    mae = np.empty(n, dtype=np.float64)
    mfe = np.empty(n, dtype=np.float64)
    for e in range(n):
        t0, t1, s = t0_idx[e], t1_idx[e], side[e]
        p0 = close[t0]
        worst, best = 0.0, 0.0
        for j in range(t0 + 1, t1 + 1):
            if s >= 0:
                fav, adv = (high[j] - p0) / p0, (low[j] - p0) / p0
            else:
                fav, adv = (p0 - low[j]) / p0, (p0 - high[j]) / p0
            if fav > best:
                best = fav
            if adv < worst:
                worst = adv
        v = vol_at_t0[e] if vol_at_t0[e] > 1e-9 else 1e-9
        mae[e] = -worst / v
        mfe[e] = best / v
    return mae, mfe


def fit_quantile(X, y, train_pos, q):
    cut = int(len(train_pos) * (1 - VAL_FRACTION))
    tr = train_pos[:cut]
    model = CatBoostRegressor(loss_function=f"Quantile:alpha={q}", depth=4, iterations=500,
                               learning_rate=0.05, l2_leaf_reg=15, random_seed=42, verbose=False,
                               thread_count=-1)
    model.fit(X.iloc[tr], y[tr])
    return model


def pinball_loss(y_true, y_pred, q):
    d = y_true - y_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


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

    print("== primary OOF (side generator) ==")
    prim = oof_run(X, y_bin, t0, t1, tag="quantile-primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    t0_meta_np = meta_labels.index.to_numpy()
    t1_meta_np = meta_labels["t1"].to_numpy()
    vol_at_meta = vol_tb[t0_nz][has_oof]

    mae_R, mfe_R = _mae_mfe_core(close, high, low, t0_meta_np, t1_meta_np, side, vol_at_meta)
    print(f"n_events={len(mae_R):,} mae_R mean={mae_R.mean():.3f} mfe_R mean={mfe_R.mean():.3f}")

    vol_daily = pd.Series(np.load(os.path.join(DS, "vol_tb.npy")))  # per-bar, reused for a simple tercile baseline
    ev_series = vol_tb
    lo_thr, hi_thr = np.nanpercentile(ev_series, [33.3, 66.7])
    vs_at_meta = np.where(vol_at_meta <= lo_thr, "low", np.where(vol_at_meta >= hi_thr, "high", "medium"))

    t0_s = pd.Series(t0_meta_np); t1_s = pd.Series(t1_meta_np)
    cv = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_bars=EMBARGO_BARS)

    results = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_events": int(len(mae_R)), "quantiles": QUANTILES, "targets": {}}

    for target_name, y_target in [("mae_R", mae_R), ("mfe_R", mfe_R)]:
        print(f"\n== target: {target_name} ==")
        target_result = {}
        for q in QUANTILES:
            oof_pred = np.full(len(y_target), np.nan)
            for fold, (train_pos, test_pos) in enumerate(cv.split(t0_s.to_numpy(), t1_s.to_numpy())):
                model = fit_quantile(X_meta, y_target, train_pos, q)
                oof_pred[test_pos] = model.predict(X_meta.iloc[test_pos])
            has_pred = np.isfinite(oof_pred)
            yp, yt = oof_pred[has_pred], y_target[has_pred]
            model_coverage = float((yt <= yp).mean())
            model_pinball = pinball_loss(yt, yp, q)

            vs_valid = vs_at_meta[has_pred]
            baseline_pred = np.zeros_like(yp)
            for vs in ("low", "medium", "high"):
                m = vs_valid == vs
                if m.sum() > 100:
                    baseline_pred[m] = np.quantile(yt[m], q)  # in-sample-by-state (simple baseline, not OOF -- optimistic on purpose, upper bound for the simple approach)
            baseline_coverage = float((yt <= baseline_pred).mean())
            baseline_pinball = pinball_loss(yt, baseline_pred, q)

            print(f"  q={q}: model_coverage={model_coverage:.3f} (target {q}) pinball={model_pinball:.4f} | "
                  f"simple_by_state_coverage={baseline_coverage:.3f} pinball={baseline_pinball:.4f}")
            target_result[str(q)] = {
                "model_coverage": model_coverage, "model_pinball": model_pinball,
                "simple_by_state_coverage": baseline_coverage, "simple_by_state_pinball": baseline_pinball,
                "model_coverage_error": abs(model_coverage - q), "baseline_coverage_error": abs(baseline_coverage - q),
            }
        results["targets"][target_name] = target_result

    out_path = os.path.join(OUT, "v3_quantile_models.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
