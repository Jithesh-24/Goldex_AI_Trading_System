"""
Phase 2 -- build + validate the v2 artifact: 26-feature schema (spread and
tick_volume dropped from the PREDICTIVE matrix per the data-semantics fix,
spread stays available for execution-cost use elsewhere) + rolling Platt
calibration on top of the meta probability.

This does NOT touch models/ (the live v1 artifacts) or retrain_daily.py's
default behavior -- it trains into models/v2/ via core.train's opt-in
--exclude-features flag, and only SIMULATES rolling calibration against the
historical OOF stream to prove the approach before any live wiring.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.train_v2_validate
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from learning.data import load_raw_m1
from features.features import build_features
from features.labeling import cusum_filter, triple_barrier_labels
from learning.train import (TB_CFG_DIR, TB_CFG_TRADE, HORIZON_VOL_SCALE, CUSUM_K,
                             assemble_dataset, label_events)
from decision.calibration import PlattCalibrator, RollingCalibrationConfig, fit_rolling
from learning.backtest import greedy_sequential
from research.audit_edge import oof_run, build_meta, manual_log_loss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
os.makedirs(OUT, exist_ok=True)
EXCLUDE = frozenset({"spread", "tick_volume"})
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def calib_report(proba, y, label):
    calib_bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    rows = []
    for i in range(len(calib_bins) - 1):
        lo, hi = calib_bins[i], calib_bins[i + 1]
        m = (proba >= lo) & (proba < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": 0})
            continue
        rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": n,
                      "mean_predicted_p": float(proba[m].mean()),
                      "observed_win_rate": float(y[m].mean())})
    brier = float(np.mean((proba - y) ** 2))
    ll = manual_log_loss(y, proba)
    print(f"  [{label}] Brier={brier:.4f} logloss={ll:.4f}")
    for r in rows:
        if r["n"] > 0:
            print(f"    {r['bin']}: n={r['n']:>7,} pred={r['mean_predicted_p']:.4f} "
                  f"obs={r['observed_win_rate']:.4f} gap={r['observed_win_rate']-r['mean_predicted_p']:+.4f}")
    return {"brier": brier, "logloss": ll, "bins": rows}


def coverage_report(proba, y, t0_arr, t1_arr, times, label):
    years = pd.DatetimeIndex(times[t0_arr]).year
    rows = []
    for thr in THRESHOLDS:
        m = proba >= thr
        cand = np.where(m)[0]
        if len(cand) < 10:
            rows.append({"threshold": thr, "n": len(cand)})
            continue
        order = cand[np.argsort(t0_arr[cand])]
        accepted = greedy_sequential(t0_arr, t1_arr, order)
        wr = float(y[accepted].mean()) if len(accepted) else None
        by_year = {}
        for yr in sorted(set(years[accepted].tolist())) if len(accepted) else []:
            ym = accepted[years[accepted] == yr]
            by_year[int(yr)] = {"n": int(len(ym)), "win_rate": float(y[ym].mean()) if len(ym) else None}
        rows.append({"threshold": thr, "n_sequential": int(len(accepted)), "win_rate": wr,
                      "by_year": by_year})
    print(f"  [{label}]")
    for r in rows:
        yby = r.get("by_year", {})
        yby_str = " ".join(f"{y}:n={v['n']}" for y, v in sorted(yby.items()))
        print(f"    thr={r['threshold']:.2f} n_seq={r.get('n_sequential', r.get('n'))} "
              f"wr={r.get('win_rate')}  [{yby_str}]")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    print("== v2 dataset: 26 features (spread, tick_volume excluded from predictive matrix) ==")
    feat, close, high, low, vol, t0_idx, feature_cols = assemble_dataset(rows=args.rows, exclude=EXCLUDE)
    times = pd.to_datetime(feat["time"].to_numpy())
    X, y_bin, t0, t1, t0_nz = label_events(close, high, low, vol, t0_idx, feature_cols, feat)
    print(f"predictive feature_cols ({len(feature_cols)}): {feature_cols}")

    print("\n== v2 primary OOF ==")
    prim = oof_run(X, y_bin, t0, t1, tag="v2-primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    print("\n== v2 meta OOF ==")
    meta = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag="v2-meta", want_importance=False)
    oof_proba = meta["oof_proba"]
    valid_meta = meta["has_oof"]
    y_meta_np = y_meta.to_numpy()
    t0_meta_np = t0_meta.to_numpy()
    t1_meta_np = t1_meta.to_numpy()

    proba_v = oof_proba[valid_meta]
    y_v = y_meta_np[valid_meta]
    t0_v = t0_meta_np[valid_meta]
    t1_v = t1_meta_np[valid_meta]

    print("\n== BEFORE calibration (raw v2 meta OOF probability) ==")
    before_calib = calib_report(proba_v, y_v, "v2 raw")
    before_cov = coverage_report(proba_v, y_v, t0_v, t1_v, times, "v2 raw, sequential coverage by year")

    print("\n== simulating rolling causal Platt calibration (monthly refit checkpoints, "
          "180-day trailing window, resolved-outcomes-only) ==")
    cfg = RollingCalibrationConfig(window_days=180, min_samples=500)
    outcomes = pd.DataFrame({
        "t0_time": times[t0_v], "t1_time": times[t1_v],
        "raw_proba": proba_v, "label": y_v,
    }).sort_values("t0_time").reset_index(drop=True)
    # Phase 3A: persist the full resolved-outcomes stream -- this is the seed the shadow
    # engine's rolling calibrator bootstraps from (a live shadow process would otherwise
    # need ~a year to accumulate min_samples=500 resolved trades of its own).
    outcomes.to_csv(os.path.join(OUT, "v2_oof_outcomes.csv"), index=False)

    # global fallback: fit once on everything resolved by the halfway point, used for cold-start
    halfway = outcomes["t0_time"].iloc[len(outcomes) // 4]
    global_fit_mask = outcomes["t1_time"] <= halfway
    global_fallback = (PlattCalibrator.fit(outcomes.loc[global_fit_mask, "raw_proba"].to_numpy(),
                                            outcomes.loc[global_fit_mask, "label"].to_numpy())
                        if global_fit_mask.sum() >= cfg.min_samples else PlattCalibrator.identity())

    month_starts = pd.date_range(outcomes["t0_time"].min().to_period("M").start_time,
                                  outcomes["t0_time"].max(), freq="MS")
    calibrated = np.full(len(outcomes), np.nan)
    checkpoints = []
    for i, m_start in enumerate(month_starts):
        cal = fit_rolling(outcomes, asof=m_start, cfg=cfg, global_fallback=global_fallback)
        m_end = month_starts[i + 1] if i + 1 < len(month_starts) else outcomes["t0_time"].max() + pd.Timedelta(days=1)
        in_month = (outcomes["t0_time"] >= m_start) & (outcomes["t0_time"] < m_end)
        calibrated[in_month.to_numpy()] = cal.apply(outcomes.loc[in_month, "raw_proba"].to_numpy())
        checkpoints.append({"asof": str(m_start), "a": cal.a, "b": cal.b, "n_samples": cal.n_samples,
                             "n_applied": int(in_month.sum())})
        last_cal = cal
    has_cal = np.isfinite(calibrated)
    # Phase 3A: persist the two calibrators the shadow engine bootstraps from --
    # global_fallback (all-history-to-halfway fit, safest cold-start) and the most
    # recent rolling checkpoint (last_cal, trailing-180d-as-of-latest-month, closer to
    # what a real nightly refit would currently believe).
    global_fallback.save(os.path.join(BASE, "models", "v2", "calibration_global_fallback.json"))
    last_cal.save(os.path.join(BASE, "models", "v2", "calibration_bootstrap.json"))
    print(f"calibrated {has_cal.sum():,}/{len(calibrated):,} events across {len(month_starts)} monthly checkpoints "
          f"(first checkpoint(s) fall back to the global fit / identity until 180d + {cfg.min_samples} samples accumulate)")

    print("\n== AFTER calibration ==")
    y_cal = outcomes["label"].to_numpy()[has_cal]
    t0_cal = outcomes["t0_time"].to_numpy()[has_cal]
    t1_cal = outcomes["t1_time"].to_numpy()[has_cal]
    proba_cal = calibrated[has_cal]
    # greedy_sequential just needs t0/t1 on a SHARED, comparable ordinal timeline (its check is
    # `t0[i] >= last_t1`) -- nanoseconds-since-epoch preserves true chronological order for both
    # arrays jointly, unlike independently-ranking each array (which would put t0 and t1 on two
    # unrelated integer scales and silently break the non-overlap comparison).
    order_rank_t0 = t0_cal.astype("datetime64[ns]").astype(np.int64)
    order_rank_t1 = t1_cal.astype("datetime64[ns]").astype(np.int64)
    after_calib = calib_report(proba_cal, y_cal, "v2 calibrated")
    years_cal = pd.DatetimeIndex(t0_cal).year
    after_cov_rows = []
    for thr in THRESHOLDS:
        m = proba_cal >= thr
        cand = np.where(m)[0]
        if len(cand) < 10:
            after_cov_rows.append({"threshold": thr, "n": len(cand)})
            continue
        order = cand[np.argsort(order_rank_t0[cand])]
        accepted = greedy_sequential(order_rank_t0, order_rank_t1, order)
        wr = float(y_cal[accepted].mean()) if len(accepted) else None
        by_year = {}
        for yr in sorted(set(years_cal[accepted].tolist())) if len(accepted) else []:
            ym = accepted[years_cal[accepted] == yr]
            by_year[int(yr)] = {"n": int(len(ym)), "win_rate": float(y_cal[ym].mean()) if len(ym) else None}
        after_cov_rows.append({"threshold": thr, "n_sequential": int(len(accepted)), "win_rate": wr,
                                "by_year": by_year})
        yby_str = " ".join(f"{y}:n={v['n']}" for y, v in sorted(by_year.items()))
        print(f"    thr={thr:.2f} n_seq={len(accepted)} wr={wr}  [{yby_str}]")

    result = {
        "excluded_features": sorted(EXCLUDE), "feature_cols": feature_cols,
        "primary_fold_metrics": prim["fold_metrics"], "meta_fold_metrics": meta["fold_metrics"],
        "before_calibration": {"calib": before_calib, "coverage": before_cov},
        "after_calibration": {"calib": after_calib, "coverage": after_cov_rows,
                               "monthly_checkpoints": checkpoints},
        "elapsed_sec": time.time() - t_start,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT, f"v2_validation_{stamp}.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(OUT, "v2_validation_latest.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDONE in {result['elapsed_sec']:.1f}s -> research/output/v2_validation_{stamp}.json")


if __name__ == "__main__":
    main()
