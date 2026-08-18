"""
Phase 2 Step 1+2 -- live/OOF parity investigation. Read-only: loads the
DEPLOYED primary.cbm/meta.cbm (does not retrain, does not touch the live
engine), scores them in-sample on the same 2025/2026 events used in Phase
1A's OOF fold 4, and compares distributions + the exact train/holdout split
boundary used to produce the deployed artifacts. Also diffs the live
feature-construction code path (ai_signal_engine.LiveEngine's buffer ->
build_features) against the training code path (core.train.assemble_dataset)
on identical recent bars, to rule out a feature-vector mismatch.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.parity_check
"""
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from core.data import load_raw_m1
from core.features import build_features
from core.labeling import cusum_filter, triple_barrier_labels
from core.train import (TB_CFG_DIR, TB_CFG_TRADE, HORIZON_VOL_SCALE, CUSUM_K,
                         VAL_FRACTION)
from research.audit_edge import oof_run, build_meta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")


def pctl_report(p, label):
    p = np.asarray(p, dtype=np.float64)
    if len(p) == 0:
        print(f"{label}: n=0")
        return
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(p, qs)
    row = " ".join(f"{q}%={v:.4f}" for q, v in zip(qs, vals))
    print(f"{label}: n={len(p):,} mean={p.mean():.4f} {row}")
    for thr in (0.60, 0.65, 0.70):
        print(f"    frac>={thr:.2f}: {(p >= thr).mean()*100:.3f}%")


def main():
    print("== rebuilding deployed 28-feature dataset ==")
    df = load_raw_m1()
    feat = build_features(df)
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    times = pd.to_datetime(feat["time"].to_numpy())

    with open(os.path.join(MODELS, "feature_cols.json")) as f:
        deployed_cfg = json.load(f)
    primary_cols = deployed_cfg["primary"]
    meta_cols = deployed_cfg["meta"]

    vol = feat["ewma_vol"].to_numpy(dtype=np.float64)
    vol_filled = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[np.isfinite(vol)]))
    threshold = np.clip(CUSUM_K * vol_filled * close, 1e-6, None)
    event_mask = cusum_filter(close, threshold)
    vol_tb = vol_filled * np.sqrt(TB_CFG_DIR.max_holding) * HORIZON_VOL_SCALE

    feature_cols = [c for c in feat.columns if c != "time"]
    print(f"live feature_cols.json primary list == locally computed feature list? "
          f"{sorted(primary_cols) == sorted(feature_cols)}")
    warmup_ok = feat[feature_cols].notna().all(axis=1).to_numpy()
    horizon_ok = np.arange(len(df)) < (len(df) - TB_CFG_DIR.max_holding - 1)
    valid = event_mask & warmup_ok & horizon_ok
    t0_idx = np.where(valid)[0]

    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, TB_CFG_DIR, side=None)
    y_raw = labels["label"].to_numpy()
    t1_raw = labels["t1"].to_numpy()
    nz = y_raw != 0
    t0_nz = t0_idx[nz]
    y_bin = pd.Series((y_raw[nz] == 1).astype(np.int64)).reset_index(drop=True)
    t0 = pd.Series(t0_nz).reset_index(drop=True)
    t1 = pd.Series(t1_raw[nz]).reset_index(drop=True)
    X_full = feat.loc[t0_nz, feature_cols].reset_index(drop=True)
    n_events = len(X_full)
    print(f"n_directional_events={n_events:,}")

    # ---- exact train/holdout boundary the DEPLOYED final model used ----
    cut = int(n_events * (1 - VAL_FRACTION))
    cut_date = times[t0_nz[cut]]
    print(f"\n== deployed final-model fit boundary (core/train.py:_fit_with_early_stopping) ==")
    print(f"cut index={cut:,} of {n_events:,} ({(1-VAL_FRACTION)*100:.0f}% train / {VAL_FRACTION*100:.0f}% early-stop holdout)")
    print(f"TRAIN (gradient-fit) events: index [0, {cut}) -> dates {times[t0_nz[0]]} -> {cut_date}")
    print(f"EARLY-STOP HOLDOUT (not gradient-fit, only monitored): index [{cut}, {n_events}) "
          f"-> dates {cut_date} -> {times[t0_nz[-1]]}")
    va_dates = times[t0_nz[cut:]]
    for yr in (2025, 2026):
        n_yr = int(((va_dates.year == yr)).sum())
        print(f"  year {yr}: {n_yr:,} events fall in the early-stop HOLDOUT slice (never gradient-fit)")

    # ---- score the DEPLOYED (in-sample-fit) models on all events ----
    print("\n== scoring DEPLOYED primary.cbm / meta.cbm (in-sample, no retraining) ==")
    dep_primary = CatBoostClassifier()
    dep_primary.load_model(os.path.join(MODELS, "primary.cbm"))
    dep_meta = CatBoostClassifier()
    dep_meta.load_model(os.path.join(MODELS, "meta.cbm"))

    dep_primary_proba = dep_primary.predict_proba(X_full[primary_cols])[:, 1]
    dep_side = np.where(dep_primary_proba >= 0.5, 1.0, -1.0)
    X_meta_dep = X_full.copy()
    X_meta_dep["assumed_side"] = dep_side
    dep_meta_proba = dep_meta.predict_proba(X_meta_dep[meta_cols])[:, 1]

    event_years = pd.DatetimeIndex(times[t0_nz]).year
    for yr in (2022, 2023, 2024, 2025, 2026):
        m = event_years == yr
        pctl_report(dep_meta_proba[m], f"DEPLOYED in-sample meta proba, year {yr}")

    # ---- honest OOF for direct side-by-side comparison ----
    print("\n== recomputing honest OOF (same methodology as Phase 1A) for comparison ==")
    prim = oof_run(X_full, y_bin, t0, t1, tag="parity-primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta = X_full.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())
    meta = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag="parity-meta", want_importance=False)
    oof_proba = meta["oof_proba"]
    valid_meta = meta["has_oof"]
    meta_years = pd.DatetimeIndex(times[t0_meta.to_numpy()]).year

    for yr in (2022, 2023, 2024, 2025, 2026):
        m = valid_meta & (meta_years == yr)
        pctl_report(oof_proba[m], f"HONEST WALK-FORWARD OOF meta proba, year {yr}")

    # ---- feature-vector parity: live code path vs training code path ----
    print("\n== feature-vector parity: LiveEngine buffer path vs core.train assemble path ==")
    import importlib
    ai_engine = importlib.import_module("app.engine")
    seed_tail = ai_engine.load_buffer_tail(n=8000)
    live_feat = build_features(seed_tail)
    # matching window from the training-path feature frame, aligned by timestamp
    common_t = seed_tail["time"].iloc[-1]
    train_row = feat[feat["time"] == common_t]
    live_row = live_feat[live_feat["time"] == common_t]
    if len(train_row) == 1 and len(live_row) == 1:
        diffs = {}
        for c in primary_cols:
            a = float(train_row[c].iloc[0])
            b = float(live_row[c].iloc[0])
            if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, rtol=1e-6, atol=1e-9):
                diffs[c] = (a, b)
        print(f"compared timestamp {common_t}: {len(primary_cols)} columns checked, "
              f"{len(diffs)} mismatched (tolerance 1e-6 rel)")
        for c, (a, b) in diffs.items():
            print(f"  MISMATCH {c}: train_path={a} live_path={b}")
    else:
        print(f"could not find matching timestamp {common_t} in both frames "
              f"(train_row={len(train_row)}, live_row={len(live_row)}) -- buffer may not "
              f"reach far enough back for warmup; skipping exact diff")

    print("\nDONE")


if __name__ == "__main__":
    main()
