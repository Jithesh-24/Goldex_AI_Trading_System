"""REFIT per-regime calibrations with the v7.7c direction-mask fix.

BUG (v7.7b, 2026-08-06): _fit_spec_calibration derived the direction mask
from `drr` (the RR VALUE, 1.3-3.0) — `drr > 0.5` matched every row, so ALL
rows were fitted into BUY curves and SELL calibration was NEVER written.
Consequence at signal time (engine best_placement): SELL lookup
cal_by_rr.get("SELL_1.3") → None → SELL used RAW overconfident model
probability while BUY was calibrated → placement sweep systematically
favored SELL (bot shorted every uptrend; 6 SL / 1 TP while gold climbed
4234->4244 on 2026-08-06 night).

This script does NOT retrain the specialists (models are fine — they were
trained on both directions). It re-streams the matrix to recover the
direction mask for each regime's OOF va rows, then re-fits ONLY the
calibration curves with the corrected mask. ~25 min stream + seconds of fit.

Usage: python regenerate_dir_prior.py  (no args; uses saved OOF arrays)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from train_regime_spec import (stream_bucket, _fit_spec_calibration,
                               MODEL_DIR, CAL_SPLIT, FEAT_CSV,
                               FEATURE_EXCLUDE)

t0 = time.time()
TMP_DIR = f"{MODEL_DIR}/../tmp_regime"   # same as main() uses
TMP_DIR = os.path.abspath(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

print(f"[refit] pass 1: stream {FEAT_CSV} → per-regime temp CSVs (recover direction)...", flush=True)
feats, coverage, nonempty = stream_bucket(FEAT_CSV, TMP_DIR)
print(f"[refit] bucket pass done ({time.time()-t0:.0f}s): {coverage}", flush=True)

for regime in sorted(nonempty):
    tmp_file = os.path.join(TMP_DIR, f"{regime}.csv")
    oof_p = f"{MODEL_DIR}/oof_spec_{regime.lower()}.npy"
    oofy_p = f"{MODEL_DIR}/oofy_spec_{regime.lower()}.npy"
    drr_p = f"{MODEL_DIR}/drr_spec_{regime.lower()}.npy"
    if not all(os.path.exists(p) for p in (oof_p, oofy_p, drr_p)):
        print(f"[refit] SKIP {regime}: saved OOF arrays missing", flush=True)
        continue
    df = pd.read_csv(tmp_file)
    times = pd.to_datetime(df["time"]).values
    order = np.argsort(times.astype("datetime64[s]").astype(np.int64))
    cut = int(len(order) * CAL_SPLIT)
    va = order[cut:]
    oof = np.load(oof_p)
    oofy = np.load(oofy_p)
    drr = np.load(drr_p)
    if len(va) != len(oof):
        # EOD appended new bars AFTER the specialists were trained (matrix
        # rebuilt 03:03, training finished 21:58). Appended rows are the
        # newest → they sit at the TAIL of the time-sorted va slice.
        # old va = first len(oof) rows of new va (verified: 124423 = 122732+1691).
        extra = len(va) - len(oof)
        if extra > 0 and extra < len(va) // 10:
            va = va[:len(oof)]
            print(f"[refit] {regime}: trimmed {extra} EOD-appended rows from va "
                  f"→ {len(va)} (matches saved OOF)", flush=True)
        else:
            print(f"[refit] SKIP {regime}: va {len(va)} vs oof {len(oof)} — "
                  f"incompatible delta {extra}", flush=True)
            continue
    dirs_va = df["direction"].values[va] > 0.5
    n_buy = int(dirs_va.sum()); n_sell = int((~dirs_va).sum())
    out = _fit_spec_calibration(regime, oof, oofy, drr, dirs_va)
    keys = sorted(k for k in out if not k.startswith("_"))
    n_buy_c = len([k for k in keys if k.startswith("BUY")])
    n_sell_c = len([k for k in keys if k.startswith("SELL")])
    print(f"[refit] {regime}: va n={len(va):,} (BUY {n_buy:,} / SELL {n_sell:,}) "
          f"→ curves BUY={n_buy_c} SELL={n_sell_c} ({time.time()-t0:.0f}s)", flush=True)

# cleanup temp CSVs (same as main())
for fn in os.listdir(TMP_DIR):
    os.remove(os.path.join(TMP_DIR, fn))
print(f"[refit] done in {time.time()-t0:.0f}s", flush=True)
