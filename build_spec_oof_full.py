#!/usr/bin/env python3
"""
build_spec_oof_full.py — FULL-MATRIX SPECIALIST OOF (2026-08-10)
=================================================================
The rating fitter must learn its threshold on the SAME P(win) scale the
live engine produces. The engine routes each market state to its regime
specialist (3 seeds) and calibrates with that regime's OWN curves
(calibration_by_drr_spec_<regime>.json).

The per-specialist OOF files (oof_spec_*.npy) only cover each regime's
VALIDATION subset (~145k rows). This script generates a FULL-MATRIX
specialist OOF: for every matrix row, compute the regime, predict with
that regime's 3 specialist seeds, average — exactly like best_placement
does live. Output: oof_spec_full.npy (raw averaged probs, float32,
row-aligned with gold_features_m5.csv), consumed by fit_signal_rating.

Memory-safe: chunked streaming, one regime's boosters loaded at a time.
"""
import gc
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import features as F
from features import vector_regime_bin

MODEL_DIR = "models"
MATRIX = "gold_features_m5.csv"
REGIMES = ["STRONG_UP", "UP", "DOWN", "STRONG_DOWN",
           "RANGE_TIGHT", "RANGE_WIDE", "HIGH_VOL", "QUIET_LOW_VOL"]
SEEDS = [42, 7, 2026]
CHUNK = 400_000
FEATURE_EXCLUDE = {"time", "target", "fwd_return", "mfe_atr", "mfa_atr"} | F.HTF_FEATURES


def _resolve_feats():
    """Exact feature list the specialists were trained on (same derivation
    as train_regime_spec.py): matrix header minus exclusions minus raw cols."""
    hdr = pd.read_csv(MATRIX, nrows=0).columns.tolist()
    return [c for c in hdr
            if c not in FEATURE_EXCLUDE and c not in F.RAW_PRICE_COLS]


def main():
    t0 = time.time()
    cfg = json.load(open(f"{MODEL_DIR}/regime_specialists.json"))
    bins = cfg.get("bins", {})
    print(f"specialists: {len(bins)} regimes × {len(SEEDS)} seeds")

    FEATS = _resolve_feats()
    print(f"features: {len(FEATS)}")

    n = int(os.popen(f"wc -l < {MATRIX}").read().strip()) - 1
    print(f"matrix rows: {n:,}")
    oof = np.zeros(n, dtype=np.float32)
    oofy = np.zeros(n, dtype=np.int8)
    drr = np.zeros(n, dtype=np.float32)   # dir*4 + rr_idx for cal curves

    read_cols = FEATS + ["target", "direction", "rr_buy", "rr_sell"]
    dtype = {c: np.float32 for c in read_cols if c not in ("target", "direction")}
    dtype["direction"] = np.int8

    # Preload all 24 boosters (each ~few MB) — keeps routing cheap.
    boosters = {}
    for reg, meta in bins.items():
        if reg not in REGIMES:
            continue
        try:
            boosters[reg] = [lgb.Booster(model_file=f"{MODEL_DIR}/{m}")
                             for m in meta["models"]]
            for b in boosters[reg]:
                b.params = dict(b.params, num_threads=4)
        except Exception as e:
            print(f"  ⚠️ {reg}: {e}")
            boosters[reg] = None
    print(f"boosters loaded ({time.time()-t0:.0f}s)")

    from features import TP_RATIOS
    tarr = np.asarray(TP_RATIOS, dtype=np.float64)

    seen = 0
    for chunk in pd.read_csv(MATRIX, usecols=read_cols, dtype=dtype,
                             chunksize=CHUNK, low_memory=False):
        mask = ~chunk[["target", "direction", "rr_buy", "rr_sell"]].isna().any(axis=1)
        n_rows = len(chunk)
        sl = slice(seen, seen + n_rows)
        bins_v = vector_regime_bin(chunk)   # computed on FULL chunk (regime keys are float)
        # per-row regime routing — vectorized predict per regime
        p = np.full(n_rows, 0.5, dtype=np.float32)
        for reg in REGIMES:
            bs = boosters.get(reg)
            if bs is None:
                continue
            m = bins_v == reg
            if not m.any():
                continue
            Xc = chunk.loc[m, FEATS].values.astype(np.float32)
            pm = np.zeros(Xc.shape[0], dtype=np.float32)
            for b in bs:
                pm += b.predict(Xc) / len(bs)
            p[m] = pm
        oof[sl] = p
        d = chunk["direction"].values
        rr = np.where(d == 1, chunk["rr_buy"].values, chunk["rr_sell"].values)
        bucket = tarr[np.argmin(np.abs(tarr[None, :] - rr[:, None]), axis=1)]
        idx = np.where(d == 1, 4, 0) + np.array(
            [TP_RATIOS.index(t) for t in bucket])
        drr[sl] = idx
        oofy[sl] = chunk["target"].fillna(0).values.astype(np.int8)
        seen += n_rows
        if seen % 2_000_000 < CHUNK:
            print(f"  routed {seen:,}/{n:,} rows ({time.time()-t0:.0f}s)",
                  flush=True)

    assert seen == n, f"rows mismatch {seen} != {n}"
    np.save(f"{MODEL_DIR}/oof_spec_full.npy", oof)
    np.save(f"{MODEL_DIR}/oof_spec_full_y.npy", oofy)
    np.save(f"{MODEL_DIR}/oof_spec_full_drr.npy", drr)
    print(f"✅ oof_spec_full.npy saved ({seen:,} rows, mean={oof.mean():.3f}) "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
