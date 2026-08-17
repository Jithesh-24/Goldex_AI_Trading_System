#!/usr/bin/env python3
"""
generate_drift_stats.py — Compute training-time feature statistics
for live drift detection.

Reads the tick matrix, computes mean/std for each feature,
saves to models/feature_drift_stats.json.

Run after retrain completes.
"""
import os, json, time
import pandas as pd
import numpy as np

BASE = "/home/jith/.hermes/profiles/trading/scripts"
FEAT_CSV = os.environ.get("FEAT_CSV", f"{BASE}/gold_features_m5_tick.csv")

# Columns to track for drift (key features only, not all 128)
TRACK_FEATURES = [
    # Price/return features
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_15", "ret_30", "ret_60",
    "ret_mom", "spread",
    # Volatility
    "atr_14", "atr_5", "vol_20", "vol_5", "vol_ratio",
    # Trend
    "trend_ema", "adx_14", "rsi_14",
    # Tick microstructure (NEW in v8.9)
    "imb_300s", "vol_rel", "cvd",
    # Macro (NEW in v8.9)
    "dxy_z", "dxy_5d_chg", "tnx_level", "tnx_5d_chg",
    "gc_5d_chg", "gld_5d_chg", "eur_5d_chg",
    # Events
    "min_to_event", "pre_event", "post_event",
]

EXCLUDE = {"time", "target", "fwd_return", "mfe_atr", "mfa_atr", "date"}

def main():
    t0 = time.time()
    print(f"Computing drift stats from {FEAT_CSV}...")
    
    # Read all cols we need
    hdr = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    track = [c for c in TRACK_FEATURES if c in hdr]
    print(f"Tracking {len(track)} features for drift detection")
    
    # Streaming mean/std (Welford's algorithm)
    n = 0
    means = {c: 0.0 for c in track}
    m2 = {c: 0.0 for c in track}
    
    for chunk in pd.read_csv(FEAT_CSV, usecols=track, chunksize=500_000):
        for c in track:
            if c not in chunk.columns:
                continue
            vals = chunk[c].dropna().values
            for x in vals:
                n_local = n + 1
                delta = x - means[c]
                means[c] += delta / n_local
                delta2 = x - means[c]
                m2[c] += delta * delta2
        n += len(chunk)
        if n % 4_000_000 == 0:
            print(f"  processed {n:,} rows ({time.time()-t0:.0f}s)")
    
    stats = {}
    for c in track:
        if c in means:
            var = m2[c] / max(n - 1, 1)
            std = max(np.sqrt(var), 1e-8)
            stats[c] = {"mean": float(means[c]), "std": float(std)}
    
    out = f"{BASE}/models/feature_drift_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"DONE: {len(stats)} features → {out} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
