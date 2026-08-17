#!/usr/bin/env python3
"""
Build the v8.9 tick matrix: gold_features_m5.csv + dukascopy tick + macro.
Memory-efficient: streams chunks, no event recomputation (already in M5).
Output: gold_features_m5_tick.csv
"""
import pandas as pd
import numpy as np
import time, os, sys

t0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# Paths
M5 = "/home/jith/.hermes/profiles/trading/scripts/gold_features_m5.csv"
DK = "/home/jith/.hermes/profiles/trading/scripts/dukascopy_m1_features.csv"
MACRO = "/home/jith/.hermes/profiles/trading/scripts/macro_daily.csv"
OUT = "/home/jith/.hermes/profiles/trading/scripts/gold_features_m5_tick.csv"

# 1) Load dukascopy tick features (small: 703K rows)
log("Loading dukascopy tick features...")
dk = pd.read_csv(DK, parse_dates=["time"])
# Rename dk_delta -> imb_300s, dk_cvd -> cvd, dk_vol_rel -> vol_rel
rename = {}
for old, new in [("dk_delta", "imb_300s"), ("dk_cvd", "cvd"), ("dk_vol_rel", "vol_rel")]:
    if old in dk.columns:
        rename[old] = new
if rename:
    dk = dk.rename(columns=rename)
dk = dk.set_index("time").sort_index()
# Keep only the 3 new tick features + time index
tick_cols = [c for c in ["imb_300s", "vol_rel", "cvd"] if c in dk.columns]
dk = dk[tick_cols]
log(f"Tick block: {len(dk)} rows, cols={tick_cols}")

# 2) Load macro features (small: 2.5K rows)
log("Loading macro features...")
macro = pd.read_csv(MACRO, parse_dates=["date"])
macro["date"] = macro["date"].dt.tz_localize(None)  # strip UTC tz
macro = macro.set_index("date").sort_index()
macro_cols = [c for c in macro.columns if c != "Unnamed: 0"]
macro = macro[macro_cols]
log(f"Macro block: {len(macro)} rows, cols={list(macro.columns)}")

# 4) Load advanced features module
log("Loading advanced features module...")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advanced_features import compute_advanced_features, ADVANCED_FEATURE_NAMES
log(f"Advanced features: {ADVANCED_FEATURE_NAMES}")

# 5) Stream M5 matrix in chunks, merge tick+macro+advanced, write output
log(f"Streaming M5 matrix from {M5}...")
chunk_size = 500_000  # 500K rows per chunk for memory safety
first = True
rows_written = 0

for chunk in pd.read_csv(M5, parse_dates=["time"], chunksize=chunk_size):
    t1 = time.time()

    # Add date column for macro merge (don't recompute events!)
    chunk["date"] = chunk["time"].dt.normalize()

    # Left-join tick features on time
    if tick_cols:
        chunk = chunk.merge(dk, left_on="time", right_index=True, how="left")

    # Left-join macro features on date
    if len(macro_cols) > 0:
        chunk = chunk.merge(macro, left_on="date", right_index=True, how="left", suffixes=("", "_m"))

    # Drop the helper date column
    chunk = chunk.drop(columns=["date"], errors="ignore")

    # Drop duplicate macro columns if any
    dupes = [c for c in chunk.columns if c.endswith("_m")]
    if dupes:
        chunk = chunk.drop(columns=dupes)

    # Compute advanced features (from price action + volume)
    try:
        chunk = compute_advanced_features(chunk)
        log(f"  Advanced features computed: {len([c for c in ADVANCED_FEATURE_NAMES if c in chunk.columns])} cols")
    except Exception as e:
        log(f"  ⚠️ Advanced features failed: {e}")
        for col in ADVANCED_FEATURE_NAMES:
            if col not in chunk.columns:
                chunk[col] = 0.0

    # Write
    chunk.to_csv(OUT, mode="w" if first else "a", index=False, header=first)
    first = False
    rows_written += len(chunk)
    elapsed = time.time() - t1
    log(f"  chunk done: +{len(chunk):,} rows total={rows_written:,} ({elapsed:.1f}s)")

log(f"DONE: {rows_written:,} rows written to {OUT}")
log(f"Total time: {time.time()-t0:.0f}s")
