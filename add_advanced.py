#!/usr/bin/env python3
"""Add advanced features to existing tick matrix. Memory-efficient."""
import pandas as pd, numpy as np, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advanced_features import compute_advanced_features, ADVANCED_FEATURE_NAMES

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "gold_features_m5_tick.csv")
OUTPUT = os.path.join(BASE, "gold_features_m5_full.csv")

MACRO = {"dxy_z", "dxy_5d_chg", "tnx_level", "tnx_5d_chg", "gc_5d_chg", "gld_5d_chg", "eur_5d_chg"}

t0 = time.time()
first = True
rows = 0

for chunk in pd.read_csv(INPUT, parse_dates=["time"], chunksize=200_000, low_memory=False):
    # Drop macro columns
    drop = [c for c in MACRO if c in chunk.columns]
    if drop:
        chunk = chunk.drop(columns=drop)
    
    # Compute advanced features
    try:
        chunk = compute_advanced_features(chunk)
    except Exception as e:
        print(f"  ⚠️ {e}")
        for col in ADVANCED_FEATURE_NAMES:
            if col not in chunk.columns:
                chunk[col] = 0.0
    
    chunk.to_csv(OUTPUT, mode="w" if first else "a", index=False, header=first)
    first = False
    rows += len(chunk)
    if rows % 1_000_000 == 0:
        print(f"  {rows:,} rows ({time.time()-t0:.0f}s)")

elapsed = time.time() - t0
print(f"DONE: {rows:,} rows, {len(chunk.columns)} cols ({elapsed:.0f}s)")
print(f"Output: {OUTPUT}")
