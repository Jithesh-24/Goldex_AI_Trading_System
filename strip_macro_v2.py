#!/usr/bin/env python3
"""Strip macro features — memory-efficient chunked approach."""
import pandas as pd, os, time

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "gold_features_m5_tick.csv")
OUTPUT = os.path.join(BASE, "gold_features_m5_pure.csv")

MACRO_COLS = ["dxy_z", "dxy_5d_chg", "tnx_level", "tnx_5d_chg",
              "gc_5d_chg", "gld_5d_chg", "eur_5d_chg"]

# First pass: identify which macro columns exist
hdr = pd.read_csv(INPUT, nrows=0)
drop = [c for c in MACRO_COLS if c in hdr.columns]
print(f"Dropping {len(drop)} macro columns: {drop}")

# Stream chunks, drop macro, write directly
t0 = time.time()
rows = 0
first = True
for chunk in pd.read_csv(INPUT, chunksize=500_000, low_memory=False):
    chunk = chunk.drop(columns=[c for c in drop if c in chunk.columns])
    chunk.to_csv(OUTPUT, mode="w" if first else "a", index=False, header=first)
    first = False
    rows += len(chunk)
    if rows % 5_000_000 == 0:
        print(f"  {rows:,} rows ({time.time()-t0:.0f}s)")

print(f"DONE: {rows:,} rows, {len(chunk.columns)} columns ({time.time()-t0:.0f}s)")
print(f"Output: {OUTPUT}")
