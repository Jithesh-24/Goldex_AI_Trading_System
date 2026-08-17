#!/usr/bin/env python3
"""Strip macro features from tick matrix — pure gold quantitative system."""
import pandas as pd, os, time

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "gold_features_m5_tick.csv")
OUTPUT = os.path.join(BASE, "gold_features_m5_pure.csv")

MACRO_COLS = ["dxy_z", "dxy_5d_chg", "tnx_level", "tnx_5d_chg",
              "gc_5d_chg", "gld_5d_chg", "eur_5d_chg"]

print(f"Reading {INPUT}...")
t0 = time.time()
chunks = []
for i, chunk in enumerate(pd.read_csv(INPUT, chunksize=500_000, low_memory=False)):
    # Drop macro columns if present
    drop = [c for c in MACRO_COLS if c in chunk.columns]
    if drop:
        chunk = chunk.drop(columns=drop)
    chunks.append(chunk)
    if (i+1) % 10 == 0:
        print(f"  processed {(i+1)*500_000:,} rows ({time.time()-t0:.0f}s)")

print(f"  Concatenating {len(chunks)} chunks...")
df = pd.concat(chunks, ignore_index=True)
print(f"  Shape: {df.shape}")
print(f"  Columns: {len(df.columns)} (removed {len(MACRO_COLS)} macro)")

df.to_csv(OUTPUT, index=False)
print(f"  Saved: {OUTPUT} ({time.time()-t0:.0f}s)")

# Verify
cols = list(df.columns)
print(f"\nAll {len(cols)} columns:")
for i, c in enumerate(cols, 1):
    print(f"  {i:3d}. {c}")
