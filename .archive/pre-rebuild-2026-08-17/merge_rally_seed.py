"""Merge Dukascopy rally history + XM seed into gold_seed_multi.csv (v6).

WHY: XM server caps M1 history at ~60 days, and the last 60 days were
BEARISH (June-July 2026: 4523 -> 4043). The model never saw a gold rally —
it only learned SELL. Dukascopy serves REAL spot XAUUSD M1 back to 2019
(free, 1440 candles/day), covering the COVID rally (1460->2075), 2024
breakout (1990->2450) and the 2025 melt-up.

STRATEGY (7GB RAM / 1 core constraint): the full 907k-bar rally archive ×
40 rows/bar would make a ~39M-row matrix — untrainable on this machine.
Instead we select SHARP CONTINUOUS WINDOWS from each regime (the steepest
up-legs, the down-leg, the chop) so every market condition is represented
with full M1 fidelity. Rolling windows and institutional levels stay
INTACT because windows are continuous (no day-sampling — that would kill
prev-day H/L/C features and break rolling ATR/percentile warmup).

LIVE ENGINE keeps reading gold_seed.csv (XM-only); only TRAINING uses the
multi-era file.

Column notes:
  - Dukascopy: tick_volume in lots (float) — features use vol_z / vol_rel
    (z/relative), so absolute scale doesn't matter; spread ~0.20 typical.
  - All times TRUE UTC.
"""
import pandas as pd
import os

BASE = "/home/jith/.hermes/profiles/trading/scripts"
RALLY = f"{BASE}/xauusd_rally.csv"
XM_SEED = f"{BASE}/gold_seed.csv"
OUT = f"{BASE}/gold_seed_multi.csv"

# Sharp continuous windows per regime (start, end) — inclusive, TRUE UTC.
# Each window is a full 24h M1 stream with all rolling/institutional
# features computable. Budget: ~150k rally bars so the full training matrix
# (24 rows/bar × ~210k bars) stays under ~5M rows — the 7GB/1-core box
# can't train the full 907k-bar archive (39M rows would take days).
WINDOWS = [
    ("covid_up",   "2020-03-20", "2020-04-20"),  # 1450 -> 1730 vertical leg
    ("covid_up2",  "2020-07-15", "2020-08-07"),  # 1790 -> 2075 blow-off top
    ("bear_2022",  "2022-03-10", "2022-04-08"),  # 2070 -> 1920 fast down
    ("range_2023", "2023-01-10", "2023-02-10"),  # 1800-1960 chop
    ("rally_2024", "2024-02-20", "2024-04-15"),  # 2020 -> 2430 breakout
    ("meltup_2025","2025-02-10", "2025-03-10"),  # 2830 -> 2930 consolidation
]

rally = pd.read_csv(RALLY)
rally["time"] = pd.to_datetime(rally["time"])
print(f"Rally archive: {len(rally)} bars | {rally['time'].iloc[0]} -> {rally['time'].iloc[-1]}")

picks = []
for label, d0, d1 in WINDOWS:
    w = rally[(rally["time"] >= d0) & (rally["time"] <= d1)].copy()
    w["src"] = "duka"
    picks.append(w)
    print(f"  {label}: {len(w)} bars ({d0} -> {d1}) | close {w['close'].min():.0f}-{w['close'].max():.0f}")

duka = pd.concat(picks, ignore_index=True)
print(f"Selected rally windows: {len(duka)} bars")

# ── XM seed (true UTC already) ──
xm = pd.read_csv(XM_SEED)
xm["time"] = pd.to_datetime(xm["time"])
print(f"XM:   {len(xm)} bars | {xm['time'].iloc[0]} -> {xm['time'].iloc[-1]}")

cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume", "src"]
for df in (duka, xm):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df["spread"] = df["spread"].astype(float)

merged = pd.concat([duka[cols], xm[cols]], ignore_index=True)
merged = merged.sort_values("time").drop_duplicates(subset="time", keep="last").reset_index(drop=True)
merged.to_csv(OUT, index=False)

n_duka = (merged["src"] == "duka").sum()
n_xm = (merged["src"] != "duka").sum()
print(f"\nMerged seed: {len(merged)} bars ({n_duka} rally / {n_xm} XM)")
print(f"Span: {merged['time'].iloc[0]} -> {merged['time'].iloc[-1]}")
print(f"Saved: {OUT}")

# Sanity: era coverage — the model must see UP, DOWN and RANGE equally
yrs = merged.groupby(merged["time"].dt.year).size()
print("\nBars per year:")
print(yrs.to_string())

# Matrix size estimate at 40 rows/bar
print(f"\nEstimated training matrix: {len(merged) * 40:,} rows x 40 cols-grid")
