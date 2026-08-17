#!/usr/bin/env python3
"""Append 2021 M1 → pre-computed features to gold_features_rally.csv (v7.12).

Why: gold_features_rally.csv (the full-build source) skips 2021 entirely —
the downloader's PERIODS list never included it, so the model has NEVER seen
a full post-COVID consolidation year. This script computes the SAME 108-col
feature schema (F._feature_block + geometry expansion, 48 rows/bar) for the
freshly downloaded 2021 M1 bars and APPENDS them to the rally cache, so a
subsequent --full matrix rebuild picks up 2021 automatically.

Density: build_full_matrix.py subsamples the rally cache every-3rd unique
time at build (applies to every year equally). 2021 is stored in the cache
at every-3rd M1 density so the matrix ends up ~every-9th ≈ 28k unique bars
for 2021 — the DENSEST year in the matrix (2020: 18k, 2024: 19k), covering
all 12 months and every regime. Memory math: FULL 2021 density would push
the matrix to ~10.4M rows → train_continue peak ~8.3GB → OOM on this 7.5GB
box. EVERY=3 keeps peak ~6.2GB. 2021 still gets the richest per-year
coverage of any year — full year, no gaps, no regime blind spots.
"""
import sys, time, os
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

M1_2021 = f"{BASE}/gold_m1_2021.csv"        # raw Dukascopy M1 (fresh download)
CACHE = f"{BASE}/gold_features_rally.csv"    # target cache to append to
EVERY = 3                                    # density: build applies its own every-3rd → ~9th overall

def periods_of(df):
    t = df["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(df)]
    return [df.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)]

def main():
    t0 = time.time()
    # cache header = schema contract
    header = list(pd.read_csv(CACHE, nrows=0).columns)
    # assert this script's output columns are exactly the cache's (defensive)
    geometry_cols = ["sl_dist_buy", "tp_dist_buy", "sl_dist_sell", "tp_dist_sell",
                     "sl_atr_buy", "sl_atr_sell", "rr_buy", "rr_sell"]
    for c in geometry_cols + ["time", "direction", "target", "open", "high", "low", "close", "spread"]:
        assert c in header, f"cache header missing {c}"

    df = pd.read_csv(M1_2021)
    df["time"] = pd.to_datetime(df["time"])
    # normalize Dukascopy columns to the XM seed shape (spread=0.20 typical)
    for c in ("spread", "real_volume"):
        if c not in df.columns:
            df[c] = 0.0
    print(f"2021 M1: {len(df):,} bars | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}", flush=True)

    # subsample every 3rd bar (build will subsample again → ~9th overall, ~28k bars)
    keep_idx = np.arange(0, len(df), EVERY)
    sub = df.iloc[keep_idx].reset_index(drop=True)
    print(f"subsampled (every {EVERY}): {len(sub):,} bars → matrix ~{len(sub)//3:,} bars after build", flush=True)

    periods = [p for p in periods_of(sub) if len(p) >= 300]
    print(f"periods: {len(periods)}", flush=True)

    total = 0
    first = True
    for pi, p in enumerate(periods):
        fdf = F._feature_block(p).dropna().reset_index(drop=True)
        if len(fdf) < 100:
            continue
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values if "spread" in fdf.columns \
            else np.full(len(fdf), F.SPREAD)
        market_cols = [c for c in header if c in fdf.columns]
        for direction in ("BUY", "SELL"):
            for m in F.SL_MULTS:
                for r in F.TP_RATIOS:
                    sl_dist = np.maximum(atr * m, F.MIN_SL_FLOOR)
                    tp_dist = (sl_dist + spr) * r
                    tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                                             sl_dist=sl_dist, tp_dist=tp_dist, direction=direction)
                    gdf = F.add_geometry_awareness(
                        fdf, sl_dist_buy=sl_dist, tp_dist_buy=tp_dist,
                        sl_dist_sell=sl_dist, tp_dist_sell=tp_dist)
                    out = fdf[market_cols].copy()
                    for c in ("open", "high", "low", "close", "spread"):
                        out[c] = fdf[c].values
                    out["time"] = fdf["time"].values
                    for c in geometry_cols:
                        out[c] = gdf[c].values
                    out["direction"] = 1.0 if direction == "BUY" else 0.0
                    out["target"] = tdf["target"].values
                    # force exact cache column order — ragged append impossible
                    out = out.reindex(columns=header).dropna().reset_index(drop=True)
                    out.to_csv(CACHE, mode="a", header=False, index=False)
                    total += len(out)
                    del out, tdf, gdf
        del fdf
        print(f"  period {pi+1}/{len(periods)}: running total {total:,} rows ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n✅ appended {total:,} feature rows (2021) to rally cache | {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
