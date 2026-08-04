"""Streaming rally feature matrix builder (v6) — cgroup-safe.

The gateway runs inside a 2GB memory cgroup. The naive
build_placement_dataset() materializes the whole matrix (4.9M rows ≈ 3GB)
and gets OOM-killed — which also kills the gateway session.

This script builds the SAME matrix but streams it: for each contiguous
period, compute the feature block ONCE, then write each of the 24
(direction × SL × TP) placement blocks to the output CSV immediately and
free memory. Peak RSS stays ~50-100MB.

Output: gold_features_rally.csv (identical schema to build_placement_dataset)
"""
import sys, time, os
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

OUT = f"{BASE}/gold_features_rally.csv"
SEED = f"{BASE}/gold_seed_multi.csv"
XM_SEED = f"{BASE}/gold_seed.csv"


def periods_of(df):
    """Split into contiguous periods (gap > 6h) — same rule as features.py."""
    t = df["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(df)]
    return [df.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)]


def main():
    t0 = time.time()
    # ATOMIC START: truncate stale output (features.py or grid changes produce a
    # different column count — append-mode leftovers make ragged CSVs that crash
    # pd.read_csv chunked reads with IndexError in _concatenate_chunks).
    if os.path.exists(OUT):
        os.remove(OUT)
    multi = pd.read_csv(SEED)
    multi["time"] = pd.to_datetime(multi["time"])
    xm0 = pd.read_csv(XM_SEED)["time"].min()
    rally = multi[multi["time"] < xm0].reset_index(drop=True)
    print(f"rally bars: {len(rally)} | {rally['time'].iloc[0]} -> {rally['time'].iloc[-1]}", flush=True)

    periods = [p for p in periods_of(rally) if len(p) >= 300]
    print(f"periods: {len(periods)}", flush=True)

    total_rows = 0
    first = True
    for pi, p in enumerate(periods):
        fdf = F._feature_block(p).dropna().reset_index(drop=True)
        if len(fdf) < 100:
            continue
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values if "spread" in fdf.columns else np.full(len(fdf), F.SPREAD)
        market_cols = [c for c in fdf.columns
                       if c not in ("time", "target", "fwd_return") and c not in F.RAW_PRICE_COLS]
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
                    for c in ("sl_dist_buy", "tp_dist_buy", "sl_dist_sell", "tp_dist_sell",
                              "sl_atr_buy", "sl_atr_sell", "rr_buy", "rr_sell"):
                        out[c] = gdf[c].values
                    out["direction"] = 1.0 if direction == "BUY" else 0.0
                    out["target"] = tdf["target"].values
                    out = out.dropna().reset_index(drop=True)
                    out.to_csv(OUT, mode="a", header=first, index=False)
                    first = False
                    total_rows += len(out)
                    del out, tdf, gdf
        del fdf
        print(f"period {pi+1}/{len(periods)} done ({len(p)} bars) — total {total_rows:,} rows", flush=True)

    print(f"DONE: {total_rows:,} rows -> {OUT} in {time.time()-t0:.0f}s", flush=True)
    print("target balance:", pd.read_csv(OUT, usecols=["target"])["target"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
