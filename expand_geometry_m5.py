"""expand_geometry_m5.py — v8.4c RE-EXPAND matrix geometry grid to 7 TP ratios.

WHY: user mandate "no hardlimits, not always inside 3" — the model must LEARN
placements beyond TP 3.0. The geometry grid in features.py is now 6 SL × 7 TP
× 2 dir = 84 rows/bar (was 48). The existing matrix has 48 rows/bar with
targets computed ONLY for TP ≤ 3.0 — the model can never honestly evaluate a
4.0/5.5/7.0 TP it was never trained on.

THIS SCRIPT re-expands the matrix WITHOUT recomputing features or losing the
gap layers (2020-H2, 2021, 2022, 2023-H2, 2024-H2, 2025-H2, 2026-H1 live ONLY
in this matrix, NOT in the raw seed):
  1. Stream gold_features_m5.csv, dedupe per bar (features identical across
     the 48 geometry rows of a bar) → per-bar feature frame (~385k bars)
  2. Split into contiguous periods (gap > 6h), same as build_m5_matrix
  3. For each period × direction × SL_MULT × TP_RATIO (84 combos):
     add_trade_target (target/mfe/mfa) + add_geometry_awareness (geometry
     cols) → precompute per-geometry arrays for the WHOLE period
  4. Emit per bar in time order: interleave the 84 geometry rows for bar i
     → output is ALREADY time-sorted → NO external sort, NO 62GB temp files
     (disk: old 18GB + new 31GB ≈ 49GB peak, fits 95GB free)
  5. Atomic swap → gold_features_m5.csv

Same code paths as build_m5_matrix.py — identical labels, just more geometry.
Run: python expand_geometry_m5.py   (~2-4h, num_threads=4, bounded RAM)
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

OUT = f"{BASE}/gold_features_m5.csv"
TMP = f"{BASE}/.m5_expand_tmp"
CHUNK = 500_000
MAX_TARGET_BARS = F.MAX_TARGET_BARS
MIN_SL_FLOOR = F.MIN_SL_FLOOR
# Row order MUST match the actual matrix (verified on-disk): SL outer,
# TP middle, direction inner (SELL=0 then BUY=1). Walk-forward splits align
# bar blocks positionally, so within-bar order must be stable.
GEOMS = [(d, m, r) for m in F.SL_MULTS for r in F.TP_RATIOS
         for d in ("SELL", "BUY")]
GEOMS_DICT = {(m, r, d): gi for gi, (d, m, r) in enumerate(GEOMS)}

def dedupe_bars():
    """Stream matrix, keep ONE row per bar (features identical across the
    48 geometry rows). Cross-chunk safe: tracks seen times (bars are
    contiguous in the time-sorted file; a bar may span a chunk edge)."""
    hdr = pd.read_csv(OUT, nrows=0).columns.tolist()
    parts = []
    prev_time = None   # last time of the previous chunk (bar may span edge)
    for chunk in pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"},
                             chunksize=CHUNK, low_memory=False):
        times = chunk["time"].astype(str).values
        first_in_chunk = np.empty(len(times), dtype=bool)
        first_in_chunk[0] = True
        np.not_equal(times[1:], times[:-1], out=first_in_chunk[1:])
        if prev_time is not None:
            first_in_chunk[times == prev_time] = False   # carried from prev chunk
        keep = first_in_chunk
        one = chunk[keep].copy()
        parts.append(one)
        prev_time = times[-1] if len(times) else None
        del chunk; gc.collect()
    df = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    del parts; gc.collect()
    return df, hdr

def main():
    t0 = time.time()
    print(f"═══ GEOMETRY RE-EXPAND: {len(F.SL_MULTS)} SL × {len(F.TP_RATIOS)} TP × 2 dir "
          f"= {len(GEOMS)} rows/bar (was 48) ═══", flush=True)
    bars, hdr = dedupe_bars()
    print(f"unique bars: {len(bars):,} ({time.time()-t0:.0f}s)", flush=True)

    t = bars["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(bars)]
    periods = [bars.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1)
               if len(bars.iloc[bounds[i]:bounds[i+1]]) >= 100]
    print(f"periods: {len(periods)}", flush=True)

    market_cols = [c for c in hdr if c not in ("time","target","fwd_return",
                   "mfe_atr","mfa_atr") and c not in F.RAW_PRICE_COLS]
    geometry_cols = ["sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
                     "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"]
    out_cols = market_cols + ["open","high","low","close","spread","time"] + \
               geometry_cols + ["direction","target","mfe_atr","mfa_atr"]
    # sanity: same column set as the source matrix
    assert set(out_cols) == set(hdr), f"col mismatch: {set(out_cols) ^ set(hdr)}"

    OUT_tmp = OUT + ".tmp"
    if os.path.exists(OUT_tmp):
        os.remove(OUT_tmp)
    first = True
    total = 0; tb = {0.0: 0, 1.0: 0}
    for pi, p in enumerate(periods):
        fdf = p.reset_index(drop=True)
        n = len(fdf)
        # Recover EXACT per-bar ATR from the stored scale-free feature
        # (atr_pct = atr_14/close*100 — features.py add_scale_free). The
        # matrix rows are post-dropna, so recomputing ATR from OHLC loses the
        # warm-up the original build had; atr_pct round-trips it exactly.
        fdf["atr_14"] = (fdf["atr_pct"] * fdf["close"] / 100.0).values
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values
        # precompute ALL 84 geometries for this period (vectorized per geom)
        targets = np.empty((len(GEOMS), n), dtype=np.float32)
        mfe = np.empty((len(GEOMS), n), dtype=np.float32)
        mfa = np.empty((len(GEOMS), n), dtype=np.float32)
        for gi, (direction, m, r) in enumerate(GEOMS):
            sl_dist = np.maximum(atr * m, MIN_SL_FLOOR)
            tp_dist = (sl_dist + spr) * r
            tdf = F.add_trade_target(fdf, max_bars=MAX_TARGET_BARS,
                                     sl_dist=sl_dist, tp_dist=tp_dist,
                                     direction=direction)
            targets[gi] = tdf["target"].values
            mfe[gi] = tdf["mfe_atr"].values
            mfa[gi] = tdf["mfa_atr"].values
        # geometry cols are per-geom; build a compact per-bar geometry matrix
        # sl_dist/tp_dist/rr per (direction × SL × TP) — recompute cheaply
        sl_m = np.array(F.SL_MULTS); tp_r = np.array(F.TP_RATIOS)
        n_sl, n_tp = len(sl_m), len(tp_r)
        sl_dist_all = np.maximum(atr[:, None] * sl_m[None, :], MIN_SL_FLOOR)  # (n,6)
        tp_dist_all = (sl_dist_all[..., None] + spr[:, None, None]) * tp_r[None, None, :]  # (n,6,7)
        dir_col = np.zeros((n, len(GEOMS)), dtype=np.float32)
        gcols = np.zeros((n, len(GEOMS), len(geometry_cols)), dtype=np.float32)
        # geometry columns order: sl_dist_buy, tp_dist_buy, sl_dist_sell,
        #   tp_dist_sell, sl_atr_buy, sl_atr_sell, rr_buy, rr_sell
        # NOTE (verified): the matrix stores ALL 8 geometry cols on EVERY row —
        # both directions' sl/tp/rr are populated regardless of row direction.
        for gi, (direction, m, r) in enumerate(GEOMS):
            dir_col[:, gi] = 1.0 if direction == "BUY" else 0.0
            si = F.SL_MULTS.index(m); ti = F.TP_RATIOS.index(r)
            gcols[:, gi, 0] = sl_dist_all[:, si]
            gcols[:, gi, 1] = tp_dist_all[:, si, ti]
            gcols[:, gi, 2] = sl_dist_all[:, si]
            gcols[:, gi, 3] = tp_dist_all[:, si, ti]
            gcols[:, gi, 4] = sl_dist_all[:, si] / (atr + 1e-9)
            gcols[:, gi, 5] = sl_dist_all[:, si] / (atr + 1e-9)
            gcols[:, gi, 6] = tp_dist_all[:, si, ti] / (sl_dist_all[:, si] + 1e-9)
            gcols[:, gi, 7] = tp_dist_all[:, si, ti] / (sl_dist_all[:, si] + 1e-9)
        # emit bar-by-bar, geometry rows in order — output stays time-sorted
        # NOTE: out_cols layout = market_cols | open..spread | time | geometry(8)
        #   | direction | target | mfe_atr | mfa_atr — block writes must SKIP the
        #   time slot (n_mc+5) and start geometry at n_mc+6 (off-by-one that
        #   corrupted v8.4c run 1: everything after spread shifted by 1).
        n_mc = len(market_cols)
        geom0 = n_mc + 6          # first geometry col (after open..spread + time)
        dir0 = geom0 + len(geometry_cols)      # direction
        tgt0 = dir0 + 1                        # target
        mfe0 = tgt0 + 1                        # mfe_atr
        mfa0 = mfe0 + 1                        # mfa_atr
        feats_np = fdf[market_cols].values.astype(np.float32)
        for c in ("open","high","low","close","spread"):
            assert c in fdf.columns, c
        raw_np = fdf[["open","high","low","close","spread"]].values.astype(np.float32)
        times = fdf["time"].values
        # build output in big blocks (1000 bars at a time) → fast CSV append
        blk = 1000
        for b0 in range(0, n, blk):
            b1 = min(b0 + blk, n)
            nb = b1 - b0
            block = np.empty((nb * len(GEOMS), len(out_cols)), dtype=np.float32)
            for j, b in enumerate(range(b0, b1)):
                rows = slice(j * len(GEOMS), (j+1) * len(GEOMS))
                # market feats (same for all geoms of bar b)
                for ci, c in enumerate(market_cols):
                    block[rows, ci] = feats_np[b, ci]
                for ci, c in enumerate(["open","high","low","close","spread"]):
                    block[rows, n_mc + ci] = raw_np[b, ci]
                block[rows, geom0:geom0 + len(geometry_cols)] = gcols[b]
                block[rows, dir0] = dir_col[b]
                block[rows, tgt0] = targets[:, b]
                block[rows, mfe0] = mfe[:, b]
                block[rows, mfa0] = mfa[:, b]
            out_df = pd.DataFrame(block, columns=out_cols)
            out_df["time"] = np.repeat(times[b0:b1], len(GEOMS))
            out_df.to_csv(OUT_tmp, mode="a", header=first, index=False)
            first = False
            total += len(out_df)
            tb[0.0] += int((out_df["target"] == 0).sum())
            tb[1.0] += int((out_df["target"] == 1).sum())
            del out_df, block; gc.collect()
        del targets, mfe, mfa, gcols, dir_col, fdf, p; gc.collect()
        print(f"  period {pi+1}/{len(periods)} done ({time.time()-t0:.0f}s)",
              flush=True)

    os.replace(OUT_tmp, OUT)  # atomic swap
    # refresh BOTH schema sidecars with the canonical fingerprint so future
    # incremental appends (build_m5_matrix --incremental / append_missing_layers)
    # see fp match and do NOT trigger a full rebuild (which would lose layers)
    from build_m5_matrix import schema_fingerprint as _sfp
    fp = _sfp()
    with open(f"{BASE}/.matrix_schema_m5.json", "w") as f:
        json.dump({"fp": fp, "built_at": time.time(), "rows": total,
                   "expanded_at": time.time()}, f)
    with open(f"{BASE}/models/matrix_schema.json", "w") as f:
        json.dump({"fp": fp, "built_at": time.time(), "rows": total,
                   "expanded_at": time.time()}, f)
    print(f"✅ RE-EXPANDED: {total:,} rows | balance {tb} | {time.time()-t0:.0f}s",
          flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
