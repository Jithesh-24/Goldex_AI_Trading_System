"""Scale test: verify expand_geometry_m5.py reproduces the ORIGINAL build's
labels for the OLD grid (first 4 TP ratios) on a small slice, and that the
NEW grid adds correct rows. Reads the first ~100 bars of the matrix,
re-expands with the 7-TP grid, compares old-grid rows vs the source matrix."""
import gc, os, sys
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F
import build_m5_matrix as B

OUT = f"{BASE}/gold_features_m5.csv"
CHUNK = 200_000

# 1) take first ~120 unique bars from the matrix
hdr = list(pd.read_csv(OUT, nrows=0).columns)
parts = []
prev_time = None
for chunk in pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"},
                         chunksize=CHUNK, low_memory=False):
    times = chunk["time"].astype(str).values
    first_in_chunk = np.empty(len(times), dtype=bool)
    first_in_chunk[0] = True
    np.not_equal(times[1:], times[:-1], out=first_in_chunk[1:])
    if prev_time is not None:
        first_in_chunk[times == prev_time] = False
    keep = first_in_chunk
    one = chunk[keep].copy()
    parts.append(one)
    prev_time = times[-1] if len(times) else None
    if sum(keep) >= 120:
        break
bars = pd.concat(parts, ignore_index=True).head(120).sort_values("time").reset_index(drop=True)
print(f"test bars: {len(bars)} | range {bars['time'].iloc[0]} → {bars['time'].iloc[-1]}")

# 2) source rows for these bars from the ORIGINAL matrix (48 rows/bar)
src = pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"},
                  nrows=120 * 48 + 10)
src = src[src["time"].isin(set(bars["time"]))].reset_index(drop=True)
print(f"source rows: {len(src)} (expect {len(bars)}*48 = {len(bars)*48})")

# 3) run the expand logic on the 120 bars (single period)
fdf = bars.reset_index(drop=True)
n = len(fdf)
raw_ohlc = fdf[["time","open","high","low","close","spread"]].copy()
fdf["atr_14"] = (fdf["atr_pct"] * fdf["close"] / 100.0).values
atr = fdf["atr_14"].values
spr = (fdf["spread"].astype(float) / 100.0).values
GEOMS = [(d, m, r) for m in F.SL_MULTS for r in F.TP_RATIOS
         for d in ("SELL", "BUY")]
old_tp = F.TP_RATIOS[:4]  # the OLD grid the source matrix was built with

# re-expand with OLD grid only (4 TP) → must MATCH source rows exactly
GEOMS_OLD = [(d, m, r) for m in F.SL_MULTS for r in old_tp
             for d in ("SELL", "BUY")]
n_old = len(GEOMS_OLD)
targets_old = np.empty((n_old, n), dtype=np.float32)
mfe_old = np.empty((n_old, n), dtype=np.float32)
mfa_old = np.empty((n_old, n), dtype=np.float32)
for gi, (direction, m, r) in enumerate(GEOMS_OLD):
    sl_dist = np.maximum(atr * m, B.MIN_SL_FLOOR)
    tp_dist = (sl_dist + spr) * r
    tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                             sl_dist=sl_dist, tp_dist=tp_dist, direction=direction)
    targets_old[gi] = tdf["target"].values
    mfe_old[gi] = tdf["mfe_atr"].values
    mfa_old[gi] = tdf["mfa_atr"].values

# geometry cols per add_geometry_awareness for old grid
sl_m = np.array(F.SL_MULTS); tp_r = np.array(old_tp)
sl_dist_all = np.maximum(atr[:, None] * sl_m[None, :], B.MIN_SL_FLOOR)
tp_dist_all = (sl_dist_all[..., None] + spr[:, None, None]) * tp_r[None, None, :]
geom_order = ["sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
              "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"]

mismatch_target = 0
mismatch_mfe = 0
mismatch_geom = 0
tot = 0
for bi in range(n):
    bar_rows = src[src["time"] == bars["time"].iloc[bi]].sort_index()
    # source rows are ordered direction outer, SL middle, TP inner (same GEOMS_OLD)
    assert len(bar_rows) == n_old, f"bar {bi}: {len(bar_rows)} src rows != {n_old}"
    for gi, (direction, m, r) in enumerate(GEOMS_OLD):
        sr = bar_rows.iloc[gi]
        tot += 1
        if float(sr["target"]) != targets_old[gi, bi]:
            mismatch_target += 1
        if abs(float(sr["mfe_atr"]) - mfe_old[gi, bi]) > 1e-4:
            mismatch_mfe += 1
        si = F.SL_MULTS.index(m); ti = old_tp.index(r)
        if direction == "BUY":
            exp = [sl_dist_all[bi, si], tp_dist_all[bi, si, ti], 0, 0,
                   sl_dist_all[bi, si]/(atr[bi]+1e-9), 0,
                   tp_dist_all[bi, si, ti]/(sl_dist_all[bi, si]+1e-9), 0]
        else:
            exp = [0, 0, sl_dist_all[bi, si], tp_dist_all[bi, si, ti],
                   0, sl_dist_all[bi, si]/(atr[bi]+1e-9),
                   0, tp_dist_all[bi, si, ti]/(sl_dist_all[bi, si]+1e-9)]
        for ci, c in enumerate(geom_order):
            if abs(float(sr[c]) - exp[ci]) > 1e-3:
                mismatch_geom += 1
print(f"compared {tot} rows (old grid) — target mismatches: {mismatch_target}, "
      f"mfe mismatches: {mismatch_mfe}, geom mismatches: {mismatch_geom}")
# direction check
dir_mm = 0
for bi in range(n):
    bar_rows = src[src["time"] == bars["time"].iloc[bi]]
    for gi, (direction, m, r) in enumerate(GEOMS_OLD):
        if float(bar_rows.iloc[gi]["direction"]) != (1.0 if direction == "BUY" else 0.0):
            dir_mm += 1
print(f"direction mismatches: {dir_mm}")
# 4) NEW grid sanity: TP 4.0 rows have higher mfe requirement — just verify targets exist & balance changes
GEOMS_NEW = [(d, m, r) for m in F.SL_MULTS for r in F.TP_RATIOS
             for d in ("SELL", "BUY")]
print(f"new grid rows/bar: {len(GEOMS_NEW)} (old {n_old}) — target sums per TP ratio:")
for r in F.TP_RATIOS:
    tot_targets = 0
    for gi, (d, m, rr) in enumerate(GEOMS_NEW):
        if rr != r:
            continue
        sl_dist = np.maximum(atr * m, B.MIN_SL_FLOOR)
        tp_dist = (sl_dist + spr) * rr
        tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                                 sl_dist=sl_dist, tp_dist=tp_dist, direction=d)
        tot_targets += int(tdf["target"].sum())
    print(f"  TP {r}: {tot_targets} wins / {n*12} rows")
print("SCALE TEST DONE")
