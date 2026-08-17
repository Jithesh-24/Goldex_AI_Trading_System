"""Verify expand_geometry_m5.py against the ORIGINAL matrix on a FULL period
(first contiguous period of the matrix, ~1 trading day of bars). Pass the FULL
period frame to add_trade_target so forward lookahead (max_bars) is intact —
the earlier test truncated lookahead by slicing to 120 bars.

Compares: target, mfe_atr, mfa_atr, direction, and all 8 geometry cols for the
OLD grid rows (48/bar) vs the source matrix rows, bar by bar."""
import sys
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F
import build_m5_matrix as B

OUT = f"{BASE}/gold_features_m5.csv"
CHUNK = 500_000

# 1) stream deduped bars until we have a full contiguous period (gap > 6h)
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
    if sum(keep) >= 400:
        break
bars = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
print(f"collected {len(bars)} unique bars | range {bars['time'].iloc[0]} → {bars['time'].iloc[-1]}")

# split into periods by >6h gap (same as build)
t = pd.to_datetime(bars["time"])
gap = t.diff().dt.total_seconds().fillna(0).values
period_ids = np.cumsum(gap > 6 * 3600)
periods = [g.reset_index(drop=True) for _, g in bars.groupby(period_ids, sort=False)]
p0 = periods[0]
print(f"period 0: {len(p0)} bars | {p0['time'].iloc[0]} → {p0['time'].iloc[-1]}")
if len(p0) < 300:
    print("WARN: period 0 too short for full lookahead — using periods 0+1 merged")
    if len(periods) > 1:
        p0 = pd.concat([p0, periods[1]]).reset_index(drop=True)

# 2) full expansion logic on the period (old grid 4 TP = source; new grid 7 TP)
fdf = p0.reset_index(drop=True)
n = len(fdf)
fdf["atr_14"] = (fdf["atr_pct"] * fdf["close"] / 100.0).values
atr = fdf["atr_14"].values
spr = (fdf["spread"].astype(float) / 100.0).values

old_tp = F.TP_RATIOS[:4]
GEOMS_OLD = [(d, m, r) for m in F.SL_MULTS for r in old_tp for d in ("SELL", "BUY")]
n_old = len(GEOMS_OLD)
targets = np.empty((n_old, n), dtype=np.float32)
mfe = np.empty((n_old, n), dtype=np.float32)
mfa = np.empty((n_old, n), dtype=np.float32)
for gi, (direction, m, r) in enumerate(GEOMS_OLD):
    sl_dist = np.maximum(atr * m, B.MIN_SL_FLOOR)
    tp_dist = (sl_dist + spr) * r
    tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                             sl_dist=sl_dist, tp_dist=tp_dist, direction=direction)
    targets[gi] = tdf["target"].values
    mfe[gi] = tdf["mfe_atr"].values
    mfa[gi] = tdf["mfa_atr"].values

# geometry cols (exact features.py formulas)
sl_m = np.array(F.SL_MULTS); tp_r = np.array(old_tp)
sl_dist_all = np.maximum(atr[:, None] * sl_m[None, :], B.MIN_SL_FLOOR)
tp_dist_all = (sl_dist_all[..., None] + spr[:, None, None]) * tp_r[None, None, :]
geom_order = ["sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
              "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"]

# 3) source rows for the period's bars from the matrix
tset = set(p0["time"].astype(str))
src = None
for chunk in pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"},
                         chunksize=CHUNK, low_memory=False):
    m = chunk["time"].astype(str).isin(tset)
    part = chunk[m]
    if src is None:
        src = part
    else:
        src = pd.concat([src, part])
    if len(src) >= n * n_old:
        break
src = src.reset_index(drop=True)
print(f"source rows matched: {len(src)} (expect {n*n_old})")

mm_t, mm_mfe, mm_mfa, mm_geom, mm_dir = 0, 0, 0, 0, 0
tot = 0
for bi in range(n):
    bt = p0["time"].iloc[bi]
    bar_rows = src[src["time"] == bt].sort_index()
    if len(bar_rows) != n_old:
        print(f"bar {bi}: {len(bar_rows)} rows != {n_old} — SKIP")
        continue
    for gi, (direction, m, r) in enumerate(GEOMS_OLD):
        sr = bar_rows.iloc[gi]
        tot += 1
        if float(sr["target"]) != targets[gi, bi]:
            mm_t += 1
        if abs(float(sr["mfe_atr"]) - mfe[gi, bi]) > 1e-4:
            mm_mfe += 1
        if abs(float(sr["mfa_atr"]) - mfa[gi, bi]) > 1e-4:
            mm_mfa += 1
        if float(sr["direction"]) != (1.0 if direction == "BUY" else 0.0):
            mm_dir += 1
        si = F.SL_MULTS.index(m); ti = old_tp.index(r)
        # NOTE: matrix stores ALL 8 geometry cols on every row (both dirs)
        exp = [sl_dist_all[bi, si], tp_dist_all[bi, si, ti],
               sl_dist_all[bi, si], tp_dist_all[bi, si, ti],
               sl_dist_all[bi, si]/(atr[bi]+1e-9),
               sl_dist_all[bi, si]/(atr[bi]+1e-9),
               tp_dist_all[bi, si, ti]/(sl_dist_all[bi, si]+1e-9),
               tp_dist_all[bi, si, ti]/(sl_dist_all[bi, si]+1e-9)]
        for ci, c in enumerate(geom_order):
            if abs(float(sr[c]) - exp[ci]) > 1e-3:
                mm_geom += 1
print(f"compared {tot} rows — target Δ: {mm_t} | mfe Δ: {mm_mfe} | mfa Δ: {mm_mfa} | "
      f"geom Δ: {mm_geom} | dir Δ: {mm_dir}")
print("PASS" if tot and mm_t == 0 and mm_mfe == 0 and mm_mfa == 0 and mm_geom == 0 and mm_dir == 0 else "CHECK MISMATCHES")
