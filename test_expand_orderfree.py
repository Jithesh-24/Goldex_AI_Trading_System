"""ORDER-AGNOSTIC verification: for each bar, match source rows to re-expanded
rows by key (direction|sl_dist|tp_dist), then compare target/mfe_atr/mfa_atr
and geometry cols. Source matrix has permuted TP order in some SL groups
(build artifact), so positional comparison is invalid."""
import sys
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F
import build_m5_matrix as B

OUT = f"{BASE}/gold_features_m5.csv"
CHUNK = 500_000

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
    parts.append(chunk[keep].copy())
    prev_time = times[-1] if len(times) else None
    if sum(keep) >= 400:
        break
bars = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
t = pd.to_datetime(bars["time"])
gap = t.diff().dt.total_seconds().fillna(0).values
period_ids = np.cumsum(gap > 6 * 3600)
periods = [g.reset_index(drop=True) for _, g in bars.groupby(period_ids, sort=False)]
p0 = periods[0]
fdf = p0.reset_index(drop=True)
n = len(fdf)
fdf["atr_14"] = (fdf["atr_pct"] * fdf["close"] / 100.0).values
atr = fdf["atr_14"].values
spr = (fdf["spread"].astype(float) / 100.0).values

old_tp = F.TP_RATIOS[:4]
GEOMS_OLD = [(d, m, r) for m in F.SL_MULTS for r in old_tp for d in ("SELL", "BUY")]
n_old = len(GEOMS_OLD)
sl_m = np.array(F.SL_MULTS); tp_r = np.array(old_tp)
sl_dist_all = np.maximum(atr[:, None] * sl_m[None, :], B.MIN_SL_FLOOR)
tp_dist_all = (sl_dist_all[..., None] + spr[:, None, None]) * tp_r[None, None, :]

# recompute my target/mfe/mfa per geom
tgt = np.empty((n_old, n), dtype=np.float32)
mfe = np.empty((n_old, n), dtype=np.float32)
mfa = np.empty((n_old, n), dtype=np.float32)
for gi, (direction, m, r) in enumerate(GEOMS_OLD):
    sl_dist = np.maximum(atr * m, B.MIN_SL_FLOOR)
    tp_dist = (sl_dist + spr) * r
    tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                             sl_dist=sl_dist, tp_dist=tp_dist, direction=direction)
    tgt[gi] = tdf["target"].values
    mfe[gi] = tdf["mfe_atr"].values
    mfa[gi] = tdf["mfa_atr"].values

def my_key(bi, gi):
    direction, m, r = GEOMS_OLD[gi]
    si = F.SL_MULTS.index(m); ti = old_tp.index(r)
    d = "1" if direction == "BUY" else "0"
    return f"{d}|{sl_dist_all[bi, si]:.5f}|{tp_dist_all[bi, si, ti]:.5f}"

# source rows
tset = set(p0["time"].astype(str))
src = None
for chunk in pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"},
                         chunksize=CHUNK, low_memory=False):
    m = chunk["time"].astype(str).isin(tset)
    part = chunk[m]
    src = part if src is None else pd.concat([src, part])
    if len(src) >= n * n_old:
        break
src = src.reset_index(drop=True)

mm_t = mm_mfe = mm_mfa = mm_g = 0
tot = 0
for bi in range(n):
    bt = p0["time"].iloc[bi]
    bar = src[src["time"] == bt]
    sdict = {}
    for _, r in bar.iterrows():
        d = "1" if float(r["direction"]) > 0.5 else "0"
        k = f"{d}|{float(r['sl_dist_buy']):.5f}|{float(r['tp_dist_buy']):.5f}"
        sdict.setdefault(k, r)
    for gi in range(n_old):
        k = my_key(bi, gi)
        tot += 1
        if k not in sdict:
            mm_g += 1
            continue
        sr = sdict[k]
        if float(sr["target"]) != tgt[gi, bi]:
            mm_t += 1
        if abs(float(sr["mfe_atr"]) - mfe[gi, bi]) > 1e-4:
            mm_mfe += 1
        if abs(float(sr["mfa_atr"]) - mfa[gi, bi]) > 1e-4:
            mm_mfa += 1
        direction, m, r = GEOMS_OLD[gi]
        si = F.SL_MULTS.index(m); ti = old_tp.index(r)
        want_sl_atr = sl_dist_all[bi, si] / (atr[bi] + 1e-9)
        want_rr = tp_dist_all[bi, si, ti] / (sl_dist_all[bi, si] + 1e-9)
        if abs(float(sr["sl_atr_buy"]) - want_sl_atr) > 1e-3 or \
           abs(float(sr["rr_buy"]) - want_rr) > 1e-3:
            mm_g += 1
print(f"compared {tot} rows (order-agnostic) — target Δ: {mm_t} | mfe Δ: {mm_mfe} | "
      f"mfa Δ: {mm_mfa} | geom/key Δ: {mm_g}")
print("PASS" if tot and mm_t == 0 and mm_mfe == 0 and mm_mfa == 0 and mm_g == 0 else "CHECK")
