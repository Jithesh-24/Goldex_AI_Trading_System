"""Deep-dive: for matching keys, print src vs mine mfe/mfa to find the pattern."""
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

# count mismatch by geometry
from collections import Counter
mm_by_gi = Counter()
ex = []
for bi in range(50):
    bt = p0["time"].iloc[bi]
    bar = src[src["time"] == bt]
    sdict = {}
    for _, r in bar.iterrows():
        d = "1" if float(r["direction"]) > 0.5 else "0"
        k = f"{d}|{float(r['sl_dist_buy']):.5f}|{float(r['tp_dist_buy']):.5f}"
        sdict.setdefault(k, r)
    for gi in range(n_old):
        k = my_key(bi, gi)
        if k in sdict:
            sr = sdict[k]
            if abs(float(sr["mfe_atr"]) - mfe[gi, bi]) > 1e-4:
                mm_by_gi[gi] += 1
                if len(ex) < 8:
                    ex.append((bi, gi, GEOMS_OLD[gi], float(sr["mfe_atr"]), mfe[gi, bi],
                               float(sr["mfa_atr"]), mfa[gi, bi], float(sr["target"]), tgt[gi, bi]))
print("mfe mismatch count by geometry (first 50 bars):")
for gi, c in mm_by_gi.most_common():
    print(f"  gi {gi:2d} {str(GEOMS_OLD[gi]):22s} mismatches: {c}/50")
print("\nsamples (bi, gi, geom, src_mfe, my_mfe, src_mfa, my_mfa, src_tgt, my_tgt):")
for e in ex:
    print(" ", e)
