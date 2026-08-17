"""Manual walk: bar 10, gi 0 (SELL m0.8 r1.3) — compare src vs mine in detail."""
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

bi = 10
# src row for bar 10, SELL m0.8 r1.3: find by key
sl_dist = np.maximum(atr[bi] * 0.8, B.MIN_SL_FLOOR)
tp_dist = (sl_dist + spr[bi]) * 1.3
key = f"0|{sl_dist:.5f}|{tp_dist:.5f}"
tset = set(p0["time"].astype(str))
src = None
for chunk in pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"},
                         chunksize=CHUNK, low_memory=False):
    m = chunk["time"].astype(str).isin(tset)
    part = chunk[m]
    src = part if src is None else pd.concat([src, part])
    if len(src) >= n * 48:
        break
src = src.reset_index(drop=True)
bt = p0["time"].iloc[bi]
bar = src[src["time"] == bt]
found = None
for _, r in bar.iterrows():
    d = "1" if float(r["direction"]) > 0.5 else "0"
    k = f"{d}|{float(r['sl_dist_buy']):.5f}|{float(r['tp_dist_buy']):.5f}"
    if k == key:
        found = r
        break
print(f"bar {bi} time {bt} | sl_dist={sl_dist:.5f} tp_dist={tp_dist:.5f}")
print(f"key={key} found={found is not None}")
if found is not None:
    print(f"SRC: tgt={float(found['target'])} mfe={float(found['mfe_atr']):.4f} mfa={float(found['mfa_atr']):.4f}")
tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                         sl_dist=np.maximum(atr*0.8, B.MIN_SL_FLOOR),
                         tp_dist=(np.maximum(atr*0.8, B.MIN_SL_FLOOR) + spr) * 1.3,
                         direction="SELL")
print(f"MINE: tgt={tdf['target'].iloc[bi]} mfe={tdf['mfe_atr'].iloc[bi]:.4f} mfa={tdf['mfa_atr'].iloc[bi]:.4f}")
# now compute by hand for this bar
closes = fdf["close"].values; highs = fdf["high"].values; lows = fdf["low"].values
i = bi
entry = closes[i]
sd = float(np.maximum(atr[i] * 0.8, B.MIN_SL_FLOOR))
td = float((sd + spr[i]) * 1.3)
sl_level = entry + sd + F.SPREAD
tp_level = entry - td
ext = F.MAX_TARGET_BARS * 4
j_end = min(i + 1 + F.MAX_TARGET_BARS, n)
seg_hi = highs[i+1:j_end]; seg_lo = lows[i+1:j_end]
sl_hit = np.where(seg_hi >= sl_level)[0]
tp_hit = np.where(seg_lo <= tp_level)[0]
print(f"entry={entry:.2f} sl_level={sl_level:.2f} tp_level={tp_level:.2f} "
      f"sl_hit={sl_hit[:2]} tp_hit={tp_hit[:2]}")
if not sl_hit.size and not tp_hit.size:
    j_end2 = min(i + 1 + ext, n)
    seg_hi = highs[i+1:j_end2]; seg_lo = lows[i+1:j_end2]
    sl_hit = np.where(seg_hi >= sl_level)[0]
    tp_hit = np.where(seg_lo <= tp_level)[0]
    print(f"extended: sl_hit={sl_hit[:2]} tp_hit={tp_hit[:2]}")
if sl_hit.size and tp_hit.size:
    tgt = 0.0 if sl_hit[0] <= tp_hit[0] else 1.0
elif sl_hit.size: tgt = 0.0
elif tp_hit.size: tgt = 1.0
else: tgt = 1.0 if closes[min(i+1+ext,n)-1] > entry else 0.0
print(f"hand-computed tgt={tgt}")
# check the source's implied levels: what sl/tp would give src mfa/mfe?
print(f"src mfa={float(found['mfa_atr']):.4f} (adverse toward SL) — implied sl_dist={float(found['mfa_atr'])*atr[i]:.3f}")
