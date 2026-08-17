"""Debug: print first mismatching rows between source matrix and re-expansion."""
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

# check ATR recovery first: use source row's sl_atr_buy as ground truth atr ratio
# For the SL=0.8 BUY row: sl_dist_buy_src / sl_atr_buy_src should equal atr
print("ATR recovery check (first 5 bars):")
for bi in range(5):
    bt = p0["time"].iloc[bi]
    bar = src[src["time"] == bt].iloc[0]  # first row of bar = SELL sl 0.8 tp 1.3
    # row order: SELL(0) BUY(1) for sl0.8 tp1.3... first row is SELL
    sl_dist_src = float(bar["sl_dist_sell"])
    sl_atr_src = float(bar["sl_atr_sell"])
    atr_src = sl_dist_src / sl_atr_src
    print(f"  bar {bi}: atr_src={atr_src:.5f} atr_recovered={atr[bi]:.5f} "
          f"close={fdf['close'].iloc[bi]:.2f} atr_pct={fdf['atr_pct'].iloc[bi]:.4f}")

# print a mismatching geom row example
print("\nFirst 3 mismatching geom examples (bar, gi, col):")
shown = 0
sl_m = np.array(F.SL_MULTS); tp_r = np.array(old_tp)
sl_dist_all = np.maximum(atr[:, None] * sl_m[None, :], B.MIN_SL_FLOOR)
tp_dist_all = (sl_dist_all[..., None] + spr[:, None, None]) * tp_r[None, None, :]
for bi in range(n):
    bt = p0["time"].iloc[bi]
    bar_rows = src[src["time"] == bt].sort_index()
    for gi, (direction, m, r) in enumerate(GEOMS_OLD):
        sr = bar_rows.iloc[gi]
        si = F.SL_MULTS.index(m); ti = old_tp.index(r)
        if direction == "BUY":
            exp = [sl_dist_all[bi, si], tp_dist_all[bi, si, ti], 0, 0,
                   sl_dist_all[bi, si]/(atr[bi]+1e-9), 0,
                   tp_dist_all[bi, si, ti]/(sl_dist_all[bi, si]+1e-9), 0]
        else:
            exp = [0, 0, sl_dist_all[bi, si], tp_dist_all[bi, si, ti],
                   0, sl_dist_all[bi, si]/(atr[bi]+1e-9),
                   0, tp_dist_all[bi, si, ti]/(sl_dist_all[bi, si]+1e-9)]
        for ci, c in enumerate(["sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
                                "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"]):
            got = float(sr[c]); want = exp[ci]
            if abs(got - want) > 1e-3:
                print(f"  bar {bi} gi {gi} ({direction} m={m} r={r}) col={c}: "
                      f"src={got:.6f} want={want:.6f} atr={atr[bi]:.4f} spr={spr[bi]:.4f}")
                shown += 1
                if shown >= 3:
                    sys.exit(0)
