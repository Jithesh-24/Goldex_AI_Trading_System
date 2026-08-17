"""Brute-force: what (sl_dist, tp_dist) produces source's (tgt=0, mfe=3.3400,
mfa=1.4803) for SELL at bar 0? Scans the placement grid."""
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
    if sum(keep) >= 200:
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

# scan grid for SELL bar 0: source says tgt=0, mfe=3.3400, mfa=1.4803
targets = (0.0, 3.3400, 1.4803)
sl_grid = np.arange(0.5, 5.0, 0.1)
tp_grid = np.arange(1.0, 16.0, 0.5)
print("scanning...")
best = []
for sd in sl_grid:
    for td in tp_grid:
        tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                                 sl_dist=sd, tp_dist=td, direction="SELL")
        t, mfe, mfa = tdf["target"].iloc[0], tdf["mfe_atr"].iloc[0], tdf["mfa_atr"].iloc[0]
        err = abs(t - targets[0]) + abs(mfe - targets[1]) + abs(mfa - targets[2])
        best.append((err, sd, td, t, mfe, mfa))
best.sort(key=lambda x: x[0])
print("top 8 matches:")
for e, sd, td, t, mfe, mfa in best[:8]:
    print(f"  err={e:.4f} sl_dist={sd:.1f} tp_dist={td:.1f} -> tgt={t} mfe={mfe:.4f} mfa={mfa:.4f}")
# what grid value corresponds?
print("\nsource stored for SL3.4 row: sl=3.6749 tp=11.6247 (r=3.0 pos) | sl=3.6749 tp=5.0374 (r=1.3 pos)")
# try sl = (atr*1.3+spr), tp = (atr*3.4+spr)*1.3 etc — the ROTATION hypothesis
atr0 = atr[0]
spr = 0.2
for m in F.SL_MULTS:
    for r in F.TP_RATIOS[:4]:
        sd = np.maximum(atr0 * m, B.MIN_SL_FLOOR)
        td = (sd + spr) * r
        tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                                 sl_dist=sd, tp_dist=td, direction="SELL")
        t, mfe, mfa = tdf["target"].iloc[0], tdf["mfe_atr"].iloc[0], tdf["mfa_atr"].iloc[0]
        mark = " <<<" if abs(mfe - 3.34) < 0.01 and abs(mfa - 1.48) < 0.01 else ""
        print(f"  m={m} r={r}: tgt={t} mfe={mfe:.4f} mfa={mfa:.4f}{mark}")
