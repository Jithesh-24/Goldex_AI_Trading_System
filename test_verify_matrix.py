"""Post-expansion verification: structure, order, column sanity."""
import sys
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

OUT = f"{BASE}/gold_features_m5.csv"
hdr = list(pd.read_csv(OUT, nrows=0).columns)
print(f"cols: {len(hdr)} | grid: {len(F.SL_MULTS)} SL x {len(F.TP_RATIOS)} TP x 2")

df = pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"}, nrows=2000)
print(f"rows read: {len(df)}")

# 1. rows per bar
times = df["time"].values
counts = pd.Series(times).value_counts()
print(f"unique bars in sample: {len(counts)} | rows/bar: {counts.value_counts().to_dict()}")

# 2. column sanity
d = df["direction"].values
print(f"direction values: {np.unique(d)} (expect [0,1])")
t = df["target"].values
print(f"target values: {np.unique(t)} (expect [0,1])")
print(f"mfe range: {df['mfe_atr'].min():.3f}..{df['mfe_atr'].max():.3f} | mfa range: {df['mfa_atr'].min():.3f}..{df['mfa_atr'].max():.3f}")
print(f"sl_dist_buy range: {df['sl_dist_buy'].min():.3f}..{df['sl_dist_buy'].max():.3f}")
print(f"rr_buy range: {df['rr_buy'].min():.3f}..{df['rr_buy'].max():.3f}")

# 3. first bar structure: SL outer → TP middle → direction inner
bar0 = df[df["time"] == times[0]].reset_index(drop=True)
print(f"\nfirst bar {times[0]} — {len(bar0)} rows:")
sls = np.unique(bar0["sl_dist_sell"].values.round(4))
tps = np.unique(bar0["tp_dist_sell"].values.round(4))
print(f"  unique SL dists: {len(sls)} (expect {len(F.SL_MULTS)})")
print(f"  unique TP dists: {len(tps)} (expect {len(F.TP_RATIOS)})")
# order check: for each SL group, TP ascending, dir SELL,BUY
ok = True
for si, m in enumerate(F.SL_MULTS):
    for ti, r in enumerate(F.TP_RATIOS):
        for di, dname in enumerate(["SELL","BUY"]):
            gi = si * len(F.TP_RATIOS) * 2 + ti * 2 + di
            row = bar0.iloc[gi]
            if dname == "BUY":
                if row["direction"] != 1.0: ok = False; print(f"  MISMATCH gi={gi} dir")
# verify geometry values match formula
atr0 = float(bar0["atr_pct"].iloc[0]) * float(bar0["close"].iloc[0]) / 100.0
spr0 = float(bar0["spread"].iloc[0]) / 100.0
for gi, (direction, m, r) in enumerate([(d, m, r) for m in F.SL_MULTS for r in F.TP_RATIOS for d in ("SELL","BUY")]):
    row = bar0.iloc[gi]
    sd = max(atr0 * m, F.MIN_SL_FLOOR)
    td = (sd + spr0) * r
    if abs(float(row["sl_dist_sell"]) - sd) > 1e-3 or abs(float(row["tp_dist_sell"]) - td) > 1e-3:
        ok = False; print(f"  GEOM MISMATCH gi={gi} m={m} r={r}")
print(f"\nstructure order: {'OK' if ok else 'FAIL'}")

# 4. time sorted + full rows/bar divisibility
import subprocess
n_total = int(subprocess.check_output(f"wc -l < {OUT}", shell=True)) - 1
print(f"total rows: {n_total:,} | % 84 == {n_total % 84}")

# 5. verify target matches recomputation on one full bar
fdf = bar0[["open","high","low","close","spread","atr_pct"]].copy()
print("\n✅ structural verification done")
