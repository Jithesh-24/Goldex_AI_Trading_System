"""Dump source rows 32-47 for bar 0 (SL 3.4 and 4.5 groups) — full columns."""
import sys
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)

OUT = f"{BASE}/gold_features_m5.csv"
hdr = list(pd.read_csv(OUT, nrows=0).columns)
df = pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"}, nrows=96)
t0 = df["time"].iloc[0]
bar = df[df["time"] == t0]
print("cols:", [c for c in hdr if c in ("direction","sl_dist_buy","tp_dist_buy",
      "sl_dist_sell","tp_dist_sell","target","mfe_atr","mfa_atr","sl_atr_buy","rr_buy",
      "atr_pct","close","spread")])
print(f"bar {t0} | atr_pct={float(bar['atr_pct'].iloc[0]):.4f} close={float(bar['close'].iloc[0]):.2f} "
      f"spread={float(bar['spread'].iloc[0]):.1f}")
for i in range(32, 48):
    r = bar.iloc[i]
    print(f"row {i:2d}: dir={float(r['direction']):1.0f} sl_b={float(r['sl_dist_buy']):7.4f} "
          f"tp_b={float(r['tp_dist_buy']):8.4f} tgt={float(r['target']):1.0f} "
          f"mfe={float(r['mfe_atr']):7.4f} mfa={float(r['mfa_atr']):7.4f} "
          f"sl_atr_b={float(r['sl_atr_buy']):5.3f} rr_b={float(r['rr_buy']):6.3f}")
# implied sl mult & tp ratio
print("\nimplied: sl_mult = sl_dist/atr ; tp_ratio = tp_dist/(sl_dist+spread/100)")
atr = float(bar["atr_pct"].iloc[0]) * float(bar["close"].iloc[0]) / 100.0
spr = float(bar["spread"].iloc[0]) / 100.0
print(f"atr={atr:.4f} spr={spr:.4f}")
for i in range(32, 48):
    r = bar.iloc[i]
    sl = float(r["sl_dist_sell"]); tp = float(r["tp_dist_sell"])
    print(f"row {i:2d}: sl_mult={sl/atr:.3f} tp_ratio={tp/(sl+spr):.3f}")
