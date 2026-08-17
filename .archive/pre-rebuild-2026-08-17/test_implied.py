"""Recover the source's IMPLIED sl_dist/tp_dist from its stored mfe/mfa.
For SL-hit resolutions: mfa ≈ sl_dist/atr. For TP-hit: mfe ≈ tp_dist/atr.
Compare implied vs stored geometry to find the source's real placements."""
import sys
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F
import build_m5_matrix as B

OUT = f"{BASE}/gold_features_m5.csv"

hdr = list(pd.read_csv(OUT, nrows=0).columns)
# read first bar only
df = pd.read_csv(OUT, dtype={c: np.float32 for c in hdr if c != "time"}, nrows=96)
t0 = df["time"].iloc[0]
bar = df[df["time"] == t0]
atr = float(bar["atr_pct"].iloc[0]) * float(bar["close"].iloc[0]) / 100.0
spr = float(bar["spread"].iloc[0]) / 100.0
print(f"bar {t0} | atr={atr:.4f} spr={spr:.4f}")
print("row | dir | stored_sl | stored_tp | tgt | mfe | mfa | implied_sl(mfa*atr) | implied_tp(mfe*atr)")
for i in range(0, 48, 2):
    r = bar.iloc[i]
    d = "SELL" if float(r["direction"]) < 0.5 else "BUY "
    sl = float(r["sl_dist_sell"]); tp = float(r["tp_dist_sell"])
    mfe = float(r["mfe_atr"]); mfa = float(r["mfa_atr"])
    imp_sl = mfa * atr
    imp_tp = mfe * atr
    flag = ""
    if abs(imp_sl - sl) > 0.5 * sl and abs(imp_sl - tp) < 0.5 * tp:
        flag = " <== mfa≈tp!"
    if abs(imp_tp - tp) > 0.5 * tp and abs(imp_tp - sl) < 0.5 * sl:
        flag = " <== mfe≈sl!"
    print(f"{i:3d} | {d} | {sl:7.4f} | {tp:8.4f} | {float(r['target']):1.0f} | "
          f"{mfe:6.3f} | {mfa:6.3f} | {imp_sl:8.4f} | {imp_tp:8.4f}{flag}")
