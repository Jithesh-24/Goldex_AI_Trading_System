"""Inspect the actual geometry row order within one bar of the source matrix."""
import pandas as pd
import numpy as np

hdr = list(pd.read_csv("/home/jith/.hermes/profiles/trading/scripts/gold_features_m5.csv", nrows=0).columns)
df = pd.read_csv("/home/jith/.hermes/profiles/trading/scripts/gold_features_m5.csv",
                 dtype={c: np.float32 for c in hdr if c != "time"}, nrows=200)
t0 = df["time"].iloc[0]
bar = df[df["time"] == t0]
print(f"bar time: {t0} | rows: {len(bar)}")
print(f"cols: direction sl_dist_buy tp_dist_buy sl_dist_sell tp_dist_sell sl_atr_buy sl_atr_sell rr_buy rr_sell")
for i in range(len(bar)):
    r = bar.iloc[i]
    print(f"{i:3d} | {float(r['direction']):4.1f} {float(r['sl_dist_buy']):8.3f} {float(r['tp_dist_buy']):8.3f} "
          f"{float(r['sl_dist_sell']):8.3f} {float(r['tp_dist_sell']):8.3f} "
          f"{float(r['sl_atr_buy']):6.3f} {float(r['sl_atr_sell']):6.3f} "
          f"{float(r['rr_buy']):6.3f} {float(r['rr_sell']):6.3f}")
# also check time format
print("time dtype sample:", repr(t0))
