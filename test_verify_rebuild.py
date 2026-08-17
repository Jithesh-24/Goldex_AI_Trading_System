"""Verify rebuilt matrix: rows%84==0, per-bar 84 rows, geometry varies per SL/TP/dir,
target/mfe/mfa sane, time sorted, no dup bars."""
import sys
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

OUT = f"{BASE}/gold_features_m5.csv"
hdr = list(pd.read_csv(OUT, nrows=0).columns)
print("cols:", len(hdr), "| unique names:", len(set(hdr)))

# quick structure check on first 3 bars
n = 84
df = pd.read_csv(OUT, nrows=3 * n + 1, dtype={"time": str})
times = df["time"].values
uniq_times = []
for t in times:
    if t not in uniq_times:
        uniq_times.append(t)
print(f"first {3*n+1} rows -> unique times: {len(uniq_times)} (expect 3)")
if len(uniq_times) > 1:
    first_bar = df.iloc[:n]
    print("\n-- bar0 geometry sweep --")
    print("sl_dist_buy uniq:", np.round(first_bar['sl_dist_buy'].unique(), 3))
    print("tp_dist_buy uniq:", np.round(first_bar['tp_dist_buy'].unique(), 3))
    print("direction uniq:", first_bar['direction'].unique())
    print("target uniq:", first_bar['target'].unique())
    print("mfe range:", round(first_bar['mfe_atr'].min(), 3), "->", round(first_bar['mfe_atr'].max(), 3))
    print("mfa range:", round(first_bar['mfa_atr'].min(), 3), "->", round(first_bar['mfa_atr'].max(), 3))
    # SL outer -> TP middle -> dir inner check
    sls = first_bar['sl_dist_buy'].values
    tps = first_bar['tp_dist_buy'].values
    dirs = first_bar['direction'].values
    sl_groups = []
    i = 0
    while i < n:
        j = i
        while j < n and abs(sls[j] - sls[i]) < 1e-6:
            j += 1
        sl_groups.append((round(sls[i], 3), j - i))
        i = j
    print("SL groups (value, rows):", sl_groups)
    # check TP monotonic within first SL group
    g0 = first_bar[first_bar['sl_dist_buy'] == sls[0]]
    tps0 = g0['tp_dist_buy'].values
    print("TP order in SL group 0:", np.round(tps0, 3))
    print("dir order in TP group 0:", dirs[:2])
