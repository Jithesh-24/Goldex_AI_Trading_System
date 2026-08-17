"""Merge ALL raw M1 sources into one canonical file for a full matrix rebuild.
Sources: 6yr seed (base), gap files (clean UTC), Dukascopy 2024/2025 (EST→+5h),
current XM seed (newest). Dedup by time — newest source wins (matches
load_raw semantics: drop_duplicates(keep=last) after concat order).

Output: gold_seed_merged_full6yr.csv with cols time,open,high,low,close,
tick_volume,spread,real_volume
"""
import pandas as pd
import numpy as np

BASE = "/home/jith/.hermes/profiles/trading/scripts"
DUKA_OFFSET_H = 5

SOURCES = [
    # (path, kind, shift_h, src_label)
    (f"{BASE}/gold_seed_full6yr.csv", "seed", 0, "seed6yr"),
    (f"{BASE}/gap_m1_2020.csv", "seed", 0, "gap2020"),
    (f"{BASE}/gold_m1_2021.csv", "seed", 0, "gap2021"),
    (f"{BASE}/gap_m1_2022a.csv", "seed", 0, "gap2022a"),
    (f"{BASE}/gap_m1_2022b.csv", "seed", 0, "gap2022b"),
    (f"{BASE}/gap_m1_2023.csv", "seed", 0, "gap2023"),
    (f"{BASE}/gap_m1_2026.csv", "seed", 0, "gap2026"),
    ("/home/jith/xau_cascade/data/raw_m1/DAT_ASCII_XAUUSD_M1_2024.csv", "duka", DUKA_OFFSET_H, "duka2024"),
    ("/home/jith/xau_cascade/data/raw_m1/DAT_ASCII_XAUUSD_M1_2025.csv", "duka", DUKA_OFFSET_H, "duka2025"),
    (f"{BASE}/gold_seed.csv", "seed", 0, "xmcur"),
]

KEEP = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

def load_one(path, kind, shift_h):
    if kind == "duka":
        raw = pd.read_csv(path, sep=";", header=None,
                          names=["dt", "open", "high", "low", "close", "vol"])
        # dt format: YYYYMMDD HHMMSS
        ts = pd.to_datetime(raw["dt"], format="%Y%m%d %H%M%S")
        ts = ts + pd.Timedelta(hours=shift_h)
        df = pd.DataFrame({
            "time": ts, "open": raw["open"], "high": raw["high"],
            "low": raw["low"], "close": raw["close"],
            "tick_volume": raw["vol"].astype(float),
            "spread": 20.0,  # F.SPREAD*100 points (synthesize like load_raw)
            "real_volume": raw["vol"].astype(float),
        })
    else:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0.0
        if "spread" not in df.columns:
            df["spread"] = 20.0
        if "real_volume" not in df.columns:
            df["real_volume"] = df["tick_volume"].fillna(0.0)  # matches load_raw()
    return df[KEEP]

parts = []
for path, kind, sh, label in SOURCES:
    df = load_one(path, kind, sh)
    print(f"{label}: {len(df):,} rows  {df['time'].min()} -> {df['time'].max()}", flush=True)
    parts.append(df)

all_df = pd.concat(parts, ignore_index=True)
all_df = all_df.drop_duplicates(subset="time", keep="last")  # newest source wins (last in concat order)
all_df = all_df.sort_values("time").reset_index(drop=True)
print(f"\nMERGED: {len(all_df):,} unique M1 rows  {all_df['time'].min()} -> {all_df['time'].max()}")

out = f"{BASE}/gold_seed_merged_full6yr.csv"
all_df.to_csv(out, index=False)
print(f"WROTE {out}")

# monthly coverage check
m = all_df.set_index("time")
mc = m.resample("MS").size()
n_months = mc[mc > 0].shape[0]
print(f"months with data: {n_months} / {mc.shape[0]}")
print("first month:", mc.index[0].strftime('%Y-%m'))
print("last month:", mc.index[-1].strftime('%Y-%m'))
