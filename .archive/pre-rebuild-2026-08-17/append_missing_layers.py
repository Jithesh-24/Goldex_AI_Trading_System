"""append_missing_layers.py — v8.4 (2026-08-08)

Incremental memory-append: add ONLY the missing M1 layers to the M5 matrix.
NO full rebuild, NO full retrain. The user mandate:
  "no full retrain only the missing data should be retrained and appended
   so the model will be aware of all kind of markets"

Recoverable layers (verified on disk):
  2021        -> gold_m1_2021.csv            (clean UTC, src=duka)
  2024 full   -> DAT_ASCII_XAUUSD_M1_2024.csv (Dukascopy EST, UTC = file+5h)
  2025 full   -> DAT_ASCII_XAUUSD_M1_2025.csv (Dukascopy EST, UTC = file+5h)
  (incl. the Dec 2025 dump: 4548 -> 4311 on Mon Dec 29 2025)

Existing matrix (gold_features_m5.csv) already covers:
  2019-12..2020-08, 2022-03..09, 2023-01..04, 2024-01..05, 2025-01..05, 2026-06..08

Pipeline per layer (mirrors build_m5_matrix.py exactly):
  M1 -> resample 5min -> feature block (108 cols) -> geometry expansion
  (2 dir x 6 SL x 4 TP = 48 rows/bar) -> append to matrix, dedup by time.

The script WRITES the appended rows to gold_features_m5.csv (atomic: build a
temp file, verify, then swap). Existing rows always win on time conflict.
"""
import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from build_m5_matrix import _feature_block_m5, to_m5, schema_fingerprint, MAX_TARGET_BARS, MIN_SL_FLOOR

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUT = f"{BASE}/gold_features_m5.csv"
SCHEMA = f"{BASE}/models/matrix_schema.json"
DUKA_OFFSET_H = 5  # Dukascopy DAT_ASCII XAUUSD = EST convention; UTC = file + 5h

LAYERS = [
    # (label, path, tz_shift_h)  — shift>0 = DAT_ASCII EST; shift==0 = seed format UTC
    ("2021", f"{BASE}/gold_m1_2021.csv", 0),
    ("2020", f"{BASE}/gap_m1_2020.csv", 0),
    ("2022a", f"{BASE}/gap_m1_2022a.csv", 0),
    ("2022b", f"{BASE}/gap_m1_2022b.csv", 0),
    ("2023", f"{BASE}/gap_m1_2023.csv", 0),
    ("2026", f"{BASE}/gap_m1_2026.csv", 0),
    ("2024", "/home/jith/xau_cascade/data/raw_m1/DAT_ASCII_XAUUSD_M1_2024.csv", DUKA_OFFSET_H),
    ("2025", "/home/jith/xau_cascade/data/raw_m1/DAT_ASCII_XAUUSD_M1_2025.csv", DUKA_OFFSET_H),
]


def load_m1(label, path, shift_h):
    if shift_h == 0:
        # seed-format CSV (2021 + gap files): time,open,high,low,close,tick_volume,spread,real_volume[,src]
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        keep = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        for c in keep:
            if c not in df.columns:
                df[c] = 0.0 if c in ("tick_volume", "real_volume", "spread") else np.nan
        df = df[keep]
    else:
        raw = pd.read_csv(path, sep=";", header=None,
                          names=["dt", "open", "high", "low", "close", "vol"])
        t = pd.to_datetime(raw["dt"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=shift_h)
        df = pd.DataFrame({
            "time": t, "open": raw["open"], "high": raw["high"],
            "low": raw["low"], "close": raw["close"],
            "tick_volume": raw["vol"], "spread": F.SPREAD * 100.0,
            "real_volume": raw["vol"],
        })
    df = df.drop_duplicates(subset="time", keep="last").sort_values("time").reset_index(drop=True)
    print(f"  {label}: {len(df):,} M1 bars | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}", flush=True)
    return df


def feature_expand(m5):
    """Mirror build_m5_matrix.py geometry expansion (48 rows/bar)."""
    t0 = time.time()
    t = m5["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(m5)]
    periods = [m5.iloc[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)
               if len(m5.iloc[bounds[i]:bounds[i + 1]]) >= 300]
    print(f"  periods: {len(periods)}", flush=True)

    header = list(pd.read_csv(OUT, nrows=0).columns)

    out_rows = []
    for pi, p in enumerate(periods):
        fdf = _feature_block_m5(p).dropna().reset_index(drop=True)
        if len(fdf) < 100:
            continue
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values
        market_cols = [c for c in fdf.columns
                       if c not in ("time", "target", "fwd_return", "mfe_atr", "mfa_atr")
                       and c not in F.RAW_PRICE_COLS]
        for direction in ("BUY", "SELL"):
            for m in F.SL_MULTS:
                for r in F.TP_RATIOS:
                    sl_dist = np.maximum(atr * m, MIN_SL_FLOOR)
                    tp_dist = (sl_dist + spr) * r
                    tdf = F.add_trade_target(fdf, max_bars=MAX_TARGET_BARS,
                                                   sl_dist=sl_dist, tp_dist=tp_dist,
                                                   direction=direction)
                    gdf = F.add_geometry_awareness(
                        fdf, sl_dist_buy=sl_dist, tp_dist_buy=tp_dist,
                        sl_dist_sell=sl_dist, tp_dist_sell=tp_dist)
                    out_f = fdf[market_cols].copy()
                    for c in ("open", "high", "low", "close", "spread"):
                        out_f[c] = fdf[c].values
                    out_f["time"] = fdf["time"].values
                    for c in ("sl_dist_buy", "tp_dist_buy", "sl_dist_sell", "tp_dist_sell",
                              "sl_atr_buy", "sl_atr_sell", "rr_buy", "rr_sell"):
                        out_f[c] = gdf[c].values
                    out_f["direction"] = 1.0 if direction == "BUY" else 0.0
                    out_f["target"] = tdf["target"].values
                    out_f["mfe_atr"] = tdf["mfe_atr"].values
                    out_f["mfa_atr"] = tdf["mfa_atr"].values
                    out_rows.append(out_f)
                    del out_f, tdf, gdf
        print(f"  period {pi + 1}/{len(periods)} done ({time.time() - t0:.0f}s)", flush=True)
        del fdf
    if not out_rows:
        return None
    block = pd.concat(out_rows, ignore_index=True)
    for c in block.columns:
        if c != "time":
            block[c] = block[c].astype(np.float32)
    return block


def main():
    t0 = time.time()
    header = list(pd.read_csv(OUT, nrows=0).columns)
    print(f"═══ APPEND MISSING LAYERS ═══\nmatrix: {OUT} | cols: {len(header)}", flush=True)

    # what's already in the matrix (time span per year) — for skip logic
    cov = {}
    existing = set()
    for c in pd.read_csv(OUT, usecols=["time"], chunksize=2_000_000):
        t = pd.to_datetime(c["time"], utc=True)
        ym = t.dt.to_period("M")
        for k in ym.unique():
            cov[str(k)] = cov.get(str(k), 0) + int((ym == k).sum())
        existing.update(t.astype("int64").tolist())
    print(f"matrix currently covers {len(cov)} months: {min(cov)} .. {max(cov)}", flush=True)

    appended_any = False
    # ONLY env var: comma-separated layer labels to process (skip others)
    only = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}
    for label, path, shift in LAYERS:
        if only and label not in only:
            continue
        m1 = load_m1(label, path, shift)
        m5 = to_m5(m1)
        del m1
        block = feature_expand(m5)
        del m5
        if block is None or len(block) == 0:
            print(f"  {label}: NO rows produced — skipped", flush=True)
            continue
        # only keep rows whose time is NOT already in matrix (dedup, existing wins)
        block_t = pd.to_datetime(block["time"], utc=True)
        mask = ~block_t.astype("int64").isin(existing)
        keep = block[mask]
        # enforce exact matrix column order (sort key = time field must be consistent)
        keep = keep[header]
        if len(keep) == 0:
            print(f"  {label}: all {len(block):,} rows already in matrix — skipped", flush=True)
            continue
        keep.to_csv(f"{BASE}/append_tmp_{label}.csv", index=False)
        print(f"  {label}: +{len(keep):,} NEW rows appended (of {len(block):,})", flush=True)
        # merge into OUT atomically: header + (matrix rows + new rows), sort by time.
        # dedup already done by mask (keep rows are guaranteed new times) — no -u!
        # NOTE: two `tail` calls in a subshell — GNU tail with 2 files prints
        # '==> file <==' headers, which would corrupt the CSV.
        time_col = header.index("time") + 1  # 1-indexed field for sort
        merged = f"{BASE}/append_tmp_merged.csv"
        cmd = (f"(head -1 {OUT} && (tail -n +2 {OUT}; tail -n +2 {BASE}/append_tmp_{label}.csv) "
               f"| LC_ALL=C sort -t, -k{time_col},{time_col} --parallel=8 --buffer-size=2G --temporary-directory={BASE}) > {merged}")
        subprocess.run(["bash", "-c", cmd], check=True)
        os.replace(merged, OUT)
        os.remove(f"{BASE}/append_tmp_{label}.csv")
        appended_any = True
        print(f"  matrix now: {os.path.getsize(OUT)/1e9:.1f} GB ({time.time()-t0:.0f}s)", flush=True)

    if appended_any:
        total = 0
        for c in pd.read_csv(OUT, usecols=["time"], chunksize=2_000_000):
            total += len(c)
        with open(SCHEMA, "w") as f:
            import json
            json.dump({"fp": schema_fingerprint(), "built_at": time.time(), "rows": total}, f)
        print(f"✅ DONE: matrix {total:,} rows | {time.time()-t0:.0f}s", flush=True)
    else:
        print("⚠️  Nothing appended (all layers already present or empty)", flush=True)


if __name__ == "__main__":
    main()
