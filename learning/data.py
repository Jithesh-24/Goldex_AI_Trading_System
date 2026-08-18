"""
Raw OHLCV loading. Single source of truth for "what is the canonical price
history" — everything downstream (labeling, features, training) should load
through here instead of each script picking its own CSV, which is how the
old system ended up with 5 different half-built feature matrices nobody
could tell apart.

Canonical raw sources (deliberately NOT the old multi-GB pre-built feature
matrices, all of which were deleted as regenerable/superseded — see
.archive/pre-rebuild-2026-08-17/ for what used to read them):
  - gold_seed_merged_full6yr.csv : 2019-12-02 -> 2026-08-07, M1, MT5+Dukascopy merged
  - gold_seed.csv                : rolling recent window (last ~2.5mo), M1, live MT5
The two are stitched with gold_seed.csv taking priority on any overlapping
timestamp (it's the fresher live-captured source).
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_CSV = os.path.join(BASE, "gold_seed_merged_full6yr.csv")
RECENT_CSV = os.path.join(BASE, "gold_seed.csv")

OHLCV_COLS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]


def load_raw_m1(hist_csv: str = HIST_CSV, recent_csv: str = RECENT_CSV,
                 verbose: bool = True) -> pd.DataFrame:
    """Load + stitch the canonical M1 gold history. Returns a DataFrame
    sorted by time, unique timestamps, columns: time (datetime64, UTC-naive
    broker time as stored), open, high, low, close, tick_volume, spread.
    Does NOT forward-fill missing minutes — gaps are left as gaps, call
    `gap_report()` separately to audit them."""
    hist = pd.read_csv(hist_csv, usecols=lambda c: c in OHLCV_COLS,
                        parse_dates=["time"])
    recent = pd.read_csv(recent_csv, usecols=lambda c: c in OHLCV_COLS,
                          parse_dates=["time"])
    for df in (hist, recent):
        for c in ("open", "high", "low", "close", "tick_volume", "spread"):
            if c in df.columns:
                df[c] = df[c].astype(np.float64)

    combined = pd.concat([hist, recent], ignore_index=True)
    combined = combined.sort_values("time")
    # recent (live MT5) wins on duplicate timestamps -> keep='last' since
    # recent rows were appended after hist rows for the same instant
    combined = combined.drop_duplicates(subset="time", keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)

    if verbose:
        span = combined["time"].iloc[-1] - combined["time"].iloc[0]
        print(f"load_raw_m1: {len(combined):,} bars, "
              f"{combined['time'].iloc[0]} -> {combined['time'].iloc[-1]} "
              f"({span.days} days)")
    return combined


def gap_report(df: pd.DataFrame, max_gap_minutes: int = 10) -> pd.DataFrame:
    """Flag intraday gaps longer than `max_gap_minutes` that are NOT the
    normal weekly close (Fri evening -> Sun evening). Returns a DataFrame of
    gap_start, gap_end, gap_minutes, is_weekend_close."""
    t = df["time"].to_numpy()
    dt_minutes = np.diff(t).astype("timedelta64[s]").astype(np.float64) / 60.0
    idx = np.where(dt_minutes > max_gap_minutes)[0]
    rows = []
    for i in idx:
        start, end = df["time"].iloc[i], df["time"].iloc[i + 1]
        dow = start.dayofweek  # 4 = Friday
        is_weekend = dow == 4 and start.hour >= 20
        rows.append({"gap_start": start, "gap_end": end,
                     "gap_minutes": dt_minutes[i], "is_weekend_close": is_weekend})
    return pd.DataFrame(rows)


def to_m5(df: pd.DataFrame) -> pd.DataFrame:
    """Resample M1 -> M5 OHLCV (for higher-timeframe context features).
    Causal by construction (pandas resample only aggregates bars within each
    completed 5-minute bucket)."""
    d = df.set_index("time")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "tick_volume": "sum"}
    if "spread" in d.columns:
        agg["spread"] = "mean"
    out = d.resample("5min").agg(agg).dropna(subset=["open"])
    return out.reset_index()


if __name__ == "__main__":
    df = load_raw_m1()
    gaps = gap_report(df)
    real_gaps = gaps[~gaps["is_weekend_close"]]
    print(f"\n{len(gaps)} total gaps >10min, {len(real_gaps)} NOT weekend closes:")
    if len(real_gaps):
        print(real_gaps.sort_values("gap_minutes", ascending=False).head(30).to_string(index=False))
