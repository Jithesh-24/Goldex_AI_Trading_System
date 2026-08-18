"""Folds xm_ticker.py's live-captured bars into gold_seed.csv (the rolling
recent window core/data.py stitches onto the 6.7yr history). xm_ticker owns
the only MT5 connection and writes xm_bars_backfill.csv (last 2000 bars,
dumped on startup/reconnect) + xm_live_bars.jsonl (every completed M1 bar
since) -- this just merges both into the seed, live rows winning on any
overlap (freshest source).

Run: python3 -m core.seed_refresh   (cheap -- gold_seed.csv is ~2.5mo of M1,
not the full 6.7yr history; safe to run every few minutes via cron)
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
SEED = os.path.join(BASE, "gold_seed.csv")
BACKFILL = os.path.join(BASE, "xm_bars_backfill.csv")
LIVE_JSONL = os.path.join(OUTDIR, "xm_live_bars.jsonl")

COLS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume", "src"]


def _load_live_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("t", 0) < 946684800:  # pre-2000 epoch -> poisoned row, drop
                continue
            rows.append({"time": pd.Timestamp(d["t"], unit="s"), "open": d["o"], "high": d["h"],
                         "low": d["l"], "close": d["c"], "tick_volume": d["v"],
                         "spread": d.get("spread", 20), "real_volume": 0, "src": "xmlive"})
    return pd.DataFrame(rows, columns=COLS)


def main():
    frames = []
    if os.path.exists(SEED):
        existing = pd.read_csv(SEED, parse_dates=["time"])
        for c in COLS:
            if c not in existing.columns:
                existing[c] = 0 if c == "real_volume" else ("seed" if c == "src" else np.nan)
        frames.append(existing[COLS])

    if os.path.exists(BACKFILL):
        back = pd.read_csv(BACKFILL, parse_dates=["time"])
        back["src"] = "mt5bar"
        for c in COLS:
            if c not in back.columns:
                back[c] = 0 if c == "real_volume" else np.nan
        frames.append(back[COLS])

    if os.path.exists(LIVE_JSONL):
        live = _load_live_jsonl(LIVE_JSONL)
        if len(live):
            frames.append(live)

    if not frames:
        print("seed_refresh: nothing to merge, no source files found")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset="time", keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)
    combined.to_csv(SEED, index=False)
    print(f"seed_refresh: {len(combined):,} bars, {combined['time'].iloc[0]} -> "
          f"{combined['time'].iloc[-1]} -> {SEED}")


if __name__ == "__main__":
    main()
