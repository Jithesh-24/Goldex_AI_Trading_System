"""Merge REAL XM MT5 history + ticker XM bars into one continuous seed CSV.
The engine seeds from this file so there's no gap between history and now.

v3 (2026-08-01): TradingView feed REMOVED from the pipeline entirely.
The old version merged spot_ohlc.jsonl (engine-written TV bars) — TV froze on
07-31 and polluted the seed tail with 40+ identical frozen bars, corrupting the
model's features. Now the only sources are REAL XM data:
  1) gold_m1_history.csv  — 60k-bar bulk download (server time)
  2) xm_bars_backfill.csv — 2000 recent bars dumped by xm_ticker via
                            mt5.copy_rates_from_pos (true UTC already)
  3) xm_live_bars.jsonl   — completed M1 bars built by xm_ticker from its own
                            25ms REAL XM tick stream (true UTC already)

Timezone: MT5 server time is UTC+3 (XM, summer). History is converted to TRUE
UTC using the ticker's persisted live-detected offset (xm_server_offset.json),
so the model's hour/session features match the live engine's real-UTC bars.
NEVER mix server-time and UTC in the same column.
"""
import pandas as pd
import json, os

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
SEED = f"{BASE}/gold_seed.csv"
OFFSET_FILE = f"{OUTDIR}/xm_server_offset.json"

# MT5 server offset: persisted by xm_ticker (live-detected when market open,
# survives weekends). Fall back to 3h (XM summer).
def mt5_utc_offset():
    try:
        with open(OFFSET_FILE) as f:
            off = float(json.load(f).get("offset_h", 3.0))
        print(f"MT5 server offset: {off:+.1f}h (persisted from ticker)")
        return off
    except Exception:
        print("offset file missing — assuming +3.0h (XM summer)")
        return 3.0

OFFSET_H = mt5_utc_offset()

# 1. MT5 bulk history (server time -> true UTC)
hist = pd.read_csv(f"{BASE}/gold_m1_history.csv")
hist["time"] = pd.to_datetime(hist["time"]) - pd.Timedelta(hours=OFFSET_H)
hist["src"] = "mt5"

# 2. Ticker backfill (REAL XM bars, already true UTC — no shift)
frames = [hist[["time","open","high","low","close","tick_volume","spread","real_volume","src"]]]
try:
    back = pd.read_csv(f"{BASE}/xm_bars_backfill.csv")
    back["time"] = pd.to_datetime(back["time"])
    back["src"] = "mt5bar"
    print(f"backfill bars: {len(back)} ({back['time'].iloc[0]} -> {back['time'].iloc[-1]})")
    frames.append(back[["time","open","high","low","close","tick_volume","spread","real_volume","src"]])
except Exception as e:
    print("backfill read warn:", e)

# 3. Ticker live bars (REAL XM M1 bars built from its 25ms tick stream)
live_rows = []
try:
    with open(f"{OUTDIR}/xm_live_bars.jsonl") as f:
        for line in f:
            try:
                d = json.loads(line)
                live_rows.append({
                    "time": pd.to_datetime(d["t"], unit="s", utc=True).tz_localize(None),
                    "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"],
                    "tick_volume": d["v"], "spread": d.get("spread", 20),
                    "real_volume": 0, "src": "xmlive",
                })
            except Exception:
                pass
except Exception as e:
    print("live read warn:", e)

if live_rows:
    live = pd.DataFrame(live_rows, columns=["time","open","high","low","close","tick_volume","spread","real_volume","src"])
    print(f"live bars: {len(live)} ({live['time'].iloc[0]} -> {live['time'].iloc[-1]})")
    frames.append(live[["time","open","high","low","close","tick_volume","spread","real_volume","src"]])

all_df = pd.concat(frames, ignore_index=True)
all_df = all_df.sort_values("time").drop_duplicates(subset="time", keep="last").reset_index(drop=True)
# POISON-DROP GUARD (2026-08-03): a stale epoch-0 tick (-10800 s) had leaked
# into xm_live_bars.jsonl and re-entered the seed every merge, corrupting
# .min() (year 1969) and silently zeroing the rally filter → poisoned training.
# Drop any bar older than the year 2000 regardless of source. XAUUSD history
# never legitimately predates 2000, so this is a safe, permanent immunity.
_poison_mask = all_df["time"].dt.year < 2000
if _poison_mask.any():
    print(f"⚠️ dropped {int(_poison_mask.sum())} poison row(s) (pre-2000 timestamps)")
    all_df = all_df[~_poison_mask].reset_index(drop=True)
all_df.to_csv(SEED, index=False)
print(f"Seed: {SEED} | {len(all_df)} bars | {all_df['time'].iloc[0]} -> {all_df['time'].iloc[-1]}")
