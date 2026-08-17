#!/usr/bin/env python3
"""MARKET-BEHAVIOR JOURNAL (v7.6, 2026-08-03) — "learn from the market, not just trades"

Every EOD this script reads the day's REAL XM bars and journals WHAT THE MARKET
DID, independent of any signal/trade:
  - regime state (trend, volatility percentile, session breakdown)
  - significant moves (pumps/dumps): when, how big, what the features looked
    like BEFORE the move (precursor state) and the follow-through after
  - daily digest: what kind of day was it?

Output: cron/output/market_regime_journal.jsonl — APPENDS forever (tiny rows).
The regime-transition trainer (train_regime_transition.py) consumes this file
plus the full feature matrix to learn "what market state precedes a big move"
— pure market-behavior learning, journaled daily, trained nightly.

Pure journaling: reads nothing it can't see, writes nothing the engine reads.
Zero effect on live signals. Safe to run at any time.
"""
import json, os, sys, time, datetime
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
SEED = f"{BASE}/gold_seed.csv"
JOURNAL = f"{OUTDIR}/market_regime_journal.jsonl"

sys.path.insert(0, BASE)
import features as F

BIG_MOVE_ATR = 2.0      # |move| >= 2×ATR(14) over 30 min = "significant"
PRE_BARS = 12           # precursor window (12 M1 bars ≈ 12 min state)
POST_BARS = 30          # follow-through window (30 min)


def atr_series(closes, highs, lows, period=14):
    tr = np.maximum(highs - lows, np.maximum(
        np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values
    return atr


def _ema(s, span):
    return pd.Series(s).ewm(span=span, adjust=False).mean().values


def _ema_sma_diff(close, atr):
    e20 = _ema(close, 20); e60 = _ema(close, 60)
    return (e20[-1] - e60[-1]) / (np.nanmax(atr[-60:]) + 1e-9)


def main():
    if not os.path.exists(SEED):
        print("no seed — skip market journal")
        return
    df = pd.read_csv(SEED, parse_dates=["time"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    if len(df) < 200:
        print(f"seed too small ({len(df)}) — skip")
        return

    t = df["time"].values
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    atr = atr_series(close, high, low)

    # ── per-day digest: journal the LAST complete day only ──
    last_day = pd.Timestamp(t[-1]).date()
    day_mask = pd.Series(t).dt.date == last_day
    idx = np.where(day_mask.values)[0]
    if len(idx) < 60:
        print(f"last day ({last_day}) has only {len(idx)} bars — skip")
        return
    i0, i1 = idx[0], idx[-1] + 1  # half-open window for this day

    # detect significant moves WITHIN the day: |30-min move| >= BIG_MOVE_ATR * ATR
    fwd = pd.Series(close).rolling(30).mean().shift(-30).values  # avg price 30m ahead
    move = np.full(len(df), np.nan)
    valid = np.arange(i0, max(i0 + 1, i1 - 30))
    move[valid] = (fwd[valid] - close[valid]) / (atr[valid] + 1e-9)

    events = []
    for i in valid:
        if not np.isfinite(move[i]):
            continue
        if abs(move[i]) >= BIG_MOVE_ATR:
            events.append({
                "ts": int(np.datetime64(t[i], 's').astype('datetime64[s]').astype(np.int64)),
                "dir": "UP" if move[i] > 0 else "DN",
                "move_atr": float(move[i]),
                "pre_state": _precursor(df, i, PRE_BARS),
                "post_ret_30m": float((fwd[i] - close[i]) / close[i]),
            })

    # dedupe consecutive events (same move counted once): keep first per 10 min
    deduped = []
    last_ts = 0
    for e in events:
        if e["ts"] - last_ts >= 600:
            deduped.append(e)
            last_ts = e["ts"]

    day = last_day
    # daily digest (LAST day window i0:i1)
    dc = close[i0:i1]; dh = high[i0:i1]; dl = low[i0:i1]
    digest = {
        "date": str(day),
        "bars": len(dc),
        "first": str(pd.Timestamp(t[i0])),
        "last": str(pd.Timestamp(t[i1 - 1])),
        "o": float(df["open"].iloc[i0]),
        "c": float(dc[-1]),
        "day_ret_pct": float((dc[-1] / df["open"].iloc[i0] - 1) * 100),
        "range_pct": float((dh.max() - dl.min()) / df["open"].iloc[i0] * 100),
        "atr_pctile_200": float(pd.Series(atr[i0:i1]).rolling(200).rank(pct=True).iloc[-1]) if len(atr[i0:i1]) > 1 else None,
        "trend_ema": float(_ema_sma_diff(dc, atr[i0:i1])),
        "vol_spike": float((df["tick_volume"].astype(float).iloc[i1 - 20:i1].mean() /
                            (df["tick_volume"].astype(float).iloc[max(i0, i1 - 200):i1 - 20].mean() + 1e-9))),
        "n_events": len(deduped),
    }
    # session breakdown of events
    sess_counts = {"asia": 0, "london": 0, "ny": 0, "late": 0}
    for e in deduped:
        h = datetime.datetime.fromtimestamp(e["ts"], datetime.timezone.utc).hour
        if 0 <= h < 7: sess_counts["asia"] += 1
        elif 7 <= h < 12: sess_counts["london"] += 1
        elif 12 <= h < 17: sess_counts["ny"] += 1
        else: sess_counts["late"] += 1
    digest["events_by_session"] = sess_counts

    row = {"t": time.time(), "day": str(day), "digest": digest, "events": deduped[-50:]}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[regime-journal] {day}: {len(deduped)} significant moves | day_ret {digest['day_ret_pct']:+.2f}% | range {digest['range_pct']:.1f}% | vol_spike {digest['vol_spike']:.2f}")


def _precursor(df, i, n):
    """Feature snapshot BEFORE bar i — what the market looked like pre-move."""
    j0 = max(0, i - n)
    window = df.iloc[j0:i]
    if len(window) < 5:
        return None
    c = window["close"].values.astype(float)
    ret = (c[-1] / c[0] - 1) * 100 if c[0] else 0.0
    return {
        "ret_12m_pct": float(ret),
        "vol_rel": float(window["tick_volume"].astype(float).mean() /
                        (df["tick_volume"].astype(float).iloc[max(0, i-200):j0].mean() + 1e-9)),
        "spread": float(window["spread"].mean()) if "spread" in window else None,
        "body_ratio": float((window["close"] - window["open"]).abs().mean() /
                            (window["high"] - window["low"] + 1e-9).mean()),
    }


if __name__ == "__main__":
    main()
