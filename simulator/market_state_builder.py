"""simulator/market_state_builder.py
Builds a contracts.market_state.MarketState snapshot as-of row i's bar-OPEN
timestamp (see docs/superpowers/specs/2026-08-26-goldex-v4-phase1-simulator-design.md
Section 4 for the bar-open-vs-close decision). Only rows [0..i-1] are treated
as "completed" (their high/low/close are known); row i itself contributes
ONLY its open price and timestamp -- its high/low/close are not yet known at
decision time and must never be read here. This function is the load-bearing
piece the no-leakage test harness (Task 8) audits.

Source is always "synthetic_replay" -- this is what
contracts.market_state.MarketState's existing source Literal was designed to
distinguish from "mt5_live"."""
from datetime import timezone

import math

import pandas as pd

from contracts.market_state import MarketState, M1BarState, DataQuality, FeedHealthState

SPREAD_POINTS_TO_PRICE = 0.01
VOL_LOOKBACK_BARS = 60

# Task 12 -- data-quality thresholds. Spread anomaly is judged against the
# same trailing spread_mean_60s/spread_std_60s this module already computes
# below (real per-file history, not a fabricated baseline). 5 std devs is a
# conservative "this is not noise" bar; the x10-of-mean fallback covers the
# early-window case where std is 0 or unavailable (e.g. i==0), using the
# example from the task brief ("spread suddenly 100x normal") scaled down
# to something that won't false-positive on ordinary widening.
SPREAD_ANOMALY_STD_MULT = 5.0
SPREAD_ANOMALY_MEAN_RATIO = 10.0


def _is_invalid_price(x: float) -> bool:
    return x is None or math.isnan(x) or math.isinf(x) or x <= 0


def _is_anomalous_spread(spread_price: float, spread_mean_60s, spread_std_60s) -> bool:
    if spread_price < 0 or math.isnan(spread_price) or math.isinf(spread_price):
        return True
    if spread_mean_60s is None or spread_mean_60s <= 0:
        return False
    if spread_std_60s and spread_std_60s > 0:
        return spread_price > spread_mean_60s + SPREAD_ANOMALY_STD_MULT * spread_std_60s
    return spread_price > spread_mean_60s * SPREAD_ANOMALY_MEAN_RATIO


def _trailing_bar_window(df: pd.DataFrame, i: int, seconds: float) -> pd.DataFrame:
    """Completed bars (indices < i, so no leakage of row i's own high/low/
    close) whose timestamp falls within `seconds` before ts. This is the
    finest-granularity real sample the historical path has: one row per
    minute, each carrying that minute's tick_volume and one spread
    reading -- vs. the live path's per-tick ring buffer. Used to compute
    real (not fabricated) tick_count_60s/300s and spread_mean/std_60s
    below. At i==0 there are no prior bars and the window is empty --
    that yields a genuine 0/None, not a hardcoded placeholder."""
    if i == 0:
        return df.iloc[0:0]
    # Compare in the DataFrame's own (possibly tz-naive) timestamp dtype --
    # ts may have been coerced to tz-aware for the MarketState fields, but
    # df["time"] itself is whatever dtype the caller's frame uses.
    cutoff = df.iloc[i]["time"] - pd.Timedelta(seconds=seconds)
    prior = df.iloc[:i]
    return prior[prior["time"] >= cutoff]


def _last_valid_mid(df: pd.DataFrame, i: int) -> float:
    """Nearest prior row (i.e. rows [0..i-1], never i or later -- no
    leakage) with a finite positive open, walking backward. Falls back to
    1.0 only if every prior row is also invalid (degenerate/corrupted
    input); data_quality=INVALID on the returned snapshot is what tells a
    consumer not to trust the price either way."""
    for j in range(i - 1, -1, -1):
        candidate = float(df.iloc[j]["open"])
        if not _is_invalid_price(candidate):
            return candidate
    return 1.0


def build_snapshot(df: pd.DataFrame, i: int, symbol: str = "XAUUSD", sequence: int = 0) -> MarketState:
    row = df.iloc[i]
    ts = row["time"].to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    raw_mid = float(row["open"])
    raw_spread_price = float(row["spread"]) * SPREAD_POINTS_TO_PRICE

    data_quality = DataQuality.VALID
    if _is_invalid_price(raw_mid):
        # Invalid price: substitute the nearest known-good mid so bid/ask
        # (both gt=0 in the contract) stay constructible, and flag it --
        # never pass the bad reading through as if it were a real price.
        mid = _last_valid_mid(df, i)
        spread_price = 0.0 if _is_invalid_price(raw_spread_price) else raw_spread_price
        data_quality = DataQuality.INVALID
    else:
        mid = raw_mid
        spread_price = raw_spread_price
    bid = mid - spread_price / 2.0
    ask = mid + spread_price / 2.0

    if i > 0:
        prev = df.iloc[i - 1]
        prev_ts = prev["time"].to_pydatetime()
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
        completed_m1 = M1BarState(
            open=float(prev["open"]), high=float(prev["high"]),
            low=float(prev["low"]), close=float(prev["close"]),
            tick_count=int(prev["tick_volume"]), start_time=prev_ts, end_time=ts, complete=True,
        )
    else:
        completed_m1 = None

    current_m1 = M1BarState(
        open=mid, high=mid, low=mid, close=mid,
        tick_count=0, start_time=ts, end_time=ts, complete=False,
    )

    window_start = max(0, i - VOL_LOOKBACK_BARS)
    window = df.iloc[window_start:i]
    if len(window) >= 2:
        returns = window["close"].pct_change().dropna()
        realized_vol_60s = float(returns.std()) if len(returns) > 0 else None
    else:
        realized_vol_60s = None

    # tick_count_60s/300s and spread_mean/std_60s: computed for real from
    # the bar-derived data we actually have (tick_volume + spread columns),
    # not hardcoded. They are structurally coarser than the live path's
    # per-tick figures -- one sample per minute instead of one per tick --
    # because bar data has no sub-minute resolution. That coarseness is a
    # real property of historical bar data, not something to paper over by
    # leaving them 0/None when real (if approximate) values are available.
    win_60s = _trailing_bar_window(df, i, 60)
    win_300s = _trailing_bar_window(df, i, 300)
    tick_count_60s = int(win_60s["tick_volume"].sum()) if len(win_60s) else 0
    tick_count_300s = int(win_300s["tick_volume"].sum()) if len(win_300s) else 0

    spreads_60s = win_60s["spread"] * SPREAD_POINTS_TO_PRICE
    if len(spreads_60s) > 0:
        spread_mean_60s = float(spreads_60s.mean())
        spread_std_60s = float(spreads_60s.std(ddof=0)) if len(spreads_60s) > 1 else 0.0
    else:
        spread_mean_60s = None
        spread_std_60s = None

    # Spread-anomaly check: only meaningful against the *current* row's raw
    # spread (not the substituted one above -- if the price itself was
    # invalid we've already flagged INVALID and a spread anomaly on top of
    # that adds no information for the consumer).
    if data_quality == DataQuality.VALID and _is_anomalous_spread(raw_spread_price, spread_mean_60s, spread_std_60s):
        data_quality = DataQuality.INVALID

    # ingestion_timestamp/processing_timestamp collapsed to market_timestamp
    # by design, not oversight: historical replay has no real ingestion or
    # processing pipeline to time -- there is nothing happening between
    # "bar observed" and "state built" to measure a delta over, unlike the
    # live path where ticks genuinely travel through a socket and a queue.
    # A configurable synthetic offset was considered and rejected: it would
    # require picking a distribution for a latency that never occurred,
    # adding fake noise to a field that's currently at least honestly zero.
    return MarketState(
        symbol=symbol, source="synthetic_replay", state_version="v1", sequence=sequence,
        market_timestamp=ts, ingestion_timestamp=ts, processing_timestamp=ts,
        bid=bid, ask=ask, mid=mid, spread=spread_price, last=mid,
        last_quality=DataQuality.VALID, data_quality=data_quality,
        tick_count_60s=tick_count_60s, tick_count_300s=tick_count_300s, tick_rate_per_sec=0.0,
        current_m1=current_m1, completed_m1=completed_m1,
        realized_vol_60s=realized_vol_60s, spread_mean_60s=spread_mean_60s, spread_std_60s=spread_std_60s,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.0,
        feed_latency_sec=0.0, state_update_latency_sec=0.0,
    )
