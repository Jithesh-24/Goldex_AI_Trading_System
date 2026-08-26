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

import pandas as pd

from contracts.market_state import MarketState, M1BarState, DataQuality, FeedHealthState

SPREAD_POINTS_TO_PRICE = 0.01
VOL_LOOKBACK_BARS = 60


def build_snapshot(df: pd.DataFrame, i: int, symbol: str = "XAUUSD", sequence: int = 0) -> MarketState:
    row = df.iloc[i]
    ts = row["time"].to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    spread_price = float(row["spread"]) * SPREAD_POINTS_TO_PRICE
    mid = float(row["open"])
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

    return MarketState(
        symbol=symbol, source="synthetic_replay", state_version="v1", sequence=sequence,
        market_timestamp=ts, ingestion_timestamp=ts, processing_timestamp=ts,
        bid=bid, ask=ask, mid=mid, spread=spread_price, last=mid,
        last_quality=DataQuality.VALID,
        tick_count_60s=0, tick_count_300s=0, tick_rate_per_sec=0.0,
        current_m1=current_m1, completed_m1=completed_m1,
        realized_vol_60s=realized_vol_60s, spread_mean_60s=None, spread_std_60s=None,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.0,
        feed_latency_sec=0.0, state_update_latency_sec=0.0,
    )
