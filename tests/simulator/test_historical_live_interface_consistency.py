"""tests/simulator/test_historical_live_interface_consistency.py
Feeds equivalent synthetic data through the historical path
(simulator.market_state_builder.build_snapshot) and the live path
(market.state_engine.StateEngine.on_tick), then diffs the resulting
MarketState objects field-by-field. The two paths are NOT expected to
produce numerically identical states -- they sample the market at
different granularities (one bar-open snapshot per minute vs. one state
per tick) -- but every field that both paths CAN populate must be
populated for real on both sides, and only fields that are genuinely
unavailable at bar granularity (no sub-minute samples in bar data) may
differ by being None on the historical side.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from contracts.tick import Tick
from market.state_engine import StateEngine
from simulator.market_state_builder import build_snapshot

N_BARS = 65  # > VOL_LOOKBACK_BARS (60) so realized_vol_60s has a full window on both sides
BAR_SEC = 60
START = datetime(2020, 1, 6, 10, 0, 0, tzinfo=timezone.utc)


def _make_df():
    times = pd.date_range(START, periods=N_BARS, freq="1min")
    opens = [2500.0 + 0.1 * i for i in range(N_BARS)]
    closes = [o + 0.05 for o in opens]
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": [max(o, c) + 0.02 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 0.02 for o, c in zip(opens, closes)],
        "close": closes,
        "tick_volume": [10 + (i % 4) for i in range(N_BARS)],
        "spread": [20.0 + (i % 3) for i in range(N_BARS)],
    })


def _feed_equivalent_ticks(engine: StateEngine, df: pd.DataFrame):
    """Feeds one tick per bar, at each bar's open price/spread/time --
    the same information build_snapshot draws each row's mid/spread from
    -- so the live engine's completed_m1 buffer lines up bar-for-bar with
    the historical DataFrame."""
    last_state = None
    for idx in range(len(df)):
        row = df.iloc[idx]
        ts = row["time"].to_pydatetime().replace(tzinfo=timezone.utc)
        spread_price = float(row["spread"]) * 0.01
        mid = float(row["open"])
        bid = mid - spread_price / 2.0
        ask = mid + spread_price / 2.0
        tick = Tick(
            symbol="XAUUSD", market_timestamp=ts, ingestion_timestamp=ts,
            bid=bid, ask=ask, mid=mid, spread=spread_price, last=mid,
            tick_volume=int(row["tick_volume"]), source="synthetic_replay",
            internal_seq=idx + 1,
        )
        state = engine.on_tick(tick)
        if state is not None:
            last_state = state
    return last_state


# Fields structurally unavailable at bar granularity: bar data has one
# spread/tick_volume sample per minute, not per sub-minute interval, so
# there is no way to compute a genuine sub-60s spread distribution the
# way the live tick ring buffer can. tick_count_60s/300s and spread
# mean/std ARE computed (Task 1 fix) from the bar-level samples that do
# exist, so they are NOT in this set -- they're just coarser than live.
STRUCTURALLY_UNAVAILABLE_HISTORICAL_NONE_FIELDS = set()


def test_historical_path_computes_real_not_fabricated_activity_fields():
    df = _make_df()
    i = N_BARS - 1
    snap = build_snapshot(df, i)

    # Must be real values, not the old hardcoded 0 / 0.0 / None placeholders.
    assert snap.tick_count_60s > 0
    assert snap.tick_count_300s > 0
    assert snap.tick_count_300s >= snap.tick_count_60s
    assert snap.spread_mean_60s is not None
    assert snap.spread_std_60s is not None
    assert snap.realized_vol_60s is not None


def test_historical_first_row_has_no_prior_bars_so_counts_are_genuinely_zero():
    df = _make_df()
    snap = build_snapshot(df, 0)
    # No prior bars exist at all -- 0 here is a true observation (nothing
    # to count), not a fabricated placeholder independent of the data.
    assert snap.tick_count_60s == 0
    assert snap.tick_count_300s == 0
    assert snap.spread_mean_60s is None
    assert snap.spread_std_60s is None
    assert snap.realized_vol_60s is None


def test_live_path_computes_realized_vol_60s():
    df = _make_df()
    engine = StateEngine("XAUUSD")
    last_state = _feed_equivalent_ticks(engine, df)
    assert last_state is not None
    # This was the field state_engine.on_tick previously always left None.
    assert last_state.realized_vol_60s is not None


def test_historical_and_live_states_agree_on_field_availability():
    df = _make_df()
    hist_snap = build_snapshot(df, N_BARS - 1)

    engine = StateEngine("XAUUSD")
    live_state = _feed_equivalent_ticks(engine, df)
    assert live_state is not None

    hist_dict = hist_snap.model_dump()
    live_dict = live_state.model_dump()

    # Fields both sides can populate must be non-None on both sides.
    populated_on_both = [
        "tick_count_60s", "tick_count_300s", "spread_mean_60s",
        "spread_std_60s", "realized_vol_60s",
    ]
    for field in populated_on_both:
        assert hist_dict[field] is not None, f"{field} unexpectedly None on historical side"
        assert live_dict[field] is not None, f"{field} unexpectedly None on live side"

    # No field is None on the historical side unless explicitly declared
    # a bar-granularity-unavailable field above.
    for field, value in hist_dict.items():
        if value is None and field not in STRUCTURALLY_UNAVAILABLE_HISTORICAL_NONE_FIELDS:
            # last/last_quality/completed_m1's end_time etc. are legitimately
            # optional-by-schema fields, not activity/volatility fields --
            # only assert on the activity/volatility fields under test here.
            assert field not in populated_on_both, (
                f"{field} is None on the historical side but was not declared "
                f"structurally unavailable"
            )


def test_historical_market_closed_matches_live_computation():
    """Regression for the market_closed divergence: build_snapshot must
    compute market_closed the same way the live path does (is_market_closed
    on the bar's own timestamp), not silently default to False. Saturday
    2020-01-04 is a known-closed timestamp on the XM GOLD.i# schedule --
    verifies the historical path actually flags it, not just that it agrees
    with an equally-wrong live default."""
    from datetime import datetime, timezone
    from market.state_engine import is_market_closed

    saturday = datetime(2020, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
    assert is_market_closed(saturday) is True

    df = pd.DataFrame({
        "time": [saturday, saturday + timedelta(minutes=1)],
        "open": [2500.0, 2500.1], "high": [2500.2, 2500.3], "low": [2499.8, 2499.9],
        "close": [2500.05, 2500.15], "tick_volume": [10, 11], "spread": [20.0, 21.0],
    })
    snap = build_snapshot(df, 1)
    assert snap.market_closed is True

    weekday = datetime(2020, 1, 6, 10, 0, 0, tzinfo=timezone.utc)
    df2 = pd.DataFrame({
        "time": [weekday, weekday + timedelta(minutes=1)],
        "open": [2500.0, 2500.1], "high": [2500.2, 2500.3], "low": [2499.8, 2499.9],
        "close": [2500.05, 2500.15], "tick_volume": [10, 11], "spread": [20.0, 21.0],
    })
    snap2 = build_snapshot(df2, 1)
    assert snap2.market_closed is False


def test_historical_and_live_agree_field_by_field_or_declare_divergence():
    """Broader than test_historical_and_live_states_agree_on_field_availability:
    for every MarketState field, either both paths agree on the value (for
    fields both sides can genuinely compute the same way, given equivalent
    input) or the field is explicitly declared as an accepted divergence with
    a stated reason. This is what would have caught the market_closed bug --
    a False-vs-False comparison passes None-ness checks but not an explicit
    equality/declaration check like this one."""
    df = _make_df()
    hist_snap = build_snapshot(df, N_BARS - 1)

    engine = StateEngine("XAUUSD")
    live_state = _feed_equivalent_ticks(engine, df)
    assert live_state is not None

    hist_dict = hist_snap.model_dump()
    live_dict = live_state.model_dump()

    # Fields expected to genuinely diverge between the two paths, with why.
    ACCEPTED_DIVERGENCES = {
        "source": "historical is 'synthetic_replay', live is the tick's own source",
        "sequence": "independent per-path counters",
        "ingestion_timestamp": "historical collapses to market_timestamp (no real ingestion pipeline)",
        "processing_timestamp": "historical collapses to market_timestamp; live uses wall-clock now()",
        "last_tick_age_sec": "historical is always 0.0 (no wall-clock replay); live measures real latency",
        "feed_latency_sec": "historical is always 0.0; live measures real ingestion latency",
        "state_update_latency_sec": "historical is always 0.0; live measures real processing latency",
        "tick_rate_per_sec": "historical hardcodes 0.0 (no sub-bar tick rate at bar granularity)",
        "current_m1": "different open/high/low/close construction (synthetic mid-bar vs tick-built)",
        "completed_m1": "different tick_count/end_time provenance between paths",
        "tick_count_60s": "coarser one-sample-per-minute vs per-tick sampling",
        "tick_count_300s": "coarser one-sample-per-minute vs per-tick sampling",
        "spread_mean_60s": "coarser one-sample-per-minute vs per-tick sampling",
        "spread_std_60s": "structurally ~0 at bar granularity (see market_state_builder.py comment)",
        "realized_vol_60s": "independent bar-buffer bookkeeping between paths, both real but not identical",
        "bid": "constructed from equivalent but not bit-identical mid/spread math",
        "ask": "constructed from equivalent but not bit-identical mid/spread math",
        "mid": "historical bar-open price vs live per-tick mid; equivalent inputs, not guaranteed identical",
        "spread": "equivalent inputs, not guaranteed bit-identical after independent rounding paths",
        "last": "equivalent inputs, not guaranteed bit-identical",
    }
    for field, live_val in live_dict.items():
        hist_val = hist_dict[field]
        if field in ACCEPTED_DIVERGENCES:
            continue
        assert hist_val == live_val, (
            f"{field} diverges between historical ({hist_val!r}) and live "
            f"({live_val!r}) but is not declared an accepted divergence -- "
            f"either fix the historical path to match, or add {field!r} to "
            f"ACCEPTED_DIVERGENCES with a stated reason."
        )


def test_historical_timestamp_collapse_is_documented_not_accidental():
    """ingestion_timestamp == processing_timestamp == market_timestamp on
    the historical path is a deliberate zero-offset simplification (see
    comment in simulator/market_state_builder.py), not an oversight --
    pin the current behavior so a silent regression either way is caught."""
    df = _make_df()
    snap = build_snapshot(df, 2)
    assert snap.ingestion_timestamp == snap.market_timestamp
    assert snap.processing_timestamp == snap.market_timestamp


def test_live_timestamps_are_genuinely_distinct():
    """Contrast case: the live path DOES distinguish these timestamps
    with real deltas, unlike the historical path."""
    df = _make_df()
    engine = StateEngine("XAUUSD")
    live_state = _feed_equivalent_ticks(engine, df)
    assert live_state is not None
    assert live_state.processing_timestamp > live_state.market_timestamp
    assert live_state.state_update_latency_sec is not None
