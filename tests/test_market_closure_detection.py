"""tests/test_market_closure_detection.py -- test market closure detection in StateEngine.on_tick()"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.market_state import DataQuality
from contracts.tick import Tick
from market.state_engine import StateEngine, is_market_closed


def _make_tick(market_timestamp, bid=1500.0, ask=1500.5, source="synthetic_replay"):
    """Helper to create a Tick with given market_timestamp."""
    return Tick(
        symbol="GOLD.i#",
        market_timestamp=market_timestamp,
        ingestion_timestamp=datetime.now(timezone.utc),
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        spread=ask - bid,
        last=None,
        source=source,
        internal_seq=1,
    )


def test_is_market_closed_friday_evening():
    """Friday 21:00 UTC is market closed."""
    friday_2100 = datetime(2026, 1, 2, 21, 0, tzinfo=timezone.utc)  # A Friday in 2026
    assert is_market_closed(friday_2100) is True


def test_is_market_closed_friday_afternoon():
    """Friday 15:00 UTC is market open."""
    friday_1500 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)  # A Friday in 2026
    assert is_market_closed(friday_1500) is False


def test_is_market_closed_saturday():
    """Saturday is completely closed."""
    saturday = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)  # A Saturday in 2026
    assert is_market_closed(saturday) is True


def test_is_market_closed_sunday_morning():
    """Sunday morning (before 22:00) is closed."""
    sunday_morning = datetime(2026, 1, 4, 10, 0, tzinfo=timezone.utc)  # A Sunday in 2026
    assert is_market_closed(sunday_morning) is True


def test_is_market_closed_sunday_evening():
    """Sunday 22:00+ UTC is market open (start of next week)."""
    sunday_2200 = datetime(2026, 1, 4, 22, 0, tzinfo=timezone.utc)  # A Sunday in 2026
    assert is_market_closed(sunday_2200) is False


def test_is_market_closed_monday():
    """Monday daytime is market open."""
    monday_1000 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)  # A Monday in 2026
    assert is_market_closed(monday_1000) is False


def test_is_market_closed_wednesday_evening():
    """Wednesday evening (21:00-22:00) is closed."""
    wednesday_2130 = datetime(2026, 1, 7, 21, 30, tzinfo=timezone.utc)  # A Wednesday in 2026
    assert is_market_closed(wednesday_2130) is True


def test_on_tick_during_closure_sets_market_closed_true():
    """A tick processed during market closure should have market_closed=True."""
    engine = StateEngine("GOLD.i#")
    friday_2100 = datetime(2026, 1, 2, 21, 0, tzinfo=timezone.utc)
    tick = _make_tick(friday_2100)

    state = engine.on_tick(tick)

    assert state is not None
    assert state.market_closed is True


def test_on_tick_during_open_sets_market_closed_false():
    """A tick processed during market open should have market_closed=False."""
    engine = StateEngine("GOLD.i#")
    monday_1000 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    tick = _make_tick(monday_1000)

    state = engine.on_tick(tick)

    assert state is not None
    assert state.market_closed is False


def test_on_tick_multiple_ticks_closure_status_changes():
    """Process ticks with different closure status (two separate engines to avoid ordering issues)."""
    # Tick during open market (Monday 10:00)
    engine_open = StateEngine("GOLD.i#")
    monday_1000 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    tick_open = _make_tick(monday_1000)
    state_open = engine_open.on_tick(tick_open)
    assert state_open.market_closed is False

    # Tick during closed market (Friday 21:30)
    engine_closed = StateEngine("GOLD.i#")
    friday_2130 = datetime(2026, 1, 2, 21, 30, tzinfo=timezone.utc)
    tick_closed = _make_tick(friday_2130)
    state_closed = engine_closed.on_tick(tick_closed)
    assert state_closed.market_closed is True


def test_on_tick_valid_tick_flagged_valid():
    """An ordinary tick gets data_quality=VALID."""
    engine = StateEngine("GOLD.i#")
    monday_1000 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    state = engine.on_tick(_make_tick(monday_1000))
    assert state.data_quality == DataQuality.VALID


def test_on_tick_nan_mid_flagged_invalid():
    """A corrupted tick (NaN mid) bypassing Tick's own pydantic validation
    (via model_construct, simulating a bad upstream record) must be flagged
    INVALID and still produce a usable MarketState with positive bid/ask --
    not silently dropped and not crashing."""
    engine = StateEngine("GOLD.i#")
    ts = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    good_tick = _make_tick(ts)
    engine.on_tick(good_tick)  # seed a known-good last_bid/last_ask

    bad_tick = Tick.model_construct(
        symbol="GOLD.i#", market_timestamp=ts.replace(second=1),
        ingestion_timestamp=datetime.now(timezone.utc),
        bid=1500.0, ask=1500.5, mid=float("nan"), spread=0.5,
        last=None, source="synthetic_replay", internal_seq=2,
    )
    state = engine.on_tick(bad_tick)
    assert state is not None
    assert state.data_quality == DataQuality.INVALID
    assert state.bid > 0
    assert state.ask > 0
    # last known-good bid/ask (from good_tick) is what got carried forward
    assert state.bid == good_tick.bid
    assert state.ask == good_tick.ask


def test_on_tick_negative_bid_flagged_invalid():
    """A corrupted tick with a negative bid (bypassing Tick's own gt=0
    validation) must be flagged rather than silently accepted."""
    engine = StateEngine("GOLD.i#")
    ts = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    bad_tick = Tick.model_construct(
        symbol="GOLD.i#", market_timestamp=ts,
        ingestion_timestamp=datetime.now(timezone.utc),
        bid=-1.0, ask=1500.5, mid=750.0, spread=1501.5,
        last=None, source="synthetic_replay", internal_seq=1,
    )
    state = engine.on_tick(bad_tick)
    assert state is not None
    assert state.data_quality == DataQuality.INVALID
    assert state.bid > 0


def test_on_tick_anomalous_spread_flagged_invalid():
    """A spread suddenly far outside the trailing window's normal range is
    flagged, even though bid/ask/mid are all individually valid."""
    engine = StateEngine("GOLD.i#")
    base = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
    # Build up a normal spread history (~0.5) over several ticks.
    for k in range(10):
        ts = base.replace(second=k)
        engine.on_tick(_make_tick(ts, bid=1500.0 + k * 0.01, ask=1500.5 + k * 0.01))

    # Now a tick with a spread 100x the established norm.
    ts = base.replace(second=11)
    spike_tick = _make_tick(ts, bid=1500.0, ask=1550.0)
    state = engine.on_tick(spike_tick)
    assert state is not None
    assert state.data_quality == DataQuality.INVALID


def test_on_tick_normal_spread_stays_valid():
    """Ordinary spread fluctuation within the ~0.5 range does not trip the
    anomaly flag."""
    engine = StateEngine("GOLD.i#")
    base = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
    state = None
    for k in range(10):
        ts = base.replace(second=k)
        state = engine.on_tick(_make_tick(ts, bid=1500.0 + k * 0.01, ask=1500.5 + k * 0.01))
    assert state is not None
    assert state.data_quality == DataQuality.VALID


if __name__ == "__main__":
    test_is_market_closed_friday_evening()
    test_is_market_closed_friday_afternoon()
    test_is_market_closed_saturday()
    test_is_market_closed_sunday_morning()
    test_is_market_closed_sunday_evening()
    test_is_market_closed_monday()
    test_is_market_closed_wednesday_evening()
    test_on_tick_during_closure_sets_market_closed_true()
    test_on_tick_during_open_sets_market_closed_false()
    test_on_tick_multiple_ticks_closure_status_changes()
    test_on_tick_valid_tick_flagged_valid()
    test_on_tick_nan_mid_flagged_invalid()
    test_on_tick_negative_bid_flagged_invalid()
    test_on_tick_anomalous_spread_flagged_invalid()
    test_on_tick_normal_spread_stays_valid()
    print("tests/test_market_closure_detection.py: OK")
