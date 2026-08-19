"""python3 tests/test_microstructure_live.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta

from contracts.market_state import MarketState, FeedHealthState, DataQuality
from features.microstructure_live import TickActivityTracker


def _state(seq, ts, bid, ask, spread):
    return MarketState(
        symbol="GOLD.i#", source="synthetic_replay", sequence=seq,
        market_timestamp=ts, ingestion_timestamp=ts, processing_timestamp=ts,
        bid=bid, ask=ask, mid=(bid + ask) / 2, spread=spread,
        last=None, last_quality=DataQuality.UNAVAILABLE,
        tick_count_60s=seq, tick_count_300s=seq, tick_rate_per_sec=1.0,
        current_m1=None, completed_m1=None,
        realized_vol_60s=None, spread_mean_60s=spread, spread_std_60s=0.0,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.01,
        feed_latency_sec=0.01, state_update_latency_sec=0.0001,
    )


def test_tick_activity_tracker_basic():
    tracker = TickActivityTracker()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = None
    for i in range(20):
        ts = base + timedelta(milliseconds=i * 300)
        state = _state(i, ts, 2000.0 + i * 0.01, 2000.2 + i * 0.01, 0.2 + (i % 3) * 0.01)
        out = tracker.update(state)
    assert out is not None
    assert set(out.keys()) == {
        "spread_change_live", "spread_shock_zscore_live",
        "tick_interarrival_mean_60s", "tick_interarrival_std_60s",
        "tick_arrival_burstiness_60s",
    }
    for v in out.values():
        assert v is None or isinstance(v, float)


if __name__ == "__main__":
    test_tick_activity_tracker_basic()
    print("tests/test_microstructure_live.py: OK")
