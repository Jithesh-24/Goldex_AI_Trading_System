"""tests/test_ev_cost.py"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import MAEOutput, MFEOutput
from decision.ev_cost import candidate_sl_tp, round_trip_cost_r


def _mae(status="VALIDATED", q75=0.5):
    return MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status=status, q50=0.3, q75=q75, q90=0.8)


def _mfe(status="VALIDATED", q75=0.9):
    return MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status=status, q50=0.5, q75=q75, q90=1.4)


def test_candidate_sl_tp_uses_q75():
    sl, tp = candidate_sl_tp(_mae(), _mfe())
    assert sl == 0.5
    assert tp == 0.9


def test_candidate_sl_tp_unavailable_returns_none():
    sl, tp = candidate_sl_tp(_mae(status="UNAVAILABLE"), _mfe())
    assert sl is None and tp is None


class _FakeMarketState:
    def __init__(self, spread, timestamp, realized_vol_60s=0.0006, mid=2350.0):
        self.spread = spread
        self.market_timestamp = timestamp
        self.realized_vol_60s = realized_vol_60s
        self.mid = mid


def test_round_trip_cost_r_fresh():
    # FIX 3 (C4, final-review fix wave): candidate_sl_distance (mae.q75) is an
    # R-multiple, already normalized by a volatility estimate -- spread (a
    # price) must go through the SAME normalization (vol * mid) to become a
    # valid R-multiple cost. The old formula ((spread*2)/candidate_sl_distance)
    # mixed price units with R-multiple units; this test asserts the CORRECT
    # formula's output, not a re-encoding of the old broken one.
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc),
                           realized_vol_60s=0.0006, mid=2350.0)
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5)
    expected = (0.02 * 2) / (0.5 * 0.0006 * 2350.0)
    assert cost == expected


def test_round_trip_cost_r_stale_returns_none():
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc) - timedelta(seconds=60))
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5, max_staleness_seconds=5.0)
    assert cost is None


def test_round_trip_cost_r_future_timestamp_returns_none():
    # FIX 9 (I5): round_trip_cost_r's own age computation must also reject a
    # future timestamp (age < 0), matching decision/ev_engine.py's check.
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc) + timedelta(seconds=60))
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5, max_staleness_seconds=5.0)
    assert cost is None


def test_round_trip_cost_r_missing_vol_returns_none():
    # FIX 3 (C4): never fabricate a vol estimate -- if realized_vol_60s is
    # None (it's an Optional field on the real MarketState contract), the
    # cost must be None, not a made-up number.
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc), realized_vol_60s=None)
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5)
    assert cost is None


def test_round_trip_cost_r_against_real_market_state_contract():
    # FIX 4 (C5): confirm this reads .market_timestamp (which the real
    # contract has), not .timestamp (which it doesn't).
    from contracts.market_state import MarketState, FeedHealthState
    now = datetime.now(timezone.utc)
    ms = MarketState(
        symbol="XAUUSD", source="synthetic_replay", sequence=1,
        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
        bid=2349.9, ask=2350.1, mid=2350.0, spread=0.2,
        tick_count_60s=10, tick_count_300s=50, tick_rate_per_sec=0.17,
        realized_vol_60s=0.0006,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.1,
    )
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5)
    assert cost is not None and cost > 0


if __name__ == "__main__":
    test_candidate_sl_tp_uses_q75()
    test_candidate_sl_tp_unavailable_returns_none()
    test_round_trip_cost_r_fresh()
    test_round_trip_cost_r_stale_returns_none()
    test_round_trip_cost_r_future_timestamp_returns_none()
    test_round_trip_cost_r_missing_vol_returns_none()
    test_round_trip_cost_r_against_real_market_state_contract()
    print("tests/test_ev_cost.py: OK")
