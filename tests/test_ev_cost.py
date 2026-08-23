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
    def __init__(self, spread, timestamp):
        self.spread = spread
        self.timestamp = timestamp


def test_round_trip_cost_r_fresh():
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc))
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5)
    assert cost == (0.02 * 2) / 0.5


def test_round_trip_cost_r_stale_returns_none():
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc) - timedelta(seconds=60))
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5, max_staleness_seconds=5.0)
    assert cost is None


if __name__ == "__main__":
    test_candidate_sl_tp_uses_q75()
    test_candidate_sl_tp_unavailable_returns_none()
    test_round_trip_cost_r_fresh()
    test_round_trip_cost_r_stale_returns_none()
    print("tests/test_ev_cost.py: OK")
