"""tests/test_phase5_leakage.py
Spec section 16/29: no future information enters EV; deterministic
replay; live/replay equivalence within tolerance; a DATA_LIMITED/
UNAVAILABLE specialist cannot produce a valid numeric decision; stale
market blocks a live decision; schema-mismatched input is rejected."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate


class _FakeMarketState:
    def __init__(self, spread, timestamp):
        self.spread = spread
        self.timestamp = timestamp


def _inputs(direction_status="VALIDATED"):
    ms = _FakeMarketState(spread=0.01, timestamp=datetime.now(timezone.utc))
    direction = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                                 model_status=direction_status, probability_long=0.6,
                                 probability_short=0.4, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_meta_v3_candidate_h15", horizon=15,
                                     model_status="VALIDATED", probability_take=0.55, calibrated=True)
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                             model_status="VALIDATED", p_tp=0.55, calibrated=True)
    mae = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.3, q75=0.5, q90=0.8)
    mfe = MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=0.9, q90=1.4)
    return ms, direction, opportunity, barrier, mae, mfe


def test_deterministic_replay():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    d1 = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    d2 = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d1.decision == d2.decision
    assert d1.ev_adj == d2.ev_adj


def test_data_limited_specialist_cannot_produce_valid_decision():
    ms, direction, opportunity, barrier, mae, mfe = _inputs(direction_status="DATA_LIMITED")
    d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


def test_stale_market_prevents_valid_decision():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    ms.timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


def test_schema_mismatch_rejected():
    with pytest.raises(ValidationError):
        DirectionOutput(model_id="x", horizon="not-an-int", model_status="VALIDATED")


def test_live_and_replay_paths_call_same_evaluate_function():
    import decision.ev_engine as live_engine
    import research.phase5_ev_engine as replay_engine
    assert replay_engine.evaluate is live_engine.evaluate


def test_unavailable_specialist_cannot_produce_valid_decision():
    ms, direction, opportunity, barrier, mae, mfe = _inputs(direction_status="UNAVAILABLE")
    d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


def test_future_timestamp_prevents_valid_decision():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    ms.timestamp = datetime.now(timezone.utc) + timedelta(minutes=5)
    d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


if __name__ == "__main__":
    test_deterministic_replay()
    test_data_limited_specialist_cannot_produce_valid_decision()
    test_stale_market_prevents_valid_decision()
    test_schema_mismatch_rejected()
    test_live_and_replay_paths_call_same_evaluate_function()
    test_unavailable_specialist_cannot_produce_valid_decision()
    test_future_timestamp_prevents_valid_decision()
    print("tests/test_phase5_leakage.py: OK")
