"""tests/test_ev_engine.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate


class _FakeMarketState:
    def __init__(self, spread, timestamp):
        self.spread = spread
        self.timestamp = timestamp


def _valid_inputs():
    ms = _FakeMarketState(spread=0.01, timestamp=datetime.now(timezone.utc))
    direction = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                                 model_status="VALIDATED", probability_long=0.6,
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


def test_evaluate_produces_a_decision():
    ms, direction, opportunity, barrier, mae, mfe = _valid_inputs()
    d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win=0.5,
                 mae_out=mae, mfe_out=mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision in ("NO_TRADE", "LONG_CANDIDATE", "SHORT_CANDIDATE")
    assert d.specialist_model_ids["direction"] == "direction_v3_candidate_h15"


def test_evaluate_unavailable_direction_forces_no_trade():
    ms, direction, opportunity, barrier, mae, mfe = _valid_inputs()
    direction.model_status = "UNAVAILABLE"
    d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win=0.5,
                 mae_out=mae, mfe_out=mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"
    assert "direction" in d.decision_reason.lower() or "unavailable" in d.decision_reason.lower()


def test_evaluate_stale_market_forces_no_trade():
    from datetime import timedelta
    ms, direction, opportunity, barrier, mae, mfe = _valid_inputs()
    ms.timestamp = datetime.now(timezone.utc) - timedelta(seconds=60)
    d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win=0.5,
                 mae_out=mae, mfe_out=mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


if __name__ == "__main__":
    test_evaluate_produces_a_decision()
    test_evaluate_unavailable_direction_forces_no_trade()
    test_evaluate_stale_market_forces_no_trade()
    print("tests/test_ev_engine.py: OK")
