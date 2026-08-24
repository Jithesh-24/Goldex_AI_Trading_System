"""tests/test_ev_engine_direction_side_lineage.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate


class _FakeMarketState:
    def __init__(self):
        self.market_timestamp = datetime.now(timezone.utc)
        self.spread = 0.02
        self.mid = 2000.0


def _base_outputs(direction_model_id="direction_v3_candidate_h15", opportunity_direction_model_id="direction_v3_candidate_h15"):
    direction = DirectionOutput(model_id=direction_model_id, horizon=15, model_status="VALIDATED",
                                 probability_long=0.6, probability_short=0.4, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                                     probability_take=0.7, calibrated=True,
                                     assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    barrier = BarrierOutput(model_id="barrier_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                             p_tp=0.6, calibrated=True, assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    mae = MAEOutput(model_id="mae_quantile_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=0.8, q90=1.1, assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    mfe = MFEOutput(model_id="mfe_quantile_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=1.2, q90=1.8, assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    return direction, opportunity, barrier, mae, mfe


def test_matching_direction_model_id_does_not_force_no_trade():
    direction, opportunity, barrier, mae, mfe = _base_outputs()
    d = evaluate(_FakeMarketState(), direction, opportunity, barrier, 0.5, mae, mfe,
                 timeout_r=0.1, timeout_r_provisional_proxy=True)
    assert d.decision != "NO_TRADE" or "Direction side lineage mismatch" not in (d.decision_reason or "")


def test_mismatched_direction_model_id_forces_no_trade():
    direction, opportunity, barrier, mae, mfe = _base_outputs(
        direction_model_id="direction_v3_candidate_h15",
        opportunity_direction_model_id="direction_v3_candidate_h45",  # WRONG horizon's Direction model
    )
    d = evaluate(_FakeMarketState(), direction, opportunity, barrier, 0.5, mae, mfe,
                 timeout_r=0.1, timeout_r_provisional_proxy=True)
    assert d.decision == "NO_TRADE"
    assert "Direction side lineage mismatch" in d.decision_reason


if __name__ == "__main__":
    test_matching_direction_model_id_does_not_force_no_trade()
    test_mismatched_direction_model_id_forces_no_trade()
    print("tests/test_ev_engine_direction_side_lineage.py: OK")
