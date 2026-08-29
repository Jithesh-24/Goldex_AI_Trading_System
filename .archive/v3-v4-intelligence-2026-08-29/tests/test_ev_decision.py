"""tests/test_ev_decision.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.ev_decision import EVDecision


def test_ev_decision_no_trade_minimal():
    d = EVDecision(
        timestamp=datetime.now(timezone.utc), direction=None, decision="NO_TRADE",
        ev_adj=0.0, ev_raw=0.0, uncertainty=1.0, decision_margin=0.0,
        candidate_sl=None, candidate_tp=None, cost_r=None, known_cost_only=True,
        specialist_model_ids={}, calibration_ids={}, feature_schema_ids={},
        ev_formula_version="v1", cost_model_version="v1", regime_state=None,
        timeout_r_provisional_proxy=True, decision_reason="required specialist unavailable",
    )
    assert d.decision == "NO_TRADE"


def test_ev_decision_long_candidate():
    d = EVDecision(
        timestamp=datetime.now(timezone.utc), direction="long", decision="LONG_CANDIDATE",
        ev_adj=0.15, ev_raw=0.20, uncertainty=0.3, decision_margin=0.05,
        candidate_sl=0.4, candidate_tp=0.9, cost_r=0.05, known_cost_only=True,
        specialist_model_ids={"direction": "direction_v3_candidate_h15"},
        calibration_ids={"direction": "direction_h15_platt"},
        feature_schema_ids={"direction": "direction_v3_h15__2026-08-22"},
        ev_formula_version="v1", cost_model_version="v1", regime_state=2,
        timeout_r_provisional_proxy=False, decision_reason="ev_adj above min_edge_threshold",
    )
    assert d.decision == "LONG_CANDIDATE"
    assert d.candidate_sl == 0.4


if __name__ == "__main__":
    test_ev_decision_no_trade_minimal()
    test_ev_decision_long_candidate()
    print("tests/test_ev_decision.py: OK")
