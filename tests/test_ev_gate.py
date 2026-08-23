"""tests/test_ev_gate.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.ev_gate import decide, MIN_EDGE_THRESHOLD, compute_side_ev
from contracts.specialist_output import BarrierOutput


def test_decide_no_trade_both_below_threshold():
    decision, direction, reason = decide(long_ev_adj=0.001, short_ev_adj=-0.05)
    assert decision == "NO_TRADE"
    assert direction is None


def test_decide_long_wins():
    decision, direction, reason = decide(long_ev_adj=0.10, short_ev_adj=0.02)
    assert decision == "LONG_CANDIDATE"
    assert direction == "long"


def test_decide_short_wins():
    decision, direction, reason = decide(long_ev_adj=-0.01, short_ev_adj=0.15)
    assert decision == "SHORT_CANDIDATE"
    assert direction == "short"


def test_decide_none_available_is_no_trade():
    decision, direction, reason = decide(long_ev_adj=None, short_ev_adj=None)
    assert decision == "NO_TRADE"
    assert "unavailable" in reason.lower()


def test_compute_side_ev_gated_off_returns_none():
    barrier = BarrierOutput(model_id="x", horizon=15, model_status="VALIDATED", p_tp=0.5, calibrated=True)
    ev = compute_side_ev(barrier, direction_gate_ok=False, p_sl_given_not_win=0.5,
                          tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.02, uncertainty=0.2)
    assert ev is None


def test_compute_side_ev_computes_when_gated_on():
    barrier = BarrierOutput(model_id="x", horizon=15, model_status="VALIDATED", p_tp=0.5, calibrated=True)
    ev = compute_side_ev(barrier, direction_gate_ok=True, p_sl_given_not_win=0.5,
                          tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.02, uncertainty=0.2, k=0.1)
    assert ev is not None


if __name__ == "__main__":
    test_decide_no_trade_both_below_threshold()
    test_decide_long_wins()
    test_decide_short_wins()
    test_decide_none_available_is_no_trade()
    test_compute_side_ev_gated_off_returns_none()
    test_compute_side_ev_computes_when_gated_on()
    print("tests/test_ev_gate.py: OK")
