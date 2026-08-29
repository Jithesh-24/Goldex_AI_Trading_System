"""tests/test_ev_formula.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import BarrierOutput
from decision.ev_formula import compute_barrier_split, raw_ev, risk_adjusted_ev


def test_compute_barrier_split_sums_to_one():
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                             model_status="VALIDATED", p_tp=0.5, calibrated=True)
    split = compute_barrier_split(barrier, p_sl_given_not_win=0.6)
    assert abs(split["p_tp"] - 0.5) < 1e-9
    assert abs(split["p_sl"] - 0.5 * 0.6) < 1e-9
    assert abs(split["p_timeout"] - 0.5 * 0.4) < 1e-9
    assert abs(split["p_tp"] + split["p_sl"] + split["p_timeout"] - 1.0) < 1e-9


def test_compute_barrier_split_unavailable_returns_none():
    barrier = BarrierOutput(model_id="x", horizon=15, model_status="UNAVAILABLE")
    split = compute_barrier_split(barrier, p_sl_given_not_win=0.6)
    assert split["p_tp"] is None and split["p_sl"] is None and split["p_timeout"] is None


def test_raw_ev_known_case():
    # p_tp=0.5 tp_r=1.0, p_sl=0.3 sl_r=0.5, p_timeout=0.2 timeout_r=0.1, cost_r=0.05
    ev = raw_ev(p_tp=0.5, p_sl=0.3, p_timeout=0.2, tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.05)
    expected = 0.5 * 1.0 - 0.3 * 0.5 - 0.2 * 0.1 - 0.05
    assert abs(ev - expected) < 1e-9


def test_raw_ev_missing_input_returns_none():
    ev = raw_ev(p_tp=None, p_sl=0.3, p_timeout=0.2, tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.05)
    assert ev is None


def test_risk_adjusted_ev_reduces_with_uncertainty():
    high_conf = risk_adjusted_ev(ev_raw=0.2, uncertainty=0.1, k=0.5)
    low_conf = risk_adjusted_ev(ev_raw=0.2, uncertainty=0.9, k=0.5)
    assert high_conf > low_conf


if __name__ == "__main__":
    test_compute_barrier_split_sums_to_one()
    test_compute_barrier_split_unavailable_returns_none()
    test_raw_ev_known_case()
    test_raw_ev_missing_input_returns_none()
    test_risk_adjusted_ev_reduces_with_uncertainty()
    print("tests/test_ev_formula.py: OK")
