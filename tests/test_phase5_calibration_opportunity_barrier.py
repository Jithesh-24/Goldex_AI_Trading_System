"""tests/test_phase5_calibration_opportunity_barrier.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_calibration import _oof_for_opportunity, _oof_for_barrier


def test_oof_for_opportunity_shapes_match():
    t0, y_full, p_full, mask = _oof_for_opportunity(max_holding=15, rows=20000)
    assert len(t0) == len(y_full) == len(p_full) == len(mask)
    y_true = y_full[mask]
    assert set(y_true.tolist()) <= {0, 1}


def test_oof_for_barrier_shapes_match():
    t0, y_full, p_full, mask = _oof_for_barrier(max_holding=15, rows=20000)
    assert len(t0) == len(y_full) == len(p_full) == len(mask)
    y_true = y_full[mask]
    assert set(y_true.tolist()) <= {0, 1}


if __name__ == "__main__":
    test_oof_for_opportunity_shapes_match()
    test_oof_for_barrier_shapes_match()
    print("tests/test_phase5_calibration_opportunity_barrier.py: OK")
