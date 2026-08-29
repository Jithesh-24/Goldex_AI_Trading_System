"""tests/test_phase5_calibration_v3b.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_calibration import _oof_for_direction, _oof_for_opportunity, _oof_for_barrier
from research.direction_side import compute_direction_oof


def test_oof_for_direction_matches_shared_helper():
    t0, y_full, p_full, m = _oof_for_direction(max_holding=15, rows=600000)
    dir_oof = compute_direction_oof(max_holding=15, rows=600000)
    assert np.array_equal(t0, dir_oof["t0_nz"])
    assert np.array_equal(m, dir_oof["has_oof"])
    np.testing.assert_allclose(p_full[m], dir_oof["p_direction_cal"][m], atol=1e-9)


def test_opportunity_and_barrier_still_shape_correctly_after_side_fix():
    t0, y_full, p_full, mask = _oof_for_opportunity(max_holding=15, rows=600000)
    assert len(t0) == len(y_full) == len(p_full) == len(mask)
    assert set(y_full[mask].tolist()) <= {0, 1}
    t0b, y_full_b, p_full_b, mask_b = _oof_for_barrier(max_holding=15, rows=600000)
    assert np.array_equal(t0, t0b)


if __name__ == "__main__":
    test_oof_for_direction_matches_shared_helper()
    test_opportunity_and_barrier_still_shape_correctly_after_side_fix()
    print("tests/test_phase5_calibration_v3b.py: OK")
