"""tests/test_phase5b_stats_utils.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics._stats_utils import (
    pointbiserial_with_ci, fit_calibration_slope_intercept, population_label,
)


def test_pointbiserial_with_ci_perfect_correlation():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=2000) >= 0.5).astype(int)
    p = np.where(y == 1, rng.uniform(0.6, 1.0, 2000), rng.uniform(0.0, 0.4, 2000))
    out = pointbiserial_with_ci(y, p)
    assert out["n"] == 2000
    assert out["r"] > 0.5
    assert out["ci_lo"] < out["r"] < out["ci_hi"]


def test_pointbiserial_with_ci_small_n_returns_none():
    out = pointbiserial_with_ci(np.array([1, 0]), np.array([0.6, 0.4]))
    assert out["r"] is None
    assert out["n"] == 2


def test_fit_calibration_slope_intercept_well_calibrated():
    rng = np.random.default_rng(1)
    n = 5000
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    out = fit_calibration_slope_intercept(y, p_true)
    assert out["n"] == n
    assert abs(out["slope"] - 1.0) < 0.15
    assert abs(out["intercept"]) < 0.15
    assert out["slope_se"] > 0
    assert out["intercept_se"] > 0


def test_population_label():
    assert population_label("oos", 12345) == {"population": "oos", "n": 12345}


if __name__ == "__main__":
    test_pointbiserial_with_ci_perfect_correlation()
    test_pointbiserial_with_ci_small_n_returns_none()
    test_fit_calibration_slope_intercept_well_calibrated()
    test_population_label()
    print("tests/test_phase5b_stats_utils.py: OK")
