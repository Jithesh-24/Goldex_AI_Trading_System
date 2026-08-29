"""tests/test_calibration_registry.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from decision.calibration_registry import CalibrationRegistry
from research.phase5_calibration import fit_and_save_calibrator


def test_fit_and_save_then_resolve_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(42)
        p_raw = rng.uniform(0, 1, 500)
        y_true = (rng.uniform(0, 1, 500) < p_raw).astype(int)
        path = fit_and_save_calibrator("direction", 15, y_true, p_raw, calibration_dir=tmp)
        assert os.path.exists(path)
        reg = CalibrationRegistry(calibration_dir=tmp)
        cal = reg.resolve("direction", 15)
        p_cal = cal.apply(p_raw[:5])
        assert len(p_cal) == 5


def test_resolve_missing_raises():
    with tempfile.TemporaryDirectory() as tmp:
        reg = CalibrationRegistry(calibration_dir=tmp)
        with pytest.raises(FileNotFoundError):
            reg.resolve("direction", 999)


if __name__ == "__main__":
    test_fit_and_save_then_resolve_roundtrip()
    test_resolve_missing_raises()
    print("tests/test_calibration_registry.py: OK")
