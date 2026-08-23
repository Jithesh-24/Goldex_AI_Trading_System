"""decision/calibration_registry.py
Static, config-driven calibrator lookup -- mirrors decision/router.py's
ModelRouter pattern (no live recalibration, no champion/challenger)."""
import json
import os

from decision.calibration import PlattCalibrator

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_DIR = os.path.join(_BASE, "models", "calibration")


class CalibrationRegistry:
    def __init__(self, calibration_dir: str = None):
        self.calibration_dir = calibration_dir if calibration_dir else CALIBRATION_DIR

    def resolve(self, role: str, horizon: int) -> PlattCalibrator:
        path = os.path.join(self.calibration_dir, f"{role}_h{horizon}_platt.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no calibrator for role={role} horizon={horizon} at {path}")
        with open(path) as f:
            d = json.load(f)
        return PlattCalibrator(a=d["a"], b=d["b"], n_samples=d["n_samples"],
                                window_start=d.get("window_start"), window_end=d.get("window_end"),
                                fit_at_utc=d.get("fit_at_utc"))
