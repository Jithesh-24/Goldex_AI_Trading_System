"""tests/test_phase5b_d3_specialist_oof_quality.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d3_specialist_oof_quality import run_d3


def test_run_d3_shape():
    result = run_d3(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    for role in ("opportunity", "barrier"):
        r = result[role]
        assert r["n"] > 20
        assert 0.0 <= r["win_rate"] <= 1.0
        assert r["baseline_win_rate"] == 0.4887
        assert "slope" in r["calibration"]
    mm = result["mae_mfe"]
    assert "long" in mm["mae_coverage_by_side"] and "short" in mm["mae_coverage_by_side"]
    assert "long" in mm["mfe_coverage_by_side"] and "short" in mm["mfe_coverage_by_side"]


if __name__ == "__main__":
    test_run_d3_shape()
    print("tests/test_phase5b_d3_specialist_oof_quality.py: OK")
