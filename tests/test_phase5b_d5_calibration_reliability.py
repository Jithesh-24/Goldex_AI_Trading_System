"""tests/test_phase5b_d5_calibration_reliability.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d5_calibration_reliability import run_d5


def test_run_d5_shape_h15_has_traded_subset():
    result = run_d5(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["global"]["n"] > 20
    assert "long" in result["by_side"] and "short" in result["by_side"]
    assert isinstance(result["traded_subset"], (dict, str))
    assert "calibration_vs_meta_label" in result
    assert result["calibration_vs_meta_label"]["n"] == result["global"]["n"]


def test_run_d5_reliability_bins_sum_to_global_n():
    result = run_d5(max_holding=15, rows=600000)
    total_binned = sum(b["n"] for b in result["global"]["reliability_bins"])
    assert total_binned == result["global"]["n"]


if __name__ == "__main__":
    test_run_d5_shape_h15_has_traded_subset()
    test_run_d5_reliability_bins_sum_to_global_n()
    print("tests/test_phase5b_d5_calibration_reliability.py: OK")
