"""tests/test_phase5b_d1_direction_quality.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d1_direction_quality import run_d1


def test_run_d1_shape():
    result = run_d1(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["oos"]["n"] > 50
    assert result["oos"]["point_biserial"]["n"] == result["oos"]["n"]
    assert len(result["oos"]["p_direction_deciles"]) == 9
    assert "long" in result["side_conditioned"] and "short" in result["side_conditioned"]
    assert result["side_conditioned"]["long"]["n"] + result["side_conditioned"]["short"]["n"] == result["oos"]["n"]


if __name__ == "__main__":
    test_run_d1_shape()
    print("tests/test_phase5b_d1_direction_quality.py: OK")
