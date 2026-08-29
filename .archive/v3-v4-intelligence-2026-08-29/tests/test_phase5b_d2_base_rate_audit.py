"""tests/test_phase5b_d2_base_rate_audit.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d2_base_rate_audit import run_d2


def test_run_d2_shape():
    result = run_d2(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    o = result["overall"]
    assert o["n"] > 50
    total = o["up_frac"] + o["down_frac"] + o["timeout_frac"]
    assert abs(total - 1.0) < 1e-6
    assert len(result["by_year"]) >= 1
    for row in result["by_year"]:
        assert "year" in row and "n" in row


if __name__ == "__main__":
    test_run_d2_shape()
    print("tests/test_phase5b_d2_base_rate_audit.py: OK")
