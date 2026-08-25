"""tests/test_phase5b_d6_long_short_conditioning.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d6_long_short_conditioning import run_d6


def test_run_d6_shape():
    result = run_d6(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    for role in ("direction", "opportunity", "barrier"):
        assert "long" in result[role] and "short" in result[role]
        assert result[role]["long"]["n"] is not None


if __name__ == "__main__":
    test_run_d6_shape()
    print("tests/test_phase5b_d6_long_short_conditioning.py: OK")
