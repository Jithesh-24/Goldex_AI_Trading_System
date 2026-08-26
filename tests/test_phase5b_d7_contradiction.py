"""tests/test_phase5b_d7_contradiction.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d7_contradiction import run_d7


def test_run_d7_shape():
    result = run_d7(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["contradiction_mask_n"] + result["non_contradiction_mask_n"] > 0
    for pop in ("contradicted", "non_contradicted"):
        assert "mean" in result["realized_r"][pop]
        assert "n" in result["realized_r"][pop]
    assert len(result["breakdown_by_barrier_probability_decile"]) == 10
    assert "long" in result["breakdown_by_side"] and "short" in result["breakdown_by_side"]
    assert set(result["breakdown_by_volatility_tercile"].keys()) == {"low", "medium", "high"}


if __name__ == "__main__":
    test_run_d7_shape()
    print("tests/test_phase5b_d7_contradiction.py: OK")
