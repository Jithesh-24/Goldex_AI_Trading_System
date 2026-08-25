"""tests/test_phase5b_d4_cross_specialist_consistency.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d4_cross_specialist_consistency import run_d4


def test_run_d4_shape():
    result = run_d4(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["n"] > 20
    for key in ("contradiction_barrier_vs_reward_risk", "contradiction_opportunity_vs_barrier"):
        c = result[key]
        assert 0.0 <= c["rate"] <= 1.0
        assert c["k"] <= c["n"]
        assert c["ci_lo"] <= c["rate"] <= c["ci_hi"]


if __name__ == "__main__":
    test_run_d4_shape()
    print("tests/test_phase5b_d4_cross_specialist_consistency.py: OK")
