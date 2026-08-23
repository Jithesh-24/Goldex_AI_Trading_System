"""python3 tests/test_phase4_mae_quantile.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_mae_quantile import run_mae_quantile_candidate


def test_run_mae_quantile_candidate_produces_real_coverage():
    result = run_mae_quantile_candidate(max_holding=45, rows=20000)
    assert result["n_events"] > 50
    for q in ("0.5", "0.75", "0.9"):
        assert q in result["global_coverage"]
        assert 0.0 <= result["global_coverage"][q] <= 1.0
        assert q in result["per_regime_coverage"], "per-vol-state coverage missing -- spec section 13 requires it"


if __name__ == "__main__":
    test_run_mae_quantile_candidate_produces_real_coverage()
    print("tests/test_phase4_mae_quantile.py: OK")
