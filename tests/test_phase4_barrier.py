"""python3 tests/test_phase4_barrier.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_barrier import run_barrier_candidate


def test_run_barrier_candidate_produces_calibration_curve():
    # NOTE: mirrors Task 5's finding -- oof_run's PurgedWalkForwardCV has a
    # hardcoded default min_train_bars=10_000 (measured in EVENTS, not raw
    # bars). rows=20000 was verified to yield too few CUSUM events for any
    # fold to clear that floor, so using rows=600000 here for the same reason.
    result = run_barrier_candidate(max_holding=45, rows=600000)
    assert result["n_events"] > 50, "too few meta-training events in dry run to trust any metric"
    assert len(result["reliability_curve"]) > 0
    for bucket in result["reliability_curve"]:
        assert 0.0 <= bucket["mean_predicted"] <= 1.0
        assert 0.0 <= bucket["actual_win_rate"] <= 1.0


if __name__ == "__main__":
    test_run_barrier_candidate_produces_calibration_curve()
    print("tests/test_phase4_barrier.py: OK")
