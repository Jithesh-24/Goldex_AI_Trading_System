"""tests/test_phase5_ev_engine.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_ev_engine import replay_and_validate


def test_replay_and_validate_on_dry_run_dataset():
    result = replay_and_validate(max_holding=15, rows=20000)
    assert "n_events" in result
    assert set(result["decisions"].keys()) == {"NO_TRADE", "LONG_CANDIDATE", "SHORT_CANDIDATE"}
    assert "baseline_comparison" in result
    assert 0.0 <= result["fragile_fraction"] <= 1.0


if __name__ == "__main__":
    test_replay_and_validate_on_dry_run_dataset()
    print("tests/test_phase5_ev_engine.py: OK")
