"""tests/test_phase5_barrier_split.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_barrier_split import run_barrier_split_candidate


def test_barrier_split_runs_on_dry_run_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        result = run_barrier_split_candidate(max_holding=15, rows=20000, registry_dir=tmp)
        assert "n_events" in result
        assert "status" in result
        assert result["status"] in ("candidate", "validated", "rejected")
        if result["n_events"] > 0:
            reg_path = os.path.join(tmp, "barrier_split_v3_candidate_h15.json")
            assert os.path.exists(reg_path)


def test_barrier_split_does_not_touch_real_registry():
    real_path = "models/registry/barrier_split_v3_candidate_h15.json"
    existed_before = os.path.exists(real_path)
    with tempfile.TemporaryDirectory() as tmp:
        run_barrier_split_candidate(max_holding=15, rows=20000, registry_dir=tmp)
    assert os.path.exists(real_path) == existed_before


if __name__ == "__main__":
    test_barrier_split_runs_on_dry_run_dataset()
    test_barrier_split_does_not_touch_real_registry()
    print("tests/test_phase5_barrier_split.py: OK")
