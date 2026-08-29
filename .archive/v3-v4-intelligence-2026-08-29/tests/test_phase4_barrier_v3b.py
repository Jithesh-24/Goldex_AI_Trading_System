"""tests/test_phase4_barrier_v3b.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_barrier import run_barrier_candidate_v3b


def test_run_barrier_candidate_v3b_conditions_on_direction_side():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_barrier_candidate_v3b(max_holding=45, rows=600000,
                                        registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert result["log_loss"] >= 0.0
    assert result["status"] in ("validated", "rejected")

    entry_path = os.path.join(tmp_registry.name, "barrier_v3b_candidate_h45.json")
    with open(entry_path) as f:
        entry = json.load(f)
    assert entry["model_id"] == "barrier_v3b_candidate_h45"
    assert "assumed_side" in entry["feature_cols"]
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


if __name__ == "__main__":
    test_run_barrier_candidate_v3b_conditions_on_direction_side()
    print("tests/test_phase4_barrier_v3b.py: OK")
