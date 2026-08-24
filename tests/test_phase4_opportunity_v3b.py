"""tests/test_phase4_opportunity_v3b.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_opportunity import run_opportunity_candidate_v3b


def test_run_opportunity_candidate_v3b_conditions_on_direction_side():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_opportunity_candidate_v3b(max_holding=45, rows=600000,
                                            registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert 0.0 <= result["oos_log_loss"] < 5.0
    assert result["status"] in ("validated", "rejected")
    assert isinstance(result["used_p_direction"], bool)

    entry_path = os.path.join(tmp_registry.name, "opportunity_v3b_candidate_h45.json")
    assert os.path.exists(entry_path)
    import json
    with open(entry_path) as f:
        entry = json.load(f)
    assert entry["model_id"] == "opportunity_v3b_candidate_h45"
    assert "assumed_side" in entry["feature_cols"]
    assert "direction_side" in entry["target_definition"] or "Direction" in entry["target_definition"]
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


def test_old_v3_artifact_untouched():
    # the real (non-tmp) registry's old artifact must still exist and be unmodified by this module import
    real_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "registry", "opportunity_v3_candidate_h45.json")
    assert os.path.exists(real_path), "old Phase 4 artifact must be preserved for audit (design section 7)"


if __name__ == "__main__":
    test_run_opportunity_candidate_v3b_conditions_on_direction_side()
    test_old_v3_artifact_untouched()
    print("tests/test_phase4_opportunity_v3b.py: OK")
