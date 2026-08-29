"""python3 tests/test_phase4_regime.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_regime import run_regime_candidate


def test_run_regime_candidate_produces_real_diagnostics():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    # NOTE: brief's Step 1 draft used rows=20000, but PurgedWalkForwardCV's
    # min_train_bars=2000 is measured in EVENTS (not raw bars, see
    # tests/test_phase4_direction.py for the same issue in Task 4/5). rows=20000
    # -> only ~2,379 CUSUM events total, so there's no room for >=2 folds each with
    # >=2000 train-eligible events plus a held-out test tail -- the fold-count
    # assertion fails before any diagnostic is computed. rows=600000 gives enough
    # events for multiple folds. Verified empirically before changing this value.
    result = run_regime_candidate(rows=600000, registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_states"] == 4
    assert result["mean_run_length"] > 1.0, "a regime that flips every bar carries no persistence"
    assert "transition_matrix_drift" in result
    assert result["status"] in ("validated", "rejected")
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


if __name__ == "__main__":
    test_run_regime_candidate_produces_real_diagnostics()
    print("tests/test_phase4_regime.py: OK")
