"""python3 tests/test_phase4_direction.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_direction import run_direction_candidate


def test_run_direction_candidate_produces_real_metrics():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    # NOTE: brief's Step 1 draft used rows=20000, but oof_run's PurgedWalkForwardCV
    # has a hardcoded default min_train_bars=10_000 (measured in EVENTS, not raw
    # bars). rows=20000 -> only ~2,379 CUSUM events, so every fold's train_eligible
    # count is below 10,000 and select_top_features raises "requires at least one
    # fold's importances" before any metric is ever computed -- not a test
    # assertion failure, an unconditional exception. rows=600000 -> ~77.5k events,
    # enough for folds 1-4 to have >=10,000 train-eligible events, so oof_run
    # actually yields folds. Verified empirically before changing this value.
    result = run_direction_candidate(max_holding=45, rows=600000,
                                      registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 100, "too few events in dry run to trust any metric"
    assert 0.0 <= result["oos_log_loss"] < 5.0
    assert 0.0 <= result["oos_brier"] <= 1.0
    assert result["status"] in ("validated", "rejected")
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


if __name__ == "__main__":
    test_run_direction_candidate_produces_real_metrics()
    print("tests/test_phase4_direction.py: OK")
