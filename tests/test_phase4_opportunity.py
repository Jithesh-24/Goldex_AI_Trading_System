"""python3 tests/test_phase4_opportunity.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_opportunity import run_opportunity_candidate


def test_run_opportunity_candidate_produces_real_metrics():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    # NOTE: mirrors Task 4's finding -- oof_run's PurgedWalkForwardCV has a
    # hardcoded default min_train_bars=10_000 (measured in EVENTS, not raw
    # bars). rows=20000 from the brief's draft was verified (Task 4 precedent)
    # to yield too few CUSUM events for any fold to clear that floor, so
    # select_top_features would raise before any metric is computed. Using
    # rows=600000 here for the same reason, verified empirically below.
    result = run_opportunity_candidate(max_holding=45, rows=600000,
                                        registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50, "too few meta-training events in dry run to trust any metric"
    assert 0.0 <= result["oos_log_loss"] < 5.0
    assert result["status"] in ("validated", "rejected")
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


if __name__ == "__main__":
    test_run_opportunity_candidate_produces_real_metrics()
    print("tests/test_phase4_opportunity.py: OK")
