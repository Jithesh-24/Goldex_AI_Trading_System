"""Contract validation smoke tests. Plain-assert, run directly:
python3 tests/test_contracts.py -- matches core/test_smoke.py convention,
no pytest dependency."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.model_registry import ModelRegistryEntry


def test_model_registry_entry_valid():
    entry = ModelRegistryEntry(
        model_id="direction_catboost_20260818",
        family="direction",
        algorithm="catboost",
        artifact_path="active/primary.cbm",
        created_at="2026-08-18T09:16:20",
        status="active",
        is_champion=True,
    )
    assert entry.model_id == "direction_catboost_20260818"
    assert entry.status == "active"


def test_model_registry_entry_rejects_bad_family():
    try:
        ModelRegistryEntry(
            model_id="x", family="not_a_real_family", algorithm="catboost",
            artifact_path="x.cbm", created_at="2026-08-18T09:16:20", status="active",
        )
        assert False, "expected validation error for bad family"
    except Exception as e:
        assert "family" in str(e).lower() or "literal" in str(e).lower()


if __name__ == "__main__":
    test_model_registry_entry_valid()
    test_model_registry_entry_rejects_bad_family()
    print("contracts/model_registry.py: OK")
