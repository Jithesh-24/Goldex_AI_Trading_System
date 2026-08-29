"""python3 tests/test_feature_registry.py"""
import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.registry import load_descriptor, load_family, load_all, build_schema, REGISTRY_DIR


def test_load_descriptor_fixture():
    fixture_dir = tempfile.mkdtemp()
    try:
        family_dir = os.path.join(fixture_dir, "test_family")
        os.makedirs(family_dir)
        payload = {
            "feature_id": "fixture_feat", "family": "test_family",
            "mathematical_definition": "x", "source_module": "features.fixture",
            "required_state": ["close"], "update_trigger": "M1_CLOSE",
            "window": 5, "causal": True, "live_compatible": True,
            "computational_cost": "LOW", "missing_value_policy": "NaN",
            "warmup_bars": 5, "historical_coverage": "FULL_HISTORY",
            "status": "REQUIRED", "status_reason": "fixture", "version": "v1",
        }
        path = os.path.join(family_dir, "fixture_feat.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        d = load_descriptor(path)
        assert d.feature_id == "fixture_feat"
        fam = load_family("test_family", registry_dir=fixture_dir)
        assert len(fam) == 1 and fam[0].feature_id == "fixture_feat"
    finally:
        shutil.rmtree(fixture_dir)


def test_load_all_real_registry_empty_before_family_tasks():
    # REGISTRY_DIR exists (created this task) but has no family JSON yet --
    # later tasks populate it. Must not crash on an empty/partial registry.
    all_descriptors = load_all()
    assert isinstance(all_descriptors, list)


def test_build_schema():
    schema = build_schema("test_schema", "v1", ["a", "b", "c"])
    assert schema.feature_ids == ["a", "b", "c"]
    assert schema.schema_id == "test_schema"


if __name__ == "__main__":
    test_load_descriptor_fixture()
    test_load_all_real_registry_empty_before_family_tasks()
    test_build_schema()
    print("tests/test_feature_registry.py: OK")
