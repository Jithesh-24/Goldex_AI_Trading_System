"""python3 tests/test_baseline_registry.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.registry import load_family, build_schema

_MODEL_REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "registry", "direction_catboost_20260818.json")


def test_baseline_registry_matches_deployed_model():
    with open(_MODEL_REGISTRY) as f:
        deployed = json.load(f)
    deployed_cols = set(deployed["feature_cols"])
    registered = {d.feature_id for d in load_family("baseline_v1")}
    assert registered == deployed_cols, registered.symmetric_difference(deployed_cols)


def test_build_baseline_schema():
    with open(_MODEL_REGISTRY) as f:
        deployed = json.load(f)
    schema = build_schema("baseline_v1", deployed["feature_schema_version"], deployed["feature_cols"])
    assert schema.feature_ids == deployed["feature_cols"]


if __name__ == "__main__":
    test_baseline_registry_matches_deployed_model()
    test_build_baseline_schema()
    print("tests/test_baseline_registry.py: OK")
