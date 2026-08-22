"""python3 tests/test_feature_set_schemas.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.registry import build_schema
from features.registry.schemas import save_schema, load_schema


def test_save_then_load_roundtrip():
    schema = build_schema("direction_v3", "2026-08-22", ["dist_from_high_20", "hour_sin"])
    with tempfile.TemporaryDirectory() as tmp:
        path = save_schema(schema, schemas_dir=tmp)
        assert os.path.exists(path)
        loaded = load_schema("direction_v3", "2026-08-22", schemas_dir=tmp)
        assert loaded.feature_ids == ["dist_from_high_20", "hour_sin"]
        assert loaded.schema_version == "2026-08-22"


if __name__ == "__main__":
    test_save_then_load_roundtrip()
    print("tests/test_feature_set_schemas.py: OK")
