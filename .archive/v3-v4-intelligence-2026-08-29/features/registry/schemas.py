"""Persists FeatureSetSchema slices (spec section 9/18/20) so a
ModelRegistryEntry.feature_schema_version can point at a real, re-loadable
artifact instead of an inline list nobody re-checks at load time."""
import json
import os

from contracts.feature_schema import FeatureSetSchema

SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")


def _path(schema_id: str, schema_version: str, schemas_dir: str) -> str:
    os.makedirs(schemas_dir, exist_ok=True)
    return os.path.join(schemas_dir, f"{schema_id}__{schema_version}.json")


def save_schema(schema: FeatureSetSchema, schemas_dir: str = SCHEMAS_DIR) -> str:
    path = _path(schema.schema_id, schema.schema_version, schemas_dir)
    with open(path, "w") as f:
        f.write(schema.model_dump_json(indent=2))
    return path


def load_schema(schema_id: str, schema_version: str, schemas_dir: str = SCHEMAS_DIR) -> FeatureSetSchema:
    path = _path(schema_id, schema_version, schemas_dir)
    with open(path) as f:
        return FeatureSetSchema(**json.load(f))
