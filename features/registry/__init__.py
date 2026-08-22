"""Feature registry loader -- reads FeatureDescriptor JSON files organized
one-subdirectory-per-family under features/registry/, and builds named
FeatureSetSchema slices for future model-specific schemas (spec section
9: the registry preserves the ENTIRE quantitative universe; nothing here
pre-selects a "final" feature set)."""
import json
import os
from datetime import datetime, timezone

from contracts.feature_schema import FeatureDescriptor, FeatureSetSchema

REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))


def load_descriptor(path: str) -> FeatureDescriptor:
    with open(path) as f:
        return FeatureDescriptor(**json.load(f))


def load_family(family: str, registry_dir: str = REGISTRY_DIR) -> list[FeatureDescriptor]:
    family_dir = os.path.join(registry_dir, family)
    if not os.path.isdir(family_dir):
        return []
    out = []
    for fname in sorted(os.listdir(family_dir)):
        if fname.endswith(".json"):
            out.append(load_descriptor(os.path.join(family_dir, fname)))
    return out


NON_FAMILY_DIRS = {"schemas"}  # features/registry/schemas/ holds saved FeatureSetSchema
# slices (features/registry/schemas.py's save_schema), not FeatureDescriptor family
# JSON -- excluded here so load_all() doesn't try to parse them as descriptors.


def load_all(registry_dir: str = REGISTRY_DIR) -> list[FeatureDescriptor]:
    out = []
    for entry in sorted(os.listdir(registry_dir)):
        full = os.path.join(registry_dir, entry)
        if os.path.isdir(full) and not entry.startswith("__") and entry not in NON_FAMILY_DIRS:
            out.extend(load_family(entry, registry_dir=registry_dir))
    return out


def build_schema(schema_id: str, schema_version: str, feature_ids: list[str]) -> FeatureSetSchema:
    return FeatureSetSchema(schema_id=schema_id, schema_version=schema_version,
                             feature_ids=feature_ids, created_at=datetime.now(timezone.utc))
