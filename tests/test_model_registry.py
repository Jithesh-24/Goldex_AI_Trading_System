"""python3 tests/test_model_registry.py"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.model_registry import ModelRegistryEntry

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
MODELS_DIR = os.path.join(BASE, "models")


def test_every_registry_entry_parses():
    paths = glob.glob(os.path.join(REGISTRY_DIR, "*.json"))
    assert len(paths) > 0, "expected at least one registry entry"
    for path in paths:
        with open(path) as f:
            ModelRegistryEntry(**json.load(f))


def test_active_artifacts_exist_on_disk():
    for path in glob.glob(os.path.join(REGISTRY_DIR, "*.json")):
        with open(path) as f:
            entry = ModelRegistryEntry(**json.load(f))
        if entry.status == "active":
            full = os.path.join(MODELS_DIR, entry.artifact_path)
            assert os.path.exists(full), f"active entry {entry.model_id} points at missing {full}"


def test_exactly_two_active_champions():
    champions = []
    for path in glob.glob(os.path.join(REGISTRY_DIR, "*.json")):
        with open(path) as f:
            entry = ModelRegistryEntry(**json.load(f))
        if entry.is_champion:
            champions.append(entry.model_id)
    assert sorted(champions) == ["direction_catboost_20260818", "opportunity_meta_catboost_20260818"]


if __name__ == "__main__":
    test_every_registry_entry_parses()
    test_active_artifacts_exist_on_disk()
    test_exactly_two_active_champions()
    print("models/registry: OK")
