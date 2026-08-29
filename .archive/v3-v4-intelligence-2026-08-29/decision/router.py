"""Static, config-driven model lookup. This is NOT a champion/challenger
engine -- it never compares live performance or picks a model based on
today's data. Model *selection* is exclusively a research process
(future phase); this class's only job at inference time is "load what
research already approved" via config/models.yaml's role -> model_id map."""
import json
import os
from typing import Optional

from contracts.model_registry import ModelRegistryEntry

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
REGISTRY_DIR = os.path.join(_MODELS_DIR, "registry")


class ModelRouter:
    def __init__(self, role_map: dict, registry_dir: str = REGISTRY_DIR, models_dir: str = _MODELS_DIR):
        self.role_map = role_map
        self.registry_dir = registry_dir
        self.models_dir = models_dir

    def resolve(self, role: str) -> Optional[ModelRegistryEntry]:
        model_id = self.role_map.get(role)
        if not model_id:
            return None
        path = os.path.join(self.registry_dir, f"{model_id}.json")
        with open(path) as f:
            return ModelRegistryEntry(**json.load(f))

    def artifact_path(self, role: str) -> Optional[str]:
        entry = self.resolve(role)
        if entry is None:
            return None
        return os.path.join(self.models_dir, entry.artifact_path)
