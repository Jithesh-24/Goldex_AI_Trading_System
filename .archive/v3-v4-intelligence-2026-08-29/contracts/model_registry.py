"""Canonical model registry contract. The live router and every training
script that registers a model both import ModelRegistryEntry from here --
nobody redeclares this shape."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

ModelFamily = Literal[
    "direction", "opportunity_meta", "regime",
    "mae_quantile", "mfe_quantile", "barrier_probability", "execution_decay",
]
ModelStatus = Literal["candidate", "validated", "active", "archived", "rejected"]


class ModelLineage(BaseModel):
    data_snapshot: Optional[str] = None
    code_commit: Optional[str] = None
    config_snapshot: Optional[str] = None


class ModelRegistryEntry(BaseModel):
    model_id: str
    family: ModelFamily
    algorithm: str
    artifact_path: str
    feature_schema_version: Optional[str] = None
    feature_cols: list[str] = Field(default_factory=list)
    target_definition: Optional[str] = None
    training_config: dict = Field(default_factory=dict)
    training_period: Optional[str] = None
    validation_period: Optional[str] = None
    created_at: datetime
    status: ModelStatus
    is_champion: bool = False
    metrics: dict = Field(default_factory=dict)
    lineage: ModelLineage = Field(default_factory=ModelLineage)
