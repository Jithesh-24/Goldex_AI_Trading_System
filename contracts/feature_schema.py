"""Canonical feature schema contract -- prevents the feature-mismatch
problems between training and inference that motivated this rebuild."""
from typing import Optional

from pydantic import BaseModel, Field


class FeatureDescriptor(BaseModel):
    name: str
    family: str
    source: str
    frequency: str
    causal: bool
    required_data: list[str] = Field(default_factory=list)
    update_mechanism: str
    version: str
    dtype: str
    valid_range: Optional[tuple[float, float]] = None
    missing_value_policy: str


class FeatureSetSchema(BaseModel):
    schema_version: str
    features: list[FeatureDescriptor]
