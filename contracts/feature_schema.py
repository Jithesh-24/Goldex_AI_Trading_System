"""Canonical feature schema contract -- prevents the feature-mismatch
problems between training and inference that motivated this rebuild.
Extended in Phase 3 with full per-feature metadata (spec section 5) and
model-routing schema slicing (spec section 9): the registry preserves the
entire quantitative universe, and FeatureSetSchema.feature_ids lets each
future specialist model construct its own slice -- no feature is
pre-selected here."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FeatureStatus(str, Enum):
    REQUIRED = "REQUIRED"
    USEFUL = "USEFUL"
    OPTIONAL = "OPTIONAL"
    UNSUPPORTED_BY_DATA = "UNSUPPORTED_BY_DATA"
    REDUNDANT = "REDUNDANT"
    REJECTED = "REJECTED"


class HistoricalCoverage(str, Enum):
    FULL_HISTORY = "FULL_HISTORY"
    PARTIAL_HISTORY = "PARTIAL_HISTORY"
    LIVE_ONLY = "LIVE_ONLY"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    UNSUPPORTED = "UNSUPPORTED"


class ComputationalCost(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    RESEARCH_ONLY = "RESEARCH_ONLY"


class UpdateTrigger(str, Enum):
    TICK = "TICK"
    M1_CLOSE = "M1_CLOSE"
    DAILY = "DAILY"
    EVENT = "EVENT"


class FeatureDescriptor(BaseModel):
    feature_id: str
    family: str
    mathematical_definition: str
    source_module: str
    required_state: list[str] = Field(default_factory=list)
    update_trigger: UpdateTrigger
    window: Optional[int] = None
    causal: bool
    live_compatible: bool
    computational_cost: ComputationalCost
    numerical_stability_notes: Optional[str] = None
    missing_value_policy: str
    warmup_bars: int
    dependencies: list[str] = Field(default_factory=list)
    units: Optional[str] = None
    normalization: Optional[str] = None
    expected_range: Optional[tuple[float, float]] = None
    historical_coverage: HistoricalCoverage
    status: FeatureStatus
    status_reason: str
    evidence_ref: Optional[str] = None
    version: str


class FeatureSetSchema(BaseModel):
    schema_id: str
    schema_version: str
    feature_ids: list[str]
    created_at: datetime
