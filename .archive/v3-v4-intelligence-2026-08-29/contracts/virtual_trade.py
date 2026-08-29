"""Canonical virtual trade contract -- the full lifecycle object a signal
becomes once the human executes it. Most forecast/EV fields are Optional
in Phase 1 (not computed until later phases build the models that fill
them)."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class VirtualTrade(BaseModel):
    trade_id: str
    signal_timestamp: datetime
    direction: int  # +1 long, -1 short
    entry: float
    sl: float
    tp: float
    expected_value: Optional[float] = None
    confidence: Optional[float] = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    feature_schema_version: Optional[str] = None
    probability_state: Optional[dict] = None
    mae_forecast: Optional[float] = None
    mfe_forecast: Optional[float] = None
    regime: Optional[str] = None
    execution_metadata: Optional[dict] = None
    management_state: Optional[dict] = None
    resolution: Optional[str] = None
    outcome: Optional[str] = None
    journal_ref: Optional[str] = None
