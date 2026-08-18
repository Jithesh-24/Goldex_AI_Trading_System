"""Canonical live market state contract. Most fields are Optional in
Phase 1 -- the shape exists so later phases populate it from a real feed
instead of inventing a new one. See market/README.md for why this isn't
wired to a live feed yet."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MarketState(BaseModel):
    timestamp: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    spread: Optional[float] = None
    mid: Optional[float] = None
    tick_state: Optional[dict] = None
    m1_state: Optional[dict] = None
    multi_horizon_state: Optional[dict] = None
    volatility_state: Optional[dict] = None
    activity_state: Optional[dict] = None
    session: Optional[str] = None
    regime: Optional[str] = None
    feature_state_ref: Optional[str] = None
