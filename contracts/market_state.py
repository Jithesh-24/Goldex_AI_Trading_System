"""Canonical live market state contract -- the single source of truth
every future V3 component reads. Feed health is authoritative: a
consumer must check feed_health before trusting price fields. Missing
data uses DataQuality.UNAVAILABLE/UNKNOWN, never a silent 0.0."""
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeedHealthState(str, Enum):
    UNKNOWN = "UNKNOWN"
    CONNECTED = "CONNECTED"
    STALE = "STALE"
    RECONNECTING = "RECONNECTING"
    DISCONNECTED = "DISCONNECTED"
    INVALID = "INVALID"


class DataQuality(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class M1BarState(BaseModel):
    open: float
    high: float
    low: float
    close: float
    tick_count: int
    start_time: datetime
    end_time: Optional[datetime] = None
    complete: bool


class MarketState(BaseModel):
    # IDENTITY
    symbol: str
    source: Literal["mt5_live", "synthetic_replay"]
    state_version: str = "v1"
    sequence: int
    # TIME
    market_timestamp: datetime
    ingestion_timestamp: datetime
    processing_timestamp: datetime
    # PRICE
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    mid: float
    spread: float
    last: Optional[float] = None
    last_quality: DataQuality = DataQuality.UNAVAILABLE
    # ACTIVITY
    tick_count_60s: int
    tick_count_300s: int
    tick_rate_per_sec: float
    # BAR STATE
    current_m1: Optional[M1BarState] = None
    completed_m1: Optional[M1BarState] = None
    # VOLATILITY STATE (raw inputs only, not the Phase 3 feature library)
    realized_vol_60s: Optional[float] = None
    spread_mean_60s: Optional[float] = None
    spread_std_60s: Optional[float] = None
    # FEED HEALTH
    feed_health: FeedHealthState
    last_tick_age_sec: float
    feed_latency_sec: Optional[float] = None
    state_update_latency_sec: Optional[float] = None
