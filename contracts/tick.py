"""Canonical normalized tick contract. internal_seq is feed_listener.py's
own monotonic counter -- MT5 provides no reliable broker-side tick
sequence ID, this is explicitly internal, never presented as broker
sequencing."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Tick(BaseModel):
    symbol: str
    market_timestamp: datetime
    ingestion_timestamp: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    mid: float
    spread: float
    last: Optional[float] = None
    tick_volume: Optional[int] = None
    source: Literal["mt5_live", "synthetic_replay"]
    internal_seq: int
