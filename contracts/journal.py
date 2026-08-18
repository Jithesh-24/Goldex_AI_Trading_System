"""Canonical journal event contracts -- one model per lifecycle stage.
payload is an intentionally open dict: exact per-event fields are defined
by the phase that produces the event (e.g. the EOD learning system defines
LearningEvent.payload's shape when that phase is built); the envelope
(schema_version, trade_id, timestamp) is what's fixed now."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SignalEvent(BaseModel):
    schema_version: str = "v1"
    trade_id: str
    timestamp: datetime
    payload: dict


class MarketStateEvent(BaseModel):
    schema_version: str = "v1"
    trade_id: Optional[str] = None
    timestamp: datetime
    payload: dict


class ManagementEvent(BaseModel):
    schema_version: str = "v1"
    trade_id: str
    timestamp: datetime
    payload: dict


class ExecutionEvent(BaseModel):
    schema_version: str = "v1"
    trade_id: str
    timestamp: datetime
    payload: dict


class ResolutionEvent(BaseModel):
    schema_version: str = "v1"
    trade_id: str
    timestamp: datetime
    payload: dict


class LearningEvent(BaseModel):
    schema_version: str = "v1"
    timestamp: datetime
    payload: dict
