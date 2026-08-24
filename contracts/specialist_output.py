"""contracts/specialist_output.py
Formal per-role contracts the Probability/EV Engine consumes. Every field
except model_status/model_id/horizon is Optional -- a non-VALIDATED/
CANDIDATE status omits misleading numeric values entirely rather than
populating them with placeholders (spec section 3/6)."""
from typing import Literal, Optional

from pydantic import BaseModel

ModelStatus = Literal["VALIDATED", "CANDIDATE", "DATA_LIMITED", "UNAVAILABLE", "STALE", "INVALID"]


class DirectionOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    probability_long: Optional[float] = None
    probability_short: Optional[float] = None
    calibrated: bool = False


class OpportunityOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    probability_take: Optional[float] = None
    calibrated: bool = False
    assumed_side: Optional[float] = None
    direction_model_id: Optional[str] = None


class RegimeOutput(BaseModel):
    model_id: str
    model_status: ModelStatus
    regime_state: Optional[int] = None
    regime_probabilities: Optional[list[float]] = None


class MAEOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    q50: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None
    assumed_side: Optional[float] = None
    direction_model_id: Optional[str] = None


class MFEOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    q50: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None
    assumed_side: Optional[float] = None
    direction_model_id: Optional[str] = None


class BarrierOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    p_tp: Optional[float] = None
    p_sl: Optional[float] = None
    p_timeout: Optional[float] = None
    calibrated: bool = False
    assumed_side: Optional[float] = None
    direction_model_id: Optional[str] = None


class ExecutionOutput(BaseModel):
    model_id: str
    model_status: ModelStatus
    drift_60s: Optional[float] = None
    drift_120s: Optional[float] = None
    data_limited: bool = True
