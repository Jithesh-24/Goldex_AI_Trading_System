"""contracts/ev_decision.py
Phase 5 EVDecision -- the live/research output of the Probability/EV
Engine. Full lineage per spec section 15/27."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

Decision = Literal["NO_TRADE", "LONG_CANDIDATE", "SHORT_CANDIDATE"]
Direction = Literal["long", "short"]


class EVDecision(BaseModel):
    timestamp: datetime
    direction: Optional[Direction] = None
    decision: Decision
    ev_adj: float
    ev_raw: float
    uncertainty: float
    decision_margin: float
    candidate_sl: Optional[float] = None
    candidate_tp: Optional[float] = None
    cost_r: Optional[float] = None
    known_cost_only: bool
    specialist_model_ids: dict[str, str]
    calibration_ids: dict[str, str]
    feature_schema_ids: dict[str, str]
    ev_formula_version: str
    cost_model_version: str
    regime_state: Optional[int] = None
    timeout_r_provisional_proxy: bool
    decision_reason: str
