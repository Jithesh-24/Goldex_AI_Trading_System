"""decision/ev_cost.py
Spec sections 7/8: candidate SL/TP from MAE/MFE q75 (conservative, not
q50); round-trip transaction cost from live spread, in R-multiples of
the candidate SL distance. Never fabricates a cost when spread data is
missing or stale -- returns None so the caller (decision/ev_gate.py) can
force NO_TRADE."""
from datetime import datetime, timezone
from typing import Optional

from contracts.specialist_output import MAEOutput, MFEOutput

_OK_STATUSES = {"VALIDATED", "CANDIDATE"}


def candidate_sl_tp(mae: MAEOutput, mfe: MFEOutput) -> tuple[Optional[float], Optional[float]]:
    if mae.model_status not in _OK_STATUSES or mfe.model_status not in _OK_STATUSES:
        return None, None
    if mae.q75 is None or mfe.q75 is None:
        return None, None
    return mae.q75, mfe.q75


def round_trip_cost_r(market_state, candidate_sl_distance: float,
                       max_staleness_seconds: float = 5.0) -> Optional[float]:
    if candidate_sl_distance is None or candidate_sl_distance <= 0:
        return None
    if market_state is None or market_state.spread is None:
        return None
    age = (datetime.now(timezone.utc) - market_state.timestamp).total_seconds()
    if age > max_staleness_seconds:
        return None
    return (market_state.spread * 2) / candidate_sl_distance
