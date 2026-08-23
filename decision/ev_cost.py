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
    """FIX 3 (C4, final-review fix wave): candidate_sl_distance (mae.q75) is an
    R-MULTIPLE -- already normalized by a volatility estimate, see
    research/audit_edge.py's _mae_mfe_core: mae[e] = -worst / v, where `worst`
    is a price-return fraction and `v` is a volatility estimate in return
    units. market_state.spread is in PRICE units. Dividing price by an
    R-multiple does not produce a valid R-multiple cost -- the same
    volatility normalization used to produce the R-multiple in the first
    place must be applied to the spread too:
        cost_R = (spread_price * 2) / (candidate_sl_distance_R * vol_estimate * mid_price)
    vol_estimate is market_state.realized_vol_60s -- the closest available
    live volatility proxy (contracts/market_state.py). This is an
    APPROXIMATION of the training-time `vol_tb` convention, not identical --
    exact parity would require replicating the full Phase 3 feature
    pipeline's EWMA-vol-scaled-to-horizon computation live, which is out of
    scope here. If realized_vol_60s is None (it's an Optional field), this
    returns None rather than fabricating a vol estimate -- extending the
    "never fabricate a cost" principle to this new input.
    """
    if candidate_sl_distance is None or candidate_sl_distance <= 0:
        return None
    if market_state is None or market_state.spread is None:
        return None
    if market_state.realized_vol_60s is None or market_state.realized_vol_60s <= 0:
        return None
    # FIX 9 (I5): apply the identical age<0 (future-timestamp) rejection that
    # decision/ev_engine.py's separate staleness check already applies.
    age = (datetime.now(timezone.utc) - market_state.market_timestamp).total_seconds()
    if age > max_staleness_seconds or age < 0:
        return None
    denom = candidate_sl_distance * market_state.realized_vol_60s * market_state.mid
    if denom <= 0:
        return None
    return (market_state.spread * 2) / denom
