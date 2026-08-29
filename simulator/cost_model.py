"""simulator/cost_model.py
Extracted from decision/ev_cost.py during the GOLDEX foundation cleanup
(2026-08-29) -- round_trip_cost_r has no dependency on the retired V3
specialist/decision architecture; it only reads market_state fields and a
plain float SL distance. See docs/superpowers/plans/2026-08-29-goldex-foundation-cleanup-plan.md
Section D.1. candidate_sl_tp (the MAEOutput/MFEOutput-keyed sibling) is
genuinely V3-specific and was archived with decision/, not carried forward.
"""
from datetime import datetime, timezone
from typing import Optional


def round_trip_cost_r(market_state, candidate_sl_distance: float,
                       max_staleness_seconds: float = 5.0) -> Optional[float]:
    """candidate_sl_distance (mae.q75) is an R-MULTIPLE -- already normalized
    by a volatility estimate, see research/audit_edge.py's _mae_mfe_core:
    mae[e] = -worst / v, where `worst` is a price-return fraction and `v` is
    a volatility estimate in return units. market_state.spread is in PRICE
    units. Dividing price by an R-multiple does not produce a valid
    R-multiple cost -- the same volatility normalization used to produce the
    R-multiple in the first place must be applied to the spread too:
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
    # apply the identical age<0 (future-timestamp) rejection that
    # decision/ev_engine.py's separate staleness check already applied.
    age = (datetime.now(timezone.utc) - market_state.market_timestamp).total_seconds()
    if age > max_staleness_seconds or age < 0:
        return None
    denom = candidate_sl_distance * market_state.realized_vol_60s * market_state.mid
    if denom <= 0:
        return None
    return (market_state.spread * 2) / denom
