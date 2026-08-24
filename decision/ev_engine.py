"""decision/ev_engine.py
Spec section 14: the live entry point -- a pure function, MarketState +
specialist outputs -> EVDecision. Called ONLY from a shadow-evaluation
path (Task 13); never wired into app/engine.py's production decision
sequence."""
from datetime import datetime, timezone
from typing import Optional

from contracts.ev_decision import EVDecision
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_cost import candidate_sl_tp, round_trip_cost_r
from decision.ev_gate import compute_side_ev, decide, MIN_EDGE_THRESHOLD
from decision.ev_formula import EV_FORMULA_VERSION

COST_MODEL_VERSION = "v1"
OPPORTUNITY_MIN_TAKE_PROBABILITY = 0.5
MARKET_STALENESS_SECONDS = 5.0
_OK = {"VALIDATED", "CANDIDATE"}
# Real feature-schema IDs registered by each Phase 4/5 specialist script
# (e.g. research/phase4_direction.py) via features.registry.build_schema +
# save_schema. All 15 (5 specialists x 3 horizons) exist on disk under
# features/registry/schemas/ as "<schema_id>__<schema_version>.json" --
# verify with: ls features/registry/schemas/*_v3_h*.json
FEATURE_SCHEMA_VERSION = "2026-08-22"


def evaluate(market_state, direction_out: DirectionOutput, opportunity_out: OpportunityOutput,
             barrier_out: BarrierOutput, p_sl_given_not_win: Optional[float],
             mae_out: MAEOutput, mfe_out: MFEOutput, timeout_r: float,
             timeout_r_provisional_proxy: bool, regime_state: Optional[int] = None) -> EVDecision:
    now = datetime.now(timezone.utc)
    uncertainty = 1.0  # Initialize at the top so all branches have a defined value
    sl_r, tp_r = candidate_sl_tp(mae_out, mfe_out)
    cost_r = round_trip_cost_r(market_state, sl_r, max_staleness_seconds=MARKET_STALENESS_SECONDS) if sl_r else None

    # FIX 4 (C5): the real contracts.market_state.MarketState has no `.timestamp`
    # field -- only market_timestamp/ingestion_timestamp/processing_timestamp.
    # market_timestamp is the semantically correct one for a staleness check:
    # it reflects when the market data itself was captured, not this system's
    # internal pipeline delay (ingestion/processing_timestamp).
    age = (now - market_state.market_timestamp).total_seconds() if market_state else None
    stale = market_state is None or age > MARKET_STALENESS_SECONDS or (age is not None and age < 0)
    # FIX 5 (I2): also require probability_short is not None -- it is
    # independently Optional per contracts/specialist_output.py, so a Direction
    # output missing only probability_short used to crash on
    # `probability_long > probability_short` below instead of forcing NO_TRADE.
    direction_available = (direction_out.model_status in _OK
                            and direction_out.probability_long is not None
                            and direction_out.probability_short is not None)
    barrier_available = barrier_out.model_status in _OK

    if stale:
        reason = "MarketState stale"
        long_ev = short_ev = None
    elif not direction_available:
        reason = "Direction specialist unavailable"
        long_ev = short_ev = None
    elif not barrier_available:
        reason = "Barrier specialist unavailable"
        long_ev = short_ev = None
    elif cost_r is None:
        reason = "cost unavailable (spread missing/stale or no candidate SL)"
        long_ev = short_ev = None
    else:
        long_gate_ok = direction_out.probability_long > direction_out.probability_short
        short_gate_ok = not long_gate_ok
        # FIX (targeted correction pass, 2026-08-24): Opportunity's veto used to
        # be skipped entirely whenever its status was untrusted (UNAVAILABLE/
        # DATA_LIMITED/INVALID/STALE), meaning an untrusted Opportunity specialist
        # INCREASED tradeable volume instead of blocking it. Opportunity has no
        # separate availability gate in this design (unlike Direction/Barrier
        # above) -- it is always expected to participate once we reach this
        # branch, so any non-trusted status (or a trusted status with a missing
        # probability_take) must fail CLOSED: force NO_TRADE on both sides.
        if opportunity_out.model_status in _OK and opportunity_out.probability_take is not None:
            if opportunity_out.probability_take < OPPORTUNITY_MIN_TAKE_PROBABILITY:
                long_gate_ok = short_gate_ok = False
        else:
            long_gate_ok = short_gate_ok = False
        uncertainty = 0.0
        if direction_out.model_status == "CANDIDATE":
            uncertainty += 0.2
        if barrier_out.model_status == "CANDIDATE":
            uncertainty += 0.2
        if opportunity_out.model_status not in _OK:
            uncertainty += 0.2
        uncertainty = min(uncertainty, 1.0)

        long_ev = compute_side_ev(barrier_out, long_gate_ok, p_sl_given_not_win, tp_r, sl_r, timeout_r, cost_r, uncertainty)
        short_ev = compute_side_ev(barrier_out, short_gate_ok, p_sl_given_not_win, tp_r, sl_r, timeout_r, cost_r, uncertainty)
        reason = None

    long_ev_adj = long_ev["ev_adj"] if long_ev else None
    short_ev_adj = short_ev["ev_adj"] if short_ev else None
    decision, direction, decide_reason = decide(long_ev_adj, short_ev_adj)
    final_reason = reason if reason else decide_reason
    chosen = {"long": long_ev, "short": short_ev}.get(direction)
    chosen_ev_adj = chosen["ev_adj"] if chosen else 0.0
    chosen_ev_raw = chosen["ev_raw"] if chosen else 0.0

    # FIX 6 (I2): decision_margin is now a real computed value instead of a
    # hardcoded 0.0. When both sides produced a real ev_adj, it's the "how
    # much better was the winning side" margin: abs(long_ev_adj - short_ev_adj).
    # Otherwise (only one side/neither available), fall back to the margin
    # above the trade gate: chosen_ev_adj - MIN_EDGE_THRESHOLD (0.0 when no
    # side was chosen at all).
    if long_ev_adj is not None and short_ev_adj is not None:
        decision_margin = abs(long_ev_adj - short_ev_adj)
    elif chosen is not None:
        decision_margin = chosen_ev_adj - MIN_EDGE_THRESHOLD
    else:
        decision_margin = 0.0

    # FIX 6 (I2): calibration_ids/feature_schema_ids populated with the real
    # specialist model_id strings already in scope, mirroring
    # specialist_model_ids's existing pattern. This is a documented
    # simplification -- a true separate calibration-artifact ID isn't
    # threaded through end-to-end yet (Task 3/4's CalibrationRegistry artifact
    # IDs), so each specialist's own model_id doubles as a placeholder here.
    calibration_ids = {"direction": direction_out.model_id, "opportunity": opportunity_out.model_id,
                        "barrier": barrier_out.model_id, "mae": mae_out.model_id, "mfe": mfe_out.model_id}
    # FIX (targeted correction pass, 2026-08-24): feature_schema_ids used to be
    # aliased to calibration_ids (literal dict reuse) -- it must carry actual
    # feature schema IDs, distinct from the specialist model_ids above. Schema
    # ID format is "<specialist>_v3_h<horizon>" (mae/mfe use "_quantile_v3_h"),
    # per features/registry/schemas/*.json filenames.
    feature_schema_ids = {
        "direction": f"direction_v3_h{direction_out.horizon}__{FEATURE_SCHEMA_VERSION}",
        "opportunity": f"opportunity_v3_h{opportunity_out.horizon}__{FEATURE_SCHEMA_VERSION}",
        "barrier": f"barrier_v3_h{barrier_out.horizon}__{FEATURE_SCHEMA_VERSION}",
        "mae": f"mae_quantile_v3_h{mae_out.horizon}__{FEATURE_SCHEMA_VERSION}",
        "mfe": f"mfe_quantile_v3_h{mfe_out.horizon}__{FEATURE_SCHEMA_VERSION}",
    }

    return EVDecision(
        timestamp=now, direction=direction, decision=decision,
        ev_adj=chosen_ev_adj, ev_raw=chosen_ev_raw, uncertainty=0.0 if stale else uncertainty,
        decision_margin=decision_margin, candidate_sl=sl_r, candidate_tp=tp_r, cost_r=cost_r, known_cost_only=True,
        specialist_model_ids={"direction": direction_out.model_id, "opportunity": opportunity_out.model_id,
                               "barrier": barrier_out.model_id, "mae": mae_out.model_id, "mfe": mfe_out.model_id},
        calibration_ids=calibration_ids, feature_schema_ids=feature_schema_ids,
        ev_formula_version=EV_FORMULA_VERSION, cost_model_version=COST_MODEL_VERSION,
        regime_state=regime_state, timeout_r_provisional_proxy=timeout_r_provisional_proxy,
        decision_reason=final_reason,
    )
