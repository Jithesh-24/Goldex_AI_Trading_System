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
from decision.ev_gate import compute_side_ev, decide
from decision.ev_formula import EV_FORMULA_VERSION

COST_MODEL_VERSION = "v1"
OPPORTUNITY_MIN_TAKE_PROBABILITY = 0.5
MARKET_STALENESS_SECONDS = 5.0
_OK = {"VALIDATED", "CANDIDATE"}


def evaluate(market_state, direction_out: DirectionOutput, opportunity_out: OpportunityOutput,
             barrier_out: BarrierOutput, p_sl_given_not_win: Optional[float],
             mae_out: MAEOutput, mfe_out: MFEOutput, timeout_r: float,
             timeout_r_provisional_proxy: bool, regime_state: Optional[int] = None) -> EVDecision:
    now = datetime.now(timezone.utc)
    uncertainty = 1.0  # Initialize at the top so all branches have a defined value
    sl_r, tp_r = candidate_sl_tp(mae_out, mfe_out)
    cost_r = round_trip_cost_r(market_state, sl_r, max_staleness_seconds=MARKET_STALENESS_SECONDS) if sl_r else None

    stale = market_state is None or (now - market_state.timestamp).total_seconds() > MARKET_STALENESS_SECONDS
    direction_available = direction_out.model_status in _OK and direction_out.probability_long is not None
    barrier_available = barrier_out.model_status in _OK

    if stale:
        reason = "MarketState stale"
        long_ev_adj = short_ev_adj = None
    elif not direction_available:
        reason = "Direction specialist unavailable"
        long_ev_adj = short_ev_adj = None
    elif not barrier_available:
        reason = "Barrier specialist unavailable"
        long_ev_adj = short_ev_adj = None
    elif cost_r is None:
        reason = "cost unavailable (spread missing/stale or no candidate SL)"
        long_ev_adj = short_ev_adj = None
    else:
        long_gate_ok = direction_out.probability_long > direction_out.probability_short
        short_gate_ok = not long_gate_ok
        if opportunity_out.model_status in _OK and opportunity_out.probability_take is not None:
            if opportunity_out.probability_take < OPPORTUNITY_MIN_TAKE_PROBABILITY:
                long_gate_ok = short_gate_ok = False
        uncertainty = 0.0
        if direction_out.model_status == "CANDIDATE":
            uncertainty += 0.2
        if barrier_out.model_status == "CANDIDATE":
            uncertainty += 0.2
        if opportunity_out.model_status not in _OK:
            uncertainty += 0.2
        uncertainty = min(uncertainty, 1.0)

        long_ev_adj = compute_side_ev(barrier_out, long_gate_ok, p_sl_given_not_win, tp_r, sl_r, timeout_r, cost_r, uncertainty)
        short_ev_adj = compute_side_ev(barrier_out, short_gate_ok, p_sl_given_not_win, tp_r, sl_r, timeout_r, cost_r, uncertainty)
        reason = None

    decision, direction, decide_reason = decide(long_ev_adj, short_ev_adj)
    final_reason = reason if reason else decide_reason
    chosen_ev_adj = {"long": long_ev_adj, "short": short_ev_adj}.get(direction, 0.0) or 0.0

    return EVDecision(
        timestamp=now, direction=direction, decision=decision,
        ev_adj=chosen_ev_adj, ev_raw=chosen_ev_adj, uncertainty=0.0 if stale else uncertainty,
        decision_margin=0.0, candidate_sl=sl_r, candidate_tp=tp_r, cost_r=cost_r, known_cost_only=True,
        specialist_model_ids={"direction": direction_out.model_id, "opportunity": opportunity_out.model_id,
                               "barrier": barrier_out.model_id, "mae": mae_out.model_id, "mfe": mfe_out.model_id},
        calibration_ids={}, feature_schema_ids={},
        ev_formula_version=EV_FORMULA_VERSION, cost_model_version=COST_MODEL_VERSION,
        regime_state=regime_state, timeout_r_provisional_proxy=timeout_r_provisional_proxy,
        decision_reason=final_reason,
    )
