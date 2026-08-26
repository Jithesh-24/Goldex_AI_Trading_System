"""research/phase5b_diagnostics/d8_selection_calibration.py
Batch 2, D8: traces WHERE h=15's Barrier calibration collapses through the
real decision pipeline, reusing D5's proven-equivalent per-event
decision/ev_engine.evaluate() loop pattern rather than reconstructing an
approximation of it. See docs/superpowers/specs/2026-08-26-golex-v3-
phase5-batch2-ev-uncertainty-design.md section D8 for why only two
per-event-varying gates exist in this static replay methodology (the
honesty_note field explains this in the output itself, not just here).
"""
from datetime import datetime, timezone

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset
from research.phase5_timeout_payoff import estimate_timeout_payoff
from decision.ev_engine import evaluate, OPPORTUNITY_MIN_TAKE_PROBABILITY
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from research.phase5b_diagnostics._stats_utils import fit_calibration_slope_intercept

HONESTY_NOTE = (
    "In this static full-history replay, model_status-based gates (Direction/"
    "Barrier/MAE/MFE availability) are CONSTANT across all events for a given "
    "horizon -- registry status does not vary per-event within one replay run. "
    "They therefore cannot produce a distinguishable sub-population within a "
    "single horizon's stage trace. Only two gates vary per-event in this "
    "methodology: the Opportunity probability_take veto, and the final EV/"
    "MIN_EDGE_THRESHOLD gate. This is a real scope boundary of the replay "
    "methodology, stated explicitly rather than glossed over."
)


class _DiagMarketState:
    def __init__(self, spread, mid, vol_60s):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = vol_60s
        self.mid = mid


def _stage_stats(name: str, y_true: np.ndarray, p: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return {"stage": name, "n": 0, "calibration": {"slope": None, "intercept": None,
                "slope_se": None, "intercept_se": None, "n": 0}, "brier": None,
                "p_mean": None, "realized_outcome_rate": None}
    cal = fit_calibration_slope_intercept(y_true, p)
    brier = float(np.mean((p - y_true) ** 2))
    return {"stage": name, "n": n, "calibration": cal, "brier": brier,
            "p_mean": float(np.mean(p)), "realized_outcome_rate": float(np.mean(y_true))}


def run_d8(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    n = data["n"]
    side = data["side"]
    touch = data["touch"]
    p_barrier_win = data["p_barrier_win"]

    from research.phase5_ev_engine import _real_model_status
    direction_status = _real_model_status(f"direction_v3_candidate_h{max_holding}", registry_dir)
    opportunity_status = _real_model_status(f"opportunity_v3b_candidate_h{max_holding}", registry_dir)
    barrier_status = _real_model_status(f"barrier_v3b_candidate_h{max_holding}", registry_dir)
    mae_status = _real_model_status(f"mae_quantile_v3b_candidate_h{max_holding}", registry_dir)
    mfe_status = _real_model_status(f"mfe_quantile_v3b_candidate_h{max_holding}", registry_dir)

    y_side_correct = np.where(side == 1.0, (touch == 1).astype(float), (touch == -1).astype(float))

    opportunity_mask = data["p_opportunity"] >= OPPORTUNITY_MIN_TAKE_PROBABILITY
    traded_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        mid_i = float(data["mid"][i])
        vol_i = float(data["vol_60s_proxy"][i])
        ms = _DiagMarketState(spread=float(data["spread"][i]), mid=mid_i, vol_60s=vol_i)
        p_long = float(data["p_direction"][i])
        direction = DirectionOutput(model_id=f"direction_v3_candidate_h{max_holding}", horizon=max_holding,
                                     model_status=direction_status, probability_long=p_long,
                                     probability_short=1 - p_long, calibrated=True)
        opportunity = OpportunityOutput(model_id=f"opportunity_v3b_candidate_h{max_holding}", horizon=max_holding,
                                         model_status=opportunity_status, probability_take=float(data["p_opportunity"][i]),
                                         calibrated=True)
        barrier = BarrierOutput(model_id=f"barrier_v3b_candidate_h{max_holding}", horizon=max_holding,
                                 model_status=barrier_status, p_tp=float(p_barrier_win[i]), calibrated=True)
        mae = MAEOutput(model_id=f"mae_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3)
        mfe = MFEOutput(model_id=f"mfe_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3)
        d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        if d.decision != "NO_TRADE":
            traded_mask[i] = True

    stage0 = _stage_stats("stage_0_full_oos", y_side_correct, p_barrier_win)
    stage1 = _stage_stats("stage_1_after_opportunity_veto", y_side_correct[opportunity_mask], p_barrier_win[opportunity_mask])
    stage2 = _stage_stats("stage_2_after_ev_gate_final_traded", y_side_correct[traded_mask], p_barrier_win[traded_mask])
    stages = [stage0, stage1, stage2]

    full_slope = stage0["calibration"]["slope"]
    degradation_begins_at = stages[0]["stage"]
    DEVIATION_THRESHOLD = 0.3  # reported explicitly, not hidden; matches the same threshold convention used elsewhere in this batch's attribution framework
    for s in stages:
        slope = s["calibration"]["slope"]
        if slope is not None and full_slope is not None and abs(slope - full_slope) > DEVIATION_THRESHOLD:
            degradation_begins_at = s["stage"]
            break

    return {"horizon": max_holding, "stages": stages, "degradation_begins_at": degradation_begins_at,
            "honesty_note": HONESTY_NOTE, "deviation_threshold_used": DEVIATION_THRESHOLD}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d8(max_holding=h)
        print(f"D8 h={h}: degradation_begins_at={r['degradation_begins_at']}")
        for s in r["stages"]:
            print(f"  {s['stage']}: n={s['n']} slope={s['calibration']['slope']}")
