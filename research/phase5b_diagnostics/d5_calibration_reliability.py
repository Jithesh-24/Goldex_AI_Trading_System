"""research/phase5b_diagnostics/d5_calibration_reliability.py
Batch 1, D5: reliability curves, Brier score, ECE, and calibration
slope/intercept -- global, by long/short side, and in the traded subset
ONLY where n_traded > 0 (per design: never fabricate a traded-subset
statistic for a zero-trade horizon; report the literal N/A string
instead). Re-runs the same per-event decision/ev_engine.evaluate() loop
research/phase5_ev_engine.py::replay_and_validate already uses, but
captures per-event probability/outcome/decision arrays that function
doesn't expose, rather than modifying that (production-adjacent, Phase
5A-reviewed) file. See docs/superpowers/specs/2026-08-26-golex-v3-
phase5-batch1-diagnostics-design.md section D5.
"""
from datetime import datetime, timezone

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset, realized_r_for_direction
from research.phase5_timeout_payoff import estimate_timeout_payoff
from decision.ev_engine import evaluate
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput

N_BINS = 10


class _DiagMarketState:
    def __init__(self, spread, mid, vol_60s):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = vol_60s
        self.mid = mid


def _reliability_and_scores(y_true: np.ndarray, p: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None, "calibration": {"intercept": None, "slope": None,
                "intercept_se": None, "slope_se": None, "n": 0}, "reliability_bins": []}
    from research.phase5b_diagnostics._stats_utils import fit_calibration_slope_intercept
    brier = float(np.mean((p - y_true) ** 2))
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
    bins = []
    ece = 0.0
    for b in range(N_BINS):
        m = bin_idx == b
        bn = int(m.sum())
        if bn == 0:
            bins.append({"bin": b, "n": 0, "mean_predicted": None, "observed_rate": None})
            continue
        mean_p = float(p[m].mean())
        obs = float(y_true[m].mean())
        bins.append({"bin": b, "n": bn, "mean_predicted": mean_p, "observed_rate": obs})
        ece += (bn / n) * abs(obs - mean_p)
    cal = fit_calibration_slope_intercept(y_true, p)
    return {"n": n, "brier": brier, "ece": float(ece), "calibration": cal, "reliability_bins": bins}


def run_d5(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    n = data["n"]
    side = data["side"]

    from research.phase5_ev_engine import _real_model_status
    direction_status = _real_model_status(f"direction_v3_candidate_h{max_holding}", registry_dir)
    opportunity_status = _real_model_status(f"opportunity_v3b_candidate_h{max_holding}", registry_dir)
    barrier_status = _real_model_status(f"barrier_v3b_candidate_h{max_holding}", registry_dir)
    mae_status = _real_model_status(f"mae_quantile_v3b_candidate_h{max_holding}", registry_dir)
    mfe_status = _real_model_status(f"mfe_quantile_v3b_candidate_h{max_holding}", registry_dir)

    p_used = np.full(n, np.nan)   # the probability that drove each event's decision (Barrier's p_tp)
    y_outcome = np.full(n, np.nan)  # 1 if the traded/proposed side's touch matched, else 0 -- NaN if NO_TRADE
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
                                 model_status=barrier_status, p_tp=float(data["p_barrier_win"][i]), calibrated=True)
        mae = MAEOutput(model_id=f"mae_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3)
        mfe = MFEOutput(model_id=f"mfe_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3)

        d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        p_used[i] = float(data["p_barrier_win"][i])
        if d.decision != "NO_TRADE":
            traded_mask[i] = True
            touch = data["touch"][i]
            y_outcome[i] = 1.0 if ((d.direction == "long" and touch == 1) or (d.direction == "short" and touch == -1)) else 0.0

    # "global" calibration uses touch-derived correctness for Barrier's OWN proposed
    # side (side array), not just traded events -- this answers "is p_barrier_win calibrated
    # against ITS side's real touch outcome", independent of whether the EV gate traded it.
    # (p_used/traded_mask, populated in the loop above, are used below for traded_subset.)
    touch_all = data["touch"]
    y_side_correct = np.where(side == 1.0, (touch_all == 1).astype(float), (touch_all == -1).astype(float))
    global_stats = _reliability_and_scores(y_side_correct, data["p_barrier_win"])

    long_mask = side == 1.0
    short_mask = side == -1.0
    by_side = {
        "long": _reliability_and_scores(y_side_correct[long_mask], data["p_barrier_win"][long_mask]),
        "short": _reliability_and_scores(y_side_correct[short_mask], data["p_barrier_win"][short_mask]),
    }

    n_traded = int(traded_mask.sum())
    if n_traded == 0:
        traded_subset = "N/A (zero trades at this horizon)"
    else:
        traded_subset = _reliability_and_scores(y_outcome[traded_mask], p_used[traded_mask])

    return {"horizon": max_holding, "global": global_stats, "by_side": by_side, "traded_subset": traded_subset}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d5(max_holding=h)
        print(f"D5 h={h}: global_n={r['global']['n']} brier={r['global']['brier']} "
              f"traded_subset={'N/A' if isinstance(r['traded_subset'], str) else r['traded_subset']['n']}")
