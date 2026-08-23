"""research/phase5_ev_engine.py
Spec sections 13/12/22/23: research-only EV replay simulator. Calls the
SAME decision/ev_engine.py.evaluate() pure function the live path uses
(spec section 14's live/replay equivalence requirement), against
historical specialist-output replay data. Computes OOS decision
distribution, expected-vs-realized R, a baseline comparison (simple
P(direction)>0.55 gate, no cost/no EV), and a sensitivity/fragility scan
(spec section 20/12). NO Telegram, no live I/O.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_engine
"""
from datetime import datetime, timezone
import json
import os
import tempfile

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset
from research.phase5_timeout_payoff import estimate_timeout_payoff
from research.phase5_barrier_split import run_barrier_split_candidate
from decision.ev_engine import evaluate
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput

SIMPLE_BASELINE_THRESHOLD = 0.55

_REAL_REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "registry")

# FIX 10 (I7+I8, final-review fix wave): registry ModelStatus -> specialist-
# output-contract model_status. A rejected model's predictions shouldn't be
# trusted numerically, so it maps to UNAVAILABLE (not a lower-confidence
# status that still lets its numbers flow into the EV formula).
_REGISTRY_STATUS_MAP = {"validated": "VALIDATED", "candidate": "CANDIDATE", "rejected": "UNAVAILABLE",
                         "active": "VALIDATED", "archived": "UNAVAILABLE"}


def _real_model_status(model_id: str, registry_dir: str = None) -> str:
    d = registry_dir if registry_dir else _REAL_REGISTRY_DIR
    path = os.path.join(d, f"{model_id}.json")
    with open(path) as f:
        entry = json.load(f)
    return _REGISTRY_STATUS_MAP.get(entry["status"], "UNAVAILABLE")


class _ReplayMarketState:
    def __init__(self, spread):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        # Research replay spread is a fixed research-only placeholder (see
        # research/phase5_ev_dataset.py's REPRESENTATIVE_SPREAD); pair it with
        # a plausible fixed vol/mid so decision/ev_cost.py's real cost formula
        # (FIX 3) can run in this OOS replay too, instead of unconditionally
        # returning None for every replayed event.
        self.realized_vol_60s = 0.0006
        self.mid = 2350.0


def replay_and_validate(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    split_info = run_barrier_split_candidate(max_holding, rows=rows, registry_dir=registry_dir)
    # Task 2's barrier_split classifier returns log-loss metrics, not probabilities.
    # The return contract does not expose per-event or aggregate P(sl|not_win) values.
    # Use 0.5 (least-informative prior) as the fixed stop-loss probability for this replay.
    # This is a known limitation: Task 2's OOF probabilities are not currently available for
    # per-event replay use. A future iteration could expose these for finer-grained analysis.
    p_sl_given_not_win = 0.5

    # FIX 10 (I7+I8): read each specialist's REAL committed registry status for
    # this horizon instead of hardcoding "VALIDATED" for every specialist at
    # every horizon. This both exercises evaluate()'s status-gating logic (it
    # never ran before) and makes `uncertainty` genuinely non-zero when a
    # specialist is CANDIDATE/rejected, so DEFAULT_K is no longer a no-op.
    direction_status = _real_model_status(f"direction_v3_candidate_h{max_holding}")
    opportunity_status = _real_model_status(f"opportunity_v3_candidate_h{max_holding}")
    barrier_status = _real_model_status(f"barrier_v3_candidate_h{max_holding}")
    mae_status = _real_model_status(f"mae_quantile_v3_candidate_h{max_holding}")
    mfe_status = _real_model_status(f"mfe_quantile_v3_candidate_h{max_holding}")

    decisions = {"NO_TRADE": 0, "LONG_CANDIDATE": 0, "SHORT_CANDIDATE": 0}
    expected_rs, realized_rs = [], []
    fragile_count = 0
    baseline_trades = 0
    baseline_realized = []

    n = data["n"]
    for i in range(n):
        ms = _ReplayMarketState(spread=float(data["spread"][i]))
        p_long = float(data["p_direction"][i])
        direction = DirectionOutput(model_id=f"direction_v3_candidate_h{max_holding}", horizon=max_holding,
                                     model_status=direction_status, probability_long=p_long,
                                     probability_short=1 - p_long, calibrated=True)
        opportunity = OpportunityOutput(model_id=f"opportunity_v3_candidate_h{max_holding}", horizon=max_holding,
                                         model_status=opportunity_status, probability_take=float(data["p_opportunity"][i]),
                                         calibrated=True)
        barrier = BarrierOutput(model_id=f"barrier_v3_candidate_h{max_holding}", horizon=max_holding,
                                 model_status=barrier_status, p_tp=float(data["p_barrier_win"][i]), calibrated=True)
        mae = MAEOutput(model_id=f"mae_quantile_v3_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3)
        mfe = MFEOutput(model_id=f"mfe_quantile_v3_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3)

        d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        decisions[d.decision] += 1
        if d.decision != "NO_TRADE":
            expected_rs.append(d.ev_adj)
            realized_rs.append(float(data["realized_r"][i]))
            perturbed = evaluate(_ReplayMarketState(spread=float(data["spread"][i]) * 1.5), direction, opportunity,
                                  barrier, p_sl_given_not_win, mae, mfe, timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                                  timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
            if (perturbed.ev_adj > 0) != (d.ev_adj > 0):
                fragile_count += 1

        if p_long > SIMPLE_BASELINE_THRESHOLD:
            baseline_trades += 1
            baseline_realized.append(float(data["realized_r"][i]))

    n_traded = len(expected_rs)
    return {
        "n_events": n, "decisions": decisions,
        "expected_vs_realized_r": {
            "mean_expected": float(np.mean(expected_rs)) if n_traded else None,
            "mean_realized": float(np.mean(realized_rs)) if n_traded else None,
            "n_traded": n_traded,
        },
        "baseline_comparison": {
            "simple_gate_n_trades": baseline_trades,
            "simple_gate_mean_realized_r": float(np.mean(baseline_realized)) if baseline_trades else None,
            "ev_engine_n_trades": n_traded,
            "ev_engine_mean_realized_r": float(np.mean(realized_rs)) if n_traded else None,
        },
        "fragile_fraction": (fragile_count / n_traded) if n_traded else 0.0,
    }


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS

    # Create a temporary directory for barrier_split registry output (research runs must not touch real registry)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run all 3 horizons sequentially with temp registry directory
        for h in HORIZONS:
            r = replay_and_validate(h, registry_dir=tmpdir)
            print(f"h={h}: {r}")
