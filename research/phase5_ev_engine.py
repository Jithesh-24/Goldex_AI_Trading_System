"""research/phase5_ev_engine.py
Spec sections 13/12/22/23: research-only EV replay simulator. Calls the
SAME decision/ev_engine.py.evaluate() pure function the live path uses
(spec section 14's live/replay equivalence requirement), against
historical specialist-output replay data. Computes OOS decision
distribution, expected-vs-realized R, a baseline comparison (simple
P(direction)>0.55 gate, no cost/no EV), and a sensitivity/fragility scan
(spec section 20/12). NO Telegram, no live I/O.

FIX (targeted correction pass, 2026-08-24): realized R for each traded
event is now looked up via research.phase5_ev_dataset.realized_r_for_direction
using the ENGINE'S OWN decided direction, not a single direction-agnostic
stream. The baseline gate always trades long (p_long > threshold), so it
always uses the "long" realized-R stream.

FIX (targeted correction pass, 2026-08-24): `_ReplayMarketState.mid` and
`.realized_vol_60s` are now real per-event historical values (close price
at entry; un-scaled EWMA vol) from research/phase5_ev_dataset.py, not the
two hardcoded constants (2350.0 / 0.0006) used previously.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_engine
"""
from datetime import datetime, timezone
import json
import os
import tempfile

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset, realized_r_for_direction
from research.phase5_timeout_payoff import estimate_timeout_payoff
from research.phase5_barrier_split import run_barrier_split_candidate
from decision.ev_engine import evaluate
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput

SIMPLE_BASELINE_THRESHOLD = 0.55

_REAL_REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "registry")

_REGISTRY_STATUS_MAP = {"validated": "VALIDATED", "candidate": "CANDIDATE", "rejected": "UNAVAILABLE",
                         "active": "VALIDATED", "archived": "UNAVAILABLE"}


def _real_model_status(model_id: str, registry_dir: str = None) -> str:
    d = registry_dir if registry_dir else _REAL_REGISTRY_DIR
    path = os.path.join(d, f"{model_id}.json")
    with open(path) as f:
        entry = json.load(f)
    return _REGISTRY_STATUS_MAP.get(entry["status"], "UNAVAILABLE")


class _ReplayMarketState:
    def __init__(self, spread, mid, vol_60s):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = vol_60s
        self.mid = mid


def replay_and_validate(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    split_info = run_barrier_split_candidate(max_holding, rows=rows, registry_dir=registry_dir)
    p_sl_given_not_win = 0.5

    direction_status = _real_model_status(f"direction_v3_candidate_h{max_holding}", registry_dir=registry_dir)
    opportunity_status = _real_model_status(f"opportunity_v3b_candidate_h{max_holding}", registry_dir=registry_dir)
    barrier_status = _real_model_status(f"barrier_v3b_candidate_h{max_holding}", registry_dir=registry_dir)
    mae_status = _real_model_status(f"mae_quantile_v3b_candidate_h{max_holding}", registry_dir=registry_dir)
    mfe_status = _real_model_status(f"mfe_quantile_v3b_candidate_h{max_holding}", registry_dir=registry_dir)

    decisions = {"NO_TRADE": 0, "LONG_CANDIDATE": 0, "SHORT_CANDIDATE": 0}
    expected_rs, realized_rs = [], []
    fragile_count = 0
    baseline_trades = 0
    baseline_realized = []

    n = data["n"]
    for i in range(n):
        mid_i = float(data["mid"][i])
        vol_i = float(data["vol_60s_proxy"][i])
        ms = _ReplayMarketState(spread=float(data["spread"][i]), mid=mid_i, vol_60s=vol_i)
        p_long = float(data["p_direction"][i])
        direction = DirectionOutput(model_id=f"direction_v3_candidate_h{max_holding}", horizon=max_holding,
                                     model_status=direction_status, probability_long=p_long,
                                     probability_short=1 - p_long, calibrated=True)
        opportunity = OpportunityOutput(model_id=f"opportunity_v3b_candidate_h{max_holding}", horizon=max_holding,
                                         model_status=opportunity_status, probability_take=float(data["p_opportunity"][i]),
                                         calibrated=True, assumed_side=float(data["side"][i]),
                                         direction_model_id=data["direction_model_id"])
        barrier = BarrierOutput(model_id=f"barrier_v3b_candidate_h{max_holding}", horizon=max_holding,
                                 model_status=barrier_status, p_tp=float(data["p_barrier_win"][i]), calibrated=True,
                                 assumed_side=float(data["side"][i]), direction_model_id=data["direction_model_id"])
        mae = MAEOutput(model_id=f"mae_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3,
                         assumed_side=float(data["side"][i]), direction_model_id=data["direction_model_id"])
        mfe = MFEOutput(model_id=f"mfe_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3,
                         assumed_side=float(data["side"][i]), direction_model_id=data["direction_model_id"])

        d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        decisions[d.decision] += 1
        if d.decision != "NO_TRADE":
            expected_rs.append(d.ev_adj)
            realized_rs.append(realized_r_for_direction(d.direction, i, data))
            perturbed = evaluate(_ReplayMarketState(spread=float(data["spread"][i]) * 1.5, mid=mid_i, vol_60s=vol_i),
                                  direction, opportunity, barrier, p_sl_given_not_win, mae, mfe,
                                  timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                                  timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
            if (perturbed.ev_adj > 0) != (d.ev_adj > 0):
                fragile_count += 1

        if p_long > SIMPLE_BASELINE_THRESHOLD:
            baseline_trades += 1
            baseline_realized.append(realized_r_for_direction("long", i, data))

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

    with tempfile.TemporaryDirectory() as tmpdir:
        for h in HORIZONS:
            r = replay_and_validate(h, registry_dir=tmpdir)
            print(f"h={h}: {r}")
