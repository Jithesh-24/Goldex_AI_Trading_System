"""research/phase5b_diagnostics/d4_cross_specialist_consistency.py
Batch 1, D4: mechanical, non-subjective contradiction rates between
Barrier/Opportunity/MAE/MFE, with every comparison keyed to the SAME
event/side/horizon/TP-SL-definition via a single assemble_replay_dataset
call -- the exact discipline Phase 5A's build_meta fix exists to enforce.
Production sets sl_r, tp_r = mae.q75, mfe.q75 directly (decision/ev_cost.py
::candidate_sl_tp) -- comparing mfe_r against itself would be circular, so
the reward-to-risk check compares Barrier's independently-fit p_barrier_win
against the independently-fit mae_r/mfe_r RATIO instead. See docs/
superpowers/specs/2026-08-26-golex-v3-phase5-batch1-diagnostics-design.md
section D4.
"""
import numpy as np
from research.phase5_ev_dataset import assemble_replay_dataset
from research.audit_edge import wilson_ci


def _rate_with_ci(mask: np.ndarray) -> dict:
    n = len(mask)
    k = int(mask.sum())
    rate = float(k / n) if n else None
    lo, hi = wilson_ci(k, n) if n else (None, None)
    return {"rate": rate, "k": k, "n": n, "ci_lo": lo, "ci_hi": hi}


def run_d4(max_holding: int, rows: int = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    n = data["n"]
    p_opportunity = data["p_opportunity"]
    p_barrier_win = data["p_barrier_win"]
    mae_r = data["mae_r"]
    mfe_r = data["mfe_r"]

    barrier_vs_reward_risk = (p_barrier_win >= 0.6) & (mfe_r <= mae_r)
    opportunity_vs_barrier = (p_opportunity >= 0.5) & (p_barrier_win < 0.5)

    return {"horizon": max_holding, "n": n,
            "contradiction_barrier_vs_reward_risk": _rate_with_ci(barrier_vs_reward_risk),
            "contradiction_opportunity_vs_barrier": _rate_with_ci(opportunity_vs_barrier)}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d4(max_holding=h)
        print(f"D4 h={h}: n={r['n']} "
              f"barrier_vs_reward_risk={r['contradiction_barrier_vs_reward_risk']['rate']:.4f} "
              f"opportunity_vs_barrier={r['contradiction_opportunity_vs_barrier']['rate']:.4f}")
