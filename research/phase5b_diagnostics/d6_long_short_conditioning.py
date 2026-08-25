"""research/phase5b_diagnostics/d6_long_short_conditioning.py
Batch 1, D6: does side-conditioning actually produce side-dependent
discriminative behavior, per specialist? Reuses D1's Direction-by-side
split directly; adds the equivalent by-side split for Opportunity/Barrier
(D3 reports them pooled across sides). See docs/superpowers/specs/
2026-08-26-golex-v3-phase5-batch1-diagnostics-design.md section D6.
"""
import numpy as np
from research.phase5b_diagnostics.d1_direction_quality import run_d1
from research.phase5_calibration import _oof_for_opportunity, _oof_for_barrier
from research.direction_side import compute_direction_oof
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci


def _by_side_pointbiserial(t0_nz, y_full, p_full, mask, side):
    y_true = y_full[mask]
    p = p_full[mask]
    side_masked = side[mask]
    long_m = side_masked == 1.0
    short_m = side_masked == -1.0
    return {"long": pointbiserial_with_ci(y_true[long_m], p[long_m]),
            "short": pointbiserial_with_ci(y_true[short_m], p[short_m])}


def run_d6(max_holding: int, rows: int = None) -> dict:
    d1 = run_d1(max_holding=max_holding, rows=rows)
    direction_by_side = {
        "long": d1["side_conditioned"]["long"]["point_biserial"],
        "short": d1["side_conditioned"]["short"]["point_biserial"],
    }

    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    side = dir_oof["side"]

    t0_o, y_o, p_o, m_o = _oof_for_opportunity(max_holding, rows=rows)
    t0_b, y_b, p_b, m_b = _oof_for_barrier(max_holding, rows=rows)
    assert np.array_equal(t0_o, dir_oof["t0_nz"]), "event index mismatch: opportunity vs direction_side"

    opportunity_by_side = _by_side_pointbiserial(t0_o, y_o, p_o, m_o, side)
    barrier_by_side = _by_side_pointbiserial(t0_b, y_b, p_b, m_b, side)

    return {"horizon": max_holding, "direction": direction_by_side,
            "opportunity": opportunity_by_side, "barrier": barrier_by_side}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d6(max_holding=h)
        print(f"D6 h={h}: direction={r['direction']} opportunity={r['opportunity']} barrier={r['barrier']}")
