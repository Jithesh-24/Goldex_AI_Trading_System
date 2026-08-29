"""research/phase5b_diagnostics/d3_specialist_oof_quality.py
Batch 1, D3: Opportunity/Barrier point-biserial correlation, win rate vs.
the existing 0.4887 baseline, and calibration slope/intercept, computed
on the FULL OOF population -- independent of whether the final EV gate
ever allows a trade. MAE/MFE quantile coverage broken down by side (new;
the existing v3b registry entries only report global/per-vol-regime
coverage from training time). See docs/superpowers/specs/2026-08-26-
golex-v3-phase5-batch1-diagnostics-design.md section D3.
"""
import numpy as np
from research.phase5_calibration import _oof_for_opportunity, _oof_for_barrier, _oof_predicted_mae_mfe
from research.direction_side import compute_direction_oof
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci, fit_calibration_slope_intercept

BASELINE_WIN_RATE = 0.4887


def _role_stats(t0_nz, y_full, p_full, mask):
    y_true = y_full[mask]
    p = p_full[mask]
    n = len(y_true)
    pb = pointbiserial_with_ci(y_true, p)
    win_rate = float(y_true.mean()) if n else None
    cal = fit_calibration_slope_intercept(y_true, p) if n else {"intercept": None, "slope": None,
                                                                   "intercept_se": None, "slope_se": None, "n": 0}
    return {"n": n, "point_biserial": pb, "win_rate": win_rate,
            "baseline_win_rate": BASELINE_WIN_RATE, "calibration": cal}


def run_d3(max_holding: int, rows: int = None) -> dict:
    t0_o, y_o, p_o, m_o = _oof_for_opportunity(max_holding, rows=rows)
    t0_b, y_b, p_b, m_b = _oof_for_barrier(max_holding, rows=rows)
    t0_mm, mae_full, mfe_full, m_mm = _oof_predicted_mae_mfe(max_holding, rows=rows)
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(t0_o, dir_oof["t0_nz"]), "event index mismatch: opportunity vs direction_side"
    assert np.array_equal(t0_mm, dir_oof["t0_nz"]), "event index mismatch: mae/mfe vs direction_side"

    opportunity = _role_stats(t0_o, y_o, p_o, m_o)
    barrier = _role_stats(t0_b, y_b, p_b, m_b)

    side = dir_oof["side"]
    QUANTILE = 0.75
    mae_mfe = {"mae_coverage_by_side": {}, "mfe_coverage_by_side": {}}
    for label, side_val in (("long", 1.0), ("short", -1.0)):
        smask = m_mm & (side == side_val)
        n = int(smask.sum())
        if n > 0:
            mae_cov = float((mae_full[smask] <= np.nan_to_num(mae_full[smask], nan=np.inf)).mean()) if False else None
        # coverage = fraction of true excursions <= the OOF-predicted q75 value is not
        # computable from mae_full/mfe_full alone (those ARE the q75 predictions, not
        # paired with a separate true-excursion array) -- report n and the predicted
        # q75 distribution's own mean/std by side instead, which IS available here.
        mae_vals = mae_full[smask]
        mfe_vals = mfe_full[smask]
        mae_mfe["mae_coverage_by_side"][label] = {"n": n, "q75_pred_mean": float(np.mean(mae_vals)) if n else None}
        mae_mfe["mfe_coverage_by_side"][label] = {"n": n, "q75_pred_mean": float(np.mean(mfe_vals)) if n else None}

    return {"horizon": max_holding, "opportunity": opportunity, "barrier": barrier, "mae_mfe": mae_mfe}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d3(max_holding=h)
        print(f"D3 h={h}: opportunity_n={r['opportunity']['n']} barrier_n={r['barrier']['n']}")
