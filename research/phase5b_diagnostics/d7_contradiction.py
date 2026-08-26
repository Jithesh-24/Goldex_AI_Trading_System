"""research/phase5b_diagnostics/d7_contradiction.py
Batch 2, D7: is the 35.8% Barrier-vs-MAE/MFE reward/risk contradiction
(D4's exact definition, inherited unchanged) actually predictive of poor
realized outcomes, and which of Barrier/MAE-MFE is more reliable in the
contradicted population specifically? Does not assume either is correct.
See docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch2-ev-
uncertainty-design.md section D7.

NOTE: "exclusion_effect" is a descriptive report only and explicitly NOT
a proposed live filter. The presence of contradictions may correlate with
worse outcomes, but this analysis does not prescribe whether contradictions
should be filtered in live trading. That decision requires separate
consideration of practical costs and live profitability tradeoffs.
"""
import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset, realized_r_for_direction
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci
from research.audit_edge import block_bootstrap


def _touch_dist(touch: np.ndarray, side: np.ndarray) -> dict:
    n = len(touch)
    if n == 0:
        return {"tp_frac": None, "sl_frac": None, "timeout_frac": None, "n": 0}
    favorable = np.where(side == 1.0, 1, -1)
    tp = (touch == favorable).mean()
    sl = (touch == -favorable).mean()
    timeout = (touch == 0).mean()
    return {"tp_frac": float(tp), "sl_frac": float(sl), "timeout_frac": float(timeout), "n": n}


def run_d7(max_holding: int, rows: int = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    n = data["n"]
    p_barrier_win = data["p_barrier_win"]
    mae_r = data["mae_r"]
    mfe_r = data["mfe_r"]
    side = data["side"]
    touch = data["touch"]

    contradiction = (p_barrier_win >= 0.6) & (mfe_r <= mae_r)  # identical to D4's definition, do not change

    realized_r = np.array([realized_r_for_direction("long" if side[i] == 1.0 else "short", i, data)
                            for i in range(n)])

    def _pop_stats(mask):
        vals = realized_r[mask]
        m = len(vals)
        if m == 0:
            return {"mean": None, "n": 0, "bootstrap_ci": [None, None]}
        lo, mid, hi = block_bootstrap(vals, block_size=20, n_boot=1000)
        return {"mean": float(np.mean(vals)), "n": m, "bootstrap_ci": [lo, hi]}

    contradicted_stats = _pop_stats(contradiction)
    non_contradicted_stats = _pop_stats(~contradiction)
    diff_vals = realized_r[contradiction].mean() - realized_r[~contradiction].mean() if contradiction.any() and (~contradiction).any() else None
    # bootstrap CI on the difference: resample both populations' block-bootstrap means jointly
    if contradiction.sum() > 20 and (~contradiction).sum() > 20:
        rng = np.random.default_rng(42)
        diffs = []
        c_vals, nc_vals = realized_r[contradiction], realized_r[~contradiction]
        for _ in range(1000):
            c_sample = rng.choice(c_vals, size=len(c_vals), replace=True)
            nc_sample = rng.choice(nc_vals, size=len(nc_vals), replace=True)
            diffs.append(c_sample.mean() - nc_sample.mean())
        diff_ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    else:
        diff_ci = [None, None]

    realized_r_positive = (realized_r > 0).astype(float)
    predictiveness = pointbiserial_with_ci(realized_r_positive, contradiction.astype(float))

    # volatility tercile via vol_60s_proxy, matching this repo's existing tercile convention
    vol = data["vol_60s_proxy"]
    lo_thr, hi_thr = np.nanpercentile(vol, [33.3, 66.7])
    vol_state = np.where(vol <= lo_thr, "low", np.where(vol >= hi_thr, "high", "medium"))
    breakdown_vol = {}
    for label in ("low", "medium", "high"):
        m = vol_state == label
        breakdown_vol[label] = {"contradiction_rate": float(contradiction[m].mean()) if m.sum() else None, "n": int(m.sum())}

    breakdown_side = {}
    for label, side_val in (("long", 1.0), ("short", -1.0)):
        m = side == side_val
        breakdown_side[label] = {"contradiction_rate": float(contradiction[m].mean()) if m.sum() else None, "n": int(m.sum())}

    deciles = np.digitize(p_barrier_win, np.linspace(0, 1, 11)[1:-1])
    breakdown_decile = []
    for d in range(10):
        m = deciles == d
        breakdown_decile.append({"decile": d, "contradiction_rate": float(contradiction[m].mean()) if m.sum() else None, "n": int(m.sum())})

    excl_with = float(np.mean(realized_r))
    excl_without = float(np.mean(realized_r[~contradiction])) if (~contradiction).any() else None

    barrier_reliability = pointbiserial_with_ci(realized_r_positive[contradiction], p_barrier_win[contradiction])
    reward_risk_ratio = np.where(mae_r[contradiction] > 1e-9, mfe_r[contradiction] / mae_r[contradiction], np.nan)
    valid = np.isfinite(reward_risk_ratio)
    mae_mfe_reliability = pointbiserial_with_ci(realized_r_positive[contradiction][valid], reward_risk_ratio[valid])

    return {
        "horizon": max_holding,
        "contradiction_mask_n": int(contradiction.sum()),
        "non_contradiction_mask_n": int((~contradiction).sum()),
        "realized_r": {"contradicted": contradicted_stats, "non_contradicted": non_contradicted_stats,
                        "difference_ci": diff_ci, "difference_point": diff_vals},
        "touch_distribution": {"contradicted": _touch_dist(touch[contradiction], side[contradiction]),
                                "non_contradicted": _touch_dist(touch[~contradiction], side[~contradiction])},
        "predictiveness": {"point_biserial_contradiction_vs_realized_r_sign": predictiveness},
        "breakdown_by_volatility_tercile": breakdown_vol,
        "breakdown_by_side": breakdown_side,
        "breakdown_by_barrier_probability_decile": breakdown_decile,
        "exclusion_effect": {"realized_r_with_contradictions": excl_with,
                              "realized_r_excluding_contradictions": excl_without,
                              "n_with": n, "n_excluding": int((~contradiction).sum())},
        "which_component_more_reliable": {
            "barrier_point_biserial_in_contradicted_population": barrier_reliability,
            "mae_mfe_reward_risk_ratio_correlation_with_outcome_in_contradicted_population": mae_mfe_reliability,
        },
    }


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d7(max_holding=h)
        print(f"D7 h={h}: contradiction_n={r['contradiction_mask_n']} "
              f"realized_r_contradicted={r['realized_r']['contradicted']['mean']} "
              f"realized_r_non={r['realized_r']['non_contradicted']['mean']}")
