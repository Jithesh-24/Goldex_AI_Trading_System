"""research/phase5b_diagnostics/d1_direction_quality.py
Batch 1, D1: Direction's OOF probability distribution and point-biserial
correlation against the true directional label, overall and split by
side. Closes the point-biserial measurement flagged as deferred in the
Phase 5A retrain-and-replay report. Read-only: computes statistics from
research.direction_side.compute_direction_oof's existing output, fits
nothing new. See docs/superpowers/specs/2026-08-26-golex-v3-phase5-
batch1-diagnostics-design.md section D1.
"""
import numpy as np
from research.direction_side import compute_direction_oof
from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci


def _population_stats(y_true, p_cal):
    n = len(y_true)
    pb = pointbiserial_with_ci(y_true, p_cal)
    deciles = list(np.percentile(p_cal, np.arange(10, 100, 10)).astype(float)) if n else []
    return {"n": n, "point_biserial": pb,
            "p_direction_mean": float(np.mean(p_cal)) if n else None,
            "p_direction_std": float(np.std(p_cal)) if n else None,
            "p_direction_deciles": deciles}


def run_d1(max_holding: int, rows: int = None) -> dict:
    oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz = ds["t0_idx"][nz]
    assert np.array_equal(t0_nz, oof["t0_nz"]), "direction_side event index mismatch"
    y_true_full = (y[nz] == 1).astype(float)

    has_oof = oof["has_oof"]
    y_true = y_true_full[has_oof]
    p_cal = oof["p_direction_cal"][has_oof]
    side = oof["side"][has_oof]

    oos = _population_stats(y_true, p_cal)
    long_mask = side == 1.0
    short_mask = side == -1.0
    side_conditioned = {
        "long": _population_stats(y_true[long_mask], p_cal[long_mask]),
        "short": _population_stats(y_true[short_mask], p_cal[short_mask]),
    }
    return {"horizon": max_holding, "oos": oos, "side_conditioned": side_conditioned}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d1(max_holding=h)
        print(f"D1 h={h}: n={r['oos']['n']} point_biserial={r['oos']['point_biserial']}")
