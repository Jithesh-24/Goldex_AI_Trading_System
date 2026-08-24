"""research/direction_side.py
Phase 5A: the single, shared source of "what side did Direction propose"
for a given event. Every downstream specialist (Opportunity, Barrier, MAE,
MFE) MUST condition on this function's output and must not compute its own
side (docs/superpowers/specs/2026-08-24-golex-v3-phase5a-specialist-
conditioning-design.md, sections 1/3/5). This is the exact pass1+pass2
OOF-fit Direction's own candidate training (research/phase4_direction.py)
already does -- extracted here so both Direction's own registry entry and
every downstream consumer compute the SAME side from the SAME model, never
two independently-fit copies.
"""
import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, select_top_features
from research.audit_edge import oof_run
from learning.train import EMBARGO_BARS as REAL_EMBARGO_BARS
from decision.calibration import PlattCalibrator
from features.labeling import TripleBarrierConfig, triple_barrier_labels

TOP_N_FEATURES = 20  # matches research/phase4_direction.py's own narrowing


def compute_direction_oof(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz = t0_idx[nz]
    t1_nz = labels["t1"].to_numpy()[nz]
    n = len(t0_nz)

    candidate_cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, candidate_cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(np.int64))
    t0, t1 = pd.Series(t0_nz), pd.Series(t1_nz)

    pass1 = oof_run(X_full, y_bin, t0, t1, tag=f"direction_side_h{max_holding}_pass1", want_importance=True)
    feature_cols = select_top_features(pass1["importances"], top_n=TOP_N_FEATURES)

    X = X_full[feature_cols]
    result = oof_run(X, y_bin, t0, t1, tag=f"direction_side_h{max_holding}", want_importance=False)
    has_oof = result["has_oof"]

    p_raw_full = np.full(n, np.nan)
    p_raw_full[has_oof] = result["oof_proba"][has_oof]

    p_cal_full = np.full(n, np.nan)
    if has_oof.any():
        y_true = y_bin.to_numpy()[has_oof]
        cal = PlattCalibrator.fit(p_raw_full[has_oof], y_true)
        p_cal_full[has_oof] = cal.apply(p_raw_full[has_oof])

    side = np.zeros(n, dtype=np.float64)
    side[has_oof] = np.where(p_raw_full[has_oof] >= 0.5, 1.0, -1.0)

    return {"t0_nz": t0_nz, "feature_cols": feature_cols,
            "p_direction_raw": p_raw_full, "p_direction_cal": p_cal_full,
            "side": side, "has_oof": has_oof,
            "model_id": f"direction_v3_candidate_h{max_holding}",
            "fold_metrics": result["fold_metrics"]}
