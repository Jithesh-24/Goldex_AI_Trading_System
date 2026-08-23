"""research/phase5_calibration.py
Fits and persists per-role-per-horizon Platt calibrators. Uses
decision.calibration.PlattCalibrator.fit (Newton's method logistic fit,
identical to the one production's rolling calibrator uses) on OOF
probability/outcome pairs.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_calibration
"""
import json
import os
import time

from decision.calibration import PlattCalibrator
from decision.calibration_registry import CALIBRATION_DIR


def fit_and_save_calibrator(role: str, max_holding: int, y_true, p_raw, calibration_dir: str = None) -> str:
    if calibration_dir is None:
        calibration_dir = CALIBRATION_DIR
    cal = PlattCalibrator.fit(p_raw, y_true)
    cal.fit_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(calibration_dir, exist_ok=True)
    path = os.path.join(calibration_dir, f"{role}_h{max_holding}_platt.json")
    with open(path, "w") as f:
        json.dump({"a": cal.a, "b": cal.b, "n_samples": cal.n_samples,
                    "window_start": cal.window_start, "window_end": cal.window_end,
                    "fit_at_utc": cal.fit_at_utc}, f, indent=2)
    return path


def _oof_for_direction(max_holding, rows=None):
    import pandas as pd
    from research.phase4_dataset import assemble_v3_dataset
    from research.audit_edge import oof_run
    from features.labeling import TripleBarrierConfig, triple_barrier_labels
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]
    cols = ds["baseline_cols"] + ds["useful_cols"]
    X = feat_v3.loc[t0_nz, cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(int))
    t0, t1 = pd.Series(t0_nz), pd.Series(t1_nz)
    result = oof_run(X, y_bin, t0, t1, tag=f"calib_direction_h{max_holding}", want_importance=False)
    m = result["has_oof"]
    return y_bin.to_numpy()[m], result["oof_proba"][m]


def _oof_for_opportunity(max_holding, rows=None):
    import numpy as np
    import pandas as pd
    from research.phase4_dataset import assemble_v3_dataset
    from research.audit_edge import oof_run, build_meta
    from features.labeling import TripleBarrierConfig, triple_barrier_labels
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]
    cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(int))
    t0, t1 = pd.Series(t0_nz), pd.Series(t1_nz)
    prim = oof_run(X_full, y_bin, t0, t1, tag=f"calib_opportunity_h{max_holding}_prim", want_importance=False)
    has_oof = prim["has_oof"]
    if not has_oof.any():
        return np.array([], dtype=int), np.array([], dtype=float)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], has_oof)
    X_meta = X_full.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = meta_labels["label"].to_numpy()
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())
    result = oof_run(X_meta, pd.Series(y_meta), t0_meta, t1_meta, tag=f"calib_opportunity_h{max_holding}", want_importance=False)
    m = result["has_oof"]
    return y_meta[m], result["oof_proba"][m]


def _oof_for_barrier(max_holding, rows=None):
    # Same target/pipeline as Opportunity (both use build_meta's binary label) --
    # kept as a separate function since Barrier's own registry entry and role
    # are distinct per spec Task 9's rationale (calibration/log-loss framing
    # vs win-rate framing), even though the underlying OOF pipeline is identical.
    return _oof_for_opportunity(max_holding, rows=rows)


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        y_true, p_raw = _oof_for_direction(h)
        print(f"direction h={h}: n={len(y_true)} -> {fit_and_save_calibrator('direction', h, y_true, p_raw)}")
        y_true, p_raw = _oof_for_opportunity(h)
        print(f"opportunity h={h}: n={len(y_true)} -> {fit_and_save_calibrator('opportunity', h, y_true, p_raw)}")
        y_true, p_raw = _oof_for_barrier(h)
        print(f"barrier h={h}: n={len(y_true)} -> {fit_and_save_calibrator('barrier', h, y_true, p_raw)}")
