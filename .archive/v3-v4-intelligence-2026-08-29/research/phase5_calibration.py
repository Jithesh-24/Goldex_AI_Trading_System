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


# --- FIX 2 (C3, final-review fix wave) ---
# Every OOF-producing function below (_oof_for_direction, _oof_for_opportunity,
# _oof_for_barrier, and phase5_ev_dataset's _oof_predicted_mae_mfe) returns its
# result as a FULL-LENGTH array aligned to `t0_nz` -- the complete non-zero-
# labeled event index for this max_holding, built by calling
# assemble_v3_dataset(max_holding=h) + triple_barrier_labels(...) with the
# identical TripleBarrierConfig every one of these functions uses -- with NaN
# in positions where that function's own OOF wasn't available, plus a
# has_oof boolean mask marking which positions are real. This lets a caller
# (phase5_ev_dataset.py, phase5_uncertainty_k.py) combine several streams'
# masks (AND) and index every array by the SAME combined mask, instead of the
# old bug of pre-filtering each stream to its own shorter array and then
# joining them by raw position -- which silently paired unrelated events
# whenever the streams' OOF-availability subsets differed (they always do,
# since Direction's CV/embargo losses and Opportunity/Barrier's second-stage
# meta-labeling losses are independent).
def _oof_for_direction(max_holding, rows=None):
    import numpy as np
    from research.phase4_dataset import assemble_v3_dataset
    from research.direction_side import compute_direction_oof
    from features.labeling import TripleBarrierConfig, triple_barrier_labels

    # Get the shared Direction OOF
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)

    # Independently recompute the TRUE direction label to avoid changing compute_direction_oof's public contract
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, high, low, vol_tb, t0_idx = (ds["close"], ds["high"],
                                        ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz_computed = t0_idx[nz]

    # Assert that both functions' independently-computed t0_nz arrays match
    assert np.array_equal(t0_nz_computed, dir_oof["t0_nz"]), "direction_side event index mismatch"

    y_full = (y[nz] == 1).astype(float)
    return dir_oof["t0_nz"], y_full, dir_oof["p_direction_cal"], dir_oof["has_oof"]


def _oof_for_opportunity(max_holding, rows=None):
    import numpy as np
    import pandas as pd
    from research.phase4_dataset import assemble_v3_dataset
    from research.audit_edge import oof_run, build_meta
    from research.direction_side import compute_direction_oof
    from features.labeling import TripleBarrierConfig, triple_barrier_labels

    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]
    n = len(t0_nz)
    cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(int))
    t0, t1 = pd.Series(t0_nz), pd.Series(t1_nz)

    # Use shared Direction OOF side instead of fitting our own primary classifier
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(dir_oof["t0_nz"], t0_nz), "direction_side event index mismatch"
    has_oof1 = dir_oof["has_oof"]
    y_full = np.full(n, np.nan)
    p_full = np.full(n, np.nan)
    mask_full = np.zeros(n, dtype=bool)
    if not has_oof1.any():
        return t0_nz, y_full, p_full, mask_full
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, dir_oof["side"], has_oof1)
    X_meta = X_full.loc[has_oof1].reset_index(drop=True)
    X_meta["assumed_side"] = side
    X_meta["p_direction"] = dir_oof["p_direction_cal"][has_oof1]
    y_meta = meta_labels["label"].to_numpy()
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())
    result = oof_run(X_meta, pd.Series(y_meta), t0_meta, t1_meta, tag=f"calib_opportunity_h{max_holding}", want_importance=False)
    has_oof2 = result["has_oof"]
    idx_has1 = np.where(has_oof1)[0]
    y_full[idx_has1] = y_meta
    p_full[idx_has1[has_oof2]] = result["oof_proba"][has_oof2]
    mask_full[idx_has1[has_oof2]] = True
    return t0_nz, y_full, p_full, mask_full


def _oof_for_barrier(max_holding, rows=None):
    # Same target/pipeline as Opportunity (both use build_meta's binary label) --
    # kept as a separate function since Barrier's own registry entry and role
    # are distinct per spec Task 9's rationale (calibration/log-loss framing
    # vs win-rate framing), even though the underlying OOF pipeline is identical.
    return _oof_for_opportunity(max_holding, rows=rows)


def _oof_predicted_mae_mfe(max_holding, rows=None):
    """FIX 1 (C1+C2, final-review fix wave): real OOF-PREDICTED q75 MAE/MFE,
    not realized excursions -- mirrors research/phase4_mae_quantile.py's
    run_mae_quantile_candidate dataset assembly (mae_R/mfe_R via
    _mae_mfe_core, the assumed_side feature, the not-nz filtering) but keeps
    the per-event OOF-predicted array instead of only aggregate coverage.
    Simplification vs phase4_mae_quantile.py: uses the full baseline+useful
    candidate column pool directly rather than running Phase 4's extra
    feature-importance pass to narrow to a role-specific top-N schema --
    this function is a research-only replay input, not the persisted
    candidate model, so the narrower schema isn't required here.
    Returns (t0_nz, mae_q75_pred, mfe_q75_pred, has_oof_mask), all aligned
    to t0_nz per the FIX-2 alignment convention documented above.
    """
    import numpy as np
    import pandas as pd
    from research.phase4_dataset import assemble_v3_dataset
    from research.audit_edge import oof_run, build_meta, _mae_mfe_core
    from research.direction_side import compute_direction_oof
    from research.v3_quantile_models import fit_quantile
    from learning.cv import PurgedWalkForwardCV
    from features.labeling import TripleBarrierConfig, triple_barrier_labels

    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]
    n = len(t0_nz)

    cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(int))
    t0, t1 = pd.Series(t0_nz), pd.Series(t1_nz)

    mae_full = np.full(n, np.nan)
    mfe_full = np.full(n, np.nan)
    mask_full = np.zeros(n, dtype=bool)

    # Use shared Direction OOF side instead of fitting our own primary classifier
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(dir_oof["t0_nz"], t0_nz), "direction_side event index mismatch"
    has_oof1 = dir_oof["has_oof"]
    if not has_oof1.any():
        return t0_nz, mae_full, mfe_full, mask_full

    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, dir_oof["side"], has_oof1)
    X_meta = X_full.loc[has_oof1].reset_index(drop=True)
    X_meta["assumed_side"] = side
    X_meta["p_direction"] = dir_oof["p_direction_cal"][has_oof1]
    t0_meta = meta_labels.index.to_numpy()
    t1_meta = meta_labels["t1"].to_numpy()
    vol_at_meta = vol_tb[t0_nz][has_oof1]

    mae_R, mfe_R = _mae_mfe_core(close, high, low, t0_meta, t1_meta, side, vol_at_meta)

    cv = PurgedWalkForwardCV(n_splits=6, embargo_bars=max_holding * 2, min_train_bars=500)
    t0_s, t1_s = pd.Series(t0_meta), pd.Series(t1_meta)

    mae_oof = np.full(len(mae_R), np.nan)
    mfe_oof = np.full(len(mfe_R), np.nan)
    for train_pos, test_pos in cv.split(t0_s.to_numpy(), t1_s.to_numpy()):
        model_mae = fit_quantile(X_meta, mae_R, train_pos, 0.75)
        mae_oof[test_pos] = model_mae.predict(X_meta.iloc[test_pos])
        model_mfe = fit_quantile(X_meta, mfe_R, train_pos, 0.75)
        mfe_oof[test_pos] = model_mfe.predict(X_meta.iloc[test_pos])

    has_oof2 = np.isfinite(mae_oof) & np.isfinite(mfe_oof)
    idx_has1 = np.where(has_oof1)[0]
    mae_full[idx_has1[has_oof2]] = mae_oof[has_oof2]
    mfe_full[idx_has1[has_oof2]] = mfe_oof[has_oof2]
    mask_full[idx_has1[has_oof2]] = True

    return t0_nz, mae_full, mfe_full, mask_full


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        _, y_full, p_full, m = _oof_for_direction(h)
        y_true, p_raw = y_full[m], p_full[m]
        print(f"direction h={h}: n={len(y_true)} -> {fit_and_save_calibrator('direction', h, y_true, p_raw)}")
        _, y_full, p_full, m = _oof_for_opportunity(h)
        y_true, p_raw = y_full[m], p_full[m]
        print(f"opportunity h={h}: n={len(y_true)} -> {fit_and_save_calibrator('opportunity', h, y_true, p_raw)}")
        _, y_full, p_full, m = _oof_for_barrier(h)
        y_true, p_raw = y_full[m], p_full[m]
        print(f"barrier h={h}: n={len(y_true)} -> {fit_and_save_calibrator('barrier', h, y_true, p_raw)}")
