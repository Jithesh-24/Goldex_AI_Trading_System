"""research/phase5_barrier_split.py
Phase 5: Barrier role only produced a binary P(win) (TP-before-SL) target
in Phase 4 -- research/phase4_barrier.py's build_meta() call already
computes a `touch` column (-1/0/1: which raw barrier was actually hit,
before collapsing to the binary `label`) but discards it. This script
reuses that same touch column to train P(sl | not-win): restricted to the
not-win (label=0) subset, does the loss touch the unfavorable barrier
(sl_hit) rather than time out (timeout_hit)? Combined with the existing
Barrier role's calibrated p_win, this yields a coherent 3-way split:
p_tp = p_win, p_sl = (1-p_win)*P(sl|not_win), p_timeout = (1-p_win)*(1-P(sl|not_win)).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_barrier_split
"""
import os

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, select_top_features
from research.audit_edge import oof_run, build_meta, manual_log_loss
from decision.calibration import PlattCalibrator
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
TOP_N_FEATURES = 20


def run_barrier_split_candidate(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    if registry_dir is None:
        registry_dir = REGISTRY_DIR
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    dir_labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = dir_labels["label"].to_numpy()
    nz = y != 0
    t0_nz = t0_idx[nz]
    t1_nz = dir_labels["t1"].to_numpy()[nz]

    candidate_cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, candidate_cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(np.int64))
    t0 = pd.Series(t0_nz)
    t1 = pd.Series(t1_nz)

    prim = oof_run(X_full, y_bin, t0, t1, tag=f"barrier_split_v3_h{max_holding}_primary", want_importance=False)
    side_in = np.where(prim["oof_pred"] == 1, 1.0, -1.0)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, side_in, prim["has_oof"])
    has_oof = prim["has_oof"]
    if not has_oof.any():
        return {"n_events": 0, "status": "rejected"}

    win_label = meta_labels["label"].to_numpy()
    touch = meta_labels["touch"].to_numpy()
    favorable = np.where(side >= 0, 1, -1)
    sl_hit = (touch == -favorable).astype(np.int64)

    not_win_mask = win_label == 0
    if not_win_mask.sum() < 200:
        return {"n_events": int(not_win_mask.sum()), "status": "rejected"}

    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    X_nw = X_meta_full.loc[not_win_mask].reset_index(drop=True)
    y_nw = pd.Series(sl_hit[not_win_mask])
    t0_nw = pd.Series(meta_labels.index.to_numpy()[not_win_mask])
    t1_nw = pd.Series(meta_labels["t1"].to_numpy()[not_win_mask])

    pass1 = oof_run(X_nw, y_nw, t0_nw, t1_nw, tag=f"barrier_split_v3_h{max_holding}_pass1", want_importance=True)
    feature_cols = select_top_features(pass1["importances"], top_n=TOP_N_FEATURES)
    if "assumed_side" not in feature_cols:
        feature_cols.append("assumed_side")
    X_narrow = X_nw[feature_cols]
    result = oof_run(X_narrow, y_nw, t0_nw, t1_nw, tag=f"barrier_split_v3_h{max_holding}")
    has_oof2 = result["has_oof"]
    if not has_oof2.any():
        return {"n_events": int(len(y_nw)), "status": "rejected"}

    y_true = y_nw.to_numpy()[has_oof2]
    p_raw = result["oof_proba"][has_oof2]
    cal = PlattCalibrator.fit(p_raw, y_true)
    p_cal = cal.apply(p_raw)
    log_loss = manual_log_loss(y_true, p_cal)
    baseline_log_loss = -np.log(0.5)
    status = "validated" if log_loss < baseline_log_loss else "rejected"

    entry = ModelRegistryEntry(
        model_id=f"barrier_split_v3_candidate_h{max_holding}", family="barrier_probability",
        algorithm="catboost", artifact_path="none-oof-only",
        feature_cols=feature_cols,
        target_definition="P(sl_hit | not-win) restricted to Barrier role's not-win subset; "
                           "combined with barrier_v3_candidate's p_win to yield p_tp/p_sl/p_timeout.",
        training_config={"max_holding": max_holding, "top_n_features": TOP_N_FEATURES},
        training_period="full available history", validation_period="OOF walk-forward folds",
        created_at=pd.Timestamp.utcnow().to_pydatetime(), status=status,
        metrics={"n_events": int(len(y_nw)), "p_sl_given_not_win_log_loss": log_loss,
                 "baseline_log_loss": float(baseline_log_loss), "platt_a": cal.a, "platt_b": cal.b},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(registry_dir, exist_ok=True)
    out_path = os.path.join(registry_dir, f"{entry.model_id}.json")
    with open(out_path, "w") as f:
        f.write(entry.model_dump_json(indent=2))

    return {"n_events": int(len(y_nw)), "p_sl_given_not_win_log_loss": log_loss,
            "baseline_log_loss": float(baseline_log_loss), "status": status}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_barrier_split_candidate(h)
        print(f"h={h}: {r}")
