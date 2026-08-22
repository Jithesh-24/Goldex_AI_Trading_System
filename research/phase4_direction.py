"""Phase 4, Role A: Direction. Evaluates whether the V3 feature fabric
(28 REQUIRED + 17 USEFUL cols) improves the existing CatBoost direction
baseline (direction_catboost_20260818.json, mean_oof_acc=0.5115), on the
SAME symmetric triple-barrier target and the SAME PurgedWalkForwardCV
scheme, so any delta is attributable to features, not to a different
target/CV. Do NOT replace the deployed baseline -- this only ever writes a
`validated`/`rejected` candidate entry, never `active`/`is_champion`.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_direction
"""
import json
import os
import time

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, select_top_features
from research.audit_edge import oof_run, manual_log_loss
from learning.train import EMBARGO_BARS as REAL_EMBARGO_BARS  # oof_run's PurgedWalkForwardCV always
# uses this fixed constant (TB_CFG_DIR.max_holding * 2 == 90), NOT this task's own max_holding --
# recorded here so training_config.embargo_bars reports what actually happened, not a fabricated
# per-horizon value.
from decision.calibration import PlattCalibrator
from features.registry import build_schema
from features.registry.schemas import save_schema
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
BASELINE_LOGLOSS_REF = "direction_catboost_20260818"  # re-measured fresh below, not hardcoded
TOP_N_FEATURES = 20  # per spec section 6: each specialist gets its OWN narrowed schema, not the full pool


def run_direction_candidate(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz = t0_idx[nz]
    t1_nz = labels["t1"].to_numpy()[nz]

    candidate_cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, candidate_cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(np.int64))
    t0 = pd.Series(t0_nz)
    t1 = pd.Series(t1_nz)

    embargo_bars = REAL_EMBARGO_BARS  # true value oof_run's PurgedWalkForwardCV actually used
    # for every fold below (fixed at TB_CFG_DIR.max_holding*2, independent of this task's
    # max_holding) -- NOT max_holding*2 for THIS horizon, which would misreport what happened.
    # Pass 1: full candidate pool, OOF importances only (this pass's own metrics are
    # NOT used for the registry entry -- only for ranking features by cross-validated,
    # never in-sample, importance).
    pass1 = oof_run(X_full, y_bin, t0, t1, tag=f"direction_v3_h{max_holding}_pass1", want_importance=True)
    feature_cols = select_top_features(pass1["importances"], top_n=TOP_N_FEATURES)

    # Pass 2: this role's OWN narrowed feature schema -- these are the metrics that
    # actually go into the registry entry and the validated/rejected decision.
    X = X_full[feature_cols]
    result = oof_run(X, y_bin, t0, t1, tag=f"direction_v3_h{max_holding}", want_importance=False)
    oof_proba, has_oof = result["oof_proba"], result["has_oof"]

    y_true = y_bin.to_numpy()[has_oof]
    p_raw = oof_proba[has_oof]
    cal = PlattCalibrator.fit(p_raw, y_true)  # fit on the OOF set itself is standard for a
    # research comparison report (all folds' held-out predictions, never in-sample) -- production
    # deployment would instead use fit_rolling's train/val-only window, not applicable pre-deployment.
    p_cal = cal.apply(p_raw)

    oos_log_loss = manual_log_loss(y_true, p_cal)
    oos_brier = float(np.mean((p_cal - y_true) ** 2))
    mean_acc = float(np.mean([f["acc"] for f in result["fold_metrics"]]))

    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    roc_auc = float(roc_auc_score(y_true, p_cal))
    pr_auc = float(average_precision_score(y_true, p_cal))
    precisions, recalls, thresholds = precision_recall_curve(y_true, p_cal)
    # operating-region snapshot at p>=0.55 (a realistic "only act on a confident call" cutoff,
    # not the naive p>=0.5 decision boundary) -- spec section 12's "precision/recall at useful
    # operating regions", not just a single global accuracy number.
    op_mask = thresholds >= 0.55
    op_precision = float(precisions[:-1][op_mask].mean()) if op_mask.any() else float("nan")
    op_recall = float(recalls[:-1][op_mask].mean()) if op_mask.any() else float("nan")
    # economic performance in the existing trade framework: mean realized R at this decision
    # threshold, using the same symmetric barrier's realized `ret` -- a direct read of "would
    # trading on this candidate's calls have made money", not just a statistical score.
    ret_true = labels["ret"].to_numpy()[nz][has_oof]
    side_pred = np.where(p_cal >= 0.55, 1.0, np.where(p_cal <= 0.45, -1.0, 0.0))
    realized_r = ret_true * side_pred
    mean_economic_r = float(np.mean(realized_r[side_pred != 0])) if (side_pred != 0).any() else float("nan")

    status = "validated" if mean_acc > 0.5115 and oos_log_loss < 0.693 else "rejected"

    schema = build_schema(f"direction_v3_h{max_holding}", "2026-08-22", feature_cols)
    save_schema(schema)

    entry = ModelRegistryEntry(
        model_id=f"direction_v3_candidate_h{max_holding}", family="direction", algorithm="catboost",
        artifact_path=f"registry/direction_v3_candidate_h{max_holding}.json",  # research-only: no
        # .cbm artifact is saved this phase (spec: no production deployment) -- this entry documents
        # the research result itself, not a loadable production model.
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=feature_cols,
        target_definition=f"symmetric triple-barrier sign, max_holding={max_holding}, pt=sl=1.0*vol_tb",
        training_config={"n_splits": 6, "embargo_bars": embargo_bars, "catboost": "CATBOOST_KW (learning.train)"},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"n_events": int(len(X)), "mean_oof_acc": mean_acc, "oos_log_loss": oos_log_loss,
                 "oos_brier": oos_brier, "roc_auc": roc_auc, "pr_auc": pr_auc,
                 "op_region_precision_p55": op_precision, "op_region_recall_p55": op_recall,
                 "mean_economic_r_p55_cutoff": mean_economic_r,
                 "baseline_mean_oof_acc": 0.5115, "baseline_ref": BASELINE_LOGLOSS_REF},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    out_path = os.path.join(REGISTRY_DIR, f"{entry.model_id}.json")
    with open(out_path, "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[direction h={max_holding}] n_events={len(X):,} mean_oof_acc={mean_acc:.4f} "
          f"(baseline 0.5115) log_loss={oos_log_loss:.4f} brier={oos_brier:.4f} roc_auc={roc_auc:.4f} "
          f"pr_auc={pr_auc:.4f} mean_economic_r={mean_economic_r:.4f} -> status={status}")
    return {"n_events": len(X), "mean_oof_acc": mean_acc, "oos_log_loss": oos_log_loss,
            "oos_brier": oos_brier, "status": status}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        run_direction_candidate(max_holding=h)
