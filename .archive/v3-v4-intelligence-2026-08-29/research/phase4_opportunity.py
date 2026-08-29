"""Phase 4, Role B: Opportunity/meta. Precision filter on this task's own
primary OOF side, using the V3 feature fabric, evaluated against the
existing opportunity_meta_catboost_20260818.json baseline
(meta_win_rate_baseline=0.4887). Meta-labeling by construction (de Prado):
the meta target is built from THIS run's own out-of-fold primary
predictions, never in-sample, so it cannot trivially overfit to its own
primary.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_opportunity
"""
import os

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, select_top_features
from research.audit_edge import oof_run, build_meta, manual_log_loss
from research.direction_side import compute_direction_oof
from learning.train import EMBARGO_BARS as REAL_EMBARGO_BARS  # oof_run's PurgedWalkForwardCV always
# uses this fixed constant (TB_CFG_DIR.max_holding * 2 == 90), NOT this task's own max_holding --
# recorded here so training_config.embargo_bars reports what actually happened, not a fabricated
# per-horizon value (same bug found and fixed in Task 4's phase4_direction.py).
from decision.calibration import PlattCalibrator
from features.registry import build_schema
from features.registry.schemas import save_schema, SCHEMAS_DIR
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
TOP_N_FEATURES = 20  # per spec section 6: this role's OWN narrowed schema, not the shared pool


def run_opportunity_candidate(max_holding: int, rows: int = None, registry_dir: str = None, schemas_dir: str = None) -> dict:
    if registry_dir is None:
        registry_dir = REGISTRY_DIR
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg_dir = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    dir_labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg_dir, side=None)
    y = dir_labels["label"].to_numpy()
    nz = y != 0
    t0_nz = t0_idx[nz]
    t1_nz = dir_labels["t1"].to_numpy()[nz]

    candidate_cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, candidate_cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(np.int64))
    t0 = pd.Series(t0_nz)
    t1 = pd.Series(t1_nz)

    embargo_bars = REAL_EMBARGO_BARS  # true value oof_run's PurgedWalkForwardCV actually used for
    # every fold below (fixed at TB_CFG_DIR.max_holding*2, independent of this task's max_holding).

    # Primary side-generator run: uses the full candidate pool -- it's an internal input to the
    # meta target (side), not itself a registered specialist, so it is not narrowed.
    prim = oof_run(X_full, y_bin, t0, t1, tag=f"opportunity_v3_h{max_holding}_primary", want_importance=False)
    side_in = np.where(prim["oof_pred"] == 1, 1.0, -1.0)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, side_in, prim["has_oof"])
    has_oof = prim["has_oof"]

    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    # Pass 1 (meta stage): full pool + assumed_side, OOF importances only.
    meta_pass1 = oof_run(X_meta_full, y_meta, t0_meta, t1_meta,
                          tag=f"opportunity_v3_h{max_holding}_meta_pass1", want_importance=True)
    feature_cols_meta = select_top_features(meta_pass1["importances"], top_n=TOP_N_FEATURES)
    if "assumed_side" not in feature_cols_meta:
        feature_cols_meta.append("assumed_side")  # always keep the side flag regardless of its importance rank

    # Pass 2: this role's OWN narrowed meta feature schema -- these metrics go into the registry entry.
    X_meta = X_meta_full[feature_cols_meta]
    meta_result = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag=f"opportunity_v3_h{max_holding}_meta")
    meta_has_oof = meta_result["has_oof"]
    y_true = y_meta.to_numpy()[meta_has_oof]
    p_raw = meta_result["oof_proba"][meta_has_oof]
    cal = PlattCalibrator.fit(p_raw, y_true)
    p_cal = cal.apply(p_raw)

    oos_log_loss = manual_log_loss(y_true, p_cal)
    win_rate = float(y_meta.mean())
    status = "validated" if win_rate > 0.4887 else "rejected"

    schema = build_schema(f"opportunity_v3_h{max_holding}", "2026-08-22", feature_cols_meta)
    save_schema(schema, schemas_dir=schemas_dir if schemas_dir else SCHEMAS_DIR)

    entry = ModelRegistryEntry(
        model_id=f"opportunity_v3_candidate_h{max_holding}", family="opportunity_meta", algorithm="catboost",
        artifact_path=f"registry/opportunity_v3_candidate_h{max_holding}.json",
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=feature_cols_meta,
        target_definition=f"meta-label: assumed-side TP before SL, max_holding={max_holding}, pt=1.5*vol_tb sl=1.0*vol_tb",
        training_config={"n_splits": 6, "embargo_bars": embargo_bars, "catboost": "CATBOOST_KW (learning.train)"},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"n_events": int(len(X_meta)), "meta_win_rate": win_rate, "oos_log_loss": oos_log_loss,
                 "baseline_meta_win_rate": 0.4887},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(registry_dir, exist_ok=True)
    with open(os.path.join(registry_dir, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[opportunity h={max_holding}] n_events={len(X_meta):,} win_rate={win_rate:.4f} "
          f"(baseline 0.4887) log_loss={oos_log_loss:.4f} -> status={status}")
    return {"n_events": len(X_meta), "oos_log_loss": oos_log_loss, "status": status}


def run_opportunity_candidate_v3b(max_holding: int, rows: int = None, registry_dir: str = None, schemas_dir: str = None) -> dict:
    """Phase 5A retrain: conditions on Direction's OOF side (research.direction_side)
    instead of this specialist's own primary-stage side-guess. See
    docs/superpowers/specs/2026-08-24-golex-v3-phase5a-specialist-conditioning-design.md."""
    if registry_dir is None:
        registry_dir = REGISTRY_DIR
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg_dir = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    dir_labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg_dir, side=None)
    y = dir_labels["label"].to_numpy()
    nz = y != 0
    t0_nz = t0_idx[nz]

    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(dir_oof["t0_nz"], t0_nz), "direction_side event index mismatch"
    has_oof = dir_oof["has_oof"]
    side_full = dir_oof["side"]

    candidate_cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, candidate_cols].reset_index(drop=True)

    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, side_full, has_oof)

    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    X_meta_full["p_direction"] = dir_oof["p_direction_cal"][has_oof]
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    meta_pass1 = oof_run(X_meta_full, y_meta, t0_meta, t1_meta,
                          tag=f"opportunity_v3b_h{max_holding}_meta_pass1", want_importance=True)
    feature_cols_meta = select_top_features(meta_pass1["importances"], top_n=TOP_N_FEATURES)
    if "assumed_side" not in feature_cols_meta:
        feature_cols_meta.append("assumed_side")
    used_p_direction = "p_direction" in feature_cols_meta
    if not used_p_direction:
        feature_cols_meta.append("p_direction")  # keep it in for the A vs A+probability comparison below

    X_meta = X_meta_full[feature_cols_meta]
    meta_result = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag=f"opportunity_v3b_h{max_holding}")
    meta_has_oof = meta_result["has_oof"]
    y_true = y_meta.to_numpy()[meta_has_oof]
    p_raw = meta_result["oof_proba"][meta_has_oof]
    cal = PlattCalibrator.fit(p_raw, y_true)
    p_cal = cal.apply(p_raw)
    oos_log_loss_with_p = manual_log_loss(y_true, p_cal)

    # A vs A+probability comparison (design section 9): refit without p_direction,
    # compare OOF log-loss; drop p_direction from the FINAL schema if it doesn't help.
    cols_no_p = [c for c in feature_cols_meta if c != "p_direction"]
    X_meta_a = X_meta_full[cols_no_p]
    meta_result_a = oof_run(X_meta_a, y_meta, t0_meta, t1_meta, tag=f"opportunity_v3b_h{max_holding}_a_only", want_importance=False)
    has_oof_a = meta_result_a["has_oof"]
    y_true_a = y_meta.to_numpy()[has_oof_a]
    p_raw_a = meta_result_a["oof_proba"][has_oof_a]
    cal_a = PlattCalibrator.fit(p_raw_a, y_true_a)
    oos_log_loss_a_only = manual_log_loss(y_true_a, cal_a.apply(p_raw_a))

    keep_p_direction = oos_log_loss_with_p < oos_log_loss_a_only - 1e-4
    if keep_p_direction:
        final_cols, final_result, final_has_oof, final_p_cal, final_y_true = (
            feature_cols_meta, meta_result, meta_has_oof, p_cal, y_true)
        oos_log_loss = oos_log_loss_with_p
    else:
        final_cols, final_result, final_has_oof, final_p_cal, final_y_true = (
            cols_no_p, meta_result_a, has_oof_a, cal_a.apply(p_raw_a), y_true_a)
        oos_log_loss = oos_log_loss_a_only

    win_rate = float(y_meta.mean())
    status = "validated" if win_rate > 0.4887 else "rejected"

    schema = build_schema(f"opportunity_v3b_h{max_holding}", "2026-08-24", final_cols)
    save_schema(schema, schemas_dir=schemas_dir if schemas_dir else SCHEMAS_DIR)

    entry = ModelRegistryEntry(
        model_id=f"opportunity_v3b_candidate_h{max_holding}", family="opportunity_meta", algorithm="catboost",
        artifact_path=f"registry/opportunity_v3b_candidate_h{max_holding}.json",
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=final_cols,
        target_definition=(f"P(TP before SL | features, direction_side); direction_side/assumed_side "
                            f"sourced from {dir_oof['model_id']}'s own OOF prediction, max_holding={max_holding}, "
                            f"pt=1.5*vol_tb sl=1.0*vol_tb"),
        training_config={"n_splits": 6, "embargo_bars": REAL_EMBARGO_BARS,
                          "direction_side_model_id": dir_oof["model_id"],
                          "p_direction_log_loss": oos_log_loss_with_p,
                          "assumed_side_only_log_loss": oos_log_loss_a_only,
                          "kept_p_direction": keep_p_direction},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"n_events": int(len(final_y_true)), "meta_win_rate": win_rate, "oos_log_loss": oos_log_loss,
                 "baseline_meta_win_rate": 0.4887},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(registry_dir, exist_ok=True)
    with open(os.path.join(registry_dir, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[opportunity_v3b h={max_holding}] n_events={len(final_y_true):,} win_rate={win_rate:.4f} "
          f"log_loss={oos_log_loss:.4f} kept_p_direction={keep_p_direction} -> status={status}")
    return {"n_events": len(final_y_true), "oos_log_loss": oos_log_loss, "status": status,
            "used_p_direction": keep_p_direction}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        run_opportunity_candidate(max_holding=h)
