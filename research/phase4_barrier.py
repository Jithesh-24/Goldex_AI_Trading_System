"""Phase 4, Role F: Barrier probability. Same meta-label target as Role B
(opportunity/meta) but evaluated as a standalone calibrated-probability
specialist (spec section 24's "P(barrier)" distributional output) rather
than a trade filter -- log loss/Brier/reliability-curve/horizon-stability
first, win-rate-lift is not this task's headline metric (see Task 9's
plan entry for the full distinction).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_barrier
"""
import os

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, select_top_features
from research.audit_edge import oof_run, build_meta, manual_log_loss
from decision.calibration import PlattCalibrator
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from features.registry import build_schema
from features.registry.schemas import save_schema, SCHEMAS_DIR
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
TOP_N_FEATURES = 20  # per spec section 6: this role's OWN narrowed schema, not the shared pool


def run_barrier_candidate(max_holding: int, rows: int = None, registry_dir: str = None, schemas_dir: str = None) -> dict:
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

    prim = oof_run(X_full, y_bin, t0, t1, tag=f"barrier_v3_h{max_holding}_primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    # Pass 1: full pool + assumed_side, OOF importances only.
    barrier_pass1 = oof_run(X_meta_full, y_meta, t0_meta, t1_meta,
                             tag=f"barrier_v3_h{max_holding}_pass1", want_importance=True)
    feature_cols_meta = select_top_features(barrier_pass1["importances"], top_n=TOP_N_FEATURES)
    if "assumed_side" not in feature_cols_meta:
        feature_cols_meta.append("assumed_side")

    # Pass 2: this role's OWN narrowed feature schema -- these metrics go into the registry entry.
    X_meta = X_meta_full[feature_cols_meta]
    meta_result = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag=f"barrier_v3_h{max_holding}")
    has_oof2 = meta_result["has_oof"]
    y_true = y_meta.to_numpy()[has_oof2]
    p_raw = meta_result["oof_proba"][has_oof2]
    cal = PlattCalibrator.fit(p_raw, y_true)
    p_cal = cal.apply(p_raw)

    log_loss = manual_log_loss(y_true, p_cal)
    brier = float(np.mean((p_cal - y_true) ** 2))

    deciles = np.digitize(p_cal, np.linspace(0, 1, 11)[1:-1])
    reliability_curve = []
    for d in sorted(set(deciles)):
        m = deciles == d
        if m.sum() < 20:
            continue
        reliability_curve.append({"decile": int(d), "n": int(m.sum()),
                                   "mean_predicted": float(p_cal[m].mean()),
                                   "actual_win_rate": float(y_true[m].mean())})

    max_calib_gap = max((abs(b["mean_predicted"] - b["actual_win_rate"]) for b in reliability_curve), default=1.0)
    status = "validated" if max_calib_gap < 0.15 else "rejected"

    schema = build_schema(f"barrier_v3_h{max_holding}", "2026-08-22", feature_cols_meta)
    save_schema(schema, schemas_dir=schemas_dir if schemas_dir else SCHEMAS_DIR)
    entry = ModelRegistryEntry(
        model_id=f"barrier_v3_candidate_h{max_holding}", family="barrier_probability", algorithm="catboost",
        artifact_path=f"registry/barrier_v3_candidate_h{max_holding}.json",
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=feature_cols_meta,
        target_definition=(
            f"P(assumed-side TP before SL within max_holding={max_holding}); same triple-barrier "
            f"meta-label as opportunity_meta, but registered as a standalone calibrated-probability "
            f"specialist (spec section 24) evaluated on log loss/Brier/reliability curve/horizon "
            f"stability rather than win-rate-lift -- see Task 9 of the Phase 4 plan for why this is "
            f"not a duplicate of the opportunity_meta role."
        ),
        training_config={"n_splits": 6, "embargo_bars": max_holding * 2},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"n_events": int(len(y_true)), "log_loss": log_loss, "brier": brier,
                 "max_calibration_gap": max_calib_gap, "reliability_curve": reliability_curve},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(registry_dir, exist_ok=True)
    with open(os.path.join(registry_dir, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[barrier h={max_holding}] n_events={len(y_true):,} log_loss={log_loss:.4f} "
          f"brier={brier:.4f} max_calib_gap={max_calib_gap:.4f} -> status={status}")
    return {"n_events": int(len(y_true)), "log_loss": log_loss, "brier": brier,
            "reliability_curve": reliability_curve, "status": status}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    horizon_results = {}
    for h in HORIZONS:
        horizon_results[h] = run_barrier_candidate(max_holding=h)
    print("\nhorizon stability (log_loss/brier per horizon):")
    for h, r in horizon_results.items():
        print(f"  h={h}: log_loss={r['log_loss']:.4f} brier={r['brier']:.4f}")
