"""Phase 4, Role E: MFE quantile. CatBoost quantile regression on mfe_R
(max favourable excursion in R-multiples), V3 feature fabric, compared
against the per-vol-state empirical-quantile baseline -- same methodology
as Role D (MAE quantile, research/phase4_mae_quantile.py) and
research/v3_quantile_models.py's proven pattern, applied to the parallel
"how much upside is available" trade question (spec section 4/23: MAE and
MFE are genuinely distinct targets on the same event set, not the same
pipeline twice). Global AND per-regime coverage reported (spec section 13).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_mfe_quantile
"""
import os

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, select_top_features
from research.audit_edge import oof_run, build_meta, _mae_mfe_core
from research.v3_quantile_models import fit_quantile, pinball_loss
from learning.cv import PurgedWalkForwardCV
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from features.registry import build_schema
from features.registry.schemas import save_schema
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
QUANTILES = [0.5, 0.75, 0.9]
TOP_N_FEATURES = 20  # per spec section 6: this role's OWN narrowed schema, not the shared pool


def run_mfe_quantile_candidate(max_holding: int, rows: int = None) -> dict:
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

    prim = oof_run(X_full, y_bin, t0, t1, tag=f"mfe_v3_h{max_holding}_primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]

    # Handle case where oof_run produces no OOF (e.g., with very small datasets)
    if not has_oof.any():
        print(f"[WARNING] No OOF predictions for h={max_holding} - dataset too small or CV constraints too strict")
        return {"n_events": 0, "global_coverage": {}, "per_regime_coverage": {}, "status": "rejected"}

    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    t0_meta = meta_labels.index.to_numpy()
    t1_meta = meta_labels["t1"].to_numpy()
    vol_at_meta = vol_tb[t0_nz][has_oof]

    _, mfe_R = _mae_mfe_core(close, high, low, t0_meta, t1_meta, side, vol_at_meta)

    lo_thr, hi_thr = np.nanpercentile(vol_tb, [33.3, 66.7])
    vol_state = np.where(vol_at_meta <= lo_thr, "low", np.where(vol_at_meta >= hi_thr, "high", "medium"))

    cv = PurgedWalkForwardCV(n_splits=6, embargo_bars=max_holding * 2, min_train_bars=500)
    t0_s, t1_s = pd.Series(t0_meta), pd.Series(t1_meta)

    # Pass 1 (q=0.5 only): full pool + assumed_side, capture per-fold quantile-model
    # feature importances to narrow to this role's OWN schema before the real runs.
    pass1_importances = []
    for train_pos, _ in cv.split(t0_s.to_numpy(), t1_s.to_numpy()):
        model = fit_quantile(X_meta_full, mfe_R, train_pos, 0.5)
        pass1_importances.append(dict(zip(X_meta_full.columns, [float(v) for v in model.get_feature_importance()])))
    feature_cols_meta = select_top_features(pass1_importances, top_n=TOP_N_FEATURES)
    if "assumed_side" not in feature_cols_meta:
        feature_cols_meta.append("assumed_side")
    X_meta = X_meta_full[feature_cols_meta]

    global_coverage, global_pinball = {}, {}
    per_regime_coverage = {}
    for q in QUANTILES:
        oof_pred = np.full(len(mfe_R), np.nan)
        for _, (train_pos, test_pos) in enumerate(cv.split(t0_s.to_numpy(), t1_s.to_numpy())):
            model = fit_quantile(X_meta, mfe_R, train_pos, q)
            oof_pred[test_pos] = model.predict(X_meta.iloc[test_pos])
        has_pred = np.isfinite(oof_pred)
        yp, yt = oof_pred[has_pred], mfe_R[has_pred]
        global_coverage[str(q)] = float((yt <= yp).mean())
        global_pinball[str(q)] = pinball_loss(yt, yp, q)

        vs_valid = vol_state[has_pred]
        per_regime_coverage[str(q)] = {}
        for vs in ("low", "medium", "high"):
            m = vs_valid == vs
            if m.sum() > 30:
                per_regime_coverage[str(q)][vs] = float((yt[m] <= yp[m]).mean())

    status = "validated" if all(abs(global_coverage[str(q)] - q) < 0.1 for q in QUANTILES) else "rejected"

    schema = build_schema(f"mfe_quantile_v3_h{max_holding}", "2026-08-22", feature_cols_meta)
    save_schema(schema)
    entry = ModelRegistryEntry(
        model_id=f"mfe_quantile_v3_candidate_h{max_holding}", family="mfe_quantile", algorithm="catboost_quantile",
        artifact_path=f"registry/mfe_quantile_v3_candidate_h{max_holding}.json",
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=feature_cols_meta,
        target_definition=f"mfe_R: max favourable excursion in R-multiples up to t1, max_holding={max_holding}",
        training_config={"quantiles": QUANTILES, "n_splits": 6, "embargo_bars": max_holding * 2},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"n_events": int(len(mfe_R)), "global_coverage": global_coverage,
                 "global_pinball": global_pinball, "per_regime_coverage": per_regime_coverage},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(os.path.join(REGISTRY_DIR, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[mfe_quantile h={max_holding}] n_events={len(mfe_R):,} "
          f"global_coverage={global_coverage} -> status={status}")
    return {"n_events": len(mfe_R), "global_coverage": global_coverage,
            "per_regime_coverage": per_regime_coverage, "status": status}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        run_mfe_quantile_candidate(max_holding=h)
