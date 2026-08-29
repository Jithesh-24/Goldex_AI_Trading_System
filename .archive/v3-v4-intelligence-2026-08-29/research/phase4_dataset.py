"""Phase 4 shared dataset builder -- connects the Phase 3 V3 feature fabric
(features/replay_engine.py, additive-only until now) to real historical
bars for specialist-model training. Every research/phase4_*.py role script
imports assemble_v3_dataset instead of re-deriving features, so "what
feature values did role X train on" is always traceable to this one
function. Does NOT touch learning/train.py's production assemble_dataset
(that stays on the old build_features-only 28-col path) -- this is a
parallel, additive dataset path for Phase 4 research only."""
import numpy as np
import pandas as pd

from learning.data import load_raw_m1
from features.features import build_tier1_features, build_features
from features.replay_engine import build_candidate_features
from features.labeling import cusum_filter
from features.registry import load_all
from contracts.feature_schema import FeatureStatus

CUSUM_K = 2.5  # matches features.replay_engine.CUSUM_K / learning.train.CUSUM_K
HORIZONS = (15, 45, 90)  # very-short / production-match / medium-short, per plan's Global Constraints
HORIZON_VOL_SCALE = 0.45  # matches learning.train.HORIZON_VOL_SCALE -- same real-evidence-backed constant


def select_top_features(importances: list, top_n: int = 20) -> list:
    """Averages per-fold OOF feature importances (from research.audit_edge.oof_run's
    `importances` list -- each fold's model.get_feature_importance() on that
    fold's OWN train split, never in-sample on the full set) and returns the
    top-`top_n` feature names by mean importance. Spec section 6: each
    specialist narrows the shared baseline+useful candidate pool to its OWN
    feature schema via OOS importance -- never by in-sample importance, and
    never the same list reused unnarrowed across specialists."""
    if not importances:
        raise ValueError("select_top_features requires at least one fold's importances")
    all_cols = list(importances[0].keys())
    mean_imp = {c: float(np.mean([fold.get(c, 0.0) for fold in importances])) for c in all_cols}
    ranked = sorted(mean_imp, key=lambda c: mean_imp[c], reverse=True)
    return ranked[:top_n]


def assemble_v3_dataset(max_holding: int, rows: int = None) -> dict:
    """Returns dict: feat_v3 (28 baseline + 92 V3 candidate cols + time),
    close/high/low (float64 arrays), vol_tb (horizon-scaled vol, same
    formula as learning.train.assemble_dataset), t0_idx (CUSUM event
    positions with full warmup satisfied for the RETURNED column set),
    baseline_cols (28 REQUIRED ids, real production order), useful_cols
    (17 USEFUL-status candidate ids, Task 26's real survivor list)."""
    df = load_raw_m1()
    if rows:
        df = df.tail(rows).reset_index(drop=True)

    base_feat = build_tier1_features(df)
    baseline_feat = build_features(df)  # the deployed 28-col matrix
    cand_feat = build_candidate_features(df, base_feat, cusum_k=CUSUM_K)  # time + 92 V3 cols

    descriptors = load_all()
    baseline_cols = [d.feature_id for d in descriptors if d.family == "baseline_v1"]
    useful_cols = [d.feature_id for d in descriptors
                   if d.family != "baseline_v1" and d.status == FeatureStatus.USEFUL]

    feat_v3 = baseline_feat.copy()
    for col in useful_cols:
        feat_v3[col] = cand_feat[col].to_numpy()

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    vol = feat_v3["ewma_vol"].to_numpy(dtype=np.float64)
    vol_filled = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[np.isfinite(vol)]))
    threshold = np.clip(CUSUM_K * vol_filled * close, 1e-6, None)
    event_mask = cusum_filter(close, threshold)

    vol_tb = vol_filled * np.sqrt(max_holding) * HORIZON_VOL_SCALE

    all_cols = baseline_cols + useful_cols
    warmup_ok = feat_v3[all_cols].notna().all(axis=1).to_numpy()
    horizon_ok = np.arange(len(df)) < (len(df) - max_holding - 1)
    valid = event_mask & warmup_ok & horizon_ok
    t0_idx = np.where(valid)[0]

    print(f"assemble_v3_dataset(max_holding={max_holding}): {len(df):,} bars -> {len(t0_idx):,} events "
          f"({len(baseline_cols)} baseline + {len(useful_cols)} useful cols)")

    return {"feat_v3": feat_v3, "close": close, "high": high, "low": low,
            "vol_tb": vol_tb, "t0_idx": t0_idx,
            "baseline_cols": baseline_cols, "useful_cols": useful_cols}


if __name__ == "__main__":
    for h in HORIZONS:
        assemble_v3_dataset(max_holding=h, rows=20000)
