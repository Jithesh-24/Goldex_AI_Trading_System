"""Phase 4, Role C: Regime. Unsupervised GaussianHMM(n_components=4) over
3 causal observables (log ewma_vol, hurst_120, kalman_residual_z),
refit per walk-forward fold. Evaluated for genuine usefulness (spec
section 15) via persistence, cross-period transition-matrix stability, and
downstream win-rate separation -- NOT deployed or wired into decision/
regardless of outcome; a `rejected` result here is a legitimate, spec-
compliant finding, not a failure of this task.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_regime
"""
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from research.phase4_dataset import assemble_v3_dataset
from research.audit_edge import wilson_ci
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from learning.cv import PurgedWalkForwardCV
from features.registry import build_schema
from features.registry.schemas import save_schema
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
OBS_COLS = ["ewma_vol", "hurst_120", "kalman_residual_z"]
N_STATES = 4


def _obs_matrix(feat_v3: pd.DataFrame) -> np.ndarray:
    log_vol = np.log(np.clip(feat_v3["ewma_vol"].to_numpy(), 1e-9, None))
    hurst = feat_v3["hurst_120"].to_numpy()
    kresid = feat_v3["kalman_residual_z"].to_numpy()
    return np.column_stack([log_vol, hurst, kresid])


def run_regime_candidate(rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=45, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    obs_full = _obs_matrix(feat_v3)
    valid_bars = np.isfinite(obs_full).all(axis=1)

    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
    dir_labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = dir_labels["label"].to_numpy()
    t1 = dir_labels["t1"].to_numpy()

    cv = PurgedWalkForwardCV(n_splits=6, embargo_bars=90, min_train_bars=2000)
    folds = list(cv.split(t0_idx, t1))
    assert len(folds) >= 2, "not enough folds in this dry run to measure cross-period stability"

    run_lengths = []
    trans_mats = []
    for train_pos, test_pos in folds:
        bar_lo, bar_hi = int(t0_idx[train_pos].min()), int(t0_idx[train_pos].max())
        train_bars = np.arange(bar_lo, bar_hi + 1)
        train_bars = train_bars[valid_bars[train_bars]]
        if len(train_bars) < 500:
            continue
        mu = obs_full[train_bars].mean(axis=0)
        sd = obs_full[train_bars].std(axis=0) + 1e-9
        model = GaussianHMM(n_components=N_STATES, covariance_type="diag", random_state=42, n_iter=50)
        model.fit((obs_full[train_bars] - mu) / sd)
        trans_mats.append(model.transmat_)

        test_bar_lo, test_bar_hi = int(t0_idx[test_pos].min()), int(t0_idx[test_pos].max())
        test_bars = np.arange(test_bar_lo, test_bar_hi + 1)
        test_bars = test_bars[valid_bars[test_bars]]
        if len(test_bars) < 50:
            continue
        states = model.predict((obs_full[test_bars] - mu) / sd)
        run_lengths.extend(_run_lengths(states))

    mean_run_length = float(np.mean(run_lengths)) if run_lengths else 0.0
    drift = float(np.linalg.norm(trans_mats[0] - trans_mats[-1])) if len(trans_mats) >= 2 else float("nan")

    # downstream usefulness: refit one HMM on ALL valid bars up to the last event (for a single
    # state-per-event lookup only -- this is descriptive evidence-gathering, not a claimed OOS
    # metric, since regime assignment here is in-sample by construction) and check win-rate
    # separation across states via wilson_ci.
    all_train_bars = np.arange(0, int(t0_idx.max()) + 1)
    all_train_bars = all_train_bars[valid_bars[all_train_bars]]
    mu = obs_full[all_train_bars].mean(axis=0)
    sd = obs_full[all_train_bars].std(axis=0) + 1e-9
    full_model = GaussianHMM(n_components=N_STATES, covariance_type="diag", random_state=42, n_iter=50)
    full_model.fit((obs_full[all_train_bars] - mu) / sd)
    event_states = full_model.predict((obs_full[t0_idx] - mu) / sd)

    nz = y != 0
    win = (y[nz] == 1).astype(int)
    states_nz = event_states[nz]
    cis = {}
    for s in range(N_STATES):
        m = states_nz == s
        if m.sum() < 30:
            continue
        cis[s] = wilson_ci(int(win[m].sum()), int(m.sum()))
    disjoint = any(cis[a][1] < cis[b][0] or cis[b][1] < cis[a][0]
                   for i, a in enumerate(cis) for b in list(cis)[i + 1:])
    status = "validated" if disjoint else "rejected"

    schema = build_schema("regime_v3", "2026-08-22", OBS_COLS)
    save_schema(schema)
    entry = ModelRegistryEntry(
        model_id="regime_v3_candidate", family="regime", algorithm="gaussian_hmm",
        artifact_path="registry/regime_v3_candidate.json",
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=OBS_COLS,
        target_definition=f"unsupervised GaussianHMM(n_components={N_STATES}) over standardized "
                           f"[log(ewma_vol), hurst_120, kalman_residual_z], refit per walk-forward fold",
        training_config={"n_states": N_STATES, "n_splits": 6, "embargo_bars": 90},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"mean_run_length": mean_run_length, "transition_matrix_drift": drift,
                 "per_state_win_rate_ci": {str(k): v for k, v in cis.items()},
                 "ci_disjoint": disjoint},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(os.path.join(REGISTRY_DIR, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[regime] mean_run_length={mean_run_length:.2f} transmat_drift={drift:.4f} "
          f"per_state_win_rate_ci={cis} -> status={status}")
    return {"n_states": N_STATES, "mean_run_length": mean_run_length,
            "transition_matrix_drift": drift, "status": status}


def _run_lengths(states: np.ndarray) -> list:
    out = []
    run = 1
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            run += 1
        else:
            out.append(run)
            run = 1
    out.append(run)
    return out


if __name__ == "__main__":
    run_regime_candidate()
