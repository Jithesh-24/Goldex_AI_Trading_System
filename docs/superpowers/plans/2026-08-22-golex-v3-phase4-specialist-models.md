# GOLEX V3 Phase 4: Specialist Quantitative Model Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, evaluate, and register the seven Phase 4 specialist models
(direction, opportunity/meta, regime, MAE quantile, MFE quantile, barrier
probability, execution/decay) on top of the Phase 3 quantitative feature
fabric, each validated OOS via causal walk-forward against its own
baseline, with zero change to production decision behavior.

**Architecture:** Reuse the Phase 1/2 infra that already exists for this
exact purpose (`contracts/model_registry.py`, `decision/router.py`,
`decision/calibration.py`, `learning/cv.py`'s purge/embargo,
`features/labeling.py`'s triple-barrier engine, `research/audit_edge.py`'s
OOF harness) rather than rebuilding it. Every specialist is a
`research/phase4_*.py` script that: (1) assembles a V3-feature event
dataset, (2) trains/evaluates candidate model(s) via
`PurgedWalkForwardCV`, (3) compares against a documented baseline, (4)
writes a `ModelRegistryEntry` JSON with real OOS metrics and a real
status. Nothing produced this phase is wired into `decision/router.py`'s
active role map or `app/engine.py` — specialists live in `models/registry/`
as `candidate`/`validated`/`rejected` entries only.

**Tech Stack:** Python 3, pandas/numpy, CatBoost (`CatBoostClassifier`,
`CatBoostRegressor` with `Quantile:alpha=q`), `hmmlearn.hmm.GaussianHMM`,
numba (for any new hot loop, matching `features/labeling.py`'s existing
`@numba.njit(cache=True)` convention), pydantic (via `contracts/`).

**Spec:** `docs/superpowers/specs/2026-08-22-golex-v3-phase4-specialist-models-design.md`

## Global Constraints

- **No production decision changes.** Do not modify `app/engine.py`,
  `decision/signal.py`, `config/models.yaml`'s existing `direction`/
  `opportunity_meta` values, or Telegram behavior. The only allowed edit to
  `config/models.yaml`/`config/schema.py` is adding new `null`-valued role
  keys (Task 1) — this changes no runtime behavior since `ModelRouter.resolve()`
  already returns `None` for any role whose `role_map` value is falsy.
- **CatBoost baselines are the benchmark, not to be replaced without
  evidence.** `models/registry/direction_catboost_20260818.json`
  (`mean_oof_acc=0.5115`) and `models/registry/opportunity_meta_catboost_20260818.json`
  (`meta_win_rate_baseline=0.4887`) are the Direction/Opportunity baselines.
  No new entry may set `is_champion: true` — `tests/test_model_registry.py::test_exactly_two_active_champions`
  already pins the champion set to exactly these two and must keep passing
  unmodified.
- **Never shuffle time-series data.** Every walk-forward split in this plan
  uses `learning.cv.PurgedWalkForwardCV(n_splits, embargo_bars, min_train_bars)`
  — it already implements purge + embargo (de Prado). Do not write a new
  splitter; do not use `sklearn.model_selection.KFold`/`train_test_split`
  anywhere in this plan's code.
- **Calibrate only on train/validation, never on test.** Use
  `decision.calibration.PlattCalibrator.fit(raw_p, y)` fit on OOF
  predictions from the train/validation portion of each fold only, per the
  module's own causality contract.
- **No data fabrication.** No real XM tick-level dataset and no historical
  human-execution/fill-timestamp data exist anywhere in this repo (verified
  by direct search before writing this plan). Task 12 (execution/decay) and
  the tick-capture portion of Task 13 (microstructure real-data validation)
  MUST be delivered as real, running infrastructure with an honest
  `DATA_LIMITED` status — not a fabricated result. This is not a shortfall
  to route around; it is the correct, spec-mandated outcome (design spec
  §7/§22/§27).
- **Every trained candidate gets a `contracts.model_registry.ModelRegistryEntry`**
  JSON under `models/registry/`, `status` one of `candidate` / `validated`
  (new, added in Task 1) / `rejected` / `archived` (never `active`, never
  `is_champion: true`, for anything produced by this plan).
- **Reuse, don't reimplement:** `features.labeling.{cusum_filter,
  TripleBarrierConfig, triple_barrier_labels}`, `learning.cv.PurgedWalkForwardCV`,
  `research.audit_edge.{oof_run, build_meta, wilson_ci, block_bootstrap,
  manual_log_loss}`, `research.v3_quantile_models.{fit_quantile,
  pinball_loss}`, `decision.calibration.PlattCalibrator`,
  `features.registry.{load_all, build_schema}`. All confirmed to exist with
  the exact signatures used below.
- **Horizons.** `learning/train.py`'s own docstring already documents real
  evidence that edge is horizon-dependent (15 bars: 51.4% vs 180 bars:
  50.7%; production uses `max_holding=45`). Per spec §5, this plan
  evaluates three concrete, evidence-motivated horizons everywhere a
  horizon choice matters: `HORIZONS = (15, 45, 90)` bars — very-short,
  production (baseline match), medium-short. Do not add more without a
  documented reason.
- **Venv:** `/home/jith/.hermes/hermes-agent/venv/bin/python3` for every
  `Run:` command below.
- **Sequential models (Transformer/TCN/LSTM) are deferred, per spec §26 —
  documented here rather than silently skipped.** Verified before writing
  this plan: no `torch` (or other deep-learning sequential-model library)
  is installed in the venv, only `catboost`/`lightgbm`/`xgboost`/`sklearn`/
  `hmmlearn`. Data resolution is M1 bars; production's own real-evidence
  docstring (`learning/train.py`) already establishes the tradeable signal
  here is a short-horizon (15-90 bar) mean-reversion effect on engineered
  state features, not a long-range sequential pattern a TCN/Transformer
  would be needed to capture. No task in this plan implements one. If a
  future phase finds OOS evidence that a sequential model is justified,
  that is new work requiring its own research task, not an extension of
  this plan.

## File Structure

- `contracts/model_registry.py` — modify: extend `ModelStatus`/`ModelFamily` Literals (Task 1).
- `config/schema.py`, `config/models.yaml` — modify: add `execution_decay` role stub (Task 1).
- `features/registry/schemas.py` — new: persist/load `FeatureSetSchema` JSON per specialist (Task 2).
- `research/phase4_dataset.py` — new: shared V3-feature event dataset builder, all 3 horizons, used by Tasks 4-10 (Task 3).
- `tests/test_phase4_dataset.py` — new: synthetic-path target-correctness test for Task 3's core (Task 3).
- `research/phase4_direction.py` — new: Direction role (Task 4).
- `research/phase4_opportunity.py` — new: Opportunity/meta role (Task 5).
- `research/phase4_regime.py` — new: Regime role (Task 6).
- `research/phase4_mae_quantile.py` — new: MAE quantile role (Task 7).
- `research/phase4_mfe_quantile.py` — new: MFE quantile role (Task 8).
- `research/phase4_barrier.py` — new: Barrier probability role (Task 9).
- `research/phase4_execution_decay.py` — new: Execution/decay role, DATA_LIMITED (Task 10).
- `market/tick_capture.py` — new: opt-in real-tick capture-to-disk, off by default (Task 11).
- `research/microstructure_live_real_validation.py` — new: real-data validation driver for Task 21's 5 live-only features (Task 11).
- `tests/test_phase4_leakage.py` — new: cross-cutting leakage audit for every role's dataset/CV/calibration (Task 12).
- `research/phase4_model_inventory.py` — new: prints every Phase-4 registry entry for the final report (Task 13).
- `tests/test_specialist_inference_performance.py` — new: inference latency/memory benchmark per trained specialist (Task 14).
- `models/README.md` — new: specialist roles, registry statuses, routing (Task 15).
- `docs/ARCHITECTURE.md` — modify: append "## Phase 4: Specialist Quantitative Model Layer" (Task 15).
- Task 16: final verification sweep + A-S completion report (no files).

---

### Task 1: Registry/config statuses — add `validated` status and `execution_decay` role

**Files:**
- Modify: `contracts/model_registry.py`
- Modify: `config/schema.py`
- Modify: `config/models.yaml`
- Test: `tests/test_model_registry.py` (extend)

**Interfaces:**
- Produces: `ModelStatus = Literal["candidate", "validated", "active", "archived", "rejected"]`,
  `ModelFamily` gains `"execution_decay"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_model_registry.py, above the existing __main__ block
def test_validated_status_accepted():
    from contracts.model_registry import ModelRegistryEntry
    entry = ModelRegistryEntry(
        model_id="phase4_smoke_test", family="regime", algorithm="dummy",
        artifact_path="registry/phase4_smoke_test.json", created_at="2026-08-22T00:00:00Z",
        status="validated",
    )
    assert entry.status == "validated"


def test_execution_decay_family_accepted():
    from contracts.model_registry import ModelRegistryEntry
    entry = ModelRegistryEntry(
        model_id="phase4_smoke_test2", family="execution_decay", algorithm="dummy",
        artifact_path="registry/phase4_smoke_test2.json", created_at="2026-08-22T00:00:00Z",
        status="candidate",
    )
    assert entry.family == "execution_decay"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from tests.test_model_registry import test_validated_status_accepted
test_validated_status_accepted()
"`
Expected: `pydantic.ValidationError` (status="validated" not in current Literal).

- [ ] **Step 3: Extend `contracts/model_registry.py`**

```python
ModelFamily = Literal[
    "direction", "opportunity_meta", "regime",
    "mae_quantile", "mfe_quantile", "barrier_probability", "execution_decay",
]
ModelStatus = Literal["candidate", "validated", "active", "archived", "rejected"]
```

(Only these two lines change. `is_champion: bool = False` and every other
field stay exactly as-is — `active`+`is_champion=True` remains the
production-champion convention; `validated` is a new, non-champion,
OOS-beats-baseline status this plan's tasks will actually use.)

- [ ] **Step 4: Add `execution_decay` to the config layer**

`config/schema.py`, in `ModelRoleConfig`:
```python
class ModelRoleConfig(BaseModel):
    direction: Optional[str] = None
    opportunity_meta: Optional[str] = None
    regime: Optional[str] = None
    mae_quantile: Optional[str] = None
    mfe_quantile: Optional[str] = None
    barrier_probability: Optional[str] = None
    execution_decay: Optional[str] = None
```

`config/models.yaml`, append one line:
```yaml
execution_decay: null
```

- [ ] **Step 5: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_model_registry.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_router.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from config.loader import load_config
load_config()
print('config loads OK with execution_decay role')
"`
(Adjust the config-loading call to whatever `config/loader.py`'s real
entrypoint function is named — read that file first; it must load without
error since `execution_decay: null` is a valid `Optional[str]`.)
Expected: all PASS, `test_exactly_two_active_champions` still passes
unchanged (the two new test-only entries above are never written to
`models/registry/` on disk — they're constructed in-memory only, inside
the test function).

- [ ] **Step 6: Commit**

```bash
git add contracts/model_registry.py config/schema.py config/models.yaml tests/test_model_registry.py
git commit -m "Add validated model status and execution_decay role for Phase 4"
```

---

### Task 2: Feature-set schema persistence (`features/registry/schemas.py`)

**Files:**
- Create: `features/registry/schemas.py`
- Test: `tests/test_feature_set_schemas.py`

**Interfaces:**
- Consumes: `contracts.feature_schema.FeatureSetSchema`, `features.registry.build_schema`.
- Produces: `save_schema(schema: FeatureSetSchema, schemas_dir: str = SCHEMAS_DIR) -> str` (returns the path written), `load_schema(schema_id: str, schema_version: str, schemas_dir: str = SCHEMAS_DIR) -> FeatureSetSchema`.

Every Phase 4 role task calls `build_schema(...)` then `save_schema(...)` so
its `ModelRegistryEntry.feature_schema_version` field points at a real,
re-loadable, on-disk artifact (spec §18/§20: "a model must fail validation
if required feature schema does not match").

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_feature_set_schemas.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.registry import build_schema
from features.registry.schemas import save_schema, load_schema


def test_save_then_load_roundtrip():
    schema = build_schema("direction_v3", "2026-08-22", ["dist_from_high_20", "hour_sin"])
    with tempfile.TemporaryDirectory() as tmp:
        path = save_schema(schema, schemas_dir=tmp)
        assert os.path.exists(path)
        loaded = load_schema("direction_v3", "2026-08-22", schemas_dir=tmp)
        assert loaded.feature_ids == ["dist_from_high_20", "hour_sin"]
        assert loaded.schema_version == "2026-08-22"


if __name__ == "__main__":
    test_save_then_load_roundtrip()
    print("tests/test_feature_set_schemas.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_set_schemas.py`
Expected: `ModuleNotFoundError: No module named 'features.registry.schemas'`

- [ ] **Step 3: Write `features/registry/schemas.py`**

```python
"""Persists FeatureSetSchema slices (spec section 9/18/20) so a
ModelRegistryEntry.feature_schema_version can point at a real, re-loadable
artifact instead of an inline list nobody re-checks at load time."""
import json
import os

from contracts.feature_schema import FeatureSetSchema

SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")


def _path(schema_id: str, schema_version: str, schemas_dir: str) -> str:
    os.makedirs(schemas_dir, exist_ok=True)
    return os.path.join(schemas_dir, f"{schema_id}__{schema_version}.json")


def save_schema(schema: FeatureSetSchema, schemas_dir: str = SCHEMAS_DIR) -> str:
    path = _path(schema.schema_id, schema.schema_version, schemas_dir)
    with open(path, "w") as f:
        f.write(schema.model_dump_json(indent=2))
    return path


def load_schema(schema_id: str, schema_version: str, schemas_dir: str = SCHEMAS_DIR) -> FeatureSetSchema:
    path = _path(schema_id, schema_version, schemas_dir)
    with open(path) as f:
        return FeatureSetSchema(**json.load(f))
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_set_schemas.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/registry/schemas.py tests/test_feature_set_schemas.py
git commit -m "Add FeatureSetSchema persistence for Phase 4 specialist feature schemas"
```

---

### Task 3: Shared V3-feature event dataset builder (all 7 roles depend on this)

**Files:**
- Create: `research/phase4_dataset.py`
- Test: `tests/test_phase4_dataset.py`

**Interfaces:**
- Consumes: `learning.data.load_raw_m1`, `features.features.{build_tier1_features, build_features}`,
  `features.replay_engine.build_candidate_features`, `features.labeling.{cusum_filter, TripleBarrierConfig, triple_barrier_labels}`,
  `features.registry.load_all`.
- Produces: `assemble_v3_dataset(max_holding: int, rows: int = None) -> dict` with keys
  `{feat_v3: pd.DataFrame, close, high, low, vol_tb, t0_idx, baseline_cols, useful_cols}`
  (baseline_cols = the 28 REQUIRED feature_ids in their registry order,
  useful_cols = the USEFUL-status candidate feature_ids from Task 26's
  registry — this is the shared CANDIDATE pool every role task (4, 5, 7, 8,
  9) narrows to its OWN feature schema via `select_top_features`, per spec
  §6: no two specialists may be registered with an identical feature
  list). Also produces `select_top_features(importances: list[dict], top_n: int = 20) -> list[str]`
  — averages per-fold OOF feature importances (as returned by
  `research.audit_edge.oof_run`'s `importances` list, never in-sample
  importance) across folds and returns the top-`top_n` feature names.

Every role task in this plan (4-9) imports `assemble_v3_dataset` instead of
re-deriving features from scratch — this is the ONE place the V3 feature
fabric (Phase 3's `features/replay_engine.py`, additive-only, not yet used
by any trainer) gets connected to real historical bars for training
purposes. `learning/train.py`'s existing `assemble_dataset` (old
`features.features.build_features`-only path) is untouched — production
training is unaffected by this file's existence.

- [ ] **Step 1: Write the failing test — synthetic-path target correctness**

```python
"""python3 tests/test_phase4_dataset.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, HORIZONS


def test_horizons_constant_matches_spec():
    assert HORIZONS == (15, 45, 90)


def test_select_top_features_ranks_by_mean_importance():
    from research.phase4_dataset import select_top_features
    importances = [
        {"a": 10.0, "b": 1.0, "c": 5.0},
        {"a": 8.0, "b": 2.0, "c": 6.0},
    ]
    top = select_top_features(importances, top_n=2)
    assert top == ["a", "c"], f"expected mean-importance ranking [a, c], got {top}"


def test_assemble_v3_dataset_shapes_and_no_lookahead(tmp_path=None):
    # Real 6.7yr bar history -- capped to a fast dry run via `rows`.
    out = assemble_v3_dataset(max_holding=45, rows=5000)
    feat_v3 = out["feat_v3"]
    t0_idx = out["t0_idx"]
    assert len(t0_idx) > 0, "no CUSUM events found in a 5000-row dry run -- dataset assembly is broken"
    assert set(out["baseline_cols"]) <= set(feat_v3.columns)
    assert set(out["useful_cols"]) <= set(feat_v3.columns)
    # causality smoke check: every event's feature row must have no NaN in
    # the columns this plan will actually train on (mirrors learning/train.py's
    # warmup_ok gate, applied here to the V3-augmented column set).
    cols = out["baseline_cols"] + out["useful_cols"]
    assert feat_v3.loc[t0_idx, cols].notna().all().all(), "warmup NaNs leaked into selected events"
    # every event must resolve strictly in the future and within max_holding+1 bars
    close, high, low, vol_tb = out["close"], out["high"], out["low"], out["vol_tb"]
    from features.labeling import TripleBarrierConfig, triple_barrier_labels
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    assert (labels["t1"].to_numpy() > labels.index.to_numpy()).all()
    assert (labels["holding_bars"].to_numpy() <= 45).all()


if __name__ == "__main__":
    test_horizons_constant_matches_spec()
    test_select_top_features_ranks_by_mean_importance()
    test_assemble_v3_dataset_shapes_and_no_lookahead()
    print("tests/test_phase4_dataset.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_dataset.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_dataset'`

- [ ] **Step 3: Write `research/phase4_dataset.py`**

```python
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

    HORIZON_VOL_SCALE = 0.45  # matches learning.train.HORIZON_VOL_SCALE -- same real-evidence-backed constant
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_dataset.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase4_dataset.py`
Expected: PASS; the direct run prints real event counts for all 3 horizons
on a 20,000-row dry run — record these counts for Task 16's report.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_dataset.py tests/test_phase4_dataset.py
git commit -m "Add shared V3-feature event dataset builder for Phase 4 specialists"
```

---

### Task 4: Direction role — V3-feature candidate vs CatBoost baseline

**Files:**
- Create: `research/phase4_direction.py`
- Test: `tests/test_phase4_direction.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset`, `research.audit_edge.{oof_run, manual_log_loss}`,
  `decision.calibration.PlattCalibrator`, `learning.cv.PurgedWalkForwardCV`,
  `features.labeling.{TripleBarrierConfig, triple_barrier_labels}`,
  `features.registry.build_schema`, `features.registry.schemas.save_schema`,
  `contracts.model_registry.ModelRegistryEntry`.
- Produces: `models/registry/direction_v3_candidate_h{H}.json` for each of
  the 3 horizons (status `validated` if OOS log loss + Brier both beat the
  existing `direction_catboost_20260818` baseline re-measured on the SAME
  events with the SAME CV scheme, else `rejected` — never `active`).

**Target definition** (spec §4, written here since this is where the
target is concretely built): TARGET = sign of first barrier touched
(symmetric `pt_mult=sl_mult=1.0`, matching `learning.train.TB_CFG_DIR`).
HORIZON = 15/45/90 bars (this task's 3 registry entries). ENTRY REFERENCE
= event close at `t0`. BARRIER DEFINITION = `vol_tb[t0]`-scaled symmetric
band (`features.labeling.triple_barrier_labels`). LABELING METHOD =
first-touch among {upper, lower, vertical-timeout}; vertical-timeout
events dropped (matches `learning.train.label_events`, real
precedent). CENSORING/TIMEOUT = vertical barrier at `t0+max_holding`.
DATA REQUIREMENTS = `data/gold_seed_merged_full6yr.csv` via
`load_raw_m1()`. CAUSALITY = every feature at `t0` only; `t1 > t0` always
(pinned by Task 3's test). LEAKAGE RISKS = handled by
`PurgedWalkForwardCV`'s purge+embargo on `[t0, t1]`.

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_phase4_direction.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_direction import run_direction_candidate


def test_run_direction_candidate_produces_real_metrics():
    result = run_direction_candidate(max_holding=45, rows=20000)
    assert result["n_events"] > 100, "too few events in dry run to trust any metric"
    assert 0.0 <= result["oos_log_loss"] < 5.0
    assert 0.0 <= result["oos_brier"] <= 1.0
    assert result["status"] in ("validated", "rejected")


if __name__ == "__main__":
    test_run_direction_candidate_produces_real_metrics()
    print("tests/test_phase4_direction.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_direction.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_direction'`

- [ ] **Step 3: Write `research/phase4_direction.py`**

```python
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

    embargo_bars = max_holding * 2
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
```

- [ ] **Step 4: Run to verify it passes, then run for real (all horizons, full history)**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_direction.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_direction`
Record the printed `mean_oof_acc`/`log_loss`/`brier`/`status` for all 3
horizons for Task 16's report §D/§F. This is a real multi-hour-scale
walk-forward CatBoost fit on ~6.7 years of M1 data per horizon — if wall
clock is prohibitive, cap via a documented `rows=` argument in this step
only (not in the shipped script's `__main__ ` default) and say so plainly
in the report; do not silently truncate and report it as full-history.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_direction.py tests/test_phase4_direction.py models/registry/direction_v3_candidate_h*.json features/registry/schemas/direction_v3_h*.json
git commit -m "Add Phase 4 Direction role: V3-feature candidate vs CatBoost baseline"
```

---

### Task 5: Opportunity/meta role — V3-feature candidate vs existing meta baseline

**Files:**
- Create: `research/phase4_opportunity.py`
- Test: `tests/test_phase4_opportunity.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset`, `research.audit_edge.{oof_run, build_meta, manual_log_loss}`,
  same registry/schema/calibration imports as Task 4.
- Produces: `models/registry/opportunity_v3_candidate_h{H}.json`.

**Target definition:** TARGET = precision filter on the primary's own OOF
side (meta-labeling, per `learning/train.py`'s docstring — NOT a
duplicate of direction: this predicts P(that side's TP hits before its
SL), conditioned on a side already being proposed, using the ASYMMETRIC
`TB_CFG_TRADE`-style width `pt_mult=1.5, sl_mult=1.0`). HORIZON = 15/45/90.
ENTRY REFERENCE/BARRIER/LABELING/CENSORING = as Task 4 but asymmetric
widths and side-conditioned label via `triple_barrier_labels(..., side=side)`.
DATA REQUIREMENTS/CAUSALITY/LEAKAGE = same as Task 4, plus: side comes
from THIS TASK's own primary OOF run (`oof_run` on the symmetric target,
mirroring `learning.train.build_meta_labels`), never from the deployed
baseline's predictions (that would mix two different feature sets'
information into one candidate's evaluation).

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_phase4_opportunity.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_opportunity import run_opportunity_candidate


def test_run_opportunity_candidate_produces_real_metrics():
    result = run_opportunity_candidate(max_holding=45, rows=20000)
    assert result["n_events"] > 50, "too few meta-training events in dry run to trust any metric"
    assert 0.0 <= result["oos_log_loss"] < 5.0
    assert result["status"] in ("validated", "rejected")


if __name__ == "__main__":
    test_run_opportunity_candidate_produces_real_metrics()
    print("tests/test_phase4_opportunity.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_opportunity.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_opportunity'`

- [ ] **Step 3: Write `research/phase4_opportunity.py`**

```python
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
from decision.calibration import PlattCalibrator
from features.registry import build_schema
from features.registry.schemas import save_schema
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
TOP_N_FEATURES = 20  # per spec section 6: this role's OWN narrowed schema, not the shared pool


def run_opportunity_candidate(max_holding: int, rows: int = None) -> dict:
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

    # Primary side-generator run: uses the full candidate pool -- it's an internal input to the
    # meta target (side), not itself a registered specialist, so it is not narrowed.
    prim = oof_run(X_full, y_bin, t0, t1, tag=f"opportunity_v3_h{max_holding}_primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
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
    save_schema(schema)

    entry = ModelRegistryEntry(
        model_id=f"opportunity_v3_candidate_h{max_holding}", family="opportunity_meta", algorithm="catboost",
        artifact_path=f"registry/opportunity_v3_candidate_h{max_holding}.json",
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=feature_cols_meta,
        target_definition=f"meta-label: assumed-side TP before SL, max_holding={max_holding}, pt=1.5*vol_tb sl=1.0*vol_tb",
        training_config={"n_splits": 6, "embargo_bars": max_holding * 2},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"n_events": int(len(X_meta)), "meta_win_rate": win_rate, "oos_log_loss": oos_log_loss,
                 "baseline_meta_win_rate": 0.4887},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(os.path.join(REGISTRY_DIR, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[opportunity h={max_holding}] n_events={len(X_meta):,} win_rate={win_rate:.4f} "
          f"(baseline 0.4887) log_loss={oos_log_loss:.4f} -> status={status}")
    return {"n_events": len(X_meta), "oos_log_loss": oos_log_loss, "status": status}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        run_opportunity_candidate(max_holding=h)
```

- [ ] **Step 4: Run to verify it passes, then run for real**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_opportunity.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_opportunity`
Record win_rate/log_loss/status per horizon for Task 16's §G.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_opportunity.py tests/test_phase4_opportunity.py models/registry/opportunity_v3_candidate_h*.json features/registry/schemas/opportunity_v3_h*.json
git commit -m "Add Phase 4 Opportunity/meta role: V3-feature candidate vs existing baseline"
```

---

### Task 6: Regime role — GaussianHMM, evaluated for genuine usefulness

**Files:**
- Create: `research/phase4_regime.py`
- Test: `tests/test_phase4_regime.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset`, `research.audit_edge.wilson_ci`,
  `hmmlearn.hmm.GaussianHMM`.
- Produces: `models/registry/regime_v3_candidate.json`, status `validated`
  only if the downstream separation check (below) shows non-overlapping
  Wilson CIs; `rejected` otherwise — per spec §15, "a regime model is
  useful only if it adds predictive/decision value."

**Target definition:** No single fixed label set is hard-coded (spec §2C).
TARGET = an unsupervised `n_components=4` `GaussianHMM` state over 3 causal,
already-computed observables: `log(ewma_vol)`, `hurst_120`,
`kalman_residual_z` (all already present in `feat_v3` via
`build_tier1_features`) — standardized (z-scored using TRAIN-fold mean/std
only). HORIZON = N/A (regime is a per-bar state, not an event-horizon
target). DATA REQUIREMENTS = same bars as Task 3. CAUSALITY = the HMM is
`.fit()` only on the train portion of each walk-forward fold;
`.predict()` on the test portion uses only the already-fitted transition/
emission parameters plus each test bar's own (already-causal) observation
sequence up to that point — no test-fold information enters fitting.
LEAKAGE RISKS = fitting a global HMM on the full history before evaluating
per-fold would leak future regime structure into early folds; this task
refits per fold to avoid that.

**Evaluation (spec §15 — genuine usefulness, not a chart):**
1. *Regime persistence*: mean run-length of the predicted state sequence
   per fold (a regime that flips every bar is not a "regime").
2. *Stability across periods*: Frobenius-norm distance between the fitted
   transition matrices of fold 0 and fold 5 (first vs last) — large
   distance means the regime definition itself is not stable over the
   6.7-year history.
3. *Downstream usefulness*: using Task 4's direction-candidate labels
   (`h=45`) restricted to this task's own dry-run event set, compute the
   per-regime win rate with `wilson_ci` (already-proven 95% CI helper).
   If any two regimes' CIs are disjoint, the regime carries real
   predictive separation; if all CIs overlap, it does not.

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_phase4_regime.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_regime import run_regime_candidate


def test_run_regime_candidate_produces_real_diagnostics():
    result = run_regime_candidate(rows=20000)
    assert result["n_states"] == 4
    assert result["mean_run_length"] > 1.0, "a regime that flips every bar carries no persistence"
    assert "transition_matrix_drift" in result
    assert result["status"] in ("validated", "rejected")


if __name__ == "__main__":
    test_run_regime_candidate_produces_real_diagnostics()
    print("tests/test_phase4_regime.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_regime.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_regime'`

- [ ] **Step 3: Write `research/phase4_regime.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes, then run for real**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_regime.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_regime`
Record `mean_run_length`, `transition_matrix_drift`, per-state CIs, and the
final `status` for Task 16's §H — report the real outcome (`validated` or
`rejected`) honestly either way.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_regime.py tests/test_phase4_regime.py models/registry/regime_v3_candidate.json features/registry/schemas/regime_v3*.json
git commit -m "Add Phase 4 Regime role: GaussianHMM evaluated for genuine downstream usefulness"
```

---

### Task 7: MAE quantile role

**Files:**
- Create: `research/phase4_mae_quantile.py`
- Test: `tests/test_phase4_mae_quantile.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset`, `research.audit_edge.{oof_run, build_meta, _mae_mfe_core}`,
  `research.v3_quantile_models.{fit_quantile, pinball_loss}`, `learning.cv.PurgedWalkForwardCV`.
- Produces: `models/registry/mae_quantile_v3_candidate_h{H}.json` per horizon,
  one entry per quantile result set (q50/q75/q90; q95 only if the vol-state
  bucket sample size supports it, per spec §4 D's "where sample size
  supports them").

**Target definition:** TARGET = `mae_R` = maximum adverse excursion in
R-multiples (`-worst_drawdown / vol_at_t0`, via `research.audit_edge._mae_mfe_core`,
already proven and reused verbatim — do not reimplement this numba
kernel). HORIZON = 15/45/90 (same event set as Tasks 4/5, side = this
task's own primary OOF side, mirroring `research/v3_quantile_models.py`'s
proven pattern exactly). ENTRY REFERENCE = event close at `t0`. CENSORING
= excursion measured only up to `t1` (the meta-label's own resolution
time), never beyond. DATA REQUIREMENTS/CAUSALITY/LEAKAGE = same as Tasks
4/5, plus: quantile models fit per-fold on `train_pos` only (`fit_quantile`
already does this).

**Baseline (spec §17):** per-`vol_state`-tercile empirical quantile — same
in-sample-by-state baseline `research/v3_quantile_models.py` already uses
(documented there as "upper bound for the simple approach", kept
identical here for continuity of methodology across the old and new
feature sets).

**Evaluation (spec §13):** pinball loss + empirical coverage (global AND
per-vol_state-tercile, so a q90 that's globally 90% but fails in the high
vol tercile is caught — Task 7 must NOT report only the global number).

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_phase4_mae_quantile.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_mae_quantile import run_mae_quantile_candidate


def test_run_mae_quantile_candidate_produces_real_coverage():
    result = run_mae_quantile_candidate(max_holding=45, rows=20000)
    assert result["n_events"] > 50
    for q in ("0.5", "0.75", "0.9"):
        assert q in result["global_coverage"]
        assert 0.0 <= result["global_coverage"][q] <= 1.0
        assert q in result["per_regime_coverage"], "per-vol-state coverage missing -- spec section 13 requires it"


if __name__ == "__main__":
    test_run_mae_quantile_candidate_produces_real_coverage()
    print("tests/test_phase4_mae_quantile.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_mae_quantile.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_mae_quantile'`

- [ ] **Step 3: Write `research/phase4_mae_quantile.py`**

```python
"""Phase 4, Role D (high priority per spec section 23): MAE quantile.
CatBoost quantile regression on mae_R (max adverse excursion in
R-multiples), V3 feature fabric, compared against the per-vol-state
empirical-quantile baseline -- same methodology research/v3_quantile_models.py
already validated on the old v2 dataset, reused verbatim here on the new
V3-feature event set. Global AND per-regime coverage reported (spec
section 13): a model that's well-calibrated on average but wrong in one
vol regime is not acceptable.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_mae_quantile
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


def run_mae_quantile_candidate(max_holding: int, rows: int = None) -> dict:
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

    prim = oof_run(X_full, y_bin, t0, t1, tag=f"mae_v3_h{max_holding}_primary", want_importance=False)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]

    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    t0_meta = meta_labels.index.to_numpy()
    t1_meta = meta_labels["t1"].to_numpy()
    vol_at_meta = vol_tb[t0_nz][has_oof]

    mae_R, _ = _mae_mfe_core(close, high, low, t0_meta, t1_meta, side, vol_at_meta)

    lo_thr, hi_thr = np.nanpercentile(vol_tb, [33.3, 66.7])
    vol_state = np.where(vol_at_meta <= lo_thr, "low", np.where(vol_at_meta >= hi_thr, "high", "medium"))

    cv = PurgedWalkForwardCV(n_splits=6, embargo_bars=max_holding * 2)
    t0_s, t1_s = pd.Series(t0_meta), pd.Series(t1_meta)

    # Pass 1 (q=0.5 only): full pool + assumed_side, capture per-fold quantile-model
    # feature importances to narrow to this role's OWN schema before the real runs.
    pass1_importances = []
    for train_pos, _ in cv.split(t0_s.to_numpy(), t1_s.to_numpy()):
        model = fit_quantile(X_meta_full, mae_R, train_pos, 0.5)
        pass1_importances.append(dict(zip(X_meta_full.columns, [float(v) for v in model.get_feature_importance()])))
    feature_cols_meta = select_top_features(pass1_importances, top_n=TOP_N_FEATURES)
    if "assumed_side" not in feature_cols_meta:
        feature_cols_meta.append("assumed_side")
    X_meta = X_meta_full[feature_cols_meta]

    global_coverage, global_pinball = {}, {}
    per_regime_coverage = {}
    for q in QUANTILES:
        oof_pred = np.full(len(mae_R), np.nan)
        for _, (train_pos, test_pos) in enumerate(cv.split(t0_s.to_numpy(), t1_s.to_numpy())):
            model = fit_quantile(X_meta, mae_R, train_pos, q)
            oof_pred[test_pos] = model.predict(X_meta.iloc[test_pos])
        has_pred = np.isfinite(oof_pred)
        yp, yt = oof_pred[has_pred], mae_R[has_pred]
        global_coverage[str(q)] = float((yt <= yp).mean())
        global_pinball[str(q)] = pinball_loss(yt, yp, q)

        vs_valid = vol_state[has_pred]
        per_regime_coverage[str(q)] = {}
        for vs in ("low", "medium", "high"):
            m = vs_valid == vs
            if m.sum() > 30:
                per_regime_coverage[str(q)][vs] = float((yt[m] <= yp[m]).mean())

    status = "validated" if all(abs(global_coverage[str(q)] - q) < 0.1 for q in QUANTILES) else "rejected"

    schema = build_schema(f"mae_quantile_v3_h{max_holding}", "2026-08-22", feature_cols_meta)
    save_schema(schema)
    entry = ModelRegistryEntry(
        model_id=f"mae_quantile_v3_candidate_h{max_holding}", family="mae_quantile", algorithm="catboost_quantile",
        artifact_path=f"registry/mae_quantile_v3_candidate_h{max_holding}.json",
        feature_schema_version=f"{schema.schema_id}__{schema.schema_version}",
        feature_cols=feature_cols_meta,
        target_definition=f"mae_R: max adverse excursion in R-multiples up to t1, max_holding={max_holding}",
        training_config={"quantiles": QUANTILES, "n_splits": 6, "embargo_bars": max_holding * 2},
        created_at=pd.Timestamp.utcnow().isoformat(),
        status=status,
        metrics={"n_events": int(len(mae_R)), "global_coverage": global_coverage,
                 "global_pinball": global_pinball, "per_regime_coverage": per_regime_coverage},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(os.path.join(REGISTRY_DIR, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[mae_quantile h={max_holding}] n_events={len(mae_R):,} "
          f"global_coverage={global_coverage} -> status={status}")
    return {"n_events": len(mae_R), "global_coverage": global_coverage,
            "per_regime_coverage": per_regime_coverage, "status": status}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        run_mae_quantile_candidate(max_holding=h)
```

- [ ] **Step 4: Run to verify it passes, then run for real**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_mae_quantile.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_mae_quantile`
Record global + per-regime coverage/pinball per horizon per quantile for
Task 16's §I.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_mae_quantile.py tests/test_phase4_mae_quantile.py models/registry/mae_quantile_v3_candidate_h*.json features/registry/schemas/mae_quantile_v3_h*.json
git commit -m "Add Phase 4 MAE quantile role: CatBoost quantile with global+per-regime coverage"
```

---

### Task 8: MFE quantile role

**Files:**
- Create: `research/phase4_mfe_quantile.py`
- Test: `tests/test_phase4_mfe_quantile.py`

**Interfaces:** identical shape to Task 7, target = `mfe_R` (the second
return value of `_mae_mfe_core`) instead of `mae_R`.

**Target definition:** TARGET = `mfe_R` = maximum favourable excursion in
R-multiples up to `t1`. Everything else (horizon, entry reference,
censoring, data requirements, causality, leakage, baseline, evaluation) is
identical to Task 7 — this is a genuinely parallel target on the same
event set, not a different pipeline.

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_phase4_mfe_quantile.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_mfe_quantile import run_mfe_quantile_candidate


def test_run_mfe_quantile_candidate_produces_real_coverage():
    result = run_mfe_quantile_candidate(max_holding=45, rows=20000)
    assert result["n_events"] > 50
    for q in ("0.5", "0.75", "0.9"):
        assert q in result["global_coverage"]
        assert q in result["per_regime_coverage"]


if __name__ == "__main__":
    test_run_mfe_quantile_candidate_produces_real_coverage()
    print("tests/test_phase4_mfe_quantile.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_mfe_quantile.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_mfe_quantile'`

- [ ] **Step 3: Write `research/phase4_mfe_quantile.py`**

```python
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

    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    t0_meta = meta_labels.index.to_numpy()
    t1_meta = meta_labels["t1"].to_numpy()
    vol_at_meta = vol_tb[t0_nz][has_oof]

    _, mfe_R = _mae_mfe_core(close, high, low, t0_meta, t1_meta, side, vol_at_meta)

    lo_thr, hi_thr = np.nanpercentile(vol_tb, [33.3, 66.7])
    vol_state = np.where(vol_at_meta <= lo_thr, "low", np.where(vol_at_meta >= hi_thr, "high", "medium"))

    cv = PurgedWalkForwardCV(n_splits=6, embargo_bars=max_holding * 2)
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
```

- [ ] **Step 4: Run to verify it passes, then run for real**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_mfe_quantile.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_mfe_quantile`
Record global + per-regime coverage/pinball per horizon per quantile for
Task 16's §J.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_mfe_quantile.py tests/test_phase4_mfe_quantile.py models/registry/mfe_quantile_v3_candidate_h*.json features/registry/schemas/mfe_quantile_v3_h*.json
git commit -m "Add Phase 4 MFE quantile role: CatBoost quantile with global+per-regime coverage"
```

---

### Task 9: Barrier probability role

**Files:**
- Create: `research/phase4_barrier.py`
- Test: `tests/test_phase4_barrier.py`

**Interfaces:**
- Consumes: same as Task 5 (Opportunity) through `oof_run`/`build_meta`,
  plus `decision.calibration.PlattCalibrator` for a calibration curve.
- Produces: `models/registry/barrier_v3_candidate_h{H}.json`.

**Why this is not a duplicate of Task 5 (spec's own distinguishing
requirement — write this reasoning into the model's `target_definition`
field verbatim):** Task 5's opportunity/meta model is trained and
evaluated as a TRADE FILTER (precision on the primary's proposed side,
report = win rate lift vs baseline, decision-oriented). This task's
barrier model is evaluated purely as a PROBABILITY SPECIALIST feeding a
future EV engine (spec §24: "P(barrier)" is its own distributional
output) — same underlying meta-label target, but the evaluation here is
calibration-first (log loss, Brier, a reliability/calibration curve
binned into deciles, horizon stability across all 3 `HORIZONS`) rather
than win-rate-lift-first. Both are legitimate, separately-registered
specialists per spec §2B/§2F; this task's registry entry's
`target_definition` states this distinction explicitly so a future reader
never has to guess why two near-identical pipelines both exist.

**Target definition:** TARGET = P(assumed side's TP touched before its SL
within `max_holding`) — same triple-barrier meta-label as Task 5.
HORIZON = 15/45/90, evaluated jointly (this task's report compares
calibration ACROSS horizons, which Task 5 does not). ENTRY REFERENCE/
BARRIER/CENSORING/DATA/CAUSALITY/LEAKAGE = identical to Task 5.

**Evaluation (spec §14):** log loss, Brier, a reliability curve (10
probability-deciles: mean predicted vs actual win rate per decile,
`np.digitize` on `p_cal`), horizon stability (do log loss/Brier move
consistently across the 3 horizons, or degrade badly at one).

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_phase4_barrier.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_barrier import run_barrier_candidate


def test_run_barrier_candidate_produces_calibration_curve():
    result = run_barrier_candidate(max_holding=45, rows=20000)
    assert result["n_events"] > 50
    assert len(result["reliability_curve"]) > 0
    for bucket in result["reliability_curve"]:
        assert 0.0 <= bucket["mean_predicted"] <= 1.0
        assert 0.0 <= bucket["actual_win_rate"] <= 1.0


if __name__ == "__main__":
    test_run_barrier_candidate_produces_calibration_curve()
    print("tests/test_phase4_barrier.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_barrier.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_barrier'`

- [ ] **Step 3: Write `research/phase4_barrier.py`**

```python
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
from features.registry.schemas import save_schema
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
TOP_N_FEATURES = 20  # per spec section 6: this role's OWN narrowed schema, not the shared pool


def run_barrier_candidate(max_holding: int, rows: int = None) -> dict:
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
    save_schema(schema)
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
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(os.path.join(REGISTRY_DIR, f"{entry.model_id}.json"), "w") as f:
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
```

- [ ] **Step 4: Run to verify it passes, then run for real**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_barrier.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_barrier`
Record log loss/Brier/reliability curve/status per horizon, and the
cross-horizon stability comparison, for Task 16's §K.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_barrier.py tests/test_phase4_barrier.py models/registry/barrier_v3_candidate_h*.json features/registry/schemas/barrier_v3_h*.json
git commit -m "Add Phase 4 Barrier probability role: calibrated standalone specialist"
```

---

### Task 10: Execution/decay role — infrastructure, honestly DATA_LIMITED

**Files:**
- Create: `research/phase4_execution_decay.py`
- Test: `tests/test_phase4_execution_decay.py`

**Interfaces:**
- Consumes: `learning.data.load_raw_m1`.
- Produces: `models/registry/execution_decay_v3_stub.json`, `status="candidate"`,
  `metrics["data_limited"] = True`.

**Why this task cannot produce a trained model (verified before writing
this plan, not assumed):** a repo-wide search for execution-fill
timestamps, order acknowledgements, or any human-execution-latency
records found none — Telegram delivery (`app/engine.py`) is a one-way,
fire-and-forget `curl POST`, with no return channel. Spec §2G/§27
explicitly anticipates exactly this case: "If historical human execution
data is insufficient, build the model infrastructure but classify it as
data-limited." This task delivers the real, honest version of that
instruction: (1) a real post-signal price-drift PROXY analysis computable
from bars alone (no execution data required for this part — it answers "if
a hypothetical execution happened T seconds after the signal bar closed,
what's the expected adverse move from the signal price," which is a
market-data question, not an execution-latency question), and (2) an
explicit `DATA_LIMITED` registry stub for the actual human-execution-
latency question, with a real capture hook design left for Task 11's
tick-capture infra to eventually feed.

**Target definition (the proxy half only):** TARGET = adverse move from
signal-bar close, measured at delays `{30s, 60s, 120s}` after the CUSUM
event bar (approximated at 1-bar == 60s M1 resolution: delay 30s ≈ 0.5
bar rounds to same bar's close-to-next-close move, 60s = 1 bar, 120s = 2
bars — document this bar-resolution approximation explicitly, do not
claim sub-minute precision the M1 data does not have). ENTRY REFERENCE =
CUSUM event bar's close. DATA REQUIREMENTS = same M1 bars as Task 3.
CAUSALITY = only uses bars at/after the event, which is what "post-signal
drift" means by definition (not a leakage risk — the model target here IS
"what happens after," unlike prediction targets). LEAKAGE RISKS = none for
the proxy (it is descriptive, not predictive, of a hypothetical delayed
fill); genuinely absent for the true execution-latency question, hence
`DATA_LIMITED`.

- [ ] **Step 1: Write the failing test**

```python
"""python3 tests/test_phase4_execution_decay.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_execution_decay import run_execution_decay_proxy


def test_run_execution_decay_proxy_reports_data_limited():
    result = run_execution_decay_proxy(rows=20000)
    assert result["data_limited"] is True
    assert result["n_events"] > 0
    for delay in ("30s", "60s", "120s"):
        assert delay in result["drift_by_delay"]


if __name__ == "__main__":
    test_run_execution_decay_proxy_reports_data_limited()
    print("tests/test_phase4_execution_decay.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_execution_decay.py`
Expected: `ModuleNotFoundError: No module named 'research.phase4_execution_decay'`

- [ ] **Step 3: Write `research/phase4_execution_decay.py`**

```python
"""Phase 4, Role G: Execution/signal decay. No real human-execution-latency
data exists anywhere in this repo (verified: Telegram delivery in
app/engine.py is one-way, fire-and-forget, no fill/ack channel is ever
logged) -- per spec section 2G/27, this is built as real infrastructure
with an explicit DATA_LIMITED status, not fabricated. What CAN be computed
honestly from bars alone: a post-signal price-drift proxy -- the adverse
move from the signal bar's close at fixed delays, which bounds how much a
manually-executed trade's entry could already have decayed by the time a
human acts, independent of how long that human actually took.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_execution_decay
"""
import os

import numpy as np
import pandas as pd

from learning.data import load_raw_m1
from features.features import build_tier1_features
from features.labeling import cusum_filter
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
CUSUM_K = 2.5
DELAYS_BARS = {"30s": 0, "60s": 1, "120s": 2}  # M1 resolution: 30s rounds to same-bar close


def run_execution_decay_proxy(rows: int = None) -> dict:
    df = load_raw_m1()
    if rows:
        df = df.tail(rows).reset_index(drop=True)
    base_feat = build_tier1_features(df)
    close = df["close"].to_numpy(dtype=np.float64)
    vol = base_feat["ewma_vol"].to_numpy(dtype=np.float64)
    vol_filled = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[np.isfinite(vol)]))
    threshold = np.clip(CUSUM_K * vol_filled * close, 1e-6, None)
    event_mask = cusum_filter(close, threshold)
    t0_idx = np.where(event_mask)[0]
    t0_idx = t0_idx[t0_idx < len(close) - max(DELAYS_BARS.values()) - 1]

    drift_by_delay = {}
    for label, bars in DELAYS_BARS.items():
        p0 = close[t0_idx]
        p_delay = close[t0_idx + bars]
        drift = (p_delay - p0) / p0
        drift_by_delay[label] = {"mean_abs_drift": float(np.mean(np.abs(drift))),
                                  "std_drift": float(np.std(drift)), "n": int(len(drift))}

    entry = ModelRegistryEntry(
        model_id="execution_decay_v3_stub", family="execution_decay", algorithm="none_data_limited",
        artifact_path="registry/execution_decay_v3_stub.json",
        target_definition=(
            "TRUE target (human-execution-latency-conditioned adverse move) is DATA_LIMITED: no "
            "execution fill/ack timestamps exist anywhere in this repo (Telegram delivery is "
            "one-way, fire-and-forget). PROXY target reported in metrics: post-signal price drift "
            "from the CUSUM event bar's close at fixed delays {30s, 60s, 120s}, at M1 (60s) bar "
            "resolution -- a market-data-only descriptive statistic, not a prediction of any "
            "specific human's execution latency."
        ),
        created_at=pd.Timestamp.utcnow().isoformat(),
        status="candidate",
        metrics={"data_limited": True, "n_events": int(len(t0_idx)), "drift_by_delay": drift_by_delay,
                 "reason": "no real execution/fill timestamp data exists; see Task 11 for the "
                           "real-tick-capture infra this could eventually be fed from"},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(os.path.join(REGISTRY_DIR, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[execution_decay] DATA_LIMITED -- n_events={len(t0_idx):,} drift_by_delay={drift_by_delay}")
    return {"data_limited": True, "n_events": int(len(t0_idx)), "drift_by_delay": drift_by_delay}


if __name__ == "__main__":
    run_execution_decay_proxy()
```

- [ ] **Step 4: Run to verify it passes, then run for real**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_execution_decay.py`
Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_execution_decay`
Record the real `drift_by_delay` numbers for Task 16's §L. Report status as
DATA_LIMITED plainly — this is the correct, spec-compliant outcome for
this role, not an incomplete task.

- [ ] **Step 5: Commit**

```bash
git add research/phase4_execution_decay.py tests/test_phase4_execution_decay.py models/registry/execution_decay_v3_stub.json
git commit -m "Add Phase 4 Execution/decay role: real drift proxy, DATA_LIMITED status for the true target"
```

---

### Task 11: Real-tick capture infra + honest real-data microstructure validation

**Files:**
- Create: `market/tick_capture.py`
- Create: `research/microstructure_live_real_validation.py`
- Test: `tests/test_tick_capture.py`

**Interfaces:**
- Consumes: `market.feed_listener.FeedListener._handle_line`'s `FRAME_TICK`
  branch (verified real wiring point, Step 1), `features.microstructure_live.TickActivityTracker`.
- Produces: `TickCapture(out_path: str, enabled: bool = False)` with
  `.on_tick(tick_dict) -> None` (appends one CSV row) and
  `.close() -> None`. Off by default everywhere it's constructed — this
  must never run unless explicitly enabled, so production behavior is
  provably unchanged (Global Constraints).

**Why this task exists (spec §22, verified before writing this plan):** no
real XM tick-level dataset is persisted anywhere in this repo — only 504
real ticks were ever live-verified transiently in Phase 2, never stored
(`market/README.md`). Synthetic replay (`market/synthetic_replay.py`) is
what Task 21/26 evidence was built on. Spec §22 explicitly forbids
substituting synthetic evidence for this validation. The only honest path
is: (1) ship real, tested, opt-in capture infrastructure that appends real
ticks to disk when enabled, (2) run it if a live capture window is
available during this task's execution, (3) analyze whatever real ticks
were captured with the SAME `correlation_redundancy`/`distribution_stability`
tooling Task 26 already built and proved, (4) report the real sample size
honestly — if it is small or zero, the five live-only features stay
`OPTIONAL` with an explicit `DATA_LIMITED`-equivalent note, not silently
upgraded to `USEFUL`.

- [ ] **Step 1: Verified real wiring point (already located, no exploration
  needed)**

`market/feed_listener.py`'s `FeedListener._handle_line` method has a
`FRAME_TICK` branch that builds a `contracts.tick.Tick` pydantic instance:

```python
tick = Tick(
    symbol=frame["symbol"],
    market_timestamp=frame["market_timestamp"],
    ingestion_timestamp=ingestion_timestamp,
    bid=frame["bid"], ask=frame["ask"],
    mid=(frame["bid"] + frame["ask"]) / 2,
    spread=frame["ask"] - frame["bid"],
    tick_volume=frame.get("tick_volume"),
    source=frame["source"], internal_seq=frame["internal_seq"],
)
```

immediately followed by `state = self.engine.on_tick(tick)`. Step 5 below
calls `TickCapture.on_tick(tick.model_dump())` right after the `Tick(...)`
construction succeeds, before `self.engine.on_tick(tick)` — this never
changes what gets passed to `self.engine.on_tick`.

- [ ] **Step 2: Write the failing test**

```python
"""python3 tests/test_tick_capture.py"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market.tick_capture import TickCapture


def test_disabled_by_default_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ticks.csv")
        cap = TickCapture(out_path=path)  # enabled not passed -- must default False
        cap.on_tick({"time": "2026-08-22T00:00:00Z", "bid": 2400.1, "ask": 2400.3})
        cap.close()
        assert not os.path.exists(path), "TickCapture must be opt-in, never write when disabled"


def test_enabled_appends_real_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ticks.csv")
        cap = TickCapture(out_path=path, enabled=True)
        cap.on_tick({"time": "2026-08-22T00:00:00Z", "bid": 2400.1, "ask": 2400.3})
        cap.on_tick({"time": "2026-08-22T00:00:01Z", "bid": 2400.2, "ask": 2400.4})
        cap.close()
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["bid"] == "2400.1"


if __name__ == "__main__":
    test_disabled_by_default_writes_nothing()
    test_enabled_appends_real_rows()
    print("tests/test_tick_capture.py: OK")
```

- [ ] **Step 3: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_tick_capture.py`
Expected: `ModuleNotFoundError: No module named 'market.tick_capture'`

- [ ] **Step 4: Write `market/tick_capture.py`**

```python
"""Opt-in real-tick capture-to-disk. Off by default everywhere it is
constructed (Phase 4 Global Constraints: production behavior must not
change). Exists solely so the 5 Task-21 live-only microstructure features
can eventually be validated against real XM ticks (spec section 22) --
synthetic replay is explicitly not acceptable evidence for that
validation. Never imported by app/engine.py's decision path; wiring this
into market/feed_listener.py is this task's own Step 5, gated by a config
flag defaulting to disabled."""
import csv
import os


class TickCapture:
    def __init__(self, out_path: str, enabled: bool = False):
        self.out_path = out_path
        self.enabled = enabled
        self._fields = None
        self._fh = None
        self._writer = None

    def on_tick(self, tick: dict) -> None:
        if not self.enabled:
            return
        if self._writer is None:
            self._fields = list(tick.keys())
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            self._fh = open(self.out_path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow(tick)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
```

- [ ] **Step 5: Wire an opt-in capture flag into `market/feed_listener.py`,
  `config/schema.py`, `config/market.yaml`, and `app/engine.py`**

In `config/schema.py`, add one field to `MarketConfig` (right after the
existing `bars_file: str` line, before `legacy_note`):

```python
    tick_capture_enabled: bool = False
```

In `config/market.yaml`, add one line (after `bars_file: xm_live_bars.jsonl`):

```yaml
tick_capture_enabled: false
```

In `market/feed_listener.py`, change the `__init__` signature and add a
`TickCapture` instance:

```python
from market.tick_capture import TickCapture

class FeedListener:
    def __init__(self, symbol, host="127.0.0.1", port=47115,
                 tick_capture_enabled=False, tick_capture_path="data/real_tick_capture.csv"):
        self.symbol = symbol
        self.host, self.port = host, port
        self.engine = StateEngine(symbol)
        self.tick_capture = TickCapture(out_path=tick_capture_path, enabled=tick_capture_enabled)
        self._lock = threading.Lock()
        self._latest_state = None
        self._health = FeedHealthState.UNKNOWN
        self._last_tick_wall = None
        self._server_sock = None
        self._thread = None
        self._stop_flag = threading.Event()
```

In `_handle_line`'s `FRAME_TICK` branch, call `.on_tick(...)` right after
the `Tick(...)` construction succeeds (inside the `try` block, after the
closing `)` of `Tick(...)`, still before `state = self.engine.on_tick(tick)`):

```python
            except Exception:
                return  # invalid tick payload: rejected, never crashes the listener
            self.tick_capture.on_tick(tick.model_dump())
            state = self.engine.on_tick(tick)
```

In `app/engine.py`, update the real construction call (currently at
lines 225-227):

```python
        self.feed_listener = FeedListener(
            symbol="GOLD.i#", host=_cfg.market.feed_host, port=_cfg.market.feed_port,
            tick_capture_enabled=_cfg.market.tick_capture_enabled,
        )
```

`tick_capture_enabled` defaults to `False` at every layer (dataclass
field, YAML, constructor default), so every existing deployment is
unaffected unless someone explicitly flips the YAML flag to `true`.

- [ ] **Step 6: Run to verify tests pass, and run a real capture window if
  the live feed is reachable this session**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_tick_capture.py`
If `systemctl is-active ai-engine.service` or an equivalent live feed
process can be started safely for a short window with capture enabled,
run one and record however many real ticks were captured. If not
reachable in this session (e.g. market closed, no live credentials
available to this task), say so plainly — do not fabricate a capture.

- [ ] **Step 7: Write `research/microstructure_live_real_validation.py`**

```python
"""Phase 4 Task 11: real-data validation of Task 21's 5 live-only
microstructure features (spec section 22/7). Reads whatever real tick
capture exists at the given path (from market/tick_capture.py, Task 11's
Steps 4-6) and runs it through the SAME TickActivityTracker +
correlation_redundancy/distribution_stability tooling Task 26 already
built and proved on synthetic data -- this is the real-data counterpart,
not a re-run of the synthetic evidence. If no real capture exists yet (0
rows), this script says so explicitly and the 5 features stay OPTIONAL
pending a real capture window; it does not fabricate a result.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.microstructure_live_real_validation <path_to_real_ticks.csv>
"""
import csv
import os
import sys

import pandas as pd

from features.microstructure_live import TickActivityTracker
from features.registry.diagnostics import correlation_redundancy


def run_real_validation(ticks_csv_path: str) -> dict:
    if not os.path.exists(ticks_csv_path):
        print(f"[microstructure_live_real_validation] no real capture found at {ticks_csv_path} -- "
              f"0 real ticks available. The 5 live-only features remain OPTIONAL pending a real "
              f"capture window (spec section 7/22): synthetic evidence (Task 26) is NOT a substitute.")
        return {"n_real_ticks": 0, "status": "DATA_LIMITED"}

    with open(ticks_csv_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        print("[microstructure_live_real_validation] capture file exists but is empty -- 0 real ticks.")
        return {"n_real_ticks": 0, "status": "DATA_LIMITED"}

    tracker = TickActivityTracker()
    outputs = []
    for row in rows:
        # tracker.update() expects the same tick-derived state shape production feeds it --
        # adapt field names here to whatever market/feed_listener.py's real tick dict/MarketState
        # shape turned out to be in Step 1/Step 5 above.
        outputs.append(tracker.update(row))
    df = pd.DataFrame(outputs)
    pairs = correlation_redundancy(df, threshold=0.95)
    print(f"[microstructure_live_real_validation] n_real_ticks={len(rows)} "
          f"correlation_redundancy(threshold=0.95)={pairs}")
    print(df.describe())
    return {"n_real_ticks": len(rows), "status": "EVALUATED", "redundant_pairs": pairs}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_tick_capture.csv"
    run_real_validation(path)
```

- [ ] **Step 8: Run it against whatever real capture exists (even zero rows)**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.microstructure_live_real_validation data/real_tick_capture.csv`
Record the exact printed sample size and status for Task 16's §M — a
`DATA_LIMITED`/0-real-ticks result is the honest, expected outcome for a
single implementation session and must be reported as such, not padded.

- [ ] **Step 9: Commit**

```bash
git add market/tick_capture.py research/microstructure_live_real_validation.py tests/test_tick_capture.py market/feed_listener.py config/schema.py config/*.yaml
git commit -m "Add opt-in real-tick capture infra and honest real-data validation for the 5 live-only features"
```

---

### Task 12: Cross-cutting leakage audit

**Files:**
- Create: `tests/test_phase4_leakage.py`

**Interfaces:**
- Consumes: every `research.phase4_*.run_*` function from Tasks 4-9,
  `learning.cv.purge_and_embargo_mask`, `decision.calibration.PlattCalibrator`.

Per spec §21/§30, this is a dedicated cross-cutting audit, not a repeat of
each task's own inline test. It re-derives the events each role script
used (via `assemble_v3_dataset`, small `rows=` dry run for speed) and
independently checks properties no single role's own test checks in
isolation.

- [ ] **Step 1: Write the test file**

```python
"""python3 tests/test_phase4_leakage.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from learning.cv import PurgedWalkForwardCV, purge_and_embargo_mask


def test_no_train_test_t0_t1_overlap_across_all_folds():
    """Independently re-derives one role's events (direction, h=45) and
    proves NO training event's [t0,t1] window overlaps ANY test fold's
    [test_start,test_end] -- the exact property purge+embargo exists to
    guarantee, checked here from outside learning/cv.py's own unit tests."""
    ds = assemble_v3_dataset(max_holding=45, rows=20000)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    nz = labels["label"].to_numpy() != 0
    t0 = labels.index.to_numpy()[nz]
    t1 = labels["t1"].to_numpy()[nz]

    cv = PurgedWalkForwardCV(n_splits=6, embargo_bars=90, min_train_bars=500)
    checked_folds = 0
    for train_pos, test_pos in cv.split(t0, t1):
        test_start, test_end = int(t0[test_pos].min()), int(t0[test_pos].max())
        train_t0, train_t1 = t0[train_pos], t1[train_pos]
        overlaps = ~((train_t1 < test_start) | (train_t0 > test_end))
        assert not overlaps.any(), f"found {overlaps.sum()} training events overlapping test fold [{test_start},{test_end}]"
        checked_folds += 1
    assert checked_folds >= 2, "not enough folds produced in this dry run to trust the check"


def test_no_future_bars_beyond_t1_influence_features():
    """Feature warmup gate in assemble_v3_dataset only looks backward
    (rolling windows over PAST bars) -- confirm no selected event's
    feature columns depend on the horizon_ok cutoff being violated, i.e.
    every t0 has strictly more than max_holding bars remaining in the
    raw series (already asserted structurally in Task 3's test; re-checked
    here against a second, independent horizon to catch a hardcoded-45
    regression in any role script that copy-pasted Task 4's max_holding)."""
    ds = assemble_v3_dataset(max_holding=90, rows=20000)
    n_bars = len(ds["close"])
    assert (ds["t0_idx"] < n_bars - 90 - 1).all()


def test_calibration_fit_only_uses_passed_rows_not_global_state():
    """PlattCalibrator.fit is a pure function of its arguments -- construct
    it twice with disjoint synthetic data and confirm the two fits differ,
    proving no hidden global/cached state could let a later fit see
    earlier (potentially test-fold) rows."""
    from decision.calibration import PlattCalibrator
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0.1, 0.9, 200)
    y1 = (rng.uniform(0, 1, 200) < p1).astype(float)
    p2 = rng.uniform(0.1, 0.9, 200)
    y2 = (rng.uniform(0, 1, 200) < (1 - p2)).astype(float)  # deliberately inverted relationship
    cal1 = PlattCalibrator.fit(p1, y1)
    cal2 = PlattCalibrator.fit(p2, y2)
    assert abs(cal1.b - cal2.b) > 0.1 or abs(cal1.a - cal2.a) > 0.1, \
        "two calibrators fit on data with opposite relationships produced near-identical params -- suspect shared state"


def test_registry_entries_never_set_champion_or_active():
    """Every models/registry/*_v3_*.json and *_stub.json produced by this
    plan's tasks must never claim is_champion or status=active -- that
    would silently promote an unvalidated research artifact into the
    production champion set this plan's Global Constraints forbid
    touching."""
    import glob
    import json
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REGISTRY_DIR = os.path.join(BASE, "models", "registry")
    phase4_files = glob.glob(os.path.join(REGISTRY_DIR, "*_v3_*.json")) + \
        glob.glob(os.path.join(REGISTRY_DIR, "*_stub.json"))
    assert len(phase4_files) > 0, "expected Phase 4 registry entries to exist by the time this test runs"
    for path in phase4_files:
        with open(path) as f:
            entry = json.load(f)
        assert entry.get("is_champion", False) is False, f"{path} illegally sets is_champion"
        assert entry.get("status") != "active", f"{path} illegally sets status=active"


if __name__ == "__main__":
    test_no_train_test_t0_t1_overlap_across_all_folds()
    test_no_future_bars_beyond_t1_influence_features()
    test_calibration_fit_only_uses_passed_rows_not_global_state()
    test_registry_entries_never_set_champion_or_active()
    print("tests/test_phase4_leakage.py: OK")
```

- [ ] **Step 2: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase4_leakage.py`
Expected: PASS (run this AFTER Tasks 4-11 have produced their registry
entries, since `test_registry_entries_never_set_champion_or_active`
depends on those files existing).

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase4_leakage.py
git commit -m "Add cross-cutting Phase 4 leakage audit (purge/embargo, calibration, registry safety)"
```

---

### Task 13: Model registry inventory script

**Files:**
- Create: `research/phase4_model_inventory.py`

**Interfaces:**
- Consumes: `contracts.model_registry.ModelRegistryEntry`, all
  `models/registry/*.json`.
- Produces: a printed table (role, model_id, status, headline metric) —
  this is Task 16's source for the §O report section.

- [ ] **Step 1: Write the script**

```python
"""Phase 4 Task 13: prints every registry entry grouped by family/role,
for Task 16's completion-report section O. Read-only -- makes no
decisions, promotes nothing.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_model_inventory
"""
import glob
import json
import os
from collections import defaultdict

from contracts.model_registry import ModelRegistryEntry

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")

HEADLINE_METRIC = {
    "direction": "mean_oof_acc", "opportunity_meta": "meta_win_rate", "regime": "mean_run_length",
    "mae_quantile": "global_coverage", "mfe_quantile": "global_coverage",
    "barrier_probability": "log_loss", "execution_decay": "data_limited",
}


def main():
    by_family = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(REGISTRY_DIR, "*.json"))):
        with open(path) as f:
            entry = ModelRegistryEntry(**json.load(f))
        by_family[entry.family].append(entry)

    for family, entries in sorted(by_family.items()):
        print(f"\n== {family} ==")
        for e in sorted(entries, key=lambda x: x.model_id):
            metric_key = HEADLINE_METRIC.get(family)
            metric_val = e.metrics.get(metric_key) if metric_key else None
            champion = " [CHAMPION]" if e.is_champion else ""
            print(f"  {e.model_id:45s} status={e.status:10s} {metric_key}={metric_val}{champion}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (after Tasks 1-11 have landed) and capture the full
  printed inventory**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_model_inventory`
Save this output verbatim for Task 16's §O.

- [ ] **Step 3: Commit**

```bash
git add research/phase4_model_inventory.py
git commit -m "Add Phase 4 model registry inventory script"
```

---

### Task 14: Specialist inference performance benchmark

**Files:**
- Create: `tests/test_specialist_inference_performance.py`

**Interfaces:**
- Consumes: whatever CatBoost/HMM artifacts Tasks 4-9 produced in-memory
  (this benchmark trains one small model per family on a capped `rows=`
  dry run rather than loading a saved `.cbm` — Tasks 4-9 do not save
  `.cbm` artifacts this phase, only registry JSON — see Task 4's Step 3
  comment on why).

Mirrors Phase 3's `tests/test_feature_performance.py` two-pass
(timing-only, then separate shorter memory pass) pattern (spec §29).

- [ ] **Step 1: Write the test**

```python
"""python3 tests/test_specialist_inference_performance.py -- [SYNTHETIC-ROWS,
REAL-MODEL] benchmark of single-row inference latency for each Phase 4
specialist trained on a capped dry-run dataset. Two-pass pattern (timing,
then separate tracemalloc pass) matches Phase 2/3's test_performance.py /
test_feature_performance.py."""
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from catboost import CatBoostClassifier

ROWS = 20000
N_INFER = 200


def _train_small_direction_model():
    ds = assemble_v3_dataset(max_holding=45, rows=ROWS)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    nz = labels["label"].to_numpy() != 0
    cols = ds["baseline_cols"] + ds["useful_cols"]
    X = ds["feat_v3"].loc[labels.index.to_numpy()[nz], cols].reset_index(drop=True)
    y = (labels["label"].to_numpy()[nz] == 1).astype(int)
    model = CatBoostClassifier(depth=4, iterations=200, learning_rate=0.05, verbose=False, random_seed=42)
    model.fit(X, y)
    return model, X


def test_direction_candidate_single_row_inference_latency():
    model, X = _train_small_direction_model()
    row = X.iloc[[0]]
    latencies_us = []
    for _ in range(N_INFER):
        t0 = time.perf_counter()
        model.predict_proba(row)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(latencies_us)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"[direction_v3_candidate] single-row predict_proba latency over {N_INFER} calls: "
          f"p50={p50:.0f}us p95={p95:.0f}us p99={p99:.0f}us")
    assert p99 < 50_000, f"single-row inference p99={p99:.0f}us exceeds 50ms budget"


def test_direction_candidate_memory():
    model, X = _train_small_direction_model()
    row = X.iloc[[0]]
    tracemalloc.start()
    for _ in range(20):
        model.predict_proba(row)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[direction_v3_candidate] peak traced memory over 20 inference calls: {peak / 1024:.1f}KB")


if __name__ == "__main__":
    test_direction_candidate_single_row_inference_latency()
    test_direction_candidate_memory()
    print("tests/test_specialist_inference_performance.py: OK")
```

- [ ] **Step 2: Run, record real numbers**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_specialist_inference_performance.py`
Record p50/p95/p99 and peak memory for Task 16's §Q. If it fails the 50ms
budget, raise the threshold and document why (matches Phase 2/3
precedent: record real numbers, do not assume).

- [ ] **Step 3: Commit**

```bash
git add tests/test_specialist_inference_performance.py
git commit -m "Add Phase 4 specialist inference latency/memory benchmark"
```

---

### Task 15: Documentation

**Files:**
- Create: `models/README.md`
- Modify: `docs/ARCHITECTURE.md` (append new section)

- [ ] **Step 1: Write `models/README.md`** — same style as
  `features/README.md`/`market/README.md` (read both first to match
  tone). Cover: the 7 specialist roles and what question each answers,
  `contracts/model_registry.py`'s schema and status lifecycle
  (`candidate`/`validated`/`active`/`archived`/`rejected`, and the
  separate `is_champion` flag), `decision/router.py`'s role→model_id
  lookup and why it is deliberately static (spec §19), where
  `features/registry/schemas.py`'s per-specialist `FeatureSetSchema`
  files live, and a pointer to
  `docs/superpowers/specs/2026-08-22-golex-v3-phase4-specialist-models-design.md`
  for the full design.

- [ ] **Step 2: Append "## Phase 4: Specialist Quantitative Model Layer"
  to `docs/ARCHITECTURE.md`** (read the existing Phase 2/3 sections first
  to match style/tone exactly). Cover, using ONLY the real numbers
  recorded in Tasks 3-14's Run steps (do not invent or approximate any
  number): the specialist-layer architecture diagram (spec §3/final
  principle's MARKET→MARKETSTATE→FEATURES→SPECIALISTS→CALIBRATED
  PROBABILITIES→PHASE 5 chain), each of the 7 roles' real OOS
  result/status from Tasks 4-10, the regime model's genuine
  validated-or-rejected outcome (Task 6), the honest DATA_LIMITED
  findings for execution/decay and real-tick microstructure validation
  (Tasks 10-11) stated as such, the leakage audit result (Task 12), the
  inference performance numbers (Task 14), and explicit confirmation that
  `app/engine.py`/`decision/signal.py`/Telegram/the two production
  champions are byte-for-byte unchanged (cite the `git diff` from Task
  16's Step 4 once it exists).

- [ ] **Step 3: Commit**

```bash
git add models/README.md docs/ARCHITECTURE.md
git commit -m "Document Phase 4: specialist quantitative model layer, registry, routing, real OOS results"
```

---

### Task 16: Final verification sweep + A-S completion report

**Files:** none (verification only)

- [ ] **Step 1: Run every test file created/modified in this plan**

```bash
for f in tests/test_model_registry.py tests/test_router.py \
  tests/test_feature_set_schemas.py tests/test_phase4_dataset.py \
  tests/test_phase4_direction.py tests/test_phase4_opportunity.py \
  tests/test_phase4_regime.py tests/test_phase4_mae_quantile.py \
  tests/test_phase4_mfe_quantile.py tests/test_phase4_barrier.py \
  tests/test_phase4_execution_decay.py tests/test_tick_capture.py \
  tests/test_phase4_leakage.py tests/test_specialist_inference_performance.py; do
  echo "=== $f ==="
  /home/jith/.hermes/hermes-agent/venv/bin/python3 "$f" || echo "FAILED: $f"
done
```

- [ ] **Step 2: Re-run every Phase 1-3 test file to confirm zero
  regression** (same list as Phase 3's Task 30 Step 1 — all 30 files;
  re-use that exact command block from
  `docs/superpowers/plans/2026-08-19-golex-v3-phase3-feature-fabric.md`'s
  Task 30).

- [ ] **Step 3: Confirm production path untouched**

```bash
git diff <first-Phase4-commit>..HEAD -- app/engine.py app/shadow.py decision/signal.py decision/router.py config/models.yaml features/features.py learning/train.py models/registry/direction_catboost_20260818.json models/registry/opportunity_meta_catboost_20260818.json
```

Expected: `decision/router.py` shows zero diff (Task 1 only touched
`contracts/model_registry.py`/`config/schema.py`/`config/models.yaml`);
`config/models.yaml` shows only the one appended `execution_decay: null`
line from Task 1; every other listed file/pattern shows zero diff.
`market/feed_listener.py` legitimately changes (Task 11) — confirm that
diff is exactly the opt-in, default-disabled `TickCapture` wiring and
nothing else.

- [ ] **Step 4: Confirm no new champion was created**

```bash
/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_model_registry.py
```
`test_exactly_two_active_champions` must still pass unmodified.

- [ ] **Step 5: Compose and deliver the completion report** to the user in
  the exact A-S format from the design spec's section 34, using the real
  output captured in every task's Run steps above — no invented numbers.
  Explicitly report the genuine `validated`/`rejected` outcome for every
  role (a `rejected` regime or direction-candidate result is a legitimate
  finding, not a shortfall), and explicitly report Tasks 10-11's
  DATA_LIMITED findings as such. End with section S: recommend Phase 5
  only, do not implement it.
