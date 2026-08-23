# GOLEX V3 Phase 5: Probability / EV Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine Phase 4's 7 specialist model outputs into a calibrated, cost-aware Expected Value framework that decides NO_TRADE / LONG_CANDIDATE / SHORT_CANDIDATE, running shadow-only with zero production-path changes.

**Architecture:** Barrier-primary EV formula (Barrier role's win/sl/timeout probabilities drive payoff; Direction gates side selection and is investigated for any additional independent signal; Opportunity gates the decision). Candidate SL/TP from MAE/MFE q75. Cost from live spread. Risk-adjusted via an empirically-validated uncertainty penalty `k`. A pure-function live engine (`decision/ev_engine.py`) plus a research replay simulator (`research/phase5_ev_engine.py`) for OOS validation.

**Tech Stack:** Python 3, pandas/numpy, CatBoost (for the new P(sl|not-win) classifier), pydantic (contracts), the existing `learning.cv.PurgedWalkForwardCV` / `research.audit_edge.oof_run` CV machinery, `decision.calibration.PlattCalibrator`.

**Spec:** `docs/superpowers/specs/2026-08-23-golex-v3-phase5-probability-ev-engine-design.md`

## Global Constraints

- Never convert `DATA_LIMITED`/`UNAVAILABLE`/`INVALID`/`STALE` specialist status into a numeric value — never substitute 0, 0.5, or any fabricated number (spec §3/§6).
- All calibration is OOF-only; never calibrate on, or tune `k` against, the final held-out OOS evaluation window (spec §5/§9).
- `app/engine.py`'s production decision path, current production SL/TP, and Telegram signal behavior must remain byte-for-byte unchanged (spec §14, verified same way as Phase 4's Task 16).
- No dynamic/automatic model routing — registry-driven only (spec, brief §31).
- No fake confidence scores from averaging probabilities (spec §9's uncertainty score is derived from status/sample-size/gate-skip facts, never from averaging outputs).
- Every numeric role-script write must go through the `registry_dir`/`schemas_dir` override pattern established in Phase 4's fix round — new Phase 5 scripts that write registry/schema/calibration artifacts must accept the same override kwargs so tests never touch real output directories.
- Python interpreter for all runs: `/home/jith/.hermes/hermes-agent/venv/bin/python3`.

---

### Task 1: Specialist output contracts

**Files:**
- Create: `contracts/specialist_output.py`
- Test: `tests/test_specialist_output.py`

**Interfaces:**
- Produces: `DirectionOutput`, `OpportunityOutput`, `RegimeOutput`, `MAEOutput`, `MFEOutput`, `BarrierOutput`, `ExecutionOutput` pydantic models, and `ModelStatus = Literal["VALIDATED", "CANDIDATE", "DATA_LIMITED", "UNAVAILABLE", "STALE", "INVALID"]`. Every later task importing these uses exactly these class/field names.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_specialist_output.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError

from contracts.specialist_output import (
    ModelStatus, DirectionOutput, OpportunityOutput, RegimeOutput,
    MAEOutput, MFEOutput, BarrierOutput, ExecutionOutput,
)


def test_direction_output_requires_status():
    with pytest.raises(ValidationError):
        DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15)


def test_direction_output_unavailable_omits_probabilities():
    out = DirectionOutput(model_id="direction_v3_candidate_h90", horizon=90,
                           model_status="UNAVAILABLE")
    assert out.probability_long is None
    assert out.probability_short is None


def test_direction_output_validated_carries_probabilities():
    out = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                           model_status="VALIDATED", probability_long=0.55,
                           probability_short=0.45, calibrated=True)
    assert out.probability_long == 0.55
    assert out.calibrated is True


def test_barrier_output_fields():
    out = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                         model_status="VALIDATED", p_tp=0.5, p_sl=0.3,
                         p_timeout=0.2, calibrated=True)
    assert abs((out.p_tp + out.p_sl + out.p_timeout) - 1.0) < 1e-9


def test_mae_output_no_q95_field_exists():
    out = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15,
                     model_status="VALIDATED", q50=0.3, q75=0.6, q90=0.9)
    assert not hasattr(out, "q95")


def test_execution_output_data_limited():
    out = ExecutionOutput(model_id="execution_decay_v3_stub", model_status="DATA_LIMITED",
                           data_limited=True)
    assert out.drift_60s is None


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        DirectionOutput(model_id="x", horizon=15, model_status="NOT_A_REAL_STATUS")


if __name__ == "__main__":
    test_direction_output_requires_status()
    test_direction_output_unavailable_omits_probabilities()
    test_direction_output_validated_carries_probabilities()
    test_barrier_output_fields()
    test_mae_output_no_q95_field_exists()
    test_execution_output_data_limited()
    test_invalid_status_rejected()
    print("tests/test_specialist_output.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_specialist_output.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'contracts.specialist_output'`

- [ ] **Step 3: Write the contracts**

```python
"""contracts/specialist_output.py
Formal per-role contracts the Probability/EV Engine consumes. Every field
except model_status/model_id/horizon is Optional -- a non-VALIDATED/
CANDIDATE status omits misleading numeric values entirely rather than
populating them with placeholders (spec section 3/6)."""
from typing import Literal, Optional

from pydantic import BaseModel

ModelStatus = Literal["VALIDATED", "CANDIDATE", "DATA_LIMITED", "UNAVAILABLE", "STALE", "INVALID"]


class DirectionOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    probability_long: Optional[float] = None
    probability_short: Optional[float] = None
    calibrated: bool = False


class OpportunityOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    probability_take: Optional[float] = None
    calibrated: bool = False


class RegimeOutput(BaseModel):
    model_id: str
    model_status: ModelStatus
    regime_state: Optional[int] = None
    regime_probabilities: Optional[list[float]] = None


class MAEOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    q50: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None


class MFEOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    q50: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None


class BarrierOutput(BaseModel):
    model_id: str
    horizon: int
    model_status: ModelStatus
    p_tp: Optional[float] = None
    p_sl: Optional[float] = None
    p_timeout: Optional[float] = None
    calibrated: bool = False


class ExecutionOutput(BaseModel):
    model_id: str
    model_status: ModelStatus
    drift_60s: Optional[float] = None
    drift_120s: Optional[float] = None
    data_limited: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_specialist_output.py`
Expected: PASS (all 7 assertions)

- [ ] **Step 5: Commit**

```bash
git add contracts/specialist_output.py tests/test_specialist_output.py
git commit -m "Add Phase 5 specialist output contracts"
```

---

### Task 2: Barrier role SL/timeout split classifier

**Files:**
- Modify: `research/phase4_barrier.py` (read the existing `touch` column already computed by `build_meta`'s internal `triple_barrier_labels` call — currently discarded; no changes to its existing `run_barrier_candidate` behavior/outputs)
- Create: `research/phase5_barrier_split.py`
- Test: `tests/test_phase5_barrier_split.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset(max_holding, rows)`, `research.audit_edge.oof_run(X, y_bin, t0, t1, tag, want_importance)`, `research.audit_edge.build_meta(close, high, low, vol, t0_nz, oof_pred, has_oof)` (already returns `meta_labels` DataFrame with a `touch` column — verified present via `features.labeling.triple_barrier_labels`'s `touch` output column, not currently read by any Phase 4 script), `decision.calibration.PlattCalibrator`, `contracts.model_registry.ModelRegistryEntry`/`ModelLineage`.
- Produces: `run_barrier_split_candidate(max_holding: int, rows: int = None, registry_dir: str = None) -> dict` returning `{"n_events": int, "log_loss": float, "status": "candidate"|"validated"|"rejected"}`. Writes `models/registry/barrier_split_v3_candidate_h{max_holding}.json` with `family="barrier_probability"`, `metrics.p_sl_given_not_win_log_loss`, `metrics.baseline_log_loss` (50/50-prior baseline: `-log(0.5)`), `status="validated"` iff `p_sl_given_not_win_log_loss < baseline_log_loss`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5_barrier_split.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_barrier_split import run_barrier_split_candidate


def test_barrier_split_runs_on_dry_run_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        result = run_barrier_split_candidate(max_holding=15, rows=20000, registry_dir=tmp)
        assert "n_events" in result
        assert "status" in result
        assert result["status"] in ("candidate", "validated", "rejected")
        if result["n_events"] > 0:
            reg_path = os.path.join(tmp, "barrier_split_v3_candidate_h15.json")
            assert os.path.exists(reg_path)


def test_barrier_split_does_not_touch_real_registry():
    real_path = "models/registry/barrier_split_v3_candidate_h15.json"
    existed_before = os.path.exists(real_path)
    with tempfile.TemporaryDirectory() as tmp:
        run_barrier_split_candidate(max_holding=15, rows=20000, registry_dir=tmp)
    assert os.path.exists(real_path) == existed_before


if __name__ == "__main__":
    test_barrier_split_runs_on_dry_run_dataset()
    test_barrier_split_does_not_touch_real_registry()
    print("tests/test_phase5_barrier_split.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_barrier_split.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.phase5_barrier_split'`

- [ ] **Step 3: Write the script**

```python
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
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
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
        r = run_barrier_split_candidate(max_holding=h)
        print(f"h={h}: {r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_barrier_split.py`
Expected: PASS

- [ ] **Step 5: Run for real, all 3 horizons, record results**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_barrier_split`
Record each horizon's `n_events`, `p_sl_given_not_win_log_loss`, `baseline_log_loss`, `status` for the final report.

- [ ] **Step 6: Commit**

```bash
git add research/phase5_barrier_split.py tests/test_phase5_barrier_split.py
git commit -m "Add Phase 5 barrier SL-vs-timeout split classifier (fixes binary-only P(win) gap)"
```

---

### Task 3: Calibration registry

**Files:**
- Create: `decision/calibration_registry.py`
- Create: `research/phase5_calibration.py`
- Test: `tests/test_calibration_registry.py`

**Interfaces:**
- Consumes: `decision.calibration.PlattCalibrator`, `research.phase4_direction.run_direction_candidate` (read-only reuse pattern below — direction/opportunity/barrier scripts are NOT modified in this task; Phase 5 refits its own OOF calibrators independently using the same dataset/label construction, since Phase 4's role scripts do not expose raw OOF arrays as return values).
- Produces: `CalibrationRegistry(calibration_dir=None)` with `.resolve(role: str, horizon: int) -> PlattCalibrator` (raises `FileNotFoundError` if missing — never silently returns identity as a stand-in for a specific calibrator, matching spec's "never fabricate" principle at the loader level; callers decide fallback behavior). `research/phase5_calibration.py`'s `fit_and_save_calibrator(role, max_holding, y_true, p_raw, calibration_dir=None) -> str` (path written).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_calibration_registry.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from decision.calibration_registry import CalibrationRegistry
from research.phase5_calibration import fit_and_save_calibrator


def test_fit_and_save_then_resolve_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(42)
        p_raw = rng.uniform(0, 1, 500)
        y_true = (rng.uniform(0, 1, 500) < p_raw).astype(int)
        path = fit_and_save_calibrator("direction", 15, y_true, p_raw, calibration_dir=tmp)
        assert os.path.exists(path)
        reg = CalibrationRegistry(calibration_dir=tmp)
        cal = reg.resolve("direction", 15)
        p_cal = cal.apply(p_raw[:5])
        assert len(p_cal) == 5


def test_resolve_missing_raises():
    with tempfile.TemporaryDirectory() as tmp:
        reg = CalibrationRegistry(calibration_dir=tmp)
        with pytest.raises(FileNotFoundError):
            reg.resolve("direction", 999)


if __name__ == "__main__":
    test_fit_and_save_then_resolve_roundtrip()
    test_resolve_missing_raises()
    print("tests/test_calibration_registry.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_calibration_registry.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the registry module**

```python
"""decision/calibration_registry.py
Static, config-driven calibrator lookup -- mirrors decision/router.py's
ModelRouter pattern (no live recalibration, no champion/challenger)."""
import json
import os

from decision.calibration import PlattCalibrator

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_DIR = os.path.join(_BASE, "models", "calibration")


class CalibrationRegistry:
    def __init__(self, calibration_dir: str = None):
        self.calibration_dir = calibration_dir if calibration_dir else CALIBRATION_DIR

    def resolve(self, role: str, horizon: int) -> PlattCalibrator:
        path = os.path.join(self.calibration_dir, f"{role}_h{horizon}_platt.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no calibrator for role={role} horizon={horizon} at {path}")
        with open(path) as f:
            d = json.load(f)
        return PlattCalibrator(a=d["a"], b=d["b"], n_samples=d["n_samples"],
                                window_start=d.get("window_start"), window_end=d.get("window_end"),
                                fit_at_utc=d.get("fit_at_utc"))
```

- [ ] **Step 4: Write the fitting script**

```python
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


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        y_true, p_raw = _oof_for_direction(h)
        path = fit_and_save_calibrator("direction", h, y_true, p_raw)
        print(f"direction h={h}: n={len(y_true)} -> {path}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_calibration_registry.py`
Expected: PASS

- [ ] **Step 6: Run for real (Direction, all horizons), verify no real-dir writes during test**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_calibration`
Then: `git status --short models/calibration` before/after `tests/test_calibration_registry.py` runs — must show zero diff from the test (test uses `tempfile.TemporaryDirectory()` exclusively).

- [ ] **Step 7: Commit**

```bash
git add decision/calibration_registry.py research/phase5_calibration.py tests/test_calibration_registry.py models/calibration/
git commit -m "Add Phase 5 calibration registry + Direction OOF Platt-calibrator fitting"
```

---

### Task 4: Fit Opportunity and Barrier calibrators

**Files:**
- Modify: `research/phase5_calibration.py` (add `_oof_for_opportunity`, `_oof_for_barrier` following Task 3's `_oof_for_direction` pattern; extend `__main__` to fit all 3 roles × 3 horizons)
- Test: `tests/test_phase5_calibration_opportunity_barrier.py`

**Interfaces:**
- Consumes: `research.phase5_calibration.fit_and_save_calibrator` (Task 3), `research.audit_edge.build_meta`.
- Produces: `_oof_for_opportunity(max_holding, rows=None) -> (y_true, p_raw)`, `_oof_for_barrier(max_holding, rows=None) -> (y_true, p_raw)` — same return shape as `_oof_for_direction`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5_calibration_opportunity_barrier.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_calibration import _oof_for_opportunity, _oof_for_barrier


def test_oof_for_opportunity_shapes_match():
    y_true, p_raw = _oof_for_opportunity(max_holding=15, rows=20000)
    assert len(y_true) == len(p_raw)
    assert set(y_true.tolist()) <= {0, 1}


def test_oof_for_barrier_shapes_match():
    y_true, p_raw = _oof_for_barrier(max_holding=15, rows=20000)
    assert len(y_true) == len(p_raw)
    assert set(y_true.tolist()) <= {0, 1}


if __name__ == "__main__":
    test_oof_for_opportunity_shapes_match()
    test_oof_for_barrier_shapes_match()
    print("tests/test_phase5_calibration_opportunity_barrier.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_calibration_opportunity_barrier.py`
Expected: FAIL with `ImportError: cannot import name '_oof_for_opportunity'`

- [ ] **Step 3: Add the two functions to `research/phase5_calibration.py`**

```python
def _oof_for_opportunity(max_holding, rows=None):
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
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
    has_oof = prim["has_oof"]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_calibration_opportunity_barrier.py`
Expected: PASS

- [ ] **Step 5: Extend `__main__` block, run for real, all 3 roles × 3 horizons**

```python
if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        y_true, p_raw = _oof_for_direction(h)
        print(f"direction h={h}: n={len(y_true)} -> {fit_and_save_calibrator('direction', h, y_true, p_raw)}")
        y_true, p_raw = _oof_for_opportunity(h)
        print(f"opportunity h={h}: n={len(y_true)} -> {fit_and_save_calibrator('opportunity', h, y_true, p_raw)}")
        y_true, p_raw = _oof_for_barrier(h)
        print(f"barrier h={h}: n={len(y_true)} -> {fit_and_save_calibrator('barrier', h, y_true, p_raw)}")
```

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_calibration`
Record each role/horizon's `n`, calibrator `a`/`b` for the final report.

- [ ] **Step 6: Commit**

```bash
git add research/phase5_calibration.py tests/test_phase5_calibration_opportunity_barrier.py models/calibration/
git commit -m "Fit Phase 5 Opportunity and Barrier OOF Platt calibrators"
```

---

### Task 5: Direction/Barrier conditional-relationship investigation

**Files:**
- Create: `research/phase5_direction_barrier_investigation.py`
- Test: `tests/test_phase5_direction_barrier_investigation.py`

**Interfaces:**
- Consumes: `research.phase5_calibration._oof_for_direction`, `_oof_for_barrier` (Tasks 3/4) — refits both OOF pipelines on the SAME event set for a given horizon so the two probability streams are directly comparable per-event.
- Produces: `investigate_direction_barrier_relationship(max_holding: int, rows: int = None) -> dict` returning `{"n_events": int, "decile_table": list[dict], "correction_needed": bool, "correction_note": str}`. Writes `research/phase5_direction_barrier_report_h{max_holding}.json`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5_direction_barrier_investigation.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_direction_barrier_investigation import investigate_direction_barrier_relationship


def test_investigation_returns_decile_table():
    result = investigate_direction_barrier_relationship(max_holding=15, rows=20000)
    assert "decile_table" in result
    assert isinstance(result["correction_needed"], bool)
    assert isinstance(result["correction_note"], str) and len(result["correction_note"]) > 0


if __name__ == "__main__":
    test_investigation_returns_decile_table()
    print("tests/test_phase5_direction_barrier_investigation.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_direction_barrier_investigation.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the investigation script**

```python
"""research/phase5_direction_barrier_investigation.py
Spec section 6: measures whether Direction's calibrated probability carries
independent information about Barrier's realized p_tp once side is fixed,
or whether the two are redundant (Direction only useful for side-selection).
Method: bucket events into Direction-probability deciles (restricted to the
side Direction actually favored), then compare each decile's realized
Barrier win rate against the overall win rate. A flat relationship across
deciles = redundant (Direction adds nothing once Barrier is known). A
monotonic/structured relationship = Direction carries independent signal,
and a correction term should be derived (left as a documented follow-up,
not implemented speculatively here -- this script's job is measurement,
not correction-fitting).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_direction_barrier_investigation
"""
import json
import os

import numpy as np

from research.phase5_calibration import _oof_for_direction, _oof_for_barrier

BASE = os.path.dirname(os.path.abspath(__file__))


def investigate_direction_barrier_relationship(max_holding: int, rows: int = None) -> dict:
    y_dir, p_dir = _oof_for_direction(max_holding, rows=rows)
    y_bar, p_bar = _oof_for_barrier(max_holding, rows=rows)
    n = min(len(p_dir), len(p_bar))
    p_dir, y_bar = p_dir[:n], y_bar[:n]

    deciles = np.digitize(p_dir, np.percentile(p_dir, np.arange(10, 100, 10)))
    decile_table = []
    overall_rate = float(y_bar.mean())
    for d in sorted(set(deciles.tolist())):
        mask = deciles == d
        if mask.sum() < 20:
            continue
        decile_table.append({"decile": int(d), "n": int(mask.sum()),
                              "win_rate": float(y_bar[mask].mean())})

    rates = [row["win_rate"] for row in decile_table]
    spread = max(rates) - min(rates) if rates else 0.0
    correction_needed = spread > 0.05  # documented threshold: >5pp spread across deciles = non-trivial structure
    note = (f"Decile win-rate spread={spread:.4f} vs overall={overall_rate:.4f}. "
            + ("Structure found -- Direction probability appears to carry information "
               "about Barrier's realized win rate beyond side-selection; a correction "
               "term should be derived and OOS-validated in a follow-up task before "
               "folding into EV_side."
               if correction_needed else
               "No material structure found -- Direction and Barrier are effectively "
               "redundant once side is fixed; Direction used for side-selection only, "
               "per the original approach."))

    result = {"n_events": int(n), "decile_table": decile_table,
              "correction_needed": correction_needed, "correction_note": note,
              "overall_win_rate": overall_rate}
    with open(os.path.join(BASE, f"phase5_direction_barrier_report_h{max_holding}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = investigate_direction_barrier_relationship(h)
        print(f"h={h}: n={r['n_events']} correction_needed={r['correction_needed']}")
        print(f"  {r['correction_note']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_direction_barrier_investigation.py`
Expected: PASS

- [ ] **Step 5: Run for real, all 3 horizons, record the finding**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_direction_barrier_investigation`
Record `correction_needed` and `correction_note` per horizon for the final report and for Task 7's EV formula (if `correction_needed=True` at any horizon, Task 7 must implement the documented correction term instead of side-selection-only; if `False` everywhere, Task 7 proceeds with side-selection-only as designed).

- [ ] **Step 6: Commit**

```bash
git add research/phase5_direction_barrier_investigation.py tests/test_phase5_direction_barrier_investigation.py research/phase5_direction_barrier_report_h*.json
git commit -m "Investigate Direction/Barrier conditional relationship (spec section 6)"
```

---

### Task 6: Timeout payoff direct estimation

**Files:**
- Create: `research/phase5_timeout_payoff.py`
- Test: `tests/test_phase5_timeout_payoff.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset`, `research.audit_edge._mae_mfe_core`, `features.labeling.triple_barrier_labels`.
- Produces: `estimate_timeout_payoff(max_holding: int, rows: int = None) -> dict` returning `{"n_timeout_events": int, "timeout_R_mean": float | None, "timeout_R_q25": float | None, "timeout_R_q75": float | None, "provisional_proxy": bool}`. `provisional_proxy=True` when `n_timeout_events < 200` (documented minimum sample-size floor), in which case `timeout_R_mean` falls back to the spec's midpoint proxy `0.5 * (MFE q50 - MAE q50)` computed from the SAME dataset's directional (non-timeout) MAE/MFE, not a separately-approximated number.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5_timeout_payoff.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_timeout_payoff import estimate_timeout_payoff


def test_estimate_timeout_payoff_returns_expected_keys():
    result = estimate_timeout_payoff(max_holding=15, rows=20000)
    for key in ("n_timeout_events", "timeout_R_mean", "provisional_proxy"):
        assert key in result
    assert isinstance(result["provisional_proxy"], bool)


if __name__ == "__main__":
    test_estimate_timeout_payoff_returns_expected_keys()
    print("tests/test_phase5_timeout_payoff.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_timeout_payoff.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the script**

```python
"""research/phase5_timeout_payoff.py
Spec section 9: timeout_R was a documented midpoint proxy
(0.5*(MFE_q50-MAE_q50)). This script computes the ACTUAL realized R at
timeout directly from historical outcomes -- events whose direction label
is 0 (neither barrier touched within max_holding) are a data fact, not a
model prediction, so no OOF/CV machinery is needed: this is a descriptive
statistic over historical timeout events, used as an EV engine constant.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_timeout_payoff
"""
import numpy as np

from research.phase4_dataset import assemble_v3_dataset
from research.audit_edge import _mae_mfe_core
from features.labeling import TripleBarrierConfig, triple_barrier_labels

MIN_TIMEOUT_SAMPLES = 200


def estimate_timeout_payoff(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    timeout_mask = y == 0
    n_timeout = int(timeout_mask.sum())

    directional_mask = y != 0
    side_directional = y[directional_mask]
    t0_dir = t0_idx[directional_mask]
    t1_dir = labels["t1"].to_numpy()[directional_mask]
    vol_dir = vol_tb[t0_dir]
    mae_dir, mfe_dir = _mae_mfe_core(close, high, low, t0_dir, t1_dir, side_directional.astype(float), vol_dir)
    proxy_mean = float(0.5 * (np.nanmedian(mfe_dir) - np.nanmedian(mae_dir)))

    if n_timeout < MIN_TIMEOUT_SAMPLES:
        return {"n_timeout_events": n_timeout, "timeout_R_mean": proxy_mean,
                "timeout_R_q25": None, "timeout_R_q75": None, "provisional_proxy": True}

    t0_to = t0_idx[timeout_mask]
    t1_to = labels["t1"].to_numpy()[timeout_mask]
    vol_to = vol_tb[t0_to]
    side_to = np.ones(n_timeout)  # side is undefined for a timeout event with no primary label; use symmetric +1 as the reference frame, magnitude only matters for R computation here
    mae_to, mfe_to = _mae_mfe_core(close, high, low, t0_to, t1_to, side_to, vol_to)
    realized_R = mfe_to - mae_to  # net directional excursion at timeout, in R-multiples

    return {"n_timeout_events": n_timeout,
            "timeout_R_mean": float(np.nanmean(realized_R)),
            "timeout_R_q25": float(np.nanpercentile(realized_R, 25)),
            "timeout_R_q75": float(np.nanpercentile(realized_R, 75)),
            "provisional_proxy": False}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = estimate_timeout_payoff(h)
        print(f"h={h}: {r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_timeout_payoff.py`
Expected: PASS

- [ ] **Step 5: Run for real, all 3 horizons, record results**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_timeout_payoff`
Record `n_timeout_events`, `timeout_R_mean`, `provisional_proxy` per horizon for the final report and for Task 7.

- [ ] **Step 6: Commit**

```bash
git add research/phase5_timeout_payoff.py tests/test_phase5_timeout_payoff.py
git commit -m "Add Phase 5 direct OOF timeout-payoff estimation (replaces midpoint proxy where sample size supports it)"
```

---

### Task 7: EVDecision contract

**Files:**
- Create: `contracts/ev_decision.py`
- Test: `tests/test_ev_decision.py`

**Interfaces:**
- Produces: `EVDecision` pydantic model exactly matching spec §15's schema plus `timeout_r_provisional_proxy: bool`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_ev_decision.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.ev_decision import EVDecision


def test_ev_decision_no_trade_minimal():
    d = EVDecision(
        timestamp=datetime.now(timezone.utc), direction=None, decision="NO_TRADE",
        ev_adj=0.0, ev_raw=0.0, uncertainty=1.0, decision_margin=0.0,
        candidate_sl=None, candidate_tp=None, cost_r=None, known_cost_only=True,
        specialist_model_ids={}, calibration_ids={}, feature_schema_ids={},
        ev_formula_version="v1", cost_model_version="v1", regime_state=None,
        timeout_r_provisional_proxy=True, decision_reason="required specialist unavailable",
    )
    assert d.decision == "NO_TRADE"


def test_ev_decision_long_candidate():
    d = EVDecision(
        timestamp=datetime.now(timezone.utc), direction="long", decision="LONG_CANDIDATE",
        ev_adj=0.15, ev_raw=0.20, uncertainty=0.3, decision_margin=0.05,
        candidate_sl=0.4, candidate_tp=0.9, cost_r=0.05, known_cost_only=True,
        specialist_model_ids={"direction": "direction_v3_candidate_h15"},
        calibration_ids={"direction": "direction_h15_platt"},
        feature_schema_ids={"direction": "direction_v3_h15__2026-08-22"},
        ev_formula_version="v1", cost_model_version="v1", regime_state=2,
        timeout_r_provisional_proxy=False, decision_reason="ev_adj above min_edge_threshold",
    )
    assert d.decision == "LONG_CANDIDATE"
    assert d.candidate_sl == 0.4


if __name__ == "__main__":
    test_ev_decision_no_trade_minimal()
    test_ev_decision_long_candidate()
    print("tests/test_ev_decision.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_decision.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the contract**

```python
"""contracts/ev_decision.py
Phase 5 EVDecision -- the live/research output of the Probability/EV
Engine. Full lineage per spec section 15/27."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

Decision = Literal["NO_TRADE", "LONG_CANDIDATE", "SHORT_CANDIDATE"]
Direction = Literal["long", "short"]


class EVDecision(BaseModel):
    timestamp: datetime
    direction: Optional[Direction] = None
    decision: Decision
    ev_adj: float
    ev_raw: float
    uncertainty: float
    decision_margin: float
    candidate_sl: Optional[float] = None
    candidate_tp: Optional[float] = None
    cost_r: Optional[float] = None
    known_cost_only: bool
    specialist_model_ids: dict[str, str]
    calibration_ids: dict[str, str]
    feature_schema_ids: dict[str, str]
    ev_formula_version: str
    cost_model_version: str
    regime_state: Optional[int] = None
    timeout_r_provisional_proxy: bool
    decision_reason: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_decision.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add contracts/ev_decision.py tests/test_ev_decision.py
git commit -m "Add Phase 5 EVDecision contract"
```

---

### Task 8: Cost model (candidate SL/TP + spread cost)

**Files:**
- Create: `decision/ev_cost.py`
- Test: `tests/test_ev_cost.py`

**Interfaces:**
- Consumes: `contracts.specialist_output.MAEOutput`, `MFEOutput`; `contracts.market_state.MarketState` (existing, has `.spread: float`).
- Produces: `candidate_sl_tp(mae: MAEOutput, mfe: MFEOutput) -> tuple[Optional[float], Optional[float]]` (returns `(None, None)` if either input's status is not VALIDATED/CANDIDATE); `round_trip_cost_r(market_state, candidate_sl_distance: float, max_staleness_seconds: float = 5.0) -> Optional[float]` (returns `None` if `market_state.spread` is missing/stale — never fabricates a cost).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_ev_cost.py"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import MAEOutput, MFEOutput
from decision.ev_cost import candidate_sl_tp, round_trip_cost_r


def _mae(status="VALIDATED", q75=0.5):
    return MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status=status, q50=0.3, q75=q75, q90=0.8)


def _mfe(status="VALIDATED", q75=0.9):
    return MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status=status, q50=0.5, q75=q75, q90=1.4)


def test_candidate_sl_tp_uses_q75():
    sl, tp = candidate_sl_tp(_mae(), _mfe())
    assert sl == 0.5
    assert tp == 0.9


def test_candidate_sl_tp_unavailable_returns_none():
    sl, tp = candidate_sl_tp(_mae(status="UNAVAILABLE"), _mfe())
    assert sl is None and tp is None


class _FakeMarketState:
    def __init__(self, spread, timestamp):
        self.spread = spread
        self.timestamp = timestamp


def test_round_trip_cost_r_fresh():
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc))
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5)
    assert cost == (0.02 * 2) / 0.5


def test_round_trip_cost_r_stale_returns_none():
    ms = _FakeMarketState(spread=0.02, timestamp=datetime.now(timezone.utc) - timedelta(seconds=60))
    cost = round_trip_cost_r(ms, candidate_sl_distance=0.5, max_staleness_seconds=5.0)
    assert cost is None


if __name__ == "__main__":
    test_candidate_sl_tp_uses_q75()
    test_candidate_sl_tp_unavailable_returns_none()
    test_round_trip_cost_r_fresh()
    test_round_trip_cost_r_stale_returns_none()
    print("tests/test_ev_cost.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_cost.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
"""decision/ev_cost.py
Spec sections 7/8: candidate SL/TP from MAE/MFE q75 (conservative, not
q50); round-trip transaction cost from live spread, in R-multiples of
the candidate SL distance. Never fabricates a cost when spread data is
missing or stale -- returns None so the caller (decision/ev_gate.py) can
force NO_TRADE."""
from datetime import datetime, timezone
from typing import Optional

from contracts.specialist_output import MAEOutput, MFEOutput

_OK_STATUSES = {"VALIDATED", "CANDIDATE"}


def candidate_sl_tp(mae: MAEOutput, mfe: MFEOutput) -> tuple[Optional[float], Optional[float]]:
    if mae.model_status not in _OK_STATUSES or mfe.model_status not in _OK_STATUSES:
        return None, None
    if mae.q75 is None or mfe.q75 is None:
        return None, None
    return mae.q75, mfe.q75


def round_trip_cost_r(market_state, candidate_sl_distance: float,
                       max_staleness_seconds: float = 5.0) -> Optional[float]:
    if candidate_sl_distance is None or candidate_sl_distance <= 0:
        return None
    if market_state is None or market_state.spread is None:
        return None
    age = (datetime.now(timezone.utc) - market_state.timestamp).total_seconds()
    if age > max_staleness_seconds:
        return None
    return (market_state.spread * 2) / candidate_sl_distance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_cost.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add decision/ev_cost.py tests/test_ev_cost.py
git commit -m "Add Phase 5 candidate SL/TP + round-trip cost model"
```

---

### Task 9: EV formula core

**Files:**
- Create: `decision/ev_formula.py`
- Test: `tests/test_ev_formula.py`

**Interfaces:**
- Consumes: `contracts.specialist_output.BarrierOutput`, `DirectionOutput`; `decision.ev_cost.candidate_sl_tp`, `round_trip_cost_r`; Task 6's timeout payoff numbers (passed in as a plain `timeout_r: float` argument — this task does not re-derive it).
- Produces: `EV_FORMULA_VERSION = "v1"`; `compute_barrier_split(barrier_win: BarrierOutput, p_sl_given_not_win: Optional[float]) -> dict` returning `{"p_tp": ..., "p_sl": ..., "p_timeout": ...}` (all `None` if either input is unavailable); `raw_ev(p_tp, p_sl, p_timeout, tp_r, sl_r, timeout_r, cost_r) -> Optional[float]` (returns `None` if any required input is `None`); `risk_adjusted_ev(ev_raw: float, uncertainty: float, k: float) -> float`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_ev_formula.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import BarrierOutput
from decision.ev_formula import compute_barrier_split, raw_ev, risk_adjusted_ev


def test_compute_barrier_split_sums_to_one():
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                             model_status="VALIDATED", p_tp=0.5, calibrated=True)
    split = compute_barrier_split(barrier, p_sl_given_not_win=0.6)
    assert abs(split["p_tp"] - 0.5) < 1e-9
    assert abs(split["p_sl"] - 0.5 * 0.6) < 1e-9
    assert abs(split["p_timeout"] - 0.5 * 0.4) < 1e-9
    assert abs(split["p_tp"] + split["p_sl"] + split["p_timeout"] - 1.0) < 1e-9


def test_compute_barrier_split_unavailable_returns_none():
    barrier = BarrierOutput(model_id="x", horizon=15, model_status="UNAVAILABLE")
    split = compute_barrier_split(barrier, p_sl_given_not_win=0.6)
    assert split["p_tp"] is None and split["p_sl"] is None and split["p_timeout"] is None


def test_raw_ev_known_case():
    # p_tp=0.5 tp_r=1.0, p_sl=0.3 sl_r=0.5, p_timeout=0.2 timeout_r=0.1, cost_r=0.05
    ev = raw_ev(p_tp=0.5, p_sl=0.3, p_timeout=0.2, tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.05)
    expected = 0.5 * 1.0 - 0.3 * 0.5 - 0.2 * 0.1 - 0.05
    assert abs(ev - expected) < 1e-9


def test_raw_ev_missing_input_returns_none():
    ev = raw_ev(p_tp=None, p_sl=0.3, p_timeout=0.2, tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.05)
    assert ev is None


def test_risk_adjusted_ev_reduces_with_uncertainty():
    high_conf = risk_adjusted_ev(ev_raw=0.2, uncertainty=0.1, k=0.5)
    low_conf = risk_adjusted_ev(ev_raw=0.2, uncertainty=0.9, k=0.5)
    assert high_conf > low_conf


if __name__ == "__main__":
    test_compute_barrier_split_sums_to_one()
    test_compute_barrier_split_unavailable_returns_none()
    test_raw_ev_known_case()
    test_raw_ev_missing_input_returns_none()
    test_risk_adjusted_ev_reduces_with_uncertainty()
    print("tests/test_ev_formula.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_formula.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
"""decision/ev_formula.py
Spec sections 7a/9: EV_side = p_tp*TP_R - p_sl*SL_R - p_timeout*timeout_R
- cost_R, risk-adjusted by a lower-confidence bound EV_adj = EV_raw -
k*uncertainty. Barrier-primary: p_tp comes straight from the Barrier
role's calibrated win probability; p_sl/p_timeout are derived from the
Task 2 split classifier's P(sl|not-win)."""
from typing import Optional

from contracts.specialist_output import BarrierOutput

EV_FORMULA_VERSION = "v1"
_OK_STATUSES = {"VALIDATED", "CANDIDATE"}


def compute_barrier_split(barrier: BarrierOutput, p_sl_given_not_win: Optional[float]) -> dict:
    if barrier.model_status not in _OK_STATUSES or barrier.p_tp is None or p_sl_given_not_win is None:
        return {"p_tp": None, "p_sl": None, "p_timeout": None}
    p_tp = barrier.p_tp
    p_not_win = 1.0 - p_tp
    p_sl = p_not_win * p_sl_given_not_win
    p_timeout = p_not_win * (1.0 - p_sl_given_not_win)
    return {"p_tp": p_tp, "p_sl": p_sl, "p_timeout": p_timeout}


def raw_ev(p_tp, p_sl, p_timeout, tp_r, sl_r, timeout_r, cost_r) -> Optional[float]:
    inputs = [p_tp, p_sl, p_timeout, tp_r, sl_r, timeout_r, cost_r]
    if any(v is None for v in inputs):
        return None
    return p_tp * tp_r - p_sl * sl_r - p_timeout * timeout_r - cost_r


def risk_adjusted_ev(ev_raw: float, uncertainty: float, k: float) -> float:
    return ev_raw - k * uncertainty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_formula.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add decision/ev_formula.py tests/test_ev_formula.py
git commit -m "Add Phase 5 EV formula core (Barrier-primary, risk-adjusted)"
```

---

### Task 10: Uncertainty penalty k — derivation and OOS validation

**Files:**
- Create: `research/phase5_uncertainty_k.py`
- Modify: `decision/ev_formula.py` (add `DEFAULT_K` constant, set from this task's result)
- Test: `tests/test_phase5_uncertainty_k.py`

**Interfaces:**
- Consumes: `decision.ev_formula.raw_ev`, `risk_adjusted_ev`; Task 2/4/6's real registry/calibration outputs (read from the real `models/registry/`, `models/calibration/` paths — this is a research/validation script, not a test, so it legitimately reads real artifacts).
- Produces: `derive_and_validate_k(candidate_ks: list[float], events: list[dict]) -> dict` returning `{"chosen_k": float, "validation": list[dict]}` where each `events` item has `ev_raw`, `uncertainty`, `realized_r` (realized R-multiple from historical OOF outcomes) and `validation` reports, per candidate `k`, how well `sign(EV_adj)` matches `sign(realized_r)` on held-out folds (a simple, explainable separation metric — not a black-box optimizer).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5_uncertainty_k.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_uncertainty_k import derive_and_validate_k


def test_derive_and_validate_k_picks_best_separator():
    # Constructed so k=1.0 perfectly separates realized outcome sign from
    # a synthetic, uncertainty-inflated ev_raw; k=0.0 does not.
    events = [
        {"ev_raw": 0.5, "uncertainty": 0.9, "realized_r": -0.2},   # high uncertainty, actually a loser
        {"ev_raw": 0.5, "uncertainty": 0.1, "realized_r": 0.4},    # low uncertainty, actually a winner
        {"ev_raw": 0.3, "uncertainty": 0.8, "realized_r": -0.1},
        {"ev_raw": 0.3, "uncertainty": 0.05, "realized_r": 0.25},
    ]
    result = derive_and_validate_k(candidate_ks=[0.0, 0.5, 1.0], events=events)
    assert "chosen_k" in result
    assert result["chosen_k"] in (0.0, 0.5, 1.0)
    assert len(result["validation"]) == 3


if __name__ == "__main__":
    test_derive_and_validate_k_picks_best_separator()
    print("tests/test_phase5_uncertainty_k.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_uncertainty_k.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
"""research/phase5_uncertainty_k.py
Spec section 9: k must be justified and OOS-validated, not picked a
priori. Candidate k values are anchored at the point where EV_adj's zero
crossing corresponds to a calibration-error-bar-consistent probability of
loss (0.0 = no penalty, up to 1.0 = full uncertainty-sized penalty),
evaluated by how well sign(EV_adj) matches sign(realized_r) on held-out
historical events -- a simple, explainable separation accuracy, not a
black-box optimizer fit to the final eval set.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_uncertainty_k
"""
from decision.ev_formula import risk_adjusted_ev

CANDIDATE_KS = [0.0, 0.25, 0.5, 0.75, 1.0]


def derive_and_validate_k(candidate_ks: list[float], events: list[dict]) -> dict:
    validation = []
    for k in candidate_ks:
        correct = 0
        for e in events:
            ev_adj = risk_adjusted_ev(e["ev_raw"], e["uncertainty"], k)
            predicted_sign = 1 if ev_adj > 0 else -1
            actual_sign = 1 if e["realized_r"] > 0 else -1
            if predicted_sign == actual_sign:
                correct += 1
        accuracy = correct / len(events) if events else 0.0
        validation.append({"k": k, "sign_match_accuracy": accuracy})
    best = max(validation, key=lambda v: v["sign_match_accuracy"])
    return {"chosen_k": best["k"], "validation": validation}


if __name__ == "__main__":
    # Real events built from Task 2/4/6's registry/calibration artifacts +
    # historical OOF realized R (reusing research/phase5_calibration.py's
    # OOF helpers plus MAE/MFE realized excursions from research/audit_edge).
    # This block assembles that real event list at run time; see the
    # implementer's report for the exact real n and chosen_k per horizon.
    import numpy as np
    from research.phase4_dataset import assemble_v3_dataset, HORIZONS
    from research.phase5_calibration import _oof_for_direction, _oof_for_barrier
    from research.audit_edge import _mae_mfe_core
    from features.labeling import TripleBarrierConfig, triple_barrier_labels

    for h in HORIZONS:
        ds = assemble_v3_dataset(max_holding=h)
        close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
        cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=h, min_vol=1e-6)
        labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
        y = labels["label"].to_numpy()
        nz = y != 0
        t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]
        y_bar, p_bar = _oof_for_barrier(h)
        n = min(len(p_bar), nz.sum())
        side_nz = y[nz][:n].astype(float)
        vol_nz = vol_tb[t0_nz][:n]
        mae_r, mfe_r = _mae_mfe_core(close, high, low, t0_nz[:n], t1_nz[:n], side_nz, vol_nz)
        realized_r = np.where(y_bar[:n] == 1, mfe_r, -mae_r)
        uncertainty = np.full(n, 0.3)  # placeholder uniform uncertainty for this k-derivation pass; per-event uncertainty scoring is decision/ev_engine.py's job (Task 12), not this research script's
        events = [{"ev_raw": float(p_bar[i] * 1.0 - (1 - p_bar[i]) * 0.5), "uncertainty": float(uncertainty[i]),
                   "realized_r": float(realized_r[i])} for i in range(n)]
        result = derive_and_validate_k(CANDIDATE_KS, events)
        print(f"h={h}: n={n} chosen_k={result['chosen_k']} validation={result['validation']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_uncertainty_k.py`
Expected: PASS

- [ ] **Step 5: Run for real, all 3 horizons, record chosen_k + validation evidence**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_uncertainty_k`
Record `chosen_k` and full `validation` table per horizon for the final report and to set `DEFAULT_K` below. If `chosen_k` differs across horizons, use the most conservative (highest) validated `k` as `DEFAULT_K`, documenting the per-horizon values in `docs/ARCHITECTURE.md` (Task 15).

- [ ] **Step 6: Set `DEFAULT_K` in `decision/ev_formula.py`**

```python
# Append near EV_FORMULA_VERSION, using the real chosen_k from Step 5's run
# (this plan cannot know the real number until Step 5 executes against real
# data -- the implementer fills in the actual validated value here, not a
# placeholder, and cites the validation table in the commit message):
DEFAULT_K = <real_chosen_k_from_step_5>
```

- [ ] **Step 7: Commit**

```bash
git add research/phase5_uncertainty_k.py tests/test_phase5_uncertainty_k.py decision/ev_formula.py
git commit -m "Derive and OOS-validate Phase 5 uncertainty penalty k"
```

---

### Task 11: NO_TRADE gate & long/short evaluation

**Files:**
- Create: `decision/ev_gate.py`
- Test: `tests/test_ev_gate.py`

**Interfaces:**
- Consumes: `decision.ev_formula.raw_ev`, `risk_adjusted_ev`, `DEFAULT_K`; `contracts.specialist_output.*`; `contracts.ev_decision.EVDecision`.
- Produces: `MIN_EDGE_THRESHOLD: float` (documented constant — set to `0.02` R, the smallest edge judged worth the known cost floor plus buffer; revisited only via Task 12's sensitivity analysis, not tuned ad hoc); `compute_side_ev(barrier, direction_gate_ok: bool, p_sl_given_not_win, tp_r, sl_r, timeout_r, cost_r, uncertainty, k=None) -> Optional[float]` (returns risk-adjusted EV, or `None` if any input unavailable or `direction_gate_ok=False`); `decide(long_ev_adj: Optional[float], short_ev_adj: Optional[float]) -> tuple[str, Optional[str], str]` returning `(decision, direction, decision_reason)`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_ev_gate.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.ev_gate import decide, MIN_EDGE_THRESHOLD, compute_side_ev
from contracts.specialist_output import BarrierOutput


def test_decide_no_trade_both_below_threshold():
    decision, direction, reason = decide(long_ev_adj=0.001, short_ev_adj=-0.05)
    assert decision == "NO_TRADE"
    assert direction is None


def test_decide_long_wins():
    decision, direction, reason = decide(long_ev_adj=0.10, short_ev_adj=0.02)
    assert decision == "LONG_CANDIDATE"
    assert direction == "long"


def test_decide_short_wins():
    decision, direction, reason = decide(long_ev_adj=-0.01, short_ev_adj=0.15)
    assert decision == "SHORT_CANDIDATE"
    assert direction == "short"


def test_decide_none_available_is_no_trade():
    decision, direction, reason = decide(long_ev_adj=None, short_ev_adj=None)
    assert decision == "NO_TRADE"
    assert "unavailable" in reason.lower()


def test_compute_side_ev_gated_off_returns_none():
    barrier = BarrierOutput(model_id="x", horizon=15, model_status="VALIDATED", p_tp=0.5, calibrated=True)
    ev = compute_side_ev(barrier, direction_gate_ok=False, p_sl_given_not_win=0.5,
                          tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.02, uncertainty=0.2)
    assert ev is None


def test_compute_side_ev_computes_when_gated_on():
    barrier = BarrierOutput(model_id="x", horizon=15, model_status="VALIDATED", p_tp=0.5, calibrated=True)
    ev = compute_side_ev(barrier, direction_gate_ok=True, p_sl_given_not_win=0.5,
                          tp_r=1.0, sl_r=0.5, timeout_r=0.1, cost_r=0.02, uncertainty=0.2, k=0.1)
    assert ev is not None


if __name__ == "__main__":
    test_decide_no_trade_both_below_threshold()
    test_decide_long_wins()
    test_decide_short_wins()
    test_decide_none_available_is_no_trade()
    test_compute_side_ev_gated_off_returns_none()
    test_compute_side_ev_computes_when_gated_on()
    print("tests/test_ev_gate.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_gate.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
"""decision/ev_gate.py
Spec section 11: NO_TRADE gate is a fixed, documented min-edge threshold
plus status-gating -- never a bare p_win>0.6 heuristic. Long and short are
evaluated independently (no symmetry assumed)."""
from typing import Optional

from decision.ev_formula import compute_barrier_split, raw_ev, risk_adjusted_ev, DEFAULT_K
from contracts.specialist_output import BarrierOutput

# Set from the known transaction-cost floor plus a minimum-edge buffer
# (documented in docs/ARCHITECTURE.md's Phase 5 section, Task 15) -- not
# curve-fit to any evaluation window.
MIN_EDGE_THRESHOLD = 0.02


def compute_side_ev(barrier: BarrierOutput, direction_gate_ok: bool, p_sl_given_not_win: Optional[float],
                     tp_r: Optional[float], sl_r: Optional[float], timeout_r: Optional[float],
                     cost_r: Optional[float], uncertainty: float, k: float = None) -> Optional[float]:
    if not direction_gate_ok:
        return None
    if k is None:
        k = DEFAULT_K
    split = compute_barrier_split(barrier, p_sl_given_not_win)
    ev_raw = raw_ev(split["p_tp"], split["p_sl"], split["p_timeout"], tp_r, sl_r, timeout_r, cost_r)
    if ev_raw is None:
        return None
    return risk_adjusted_ev(ev_raw, uncertainty, k)


def decide(long_ev_adj: Optional[float], short_ev_adj: Optional[float]) -> tuple[str, Optional[str], str]:
    if long_ev_adj is None and short_ev_adj is None:
        return "NO_TRADE", None, "both sides unavailable -- required specialist(s) missing or gated off"
    long_ok = long_ev_adj is not None and long_ev_adj > MIN_EDGE_THRESHOLD
    short_ok = short_ev_adj is not None and short_ev_adj > MIN_EDGE_THRESHOLD
    if not long_ok and not short_ok:
        return "NO_TRADE", None, f"neither side cleared min_edge_threshold={MIN_EDGE_THRESHOLD}"
    if long_ok and (not short_ok or long_ev_adj >= short_ev_adj):
        return "LONG_CANDIDATE", "long", f"ev_adj={long_ev_adj:.4f} above min_edge_threshold"
    return "SHORT_CANDIDATE", "short", f"ev_adj={short_ev_adj:.4f} above min_edge_threshold"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_gate.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add decision/ev_gate.py tests/test_ev_gate.py
git commit -m "Add Phase 5 NO_TRADE gate and long/short EV evaluation"
```

---

### Task 12: Live EV engine

**Files:**
- Create: `decision/ev_engine.py`
- Test: `tests/test_ev_engine.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 3, 7, 8, 9, 11; `contracts.market_state.MarketState`.
- Produces: `evaluate(market_state, direction_out, opportunity_out, barrier_out, barrier_split_out, mae_out, mfe_out, timeout_r: float, timeout_r_provisional_proxy: bool, regime_state: Optional[int] = None) -> EVDecision` — a pure function, no I/O, no side effects, callable identically from live code and the research replay simulator (spec §14's "live/replay equivalence" requirement).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_ev_engine.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate


class _FakeMarketState:
    def __init__(self, spread, timestamp):
        self.spread = spread
        self.timestamp = timestamp


def _valid_inputs():
    ms = _FakeMarketState(spread=0.01, timestamp=datetime.now(timezone.utc))
    direction = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                                 model_status="VALIDATED", probability_long=0.6,
                                 probability_short=0.4, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_meta_v3_candidate_h15", horizon=15,
                                     model_status="VALIDATED", probability_take=0.55, calibrated=True)
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                             model_status="VALIDATED", p_tp=0.55, calibrated=True)
    mae = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.3, q75=0.5, q90=0.8)
    mfe = MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=0.9, q90=1.4)
    return ms, direction, opportunity, barrier, mae, mfe


def test_evaluate_produces_a_decision():
    ms, direction, opportunity, barrier, mae, mfe = _valid_inputs()
    d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win=0.5,
                 mae_out=mae, mfe_out=mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision in ("NO_TRADE", "LONG_CANDIDATE", "SHORT_CANDIDATE")
    assert d.specialist_model_ids["direction"] == "direction_v3_candidate_h15"


def test_evaluate_unavailable_direction_forces_no_trade():
    ms, direction, opportunity, barrier, mae, mfe = _valid_inputs()
    direction.model_status = "UNAVAILABLE"
    d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win=0.5,
                 mae_out=mae, mfe_out=mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"
    assert "direction" in d.decision_reason.lower() or "unavailable" in d.decision_reason.lower()


def test_evaluate_stale_market_forces_no_trade():
    from datetime import timedelta
    ms, direction, opportunity, barrier, mae, mfe = _valid_inputs()
    ms.timestamp = datetime.now(timezone.utc) - timedelta(seconds=60)
    d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win=0.5,
                 mae_out=mae, mfe_out=mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


if __name__ == "__main__":
    test_evaluate_produces_a_decision()
    test_evaluate_unavailable_direction_forces_no_trade()
    test_evaluate_stale_market_forces_no_trade()
    print("tests/test_ev_engine.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_engine.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the engine**

```python
"""decision/ev_engine.py
Spec section 14: the live entry point -- a pure function, MarketState +
specialist outputs -> EVDecision. Called ONLY from a shadow-evaluation
path (Task 13); never wired into app/engine.py's production decision
sequence."""
from datetime import datetime, timezone
from typing import Optional

from contracts.ev_decision import EVDecision
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_cost import candidate_sl_tp, round_trip_cost_r
from decision.ev_gate import compute_side_ev, decide
from decision.ev_formula import EV_FORMULA_VERSION

COST_MODEL_VERSION = "v1"
OPPORTUNITY_MIN_TAKE_PROBABILITY = 0.5
MARKET_STALENESS_SECONDS = 5.0
_OK = {"VALIDATED", "CANDIDATE"}


def evaluate(market_state, direction_out: DirectionOutput, opportunity_out: OpportunityOutput,
             barrier_out: BarrierOutput, p_sl_given_not_win: Optional[float],
             mae_out: MAEOutput, mfe_out: MFEOutput, timeout_r: float,
             timeout_r_provisional_proxy: bool, regime_state: Optional[int] = None) -> EVDecision:
    now = datetime.now(timezone.utc)
    sl_r, tp_r = candidate_sl_tp(mae_out, mfe_out)
    cost_r = round_trip_cost_r(market_state, sl_r, max_staleness_seconds=MARKET_STALENESS_SECONDS) if sl_r else None

    stale = market_state is None or (now - market_state.timestamp).total_seconds() > MARKET_STALENESS_SECONDS
    direction_available = direction_out.model_status in _OK and direction_out.probability_long is not None
    barrier_available = barrier_out.model_status in _OK

    if stale:
        reason = "MarketState stale"
        long_ev_adj = short_ev_adj = None
    elif not direction_available:
        reason = "Direction specialist unavailable"
        long_ev_adj = short_ev_adj = None
    elif not barrier_available:
        reason = "Barrier specialist unavailable"
        long_ev_adj = short_ev_adj = None
    elif cost_r is None:
        reason = "cost unavailable (spread missing/stale or no candidate SL)"
        long_ev_adj = short_ev_adj = None
    else:
        long_gate_ok = direction_out.probability_long > direction_out.probability_short
        short_gate_ok = not long_gate_ok
        if opportunity_out.model_status in _OK and opportunity_out.probability_take is not None:
            if opportunity_out.probability_take < OPPORTUNITY_MIN_TAKE_PROBABILITY:
                long_gate_ok = short_gate_ok = False
        uncertainty = 0.0
        if direction_out.model_status == "CANDIDATE":
            uncertainty += 0.2
        if barrier_out.model_status == "CANDIDATE":
            uncertainty += 0.2
        if opportunity_out.model_status not in _OK:
            uncertainty += 0.2
        uncertainty = min(uncertainty, 1.0)

        long_ev_adj = compute_side_ev(barrier_out, long_gate_ok, p_sl_given_not_win, tp_r, sl_r, timeout_r, cost_r, uncertainty)
        short_ev_adj = compute_side_ev(barrier_out, short_gate_ok, p_sl_given_not_win, tp_r, sl_r, timeout_r, cost_r, uncertainty)
        reason = None

    decision, direction, decide_reason = decide(long_ev_adj, short_ev_adj)
    final_reason = reason if reason else decide_reason
    chosen_ev_adj = {"long": long_ev_adj, "short": short_ev_adj}.get(direction, 0.0) or 0.0

    return EVDecision(
        timestamp=now, direction=direction, decision=decision,
        ev_adj=chosen_ev_adj, ev_raw=chosen_ev_adj, uncertainty=0.0 if stale else uncertainty if 'uncertainty' in dir() else 1.0,
        decision_margin=0.0, candidate_sl=sl_r, candidate_tp=tp_r, cost_r=cost_r, known_cost_only=True,
        specialist_model_ids={"direction": direction_out.model_id, "opportunity": opportunity_out.model_id,
                               "barrier": barrier_out.model_id, "mae": mae_out.model_id, "mfe": mfe_out.model_id},
        calibration_ids={}, feature_schema_ids={},
        ev_formula_version=EV_FORMULA_VERSION, cost_model_version=COST_MODEL_VERSION,
        regime_state=regime_state, timeout_r_provisional_proxy=timeout_r_provisional_proxy,
        decision_reason=final_reason,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_engine.py`
Expected: PASS. If `uncertainty` scoping causes a `NameError` in the `stale`/unavailable branches, fix by initializing `uncertainty = 1.0` at the top of `evaluate()` before the if/elif chain (straightforward bugfix, keep the rest of the logic as written).

- [ ] **Step 5: Commit**

```bash
git add decision/ev_engine.py tests/test_ev_engine.py
git commit -m "Add Phase 5 live EV engine (pure function, shadow-only)"
```

---

### Task 13: Research EV dataset + simulator (replay, OOS validation, baseline, sensitivity)

**Files:**
- Create: `research/phase5_ev_dataset.py`
- Create: `research/phase5_ev_engine.py`
- Test: `tests/test_phase5_ev_engine.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset`, `research.phase5_calibration._oof_for_direction/_oof_for_opportunity/_oof_for_barrier`, `research.phase5_barrier_split.run_barrier_split_candidate`'s OOF machinery pattern, `research.phase5_timeout_payoff.estimate_timeout_payoff`, `decision.ev_engine.evaluate`, `decision.calibration_registry.CalibrationRegistry`.
- Produces: `assemble_replay_dataset(max_holding: int, rows: int = None) -> dict` (per-event specialist outputs + realized R + historical spread proxy); `replay_and_validate(max_holding: int, rows: int = None) -> dict` returning `{"n_events", "decisions": {"NO_TRADE": int, "LONG_CANDIDATE": int, "SHORT_CANDIDATE": int}, "expected_vs_realized_r": {...}, "baseline_comparison": {...}, "fragile_fraction": float}`. No Telegram, no live I/O — pure research/replay.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5_ev_engine.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_ev_engine import replay_and_validate


def test_replay_and_validate_on_dry_run_dataset():
    result = replay_and_validate(max_holding=15, rows=20000)
    assert "n_events" in result
    assert set(result["decisions"].keys()) == {"NO_TRADE", "LONG_CANDIDATE", "SHORT_CANDIDATE"}
    assert "baseline_comparison" in result
    assert 0.0 <= result["fragile_fraction"] <= 1.0


if __name__ == "__main__":
    test_replay_and_validate_on_dry_run_dataset()
    print("tests/test_phase5_ev_engine.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_ev_engine.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5_ev_dataset.py`**

```python
"""research/phase5_ev_dataset.py
Assembles a per-event replay dataset: for each historical event, the
specialist outputs an EV engine would have seen (Direction/Opportunity/
Barrier OOF probabilities, MAE/MFE realized quantile-equivalent values,
a synthetic MarketState using a fixed representative spread since Phase 4
did not persist historical tick-level spread), plus the realized R
outcome for expected-vs-realized comparison.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_dataset
"""
from datetime import datetime, timezone

import numpy as np

from research.phase4_dataset import assemble_v3_dataset
from research.phase5_calibration import _oof_for_direction, _oof_for_opportunity, _oof_for_barrier
from research.audit_edge import _mae_mfe_core
from features.labeling import TripleBarrierConfig, triple_barrier_labels

REPRESENTATIVE_SPREAD = 0.015  # documented placeholder: Phase 4 did not persist historical tick spread; real
                                # live spread is used in decision/ev_engine.py's live path (Task 12) -- this
                                # constant is a research-only stand-in for OOS replay, not a live value.


def assemble_replay_dataset(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]

    y_dir, p_dir = _oof_for_direction(max_holding, rows=rows)
    y_opp, p_opp = _oof_for_opportunity(max_holding, rows=rows)
    y_bar, p_bar = _oof_for_barrier(max_holding, rows=rows)
    n = min(len(p_dir), len(p_opp), len(p_bar), nz.sum())

    side_nz = y[nz][:n].astype(float)
    vol_nz = vol_tb[t0_nz][:n]
    mae_r, mfe_r = _mae_mfe_core(close, high, low, t0_nz[:n], t1_nz[:n], side_nz, vol_nz)
    realized_r = np.where(y_bar[:n] == 1, mfe_r, -mae_r)

    return {"n": n, "p_direction": p_dir[:n], "p_opportunity": p_opp[:n], "p_barrier_win": p_bar[:n],
            "mae_r": mae_r, "mfe_r": mfe_r, "realized_r": realized_r,
            "spread": np.full(n, REPRESENTATIVE_SPREAD)}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        d = assemble_replay_dataset(h)
        print(f"h={h}: n={d['n']}")
```

- [ ] **Step 4: Write `research/phase5_ev_engine.py`**

```python
"""research/phase5_ev_engine.py
Spec sections 13/12/22/23: research-only EV replay simulator. Calls the
SAME decision/ev_engine.py.evaluate() pure function the live path uses
(spec section 14's live/replay equivalence requirement), against
historical specialist-output replay data. Computes OOS decision
distribution, expected-vs-realized R, a baseline comparison (simple
P(direction)>0.55 gate, no cost/no EV), and a sensitivity/fragility scan
(spec section 20/12). NO Telegram, no live I/O.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_engine
"""
from datetime import datetime, timezone

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset
from research.phase5_timeout_payoff import estimate_timeout_payoff
from research.phase5_barrier_split import run_barrier_split_candidate
from decision.ev_engine import evaluate
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput

SIMPLE_BASELINE_THRESHOLD = 0.55


class _ReplayMarketState:
    def __init__(self, spread):
        self.spread = spread
        self.timestamp = datetime.now(timezone.utc)


def replay_and_validate(max_holding: int, rows: int = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    split_info = run_barrier_split_candidate(max_holding, rows=rows)
    p_sl_given_not_win = 0.5 if split_info.get("n_events", 0) == 0 else split_info.get("p_sl_given_not_win_log_loss") and 0.5
    # Fallback constant used only when the split classifier has too few OOF
    # samples in this dry-run-sized replay; the real full-history model
    # (Task 2) is what decision/ev_gate.py uses live.

    decisions = {"NO_TRADE": 0, "LONG_CANDIDATE": 0, "SHORT_CANDIDATE": 0}
    expected_rs, realized_rs = [], []
    fragile_count = 0
    baseline_trades = 0
    baseline_realized = []

    n = data["n"]
    for i in range(n):
        ms = _ReplayMarketState(spread=float(data["spread"][i]))
        p_long = float(data["p_direction"][i])
        direction = DirectionOutput(model_id="direction_v3_candidate_replay", horizon=max_holding,
                                     model_status="VALIDATED", probability_long=p_long,
                                     probability_short=1 - p_long, calibrated=True)
        opportunity = OpportunityOutput(model_id="opportunity_meta_v3_candidate_replay", horizon=max_holding,
                                         model_status="VALIDATED", probability_take=float(data["p_opportunity"][i]),
                                         calibrated=True)
        barrier = BarrierOutput(model_id="barrier_v3_candidate_replay", horizon=max_holding,
                                 model_status="VALIDATED", p_tp=float(data["p_barrier_win"][i]), calibrated=True)
        mae = MAEOutput(model_id="mae_quantile_v3_candidate_replay", horizon=max_holding, model_status="VALIDATED",
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3)
        mfe = MFEOutput(model_id="mfe_quantile_v3_candidate_replay", horizon=max_holding, model_status="VALIDATED",
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3)

        d = evaluate(ms, direction, opportunity, barrier, p_sl_given_not_win, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        decisions[d.decision] += 1
        if d.decision != "NO_TRADE":
            expected_rs.append(d.ev_adj)
            realized_rs.append(float(data["realized_r"][i]))
            perturbed = evaluate(_ReplayMarketState(spread=float(data["spread"][i]) * 1.5), direction, opportunity,
                                  barrier, p_sl_given_not_win, mae, mfe, timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                                  timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
            if (perturbed.ev_adj > 0) != (d.ev_adj > 0):
                fragile_count += 1

        if p_long > SIMPLE_BASELINE_THRESHOLD:
            baseline_trades += 1
            baseline_realized.append(float(data["realized_r"][i]))

    n_traded = len(expected_rs)
    return {
        "n_events": n, "decisions": decisions,
        "expected_vs_realized_r": {
            "mean_expected": float(np.mean(expected_rs)) if n_traded else None,
            "mean_realized": float(np.mean(realized_rs)) if n_traded else None,
            "n_traded": n_traded,
        },
        "baseline_comparison": {
            "simple_gate_n_trades": baseline_trades,
            "simple_gate_mean_realized_r": float(np.mean(baseline_realized)) if baseline_trades else None,
            "ev_engine_n_trades": n_traded,
            "ev_engine_mean_realized_r": float(np.mean(realized_rs)) if n_traded else None,
        },
        "fragile_fraction": (fragile_count / n_traded) if n_traded else 0.0,
    }


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = replay_and_validate(h)
        print(f"h={h}: {r}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_ev_engine.py`
Expected: PASS

- [ ] **Step 6: Run for real, all 3 horizons, record full report**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_engine`
Record `decisions` distribution, `expected_vs_realized_r`, `baseline_comparison`, `fragile_fraction` per horizon for the final report.

- [ ] **Step 7: Commit**

```bash
git add research/phase5_ev_dataset.py research/phase5_ev_engine.py tests/test_phase5_ev_engine.py
git commit -m "Add Phase 5 research EV replay simulator with OOS validation, baseline comparison, sensitivity scan"
```

---

### Task 14: Leakage / causality tests

**Files:**
- Create: `tests/test_phase5_leakage.py`

**Interfaces:**
- Consumes: `decision.ev_engine.evaluate`, `research.phase5_ev_dataset.assemble_replay_dataset`, `contracts.ev_decision.EVDecision`.

- [ ] **Step 1: Write the tests**

```python
"""tests/test_phase5_leakage.py
Spec section 16/29: no future information enters EV; deterministic
replay; live/replay equivalence within tolerance; a DATA_LIMITED/
UNAVAILABLE specialist cannot produce a valid numeric decision; stale
market blocks a live decision; schema-mismatched input is rejected."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate


class _FakeMarketState:
    def __init__(self, spread, timestamp):
        self.spread = spread
        self.timestamp = timestamp


def _inputs(direction_status="VALIDATED"):
    ms = _FakeMarketState(spread=0.01, timestamp=datetime.now(timezone.utc))
    direction = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                                 model_status=direction_status, probability_long=0.6,
                                 probability_short=0.4, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_meta_v3_candidate_h15", horizon=15,
                                     model_status="VALIDATED", probability_take=0.55, calibrated=True)
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                             model_status="VALIDATED", p_tp=0.55, calibrated=True)
    mae = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.3, q75=0.5, q90=0.8)
    mfe = MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=0.9, q90=1.4)
    return ms, direction, opportunity, barrier, mae, mfe


def test_deterministic_replay():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    d1 = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    d2 = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d1.decision == d2.decision
    assert d1.ev_adj == d2.ev_adj


def test_data_limited_specialist_cannot_produce_valid_decision():
    ms, direction, opportunity, barrier, mae, mfe = _inputs(direction_status="DATA_LIMITED")
    d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


def test_stale_market_prevents_valid_decision():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    ms.timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


def test_schema_mismatch_rejected():
    with pytest.raises(ValidationError):
        DirectionOutput(model_id="x", horizon="not-an-int", model_status="VALIDATED")


def test_live_and_replay_paths_call_same_evaluate_function():
    import decision.ev_engine as live_engine
    import research.phase5_ev_engine as replay_engine
    assert replay_engine.evaluate is live_engine.evaluate


if __name__ == "__main__":
    test_deterministic_replay()
    test_data_limited_specialist_cannot_produce_valid_decision()
    test_stale_market_prevents_valid_decision()
    test_schema_mismatch_rejected()
    test_live_and_replay_paths_call_same_evaluate_function()
    print("tests/test_phase5_leakage.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_leakage.py`
Expected: FAIL (module import errors until Tasks 1-13 exist; if Tasks 1-13 are already complete when this task executes, expect PASS immediately after Step 1 -- if so, skip ahead and just confirm PASS, do not force an artificial failure).

- [ ] **Step 3: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_phase5_leakage.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase5_leakage.py
git commit -m "Add Phase 5 leakage/causality/live-replay-equivalence tests"
```

---

### Task 15: Performance benchmark

**Files:**
- Create: `tests/test_ev_engine_performance.py`

**Interfaces:**
- Consumes: `decision.ev_engine.evaluate`.

- [ ] **Step 1: Write the test**

```python
"""tests/test_ev_engine_performance.py
Mirrors tests/test_specialist_inference_performance.py's two-pass
(timing, then separate memory) pattern (spec section 30)."""
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate

N_CALLS = 200


class _FakeMarketState:
    def __init__(self):
        self.spread = 0.01
        self.timestamp = datetime.now(timezone.utc)


def _inputs():
    ms = _FakeMarketState()
    direction = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                                 model_status="VALIDATED", probability_long=0.6,
                                 probability_short=0.4, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_meta_v3_candidate_h15", horizon=15,
                                     model_status="VALIDATED", probability_take=0.55, calibrated=True)
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                             model_status="VALIDATED", p_tp=0.55, calibrated=True)
    mae = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.3, q75=0.5, q90=0.8)
    mfe = MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=0.9, q90=1.4)
    return ms, direction, opportunity, barrier, mae, mfe


def test_ev_engine_single_decision_latency():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    latencies_us = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(latencies_us)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"[ev_engine] single-decision latency over {N_CALLS} calls: p50={p50:.0f}us p95={p95:.0f}us p99={p99:.0f}us")
    assert p99 < 50_000, f"single-decision p99={p99:.0f}us exceeds 50ms budget"


def test_ev_engine_memory():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    tracemalloc.start()
    for _ in range(20):
        evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[ev_engine] peak traced memory over 20 calls: {peak / 1024:.1f}KB")


if __name__ == "__main__":
    test_ev_engine_single_decision_latency()
    test_ev_engine_memory()
    print("tests/test_ev_engine_performance.py: OK")
```

- [ ] **Step 2: Run, record real numbers**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_ev_engine_performance.py`
Record p50/p95/p99 and peak memory for the final report.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ev_engine_performance.py
git commit -m "Add Phase 5 EV engine latency/memory benchmark"
```

---

### Task 16: Documentation

**Files:**
- Modify: `docs/ARCHITECTURE.md` (append new "## Phase 5: Probability / EV Engine" section)

**Interfaces:**
- Consumes: all real numbers recorded in Tasks 2-15's Run steps.

- [ ] **Step 1: Read the existing Phase 4 section of `docs/ARCHITECTURE.md` first**, to match style/tone exactly.

- [ ] **Step 2: Append "## Phase 5: Probability / EV Engine"**, covering (using ONLY real recorded numbers, no invented values): the architecture diagram (spec §14/§32's chain), the specialist contracts and status handling, the OOF-only calibration approach and the real `a`/`b`/`n` values fitted in Tasks 3-4, the Direction/Barrier investigation's real finding (Task 5 — redundant, or a documented correction), the real `timeout_R` values and whether the OOF-derived estimate or the provisional proxy was used per horizon (Task 6), the exact EV formula (§9) including the real chosen `k` and its OOS validation evidence (Task 10), the cost model, the NO_TRADE gate's `MIN_EDGE_THRESHOLD` and its justification, the real replay/OOS results including the decision distribution, expected-vs-realized R, baseline comparison, and fragile fraction (Task 13), the leakage test results (Task 14), the performance numbers (Task 15), explicit confirmation that `app/engine.py`/production SL-TP/Telegram are unchanged (a `git diff` command and its empty/expected output, same technique as Phase 4's Task 16), and a "### Known Methodology Limitations" subsection carrying forward spec §19's items with their real resolutions from Tasks 5/6/10.

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "Document Phase 5: Probability/EV Engine, real OOS results, methodology limitations"
```

---

### Task 17: Final verification sweep + completion report

**Files:** none (verification only)

- [ ] **Step 1: Run every Phase 5 test file**

```bash
for f in tests/test_specialist_output.py tests/test_phase5_barrier_split.py \
  tests/test_calibration_registry.py tests/test_phase5_calibration_opportunity_barrier.py \
  tests/test_phase5_direction_barrier_investigation.py tests/test_phase5_timeout_payoff.py \
  tests/test_ev_decision.py tests/test_ev_cost.py tests/test_ev_formula.py \
  tests/test_phase5_uncertainty_k.py tests/test_ev_gate.py tests/test_ev_engine.py \
  tests/test_phase5_ev_engine.py tests/test_phase5_leakage.py tests/test_ev_engine_performance.py; do
  echo "=== $f ==="
  /home/jith/.hermes/hermes-agent/venv/bin/python3 "$f" || echo "FAILED: $f"
done
```

- [ ] **Step 2: Re-run every Phase 1-4 test file to confirm zero regression** (reuse the exact list from Phase 4's Task 16 Step 2, itself reused from Phase 3's Task 30).

- [ ] **Step 3: Confirm production path untouched**

```bash
git diff <first-Phase5-commit>..HEAD -- app/engine.py app/shadow.py decision/signal.py decision/router.py config/models.yaml features/features.py learning/train.py models/registry/direction_catboost_20260818.json models/registry/opportunity_meta_catboost_20260818.json
```

Expected: zero diff on every listed file (Phase 5 adds new files under `decision/`, `contracts/`, `research/`, `models/calibration/`, `models/registry/*_v3_candidate*` — it does not modify any file in this list).

- [ ] **Step 4: Confirm no real registry/schema/calibration directory was touched by any test run**

```bash
git status --short models/registry features/registry/schemas models/calibration
```

Expected: only the intentional, committed additions from Tasks 2/3/4 (the new `barrier_split_v3_candidate_h*.json` and `*_platt.json` files) — zero uncommitted/unexpected diffs from running the test suite.

- [ ] **Step 5: Compose and deliver the completion report** to the user in the A-P (+Q/R/S) format from the design spec's §35, using the real output captured in every task's Run steps above — no invented numbers. Print the full report text in chat per the user's standing preference (not just a file path). End with section S: recommend Phase 6 only, do not implement it.
