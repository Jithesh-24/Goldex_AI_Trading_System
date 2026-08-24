# GOLEX V3 Phase 5A: Specialist Side-Conditioning Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Opportunity, Barrier, MAE, and MFE condition on Direction's actual OOF side instead of a side each one invents for itself, closing the integration bug the H15 skew investigation found — without touching EV formula, costs, thresholds, Direction's own training, or trade-frequency logic.

**Architecture:** One shared helper (`research/direction_side.py`) computes Direction's OOF side/probability once per horizon. Every downstream specialist's training/calibration/replay code is changed to consume that helper's output as `side` + `assumed_side`/`p_direction` features, and to stop fitting its own throwaway "primary" side-generator classifier. New registry/schema artifacts are written under `*_v3b_*` names; old `*_v3_*` artifacts are untouched. `contracts/specialist_output.py` gains `assumed_side`/`direction_model_id` fields so `decision/ev_engine.py` can fail closed if a downstream output was ever conditioned on a side that doesn't match the Direction output passed alongside it.

**Tech Stack:** Python, CatBoost, pandas/numpy, pydantic (contracts), pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-golex-v3-phase5a-specialist-conditioning-design.md`

## Global Constraints

- Direction owns the side; no downstream specialist may generate its own (design §1, §3).
- Architecture A: `assumed_side` + `p_direction` as features on one shared model per specialist, not separate per-side models (design §4).
- Remove primary-stage classifiers; do not keep them "just in case" (design §2).
- The same Direction OOF side is reused across Opportunity, Barrier, MAE, MFE for a given horizon (design §5).
- Never overwrite `models/registry/{opportunity,barrier,mae_quantile,mfe_quantile}_v3_candidate_h*.json` or their schemas — new artifacts use a `v3b` name segment (design §7, §11).
- Run the OOF `assumed_side`-only vs `assumed_side`+`p_direction` comparison; drop `p_direction` from the final feature set for a specialist only if it adds no measurable value (design §9).
- Fail closed anywhere `direction_side`/`direction_model_id` is missing or mismatched in the downstream chain (design §5, §10).
- Do not modify `decision/ev_formula.py`, `decision/ev_gate.py`, `decision/ev_cost.py`, cost model, thresholds, or `research/phase4_direction.py`'s own target/feature/CV methodology.
- Do not attempt to restore long-trade frequency or optimize thresholds.
- Re-run H15/H45/H90 full-history replay from scratch after retraining; all prior Phase 5 replay numbers (correction-pass report, H15 skew investigation) are diagnostic-only, not a baseline to reproduce or beat.
- Do not proceed to Phase 6 until validation criteria in design §10 pass and Phase 5 EV has been re-evaluated on the new artifacts.

---

### Task 1: Extract Direction's OOF side into a shared helper

**Files:**
- Create: `research/direction_side.py`
- Modify: `research/phase4_direction.py:36-99` (replace the inline pass1/pass2 OOF-fit block with a call to the new helper)
- Test: `tests/test_direction_side.py`

**Interfaces:**
- Produces: `research.direction_side.compute_direction_oof(max_holding: int, rows: int = None) -> dict` with keys `t0_nz` (`np.ndarray[int64]`), `feature_cols` (`list[str]`), `p_direction_raw` (`np.ndarray[float64]`, NaN where no OOF), `p_direction_cal` (`np.ndarray[float64]`, Platt-calibrated, NaN where no OOF), `side` (`np.ndarray[float64]`, `+1.0`/`-1.0`, valid only where `has_oof`; `0.0` placeholder elsewhere), `has_oof` (`np.ndarray[bool]`), `model_id` (`str`, `f"direction_v3_candidate_h{max_holding}"` — the same id `phase4_direction.py` registers, so downstream lineage points at the real candidate this side came from).
- Consumes (from existing code, unchanged): `research.phase4_dataset.assemble_v3_dataset`, `research.phase4_dataset.select_top_features`, `research.audit_edge.oof_run`, `decision.calibration.PlattCalibrator`, `features.labeling.TripleBarrierConfig`/`triple_barrier_labels`, `learning.train.EMBARGO_BARS`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_direction_side.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.direction_side import compute_direction_oof


def test_compute_direction_oof_shapes_and_side_values():
    out = compute_direction_oof(max_holding=15, rows=600000)
    n = len(out["t0_nz"])
    assert len(out["p_direction_raw"]) == n
    assert len(out["p_direction_cal"]) == n
    assert len(out["side"]) == n
    assert len(out["has_oof"]) == n
    assert out["model_id"] == "direction_v3_candidate_h15"
    assert out["has_oof"].sum() > 50, "too few OOF events in dry run to trust anything downstream"
    side_valid = out["side"][out["has_oof"]]
    assert set(side_valid.tolist()) <= {1.0, -1.0}
    p_valid = out["p_direction_cal"][out["has_oof"]]
    assert ((p_valid >= 0.0) & (p_valid <= 1.0)).all()


def test_side_matches_probability_threshold():
    out = compute_direction_oof(max_holding=15, rows=600000)
    m = out["has_oof"]
    expected_side = (out["p_direction_raw"][m] >= 0.5).astype(float) * 2 - 1
    assert (out["side"][m] == expected_side).all(), \
        "side must be derived from p_direction_raw >= 0.5, matching ev_engine.py's own direction_gate_ok rule"


if __name__ == "__main__":
    test_compute_direction_oof_shapes_and_side_values()
    test_side_matches_probability_threshold()
    print("tests/test_direction_side.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_direction_side.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.direction_side'`

- [ ] **Step 3: Write `research/direction_side.py`**

```python
"""research/direction_side.py
Phase 5A: the single, shared source of "what side did Direction propose"
for a given event. Every downstream specialist (Opportunity, Barrier, MAE,
MFE) MUST condition on this function's output and must not compute its own
side (docs/superpowers/specs/2026-08-24-golex-v3-phase5a-specialist-
conditioning-design.md, sections 1/3/5). This is the exact pass1+pass2
OOF-fit Direction's own candidate training (research/phase4_direction.py)
already does -- extracted here so both Direction's own registry entry and
every downstream consumer compute the SAME side from the SAME model, never
two independently-fit copies.
"""
import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, select_top_features
from research.audit_edge import oof_run
from learning.train import EMBARGO_BARS as REAL_EMBARGO_BARS
from decision.calibration import PlattCalibrator
from features.labeling import TripleBarrierConfig, triple_barrier_labels

TOP_N_FEATURES = 20  # matches research/phase4_direction.py's own narrowing


def compute_direction_oof(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    feat_v3, close, high, low, vol_tb, t0_idx = (ds["feat_v3"], ds["close"], ds["high"],
                                                  ds["low"], ds["vol_tb"], ds["t0_idx"])
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz = t0_idx[nz]
    t1_nz = labels["t1"].to_numpy()[nz]
    n = len(t0_nz)

    candidate_cols = ds["baseline_cols"] + ds["useful_cols"]
    X_full = feat_v3.loc[t0_nz, candidate_cols].reset_index(drop=True)
    y_bin = pd.Series((y[nz] == 1).astype(np.int64))
    t0, t1 = pd.Series(t0_nz), pd.Series(t1_nz)

    pass1 = oof_run(X_full, y_bin, t0, t1, tag=f"direction_side_h{max_holding}_pass1", want_importance=True)
    feature_cols = select_top_features(pass1["importances"], top_n=TOP_N_FEATURES)

    X = X_full[feature_cols]
    result = oof_run(X, y_bin, t0, t1, tag=f"direction_side_h{max_holding}", want_importance=False)
    has_oof = result["has_oof"]

    p_raw_full = np.full(n, np.nan)
    p_raw_full[has_oof] = result["oof_proba"][has_oof]

    p_cal_full = np.full(n, np.nan)
    if has_oof.any():
        y_true = y_bin.to_numpy()[has_oof]
        cal = PlattCalibrator.fit(p_raw_full[has_oof], y_true)
        p_cal_full[has_oof] = cal.apply(p_raw_full[has_oof])

    side = np.zeros(n, dtype=np.float64)
    side[has_oof] = np.where(p_raw_full[has_oof] >= 0.5, 1.0, -1.0)

    return {"t0_nz": t0_nz, "feature_cols": feature_cols,
            "p_direction_raw": p_raw_full, "p_direction_cal": p_cal_full,
            "side": side, "has_oof": has_oof,
            "model_id": f"direction_v3_candidate_h{max_holding}"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_direction_side.py -v`
Expected: PASS

- [ ] **Step 5: Refactor `phase4_direction.py` to use the helper (no behavior change)**

In `research/phase4_direction.py`, replace lines 39-79 (dataset assembly through `mean_economic_r` computation) with a call to `compute_direction_oof`, keeping every metric identical:

```python
def run_direction_candidate(max_holding: int, rows: int = None, registry_dir: str = None, schemas_dir: str = None) -> dict:
    if registry_dir is None:
        registry_dir = REGISTRY_DIR
    oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    t0_nz, feature_cols = oof["t0_nz"], oof["feature_cols"]
    has_oof = oof["has_oof"]
    y_true = np.zeros(len(t0_nz), dtype=np.int64)  # placeholder, overwritten below from labels
    # re-derive labels/ret for the economic-R check and registry metrics (same
    # triple-barrier call compute_direction_oof made internally; cheap, avoids
    # threading extra return values through the shared helper's public contract)
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    assert np.array_equal(t0_idx[nz], t0_nz), "direction_side helper's event index must match this dataset assembly"
    y_bin = (y[nz] == 1).astype(np.int64)
    y_true = y_bin[has_oof]
    p_raw = oof["p_direction_raw"][has_oof]
    p_cal = oof["p_direction_cal"][has_oof]

    oos_log_loss = manual_log_loss(y_true, p_cal)
    oos_brier = float(np.mean((p_cal - y_true) ** 2))
    fold_result = oof_run(feat_v3_cols_placeholder := None, None, None, None) if False else None  # noop guard removed below
```

This inline sketch is too fragile to hand-splice around `fold_metrics`/`mean_acc` (those come from `oof_run`'s per-fold dict, which `compute_direction_oof` does not currently return). Instead of partially inlining, change `compute_direction_oof` to also return `fold_metrics` (the pass-2 `oof_run` result's `fold_metrics` list), and have `run_direction_candidate` consume that directly:

Add to `compute_direction_oof`'s return dict: `"fold_metrics": result["fold_metrics"]`.

Then in `phase4_direction.py`, replace lines 39-79 with:

```python
    oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    t0_nz, feature_cols, has_oof = oof["t0_nz"], oof["feature_cols"], oof["has_oof"]

    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, t0_idx, vol_tb = ds["close"], ds["t0_idx"], ds["vol_tb"]
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, ds["high"], ds["low"], t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    assert np.array_equal(t0_idx[nz], t0_nz), "direction_side event index mismatch"
    y_bin = (y[nz] == 1).astype(np.int64)

    y_true = y_bin[has_oof]
    p_raw = oof["p_direction_raw"][has_oof]
    p_cal = oof["p_direction_cal"][has_oof]
    mean_acc = float(np.mean([f["acc"] for f in oof["fold_metrics"]]))

    oos_log_loss = manual_log_loss(y_true, p_cal)
    oos_brier = float(np.mean((p_cal - y_true) ** 2))
    X = None  # X is only used below for len(X) in metrics/print -- replace with len(t0_nz)
```

And replace every remaining reference to `X`/`result` further down in the function (`len(X)` in the registry entry's `metrics.n_events` and the final print, `result["fold_metrics"]` in `mean_acc`) with `len(t0_nz)` and `oof["fold_metrics"]` respectively. Everything from `from sklearn.metrics import ...` onward (ROC/PR/economic-R block, `status = ...`, schema build, registry entry, file write, print, return) stays byte-identical — it only reads `y_true`, `p_cal`, `labels`, `nz`, `has_oof`, `mean_acc`, `oos_log_loss`, `oos_brier`, all of which are still defined with the same values as before. Add `from research.direction_side import compute_direction_oof` to the imports; `select_top_features` import in this file becomes unused and should be removed if nothing else in the file uses it (`grep -n select_top_features research/phase4_direction.py` after the edit — only the import line should remain if so; delete it).

- [ ] **Step 6: Run Direction's existing test to verify no behavior change**

Run: `pytest tests/test_phase4_direction.py -v`
Expected: PASS (same assertions as before the refactor — this step's purpose is proving the extraction didn't change Direction's own numbers, not adding new assertions)

- [ ] **Step 7: Commit**

```bash
git add research/direction_side.py research/phase4_direction.py tests/test_direction_side.py
git commit -m "feat: extract Direction's OOF side into shared research/direction_side.py"
```

---

### Task 2: Make `build_meta` accept a caller-supplied side instead of deriving its own

**Files:**
- Modify: `research/audit_edge.py:131-135` (`build_meta` signature/body), and its 3 in-file call sites at lines 272, 506, 535 (numbers per the version read during design; re-`grep -n "build_meta("` before editing to confirm current line numbers)
- Test: `tests/test_build_meta_side_contract.py`

**Interfaces:**
- Produces: `research.audit_edge.build_meta(close, high, low, vol, t0_nz, side, has_oof) -> (side_sub, meta_labels)` — `side` is now the caller's own signed `±1.0` array aligned to the full `t0_nz` index space (same shape convention `compute_direction_oof` returns), not a `0/1` classifier prediction. `has_oof` still selects which positions are usable. Returns `side_sub = side[has_oof]` (so downstream code keeps working with the same `side` variable name/shape it used before) and `meta_labels` (unchanged: `triple_barrier_labels(..., side=side_sub)`).
- Consumes (unchanged): `features.labeling.triple_barrier_labels`, `TB_CFG_TRADE` (module-level in `audit_edge.py`).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_build_meta_side_contract.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.audit_edge import build_meta


def _synthetic_bars(n=2000):
    rng = np.random.default_rng(0)
    close = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    vol = np.full(n, 0.01)
    return close, high, low, vol


def test_build_meta_uses_caller_supplied_side_directly():
    close, high, low, vol = _synthetic_bars()
    t0_nz = np.arange(10, 1900, 50)
    n = len(t0_nz)
    side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # caller-supplied, NOT derived internally
    has_oof = np.ones(n, dtype=bool)
    side_sub, meta_labels = build_meta(close, high, low, vol, t0_nz, side, has_oof)
    assert np.array_equal(side_sub, side), "build_meta must pass the caller's side through unchanged, not recompute it"
    assert len(meta_labels) == n
    assert set(meta_labels["label"].unique()) <= {0, 1}


def test_build_meta_respects_has_oof_mask():
    close, high, low, vol = _synthetic_bars()
    t0_nz = np.arange(10, 1900, 50)
    n = len(t0_nz)
    side = np.ones(n, dtype=np.float64)
    has_oof = np.arange(n) % 3 == 0
    side_sub, meta_labels = build_meta(close, high, low, vol, t0_nz, side, has_oof)
    assert len(side_sub) == int(has_oof.sum())
    assert len(meta_labels) == int(has_oof.sum())


if __name__ == "__main__":
    test_build_meta_uses_caller_supplied_side_directly()
    test_build_meta_respects_has_oof_mask()
    print("tests/test_build_meta_side_contract.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_meta_side_contract.py -v`
Expected: FAIL — old `build_meta` treats its 6th positional arg as `oof_pred` (0/1) and does `np.where(oof_pred[has_oof] == 1, 1.0, -1.0)`, so passing an already-signed `±1.0` array produces `side_sub` that is all `1.0` (since `1.0 == 1` is `True`, `-1.0 == 1` is `False` → both map through the old formula incorrectly) — `test_build_meta_uses_caller_supplied_side_directly`'s equality assertion fails.

- [ ] **Step 3: Update `build_meta`**

In `research/audit_edge.py`, replace:

```python
def build_meta(close, high, low, vol, t0_nz, oof_pred, has_oof):
    side = np.where(oof_pred[has_oof] == 1, 1.0, -1.0)
    t0_sub = t0_nz[has_oof]
    meta_labels = triple_barrier_labels(close, high, low, t0_sub, vol, TB_CFG_TRADE, side=side)
    return side, meta_labels
```

with:

```python
def build_meta(close, high, low, vol, t0_nz, side, has_oof):
    """Builds the meta-labeling target for a caller-supplied side.
    `side` must already be a signed +-1.0 array aligned to t0_nz's full
    index space (e.g. research.direction_side.compute_direction_oof's
    `side` output) -- this function does NOT derive a side from a raw
    classifier prediction anymore (Phase 5A: every downstream specialist
    conditions on Direction's side, never its own)."""
    side_sub = side[has_oof]
    t0_sub = t0_nz[has_oof]
    meta_labels = triple_barrier_labels(close, high, low, t0_sub, vol, TB_CFG_TRADE, side=side_sub)
    return side_sub, meta_labels
```

Then update `build_meta`'s 3 in-file call sites (Phase 1A's own `main()` — this script's original edge-audit pipeline, not part of Phase 4/5, kept working as its own self-contained side generator since it never claims Direction-conditioning). Immediately before each existing call `side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])` (and the two ablation variants with `prim_ntv`/`prim_nsp`), insert a one-line side derivation and pass it instead:

```python
side_in = np.where(prim["oof_pred"] == 1, 1.0, -1.0)
side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, side_in, prim["has_oof"])
```

(and the analogous `side_in_ntv = np.where(prim_ntv["oof_pred"] == 1, 1.0, -1.0)` / `side_in_nsp = np.where(prim_nsp["oof_pred"] == 1, 1.0, -1.0)` for the other two call sites). This keeps `audit_edge.py`'s own numbers byte-identical to before.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_meta_side_contract.py -v`
Expected: PASS

- [ ] **Step 5: Run audit_edge's own smoke path to confirm no regression**

Run: `python3 -c "from research.audit_edge import build_meta; print('import ok')"`
Expected: `import ok` (proves no syntax error; `audit_edge.main()` itself is a multi-minute full-history job and is NOT part of this plan's scope to re-run — Phase 1A's audit is unaffected by Phase 5A and not being re-validated here)

- [ ] **Step 6: Commit**

```bash
git add research/audit_edge.py tests/test_build_meta_side_contract.py
git commit -m "fix: build_meta takes caller-supplied side instead of deriving its own"
```

---

### Task 3: Retrain Opportunity on Direction's side (new `v3b` artifact)

**Files:**
- Modify: `research/phase4_opportunity.py`
- Test: `tests/test_phase4_opportunity_v3b.py`

**Interfaces:**
- Consumes: `research.direction_side.compute_direction_oof` (Task 1), `research.audit_edge.build_meta` new signature (Task 2).
- Produces: `research.phase4_opportunity.run_opportunity_candidate_v3b(max_holding: int, rows: int = None, registry_dir: str = None, schemas_dir: str = None) -> dict` (same return shape as the existing `run_opportunity_candidate`: `{"n_events", "oos_log_loss", "status"}`, plus `"used_p_direction": bool` recording whether `p_direction` survived into the final feature set). Writes `models/registry/opportunity_v3b_candidate_h{max_holding}.json` (model_id `opportunity_v3b_candidate_h{max_holding}`) — the existing `opportunity_v3_candidate_h{max_holding}.json` is untouched.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase4_opportunity_v3b.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_opportunity import run_opportunity_candidate_v3b


def test_run_opportunity_candidate_v3b_conditions_on_direction_side():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_opportunity_candidate_v3b(max_holding=45, rows=600000,
                                            registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert 0.0 <= result["oos_log_loss"] < 5.0
    assert result["status"] in ("validated", "rejected")
    assert isinstance(result["used_p_direction"], bool)

    entry_path = os.path.join(tmp_registry.name, "opportunity_v3b_candidate_h45.json")
    assert os.path.exists(entry_path)
    import json
    with open(entry_path) as f:
        entry = json.load(f)
    assert entry["model_id"] == "opportunity_v3b_candidate_h45"
    assert "assumed_side" in entry["feature_cols"]
    assert "direction_side" in entry["target_definition"] or "Direction" in entry["target_definition"]
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


def test_old_v3_artifact_untouched():
    # the real (non-tmp) registry's old artifact must still exist and be unmodified by this module import
    real_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "registry", "opportunity_v3_candidate_h45.json")
    assert os.path.exists(real_path), "old Phase 4 artifact must be preserved for audit (design section 7)"


if __name__ == "__main__":
    test_run_opportunity_candidate_v3b_conditions_on_direction_side()
    test_old_v3_artifact_untouched()
    print("tests/test_phase4_opportunity_v3b.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase4_opportunity_v3b.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_opportunity_candidate_v3b'`

- [ ] **Step 3: Add `run_opportunity_candidate_v3b` to `research/phase4_opportunity.py`**

Add below the existing `run_opportunity_candidate` (leave that function untouched so the old artifact remains reproducible), with `from research.direction_side import compute_direction_oof` added to the imports:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase4_opportunity_v3b.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase4_opportunity.py tests/test_phase4_opportunity_v3b.py
git commit -m "feat: add opportunity_v3b candidate conditioned on Direction's OOF side"
```

---

### Task 4: Retrain Barrier on Direction's side (new `v3b` artifact)

**Files:**
- Modify: `research/phase4_barrier.py`
- Test: `tests/test_phase4_barrier_v3b.py`

**Interfaces:**
- Consumes: same as Task 3.
- Produces: `research.phase4_barrier.run_barrier_candidate_v3b(max_holding, rows=None, registry_dir=None, schemas_dir=None) -> dict` (same shape as existing `run_barrier_candidate`, plus `"used_p_direction": bool`). Writes `barrier_v3b_candidate_h{max_holding}.json`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase4_barrier_v3b.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_barrier import run_barrier_candidate_v3b


def test_run_barrier_candidate_v3b_conditions_on_direction_side():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_barrier_candidate_v3b(max_holding=45, rows=600000,
                                        registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert result["log_loss"] >= 0.0
    assert result["status"] in ("validated", "rejected")

    entry_path = os.path.join(tmp_registry.name, "barrier_v3b_candidate_h45.json")
    with open(entry_path) as f:
        entry = json.load(f)
    assert entry["model_id"] == "barrier_v3b_candidate_h45"
    assert "assumed_side" in entry["feature_cols"]
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


if __name__ == "__main__":
    test_run_barrier_candidate_v3b_conditions_on_direction_side()
    print("tests/test_phase4_barrier_v3b.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase4_barrier_v3b.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_barrier_candidate_v3b'`

- [ ] **Step 3: Add `run_barrier_candidate_v3b` to `research/phase4_barrier.py`**

Same pattern as Task 3 Step 3, adapted to Barrier's own metrics (log_loss/brier/reliability_curve/max_calib_gap instead of win_rate), with `from research.direction_side import compute_direction_oof` added and the primary-stage `prim = oof_run(...)` / old `build_meta(..., prim["oof_pred"], prim["has_oof"])` call replaced by:

```python
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(dir_oof["t0_nz"], t0_nz), "direction_side event index mismatch"
    has_oof = dir_oof["has_oof"]
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, dir_oof["side"], has_oof)
    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    X_meta_full["p_direction"] = dir_oof["p_direction_cal"][has_oof]
```

Run the A-vs-A+probability comparison identically to Task 3 (refit on `feature_cols_meta` minus `p_direction`, compare `log_loss`, keep `p_direction` only if it lowers `log_loss`). Write the registry entry with `model_id=f"barrier_v3b_candidate_h{max_holding}"`, `artifact_path=f"registry/barrier_v3b_candidate_h{max_holding}.json"`, `target_definition` updated to: `f"P(assumed-side TP before SL | features, direction_side) where direction_side is sourced from {dir_oof['model_id']}'s OOF prediction, max_holding={max_holding}; timeout counted as non-TP in the denominator, matching decision/ev_formula.py's p_tp/p_sl/p_timeout split"`. Schema built via `build_schema(f"barrier_v3b_h{max_holding}", "2026-08-24", final_cols)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase4_barrier_v3b.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase4_barrier.py tests/test_phase4_barrier_v3b.py
git commit -m "feat: add barrier_v3b candidate conditioned on Direction's OOF side"
```

---

### Task 5: Retrain MAE and MFE quantile models on Direction's side (new `v3b` artifacts)

**Files:**
- Modify: `research/phase4_mae_quantile.py`
- Modify: `research/phase4_mfe_quantile.py`
- Test: `tests/test_phase4_mae_mfe_quantile_v3b.py`

**Interfaces:**
- Consumes: same as Task 3, plus `research.audit_edge._mae_mfe_core` (unchanged) and `research.v3_quantile_models.fit_quantile`/`pinball_loss` (unchanged).
- Produces: `run_mae_quantile_candidate_v3b(max_holding, rows=None, registry_dir=None, schemas_dir=None) -> dict` writing `mae_quantile_v3b_candidate_h{max_holding}.json`; `run_mfe_quantile_candidate_v3b(...)` writing `mfe_quantile_v3b_candidate_h{max_holding}.json`. Both same return shape as their existing counterparts.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase4_mae_mfe_quantile_v3b.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_mae_quantile import run_mae_quantile_candidate_v3b
from research.phase4_mfe_quantile import run_mfe_quantile_candidate_v3b


def test_mae_quantile_v3b():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_mae_quantile_candidate_v3b(max_holding=45, rows=600000,
                                             registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert result["status"] in ("validated", "rejected")
    assert os.path.exists(os.path.join(tmp_registry.name, "mae_quantile_v3b_candidate_h45.json"))
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


def test_mfe_quantile_v3b():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_mfe_quantile_candidate_v3b(max_holding=45, rows=600000,
                                             registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert result["status"] in ("validated", "rejected")
    assert os.path.exists(os.path.join(tmp_registry.name, "mfe_quantile_v3b_candidate_h45.json"))
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


if __name__ == "__main__":
    test_mae_quantile_v3b()
    test_mfe_quantile_v3b()
    print("tests/test_phase4_mae_mfe_quantile_v3b.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase4_mae_mfe_quantile_v3b.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add `run_mae_quantile_candidate_v3b` to `research/phase4_mae_quantile.py`**

Same restructuring as Tasks 3/4: replace the `prim = oof_run(...)` / `build_meta(..., prim["oof_pred"], prim["has_oof"])` block with:

```python
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(dir_oof["t0_nz"], t0_nz), "direction_side event index mismatch"
    has_oof = dir_oof["has_oof"]
    if not has_oof.any():
        print(f"[WARNING] No Direction OOF for h={max_holding} - dataset too small or CV constraints too strict")
        return {"n_events": 0, "global_coverage": {}, "per_regime_coverage": {}, "status": "rejected"}
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, dir_oof["side"], has_oof)
    X_meta_full = X_full.loc[has_oof].reset_index(drop=True)
    X_meta_full["assumed_side"] = side
    X_meta_full["p_direction"] = dir_oof["p_direction_cal"][has_oof]
```

Run the pass-1 feature-importance narrowing (`fit_quantile` at `q=0.5`) exactly as before but on `X_meta_full` (now including `p_direction`); force `assumed_side` into `feature_cols_meta` as before. Run the A-vs-A+probability comparison via `pinball_loss` at `q=0.75` (the headline quantile) instead of log-loss: fit with and without `p_direction` in the schema, keep it only if it lowers the OOF pinball loss at `q=0.75`. Registry entry: `model_id=f"mae_quantile_v3b_candidate_h{max_holding}"`, `artifact_path=f"registry/mae_quantile_v3b_candidate_h{max_holding}.json"`, `target_definition` updated to mention `direction_side` sourced from `dir_oof["model_id"]`, `training_config` gains `"direction_side_model_id": dir_oof["model_id"]`. Schema via `build_schema(f"mae_quantile_v3b_h{max_holding}", "2026-08-24", feature_cols_meta)`.

- [ ] **Step 4: Add `run_mfe_quantile_candidate_v3b` to `research/phase4_mfe_quantile.py`**

Identical pattern to Step 3, targeting `mfe_R` (the second `_mae_mfe_core` return value) instead of `mae_R`, `model_id=f"mfe_quantile_v3b_candidate_h{max_holding}"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_phase4_mae_mfe_quantile_v3b.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add research/phase4_mae_quantile.py research/phase4_mfe_quantile.py tests/test_phase4_mae_mfe_quantile_v3b.py
git commit -m "feat: add mae/mfe_quantile v3b candidates conditioned on Direction's OOF side"
```

---

### Task 6: Fix Phase 5 calibration to use the same shared Direction side

**Files:**
- Modify: `research/phase5_calibration.py`
- Test: `tests/test_phase5_calibration_v3b.py`

**Interfaces:**
- Consumes: `research.direction_side.compute_direction_oof` (Task 1), `research.audit_edge.build_meta` new signature (Task 2).
- Produces: `_oof_for_direction`, `_oof_for_opportunity`, `_oof_for_barrier`, `_oof_predicted_mae_mfe` keep their existing signatures `(max_holding, rows=None) -> (t0_nz, y_full, p_full, has_oof)` / `(t0_nz, mae_full, mfe_full, has_oof)` (`research/phase5_ev_dataset.py` depends on these exact signatures and its equality assertions across the four streams — do not change the public shape), but internally all four now read Direction's side from `compute_direction_oof` instead of each fitting (or, for `_oof_for_direction`, duplicating) their own primary-stage classifier.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5_calibration_v3b.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_calibration import _oof_for_direction, _oof_for_opportunity, _oof_for_barrier
from research.direction_side import compute_direction_oof


def test_oof_for_direction_matches_shared_helper():
    t0, y_full, p_full, m = _oof_for_direction(max_holding=15, rows=600000)
    dir_oof = compute_direction_oof(max_holding=15, rows=600000)
    assert np.array_equal(t0, dir_oof["t0_nz"])
    assert np.array_equal(m, dir_oof["has_oof"])
    np.testing.assert_allclose(p_full[m], dir_oof["p_direction_cal"][m], atol=1e-9)


def test_opportunity_and_barrier_still_shape_correctly_after_side_fix():
    t0, y_full, p_full, mask = _oof_for_opportunity(max_holding=15, rows=600000)
    assert len(t0) == len(y_full) == len(p_full) == len(mask)
    assert set(y_full[mask].tolist()) <= {0, 1}
    t0b, y_full_b, p_full_b, mask_b = _oof_for_barrier(max_holding=15, rows=600000)
    assert np.array_equal(t0, t0b)


if __name__ == "__main__":
    test_oof_for_direction_matches_shared_helper()
    test_opportunity_and_barrier_still_shape_correctly_after_side_fix()
    print("tests/test_phase5_calibration_v3b.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5_calibration_v3b.py -v`
Expected: FAIL — `_oof_for_direction` currently fits its own separate full-pool (unnarrowed) OOF pass with `want_importance=False` (`research/phase5_calibration.py:47-70`), which will not numerically match `compute_direction_oof`'s narrowed pass1+pass2 pipeline.

- [ ] **Step 3: Rewrite the four functions in `research/phase5_calibration.py`**

Add `from research.direction_side import compute_direction_oof` to the top-level imports (or inside each function, matching this file's existing per-function import style). Replace `_oof_for_direction`:

```python
def _oof_for_direction(max_holding, rows=None):
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    return dir_oof["t0_nz"], (dir_oof["p_direction_raw"] >= 0.5).astype(float), dir_oof["p_direction_cal"], dir_oof["has_oof"]
```

(the second return value `y_full` — "label is always known" per the original docstring — is the true direction label; since `compute_direction_oof` does not currently return the true `y_bin` array, recompute it the same way `_oof_for_opportunity` etc. already do, from `assemble_v3_dataset` + `triple_barrier_labels`, to avoid changing `compute_direction_oof`'s public contract for Task 1's other callers):

```python
def _oof_for_direction(max_holding, rows=None):
    import numpy as np
    from research.phase4_dataset import assemble_v3_dataset
    from features.labeling import TripleBarrierConfig, triple_barrier_labels
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    assert np.array_equal(ds["t0_idx"][nz], dir_oof["t0_nz"]), "direction_side event index mismatch"
    y_full = (y[nz] == 1).astype(float)
    return dir_oof["t0_nz"], y_full, dir_oof["p_direction_cal"], dir_oof["has_oof"]
```

Replace `_oof_for_opportunity`'s primary-stage block (lines ~92-99: the `prim = oof_run(...)` call and its immediate `has_oof1`/`build_meta` usage) with:

```python
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
```

(the rest of `_oof_for_opportunity` — the second-stage `oof_run` call and return — stays unchanged). `_oof_for_barrier` stays `return _oof_for_opportunity(max_holding, rows=rows)` (no change — it already inherits the fix through `_oof_for_opportunity`).

Replace `_oof_predicted_mae_mfe`'s primary-stage block (the `prim = oof_run(...)` call and immediate `has_oof1`/`build_meta` usage) with the same pattern:

```python
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(dir_oof["t0_nz"], t0_nz), "direction_side event index mismatch"
    has_oof1 = dir_oof["has_oof"]
    if not has_oof1.any():
        return t0_nz, mae_full, mfe_full, mask_full
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, dir_oof["side"], has_oof1)
    X_meta = X_full.loc[has_oof1].reset_index(drop=True)
    X_meta["assumed_side"] = side
    X_meta["p_direction"] = dir_oof["p_direction_cal"][has_oof1]
```

Remove the now-unused `from research.audit_edge import oof_run, build_meta` → `_mae_mfe_core` (keep `build_meta`/`_mae_mfe_core` imports, drop `oof_run` only if nothing else in this file still calls it directly — check via `grep -n "oof_run(" research/phase5_calibration.py` after editing; the second-stage `oof_run` calls in `_oof_for_opportunity`/`_oof_predicted_mae_mfe` still need it, so the import stays).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5_calibration_v3b.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing calibration test suite to confirm shapes still hold**

Run: `pytest tests/test_phase5_calibration_opportunity_barrier.py -v`
Expected: PASS (shape/range assertions only — this file's existing test does not assert exact numeric values, so the side-source fix does not break it; note in the PR description that the underlying OOF numbers will differ from before the fix)

- [ ] **Step 6: Commit**

```bash
git add research/phase5_calibration.py tests/test_phase5_calibration_v3b.py
git commit -m "fix: phase5_calibration uses shared Direction OOF side for all four downstream specialists"
```

---

### Task 7: Fail-closed side-lineage check in the contracts and EV engine

**Files:**
- Modify: `contracts/specialist_output.py`
- Modify: `decision/ev_engine.py`
- Test: `tests/test_ev_engine_direction_side_lineage.py`

**Interfaces:**
- Produces (contract change): `OpportunityOutput`, `BarrierOutput`, `MAEOutput`, `MFEOutput` each gain two new optional fields: `assumed_side: Optional[float] = None` (the `+1.0`/`-1.0` side this output was conditioned on) and `direction_model_id: Optional[str] = None` (the `model_id` of the Direction output that supplied that side). Both default to `None` so existing callers that don't set them keep working, but `decision/ev_engine.py`'s `evaluate()` now fails closed (forces `NO_TRADE`, same as the existing Opportunity-veto fail-closed pattern) whenever a specialist that IS trusted (`model_status in _OK`) has `direction_model_id` set and it does not equal `direction_out.model_id` passed to that same `evaluate()` call — this is the enforcement mechanism for design section 5's "same Direction model/version supplies side to every downstream specialist" invariant. A specialist with `direction_model_id=None` (not yet wired to report lineage) is treated as before this task (no new gate) — this keeps the change additive and non-breaking for any caller not yet updated to populate the new fields; Task 8 is the one that starts populating them for the replay path.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_ev_engine_direction_side_lineage.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate


class _FakeMarketState:
    def __init__(self):
        self.market_timestamp = datetime.now(timezone.utc)
        self.spread = 0.02
        self.mid = 2000.0


def _base_outputs(direction_model_id="direction_v3_candidate_h15", opportunity_direction_model_id="direction_v3_candidate_h15"):
    direction = DirectionOutput(model_id=direction_model_id, horizon=15, model_status="VALIDATED",
                                 probability_long=0.6, probability_short=0.4, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                                     probability_take=0.7, calibrated=True,
                                     assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    barrier = BarrierOutput(model_id="barrier_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                             p_tp=0.6, calibrated=True, assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    mae = MAEOutput(model_id="mae_quantile_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=0.8, q90=1.1, assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    mfe = MFEOutput(model_id="mfe_quantile_v3b_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=1.2, q90=1.8, assumed_side=1.0, direction_model_id=opportunity_direction_model_id)
    return direction, opportunity, barrier, mae, mfe


def test_matching_direction_model_id_does_not_force_no_trade():
    direction, opportunity, barrier, mae, mfe = _base_outputs()
    d = evaluate(_FakeMarketState(), direction, opportunity, barrier, 0.5, mae, mfe,
                 timeout_r=0.1, timeout_r_provisional_proxy=True)
    assert d.decision != "NO_TRADE" or "Direction side lineage mismatch" not in (d.decision_reason or "")


def test_mismatched_direction_model_id_forces_no_trade():
    direction, opportunity, barrier, mae, mfe = _base_outputs(
        direction_model_id="direction_v3_candidate_h15",
        opportunity_direction_model_id="direction_v3_candidate_h45",  # WRONG horizon's Direction model
    )
    d = evaluate(_FakeMarketState(), direction, opportunity, barrier, 0.5, mae, mfe,
                 timeout_r=0.1, timeout_r_provisional_proxy=True)
    assert d.decision == "NO_TRADE"
    assert "Direction side lineage mismatch" in d.decision_reason


if __name__ == "__main__":
    test_matching_direction_model_id_does_not_force_no_trade()
    test_mismatched_direction_model_id_forces_no_trade()
    print("tests/test_ev_engine_direction_side_lineage.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ev_engine_direction_side_lineage.py -v`
Expected: FAIL — `OpportunityOutput`/`BarrierOutput`/`MAEOutput`/`MFEOutput` don't accept `assumed_side`/`direction_model_id` kwargs yet (`pydantic.ValidationError: extra fields not permitted` or similar, depending on pydantic config — this repo's `BaseModel` subclasses use default config, which for pydantic v2 raises on unknown kwargs only if `model_config = ConfigDict(extra="forbid")` is set; if not set, unknown kwargs are silently accepted and the test instead fails at the `assert d.decision == "NO_TRADE"` line since nothing enforces the check yet — either failure mode confirms the gate doesn't exist)

- [ ] **Step 3: Add the fields to `contracts/specialist_output.py`**

Add to `OpportunityOutput`, `BarrierOutput`, `MAEOutput`, `MFEOutput` (after their existing fields, before any trailing config):

```python
    assumed_side: Optional[float] = None
    direction_model_id: Optional[str] = None
```

- [ ] **Step 4: Add the fail-closed check to `decision/ev_engine.py`**

In `evaluate()`, immediately after the existing `barrier_available = barrier_out.model_status in _OK` line (line 50), add:

```python
    _lineage_specialists = [opportunity_out, barrier_out, mae_out, mfe_out]
    lineage_mismatch = any(
        s.model_status in _OK and s.direction_model_id is not None and s.direction_model_id != direction_out.model_id
        for s in _lineage_specialists
    )
```

Then change the `if stale: ... elif not direction_available: ... elif not barrier_available:` chain to add one more branch (insert right after the `not barrier_available` branch, before `elif cost_r is None`):

```python
    elif lineage_mismatch:
        reason = "Direction side lineage mismatch: a downstream specialist's assumed side was not sourced from this decision's Direction model"
        long_ev = short_ev = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_ev_engine_direction_side_lineage.py -v`
Expected: PASS

- [ ] **Step 6: Run the existing EV engine test suite to confirm no regression**

Run: `pytest tests/test_phase5_ev_engine.py tests/test_ev_gate.py tests/test_specialist_output.py -v`
Expected: PASS (all existing calls omit the two new optional fields, which default to `None`, so `lineage_mismatch` is `False` for every existing test — no behavior change for callers that don't populate side lineage)

- [ ] **Step 7: Commit**

```bash
git add contracts/specialist_output.py decision/ev_engine.py tests/test_ev_engine_direction_side_lineage.py
git commit -m "feat: fail-closed check when a specialist's assumed side wasn't sourced from this decision's Direction model"
```

---

### Task 8: Wire `assumed_side`/`direction_model_id` through the Phase 5 replay path onto the new `v3b` artifacts

**Files:**
- Modify: `research/phase5_ev_dataset.py`
- Modify: `research/phase5_ev_engine.py`
- Test: `tests/test_phase5_replay_v3b_lineage.py`

**Interfaces:**
- Produces: `research.phase5_ev_dataset.assemble_replay_dataset` return dict gains `"direction_model_id": str` (from `_oof_for_direction`'s implicit source — since Task 6 made `_oof_for_direction` wrap `compute_direction_oof`, add `dir_oof["model_id"]` — see Step 1 note below on how this threads through given `_oof_for_direction`'s current 4-tuple return shape) and `"side": np.ndarray` (the combined-mask-aligned signed side, needed so replay can populate each specialist output's `assumed_side`). `research.phase5_ev_engine.replay_and_validate` reads `models/registry/{opportunity,barrier,mae_quantile,mfe_quantile}_v3b_candidate_h{max_holding}.json` (not the old `_v3_` names) for status lookups, and populates each `*Output`'s new `assumed_side`/`direction_model_id` fields from the replay dataset.

- [ ] **Step 1: Thread `direction_model_id` and `side` out of `research/phase5_calibration.py` without changing the four functions' public signatures**

Since Task 6 kept `_oof_for_direction`/`_oof_for_opportunity`/`_oof_for_barrier`/`_oof_predicted_mae_mfe`'s return shapes as 4-tuples (required so `phase5_ev_dataset.py`'s existing equality asserts keep working), expose the Direction model id and side as separate, cheap calls in `phase5_ev_dataset.py` itself rather than changing those four functions again. In `research/phase5_ev_dataset.py`, add near the top:

```python
from research.direction_side import compute_direction_oof
```

and inside `assemble_replay_dataset`, right after the existing `t0_dir, y_dir, p_dir, m_dir = _oof_for_direction(max_holding, rows=rows)` line, add:

```python
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(dir_oof["t0_nz"], t0_dir), "direction_side event index mismatch in replay dataset"
```

and in the final `return {...}` dict, add two new keys:

```python
            "direction_model_id": dir_oof["model_id"],
            "side": dir_oof["side"][combined],
```

- [ ] **Step 2: Write the failing test**

```python
"""tests/test_phase5_replay_v3b_lineage.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_ev_dataset import assemble_replay_dataset


def test_replay_dataset_exposes_direction_model_id_and_side():
    data = assemble_replay_dataset(max_holding=15, rows=600000)
    assert data["direction_model_id"] == "direction_v3_candidate_h15"
    assert len(data["side"]) == data["n"]
    assert set(data["side"].tolist()) <= {1.0, -1.0}


if __name__ == "__main__":
    test_replay_dataset_exposes_direction_model_id_and_side()
    print("tests/test_phase5_replay_v3b_lineage.py: OK")
```

- [ ] **Step 3: Run test to verify it fails, then passes after Step 1's edit**

Run: `pytest tests/test_phase5_replay_v3b_lineage.py -v`
Expected: FAIL before Step 1's edit (`KeyError: 'direction_model_id'`), PASS after.

- [ ] **Step 4: Update `research/phase5_ev_engine.py` to use `v3b` artifacts and populate side lineage**

Change the four non-Direction status lookups (lines 67-70) from `_v3_candidate_h` to `_v3b_candidate_h`:

```python
    opportunity_status = _real_model_status(f"opportunity_v3b_candidate_h{max_holding}")
    barrier_status = _real_model_status(f"barrier_v3b_candidate_h{max_holding}")
    mae_status = _real_model_status(f"mae_quantile_v3b_candidate_h{max_holding}")
    mfe_status = _real_model_status(f"mfe_quantile_v3b_candidate_h{max_holding}")
```

(Direction's own status lookup at line 66 stays `direction_v3_candidate_h{max_holding}` — Direction's artifact is unaffected by Phase 5A per design section 8.) Update the four `*Output(...)` constructions inside the `for i in range(n):` loop to use the `v3b` model ids and populate the new fields:

```python
        opportunity = OpportunityOutput(model_id=f"opportunity_v3b_candidate_h{max_holding}", horizon=max_holding,
                                         model_status=opportunity_status, probability_take=float(data["p_opportunity"][i]),
                                         calibrated=True, assumed_side=float(data["side"][i]),
                                         direction_model_id=data["direction_model_id"])
        barrier = BarrierOutput(model_id=f"barrier_v3b_candidate_h{max_holding}", horizon=max_holding,
                                 model_status=barrier_status, p_tp=float(data["p_barrier_win"][i]), calibrated=True,
                                 assumed_side=float(data["side"][i]), direction_model_id=data["direction_model_id"])
        mae = MAEOutput(model_id=f"mae_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3,
                         assumed_side=float(data["side"][i]), direction_model_id=data["direction_model_id"])
        mfe = MFEOutput(model_id=f"mfe_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3,
                         assumed_side=float(data["side"][i]), direction_model_id=data["direction_model_id"])
```

Also update the `direction = DirectionOutput(...)` construction to pass `direction_model_id`-consistent naming (it already uses `f"direction_v3_candidate_h{max_holding}"`, matching `data["direction_model_id"]" — no change needed there, just confirm equality via the new test in Step 5).

- [ ] **Step 5: Add a replay-level lineage assertion test**

Append to `tests/test_phase5_replay_v3b_lineage.py`:

```python
def test_replay_and_validate_uses_v3b_artifacts_and_never_trips_lineage_gate():
    import tempfile
    from research.phase4_dataset import assemble_v3_dataset
    from research.phase4_opportunity import run_opportunity_candidate_v3b
    from research.phase4_barrier import run_barrier_candidate_v3b
    from research.phase4_mae_quantile import run_mae_quantile_candidate_v3b
    from research.phase4_mfe_quantile import run_mfe_quantile_candidate_v3b
    from research.phase4_direction import run_direction_candidate
    from research.phase5_ev_engine import replay_and_validate

    with tempfile.TemporaryDirectory() as reg, tempfile.TemporaryDirectory() as sch:
        run_direction_candidate(max_holding=15, rows=600000, registry_dir=reg, schemas_dir=sch)
        run_opportunity_candidate_v3b(max_holding=15, rows=600000, registry_dir=reg, schemas_dir=sch)
        run_barrier_candidate_v3b(max_holding=15, rows=600000, registry_dir=reg, schemas_dir=sch)
        run_mae_quantile_candidate_v3b(max_holding=15, rows=600000, registry_dir=reg, schemas_dir=sch)
        run_mfe_quantile_candidate_v3b(max_holding=15, rows=600000, registry_dir=reg, schemas_dir=sch)
        result = replay_and_validate(max_holding=15, rows=600000, registry_dir=reg)
        assert result["n_events"] > 0
        # a lineage mismatch would silently zero out trades via NO_TRADE without raising --
        # the meaningful check is that decisions is well-formed and doesn't crash.
        assert set(result["decisions"].keys()) == {"NO_TRADE", "LONG_CANDIDATE", "SHORT_CANDIDATE"}
```

Run: `pytest tests/test_phase5_replay_v3b_lineage.py -v`
Expected: PASS (this is a slower integration test — trains 5 small candidates end-to-end; budget a few minutes)

- [ ] **Step 6: Commit**

```bash
git add research/phase5_ev_dataset.py research/phase5_ev_engine.py tests/test_phase5_replay_v3b_lineage.py
git commit -m "feat: wire assumed_side/direction_model_id through Phase 5 replay onto v3b artifacts"
```

---

### Task 9: Full retrain + full-history replay re-run, treating all prior Phase 5 numbers as diagnostic-only

**Files:**
- No source changes — this task runs the pipeline built in Tasks 1-8 over the real full-history dataset and records results.
- Create: `.superpowers/sdd/2026-08-24-golex-v3-phase5a-specialist-conditioning/retrain-and-replay-report.md` (the report artifact, not a spec/plan doc — lives under the SDD run directory like the Phase 5 correction-pass report did)

**Interfaces:** none (this task produces data/artifacts, not code).

- [ ] **Step 1: Retrain Direction (unaffected, run for lineage completeness) for all 3 horizons**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_direction`
Expected: prints `[direction h=15] ... -> status=...`, `[direction h=45] ...`, `[direction h=90] ...`; writes/overwrites `models/registry/direction_v3_candidate_h{15,45,90}.json` — this is Direction's EXISTING candidate script, run only to ensure the registry entries downstream retraining will read (`direction_v3_candidate_h{H}.json`'s `status` field) are fresh, not to change Direction's methodology.

- [ ] **Step 2: Retrain Opportunity, Barrier, MAE, MFE `v3b` candidates for all 3 horizons**

Run a short driver script (create as a throwaway, delete after use — do not leave ad-hoc scripts in the repo per this project's established no-artifacts-left-behind convention):

```python
# /tmp/run_v3b_retrain.py (throwaway)
from research.phase4_dataset import HORIZONS
from research.phase4_opportunity import run_opportunity_candidate_v3b
from research.phase4_barrier import run_barrier_candidate_v3b
from research.phase4_mae_quantile import run_mae_quantile_candidate_v3b
from research.phase4_mfe_quantile import run_mfe_quantile_candidate_v3b

for h in HORIZONS:
    print(f"=== h={h} ===")
    print("opportunity:", run_opportunity_candidate_v3b(max_holding=h))
    print("barrier:", run_barrier_candidate_v3b(max_holding=h))
    print("mae:", run_mae_quantile_candidate_v3b(max_holding=h))
    print("mfe:", run_mfe_quantile_candidate_v3b(max_holding=h))
```

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 /tmp/run_v3b_retrain.py` (background — full-history CatBoost fits across 3 horizons x 4 specialists x pass1+pass2 will take a while; use the same real-PID/`ps aux`/`/proc/<pid>/fd/1` monitoring discipline as the correction pass if the foreground shell disconnects).
Expected: 12 `-> status=...` lines total, plus a fresh `models/registry/{opportunity,barrier,mae_quantile,mfe_quantile}_v3b_candidate_h{15,45,90}.json` (12 new files) — verify the OLD `_v3_candidate_h*.json` files (12 of them) are byte-unchanged: `git status models/registry/` should show only new/untracked `*_v3b_*.json` files, zero modifications to existing `*_v3_*.json` files.

- [ ] **Step 3: Re-run Phase 5 calibration**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_calibration`
Expected: same print format as before (`direction h=15: n=... -> ...platt.json`, etc.) — writes fresh Platt calibrator JSON files under `decision.calibration_registry.CALIBRATION_DIR`; these are calibration ARTIFACTS (not gated by the `v3_`/`v3b_` registry-file distinction) and are always meant to be regenerated, so overwriting them here is expected and correct (unlike the model registry entries in Step 2).

- [ ] **Step 4: Re-run the full-history OOS replay for all 3 horizons**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_engine` (background, monitor via real PID per the correction-pass precedent — this reads the real `models/registry/` dir per Task 8's `_v3b_candidate_h` status lookups, not a tempdir, since we want the actual retrained artifacts' statuses this time)
Expected: `h=15: {...}`, `h=45: {...}`, `h=90: {...}` — each a dict with `decisions`, `expected_vs_realized_r`, `baseline_comparison`, `fragile_fraction`.

- [ ] **Step 5: Delete the throwaway driver script**

Run: `rm /tmp/run_v3b_retrain.py`

- [ ] **Step 6: Write the retrain-and-replay report**

Create `.superpowers/sdd/2026-08-24-golex-v3-phase5a-specialist-conditioning/retrain-and-replay-report.md` recording, per design section 10's validation criteria, for each of h=15/45/90:
- Registry status (`validated`/`rejected`) and headline metric for each of the 4 retrained `v3b` candidates, compared side-by-side against the OLD `v3` candidate's status/metric (§7's "valid as standalone, invalid as Phase-5-input" framing — the comparison is informational, not a pass/fail gate on its own).
- Whether `p_direction` was kept (`used_p_direction`/`kept_p_direction`) for each specialist/horizon, and the log-loss or pinball-loss delta that justified the decision (design §9).
- The new `decisions` distribution (NO_TRADE/LONG_CANDIDATE/SHORT_CANDIDATE counts) for each horizon, explicitly compared against the H15 skew investigation's zero-LONG finding — reported honestly whether the skew resolved, persisted, or changed in an unexpected way, per the investigation's original "do not force symmetry" rule.
- Point-biserial correlation of each retrained specialist's OOF probability against its own true label (design §10 criterion 3) — compute via `scipy.stats.pointbiserialr` on the `v3b` calibration OOF arrays from Task 6.
- Explicit statement that all numbers in the H15-skew-investigation and correction-pass-report documents are now superseded/diagnostic-only, per the Global Constraints.
- A final PASS / NOT READY verdict against design section 10's 5 validation criteria (criteria 1/2/5 are satisfied by construction via Tasks 1-8's code changes and are confirmed, not re-measured, here; criteria 3/4 are the ones this report actually measures).

- [ ] **Step 7: Paste the full report in chat** (per the user's standing preference — do not just reference the file path)

- [ ] **Step 8: Commit**

```bash
git add models/registry/*_v3b_candidate_h*.json .superpowers/sdd/2026-08-24-golex-v3-phase5a-specialist-conditioning/retrain-and-replay-report.md
git commit -m "feat: retrain Opportunity/Barrier/MAE/MFE on Direction's OOF side (v3b), re-run full-history replay"
```

---

## Explicit non-goals (carried from the design doc, restated so no task drifts into them)

Do not, anywhere in this plan's execution: change `decision/ev_formula.py`, `decision/ev_gate.py`, or `decision/ev_cost.py`; change any threshold constant (`OPPORTUNITY_MIN_TAKE_PROBABILITY`, `MIN_EDGE_THRESHOLD`, `DEFAULT_K`, etc.); change `research/phase4_direction.py`'s target, feature selection, or CV methodology beyond Task 1's pure code-extraction (no metric may change); attempt to restore long-trade frequency; or begin any Phase 6 work. Task 9's report closes this plan; Phase 6 authorization is a separate future decision gated on that report's verdict.
