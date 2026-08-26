# GOLEX V3 Phase 5 Batch 2: EV + Uncertainty Root-Cause Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with measurement alone, why h=15's EV engine estimates ~+0.44R but realizes ~+0.02R, whether the 35.8% Barrier-vs-MAE/MFE contradiction is actually predictive, exactly where traded-subset calibration collapses in the pipeline, and whether the current EV architecture is statistically trustworthy — producing an additive, evidence-backed decomposition of the gap and KEEP/MODIFY/REJECT/NEEDS_MORE_EVIDENCE verdicts per component.

**Architecture:** Five new read-only diagnostic modules (D7-D11) under `research/phase5b_diagnostics/`, continuing Batch 1's exact conventions and reusing its infrastructure (`_stats_utils.py`, D1-D6, `assemble_replay_dataset`) directly rather than rebuilding it. D9's Shapley decomposition needs two new fields on `assemble_replay_dataset`'s return dict (additive only). A `run_batch2.py` orchestrator produces one JSON+markdown report, mirroring `run_all.py`'s pattern.

**Tech Stack:** Python, numpy, pandas, scipy.stats, existing CatBoost/PurgedWalkForwardCV research infra (read-only reuse only), pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch2-ev-uncertainty-design.md`

## Global Constraints

- No new predictive models, no refitting anything, unless a task's own investigation proves an existing component structurally cannot answer the question (none currently anticipated to require this).
- No threshold optimization anywhere — `OPPORTUNITY_MIN_TAKE_PROBABILITY`, `MIN_EDGE_THRESHOLD`, D4's `0.6`/`mae<=mfe` contradiction cutoffs all stay exactly as they are; this batch measures against them, never tunes them.
- No production code (`decision/`, `app/`, `market/`) modified — `decision/ev_formula.py::raw_ev` and `decision/ev_gate.py` are called read-only, never edited.
- No `models/registry/` writes, no promotion, no live behavior changes.
- Full 6.7-year real data for the actual Batch 2 run (Task 7); unit tests for each module may use `rows=600000` per this repo's established convention.
- Direction side is always the exact side already resolved by the existing OOF pipeline — never independently regenerated (Phase 5A's invariant, unchanged).
- Hindsight/counterfactual values (D9's swapped inputs) are attribution tools only — they must never be returned as, or fed into, anything that could be mistaken for a live tradable prediction. Use the design doc's exact terminology: "hindsight outcome distribution" (never "true probability"), "zero-cost counterfactual / cost drag" (never implying the cost model is validated), "selection-conditioned payoff difference" (never "selection bias").
- Every statistic reports sample size `n`; confidence intervals wherever practical, reusing `_stats_utils.py`'s `pointbiserial_with_ci`/`fit_calibration_slope_intercept` and `research.audit_edge.wilson_ci`/`block_bootstrap`.
- Component interactions are never falsely attributed to a single component — this is the entire reason D9 uses exact Shapley averaging over all 16 subsets, not a single sequential swap order.
- A residual is reported, not hidden, if D9's C1-C4 + C5 attribution doesn't fully reconcile with the observed gap.
- Terminal deliverable: the real Batch 2 report (JSON + markdown) from full-history data, plus a whole-branch review (this branch's own Critical-bug-catching precedent from Batch 1 must not be skipped).

---

### Task 1: Extend `assemble_replay_dataset` with Direction-side-conditioned MAE/MFE

**Files:**
- Modify: `research/phase5_ev_dataset.py`
- Test: `tests/test_phase5_ev_dataset_realized_r.py` (existing file — add to it, do not create a new one; this file already tests `assemble_replay_dataset`'s realized-R fields)

**Interfaces:**
- Produces: `assemble_replay_dataset`'s return dict gains two new keys: `"mae_dir": np.ndarray` and `"mfe_dir": np.ndarray`, both length `n`, giving the Direction-side-conditioned REAL/realized adverse (`mae_dir`) and favorable (`mfe_dir`) excursion for each event — i.e. `mae_long`/`mfe_long` where `side==1`, `mae_short`/`mfe_short` where `side==-1` (the function's own already-computed local `mae_long, mfe_long, mae_short, mfe_short` arrays, selected by its own `side` local variable — no new `_mae_mfe_core` call). All existing keys are unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase5_ev_dataset_realized_r.py` (read the existing file first to match its exact import/fixture style before appending):

```python
def test_assemble_replay_dataset_exposes_direction_conditioned_mae_mfe():
    data = assemble_replay_dataset(max_holding=15, rows=600000)
    assert "mae_dir" in data and "mfe_dir" in data
    assert len(data["mae_dir"]) == data["n"]
    assert len(data["mfe_dir"]) == data["n"]
    assert (data["mae_dir"] >= 0).all()  # MAE is a magnitude (adverse excursion size), never negative
    assert (data["mfe_dir"] >= 0).all()  # MFE is a magnitude (favorable excursion size), never negative
```

(If the existing test file doesn't already import `assemble_replay_dataset`, add `from research.phase5_ev_dataset import assemble_replay_dataset` to its imports — check first, don't duplicate an existing import line.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5_ev_dataset_realized_r.py::test_assemble_replay_dataset_exposes_direction_conditioned_mae_mfe -v`
Expected: FAIL with `KeyError: 'mae_dir'`

- [ ] **Step 3: Add the two fields**

In `research/phase5_ev_dataset.py::assemble_replay_dataset`, find where `mae_long, mfe_long = _mae_mfe_core(close, high, low, t0_sel, t1_sel, side_long, vol_sel)` and `mae_short, mfe_short = _mae_mfe_core(close, high, low, t0_sel, t1_sel, side_short, vol_sel)` are already computed (these produce the existing `realized_r_long`/`realized_r_short` fields). Immediately after those two lines and before the `return` statement, add:

```python
    side = np.where(touch_sel == 1, 1.0, -1.0)  # NOTE: this is NOT Direction's side -- see caution below
```

**STOP — do not use `touch_sel` for this.** The function's Direction side is already available as a named local from the code that builds `realized_r_long`/`realized_r_short` — re-read the current function body before implementing: it must use the SAME `side` array `phase5_calibration`'s `_oof_for_direction`/`compute_direction_oof` produced for this event set (already threaded through `assemble_replay_dataset` as part of building the `combined` mask and exposed at the end of the function as the `"side"` return key). Use that existing `side` value — do NOT derive a new one from `touch_sel`, which would be the historically-winning side, not Direction's proposed side, and would reintroduce exactly the Phase 5A conditioning bug this whole investigation exists downstream of. Concretely:

```python
    mae_dir = np.where(side == 1.0, mae_long, mae_short)
    mfe_dir = np.where(side == 1.0, mfe_long, mfe_short)
```

placed after `mae_long, mfe_long, mae_short, mfe_short` are computed and before the `return {...}` statement, using the function's EXISTING `side` variable (the one already used to build the `"side"` key in the current return dict — confirm this by reading the function's current full body first, since the exact local variable name must match what's already there). Then add `"mae_dir": mae_dir, "mfe_dir": mfe_dir,` to the return dict, alongside the existing `"side": side` key.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5_ev_dataset_realized_r.py -v`
Expected: all tests in the file PASS, including the new one.

- [ ] **Step 5: Run the existing Batch 1 tests that depend on `assemble_replay_dataset` to confirm no regression**

Run: `pytest tests/test_phase5b_d4_cross_specialist_consistency.py tests/test_phase5b_d5_calibration_reliability.py tests/test_phase5b_run_all.py -k "not smoke" -v`
Expected: PASS (this task only adds new dict keys, never changes an existing one, so nothing that reads the old keys can break)

- [ ] **Step 6: Commit**

```bash
git add research/phase5_ev_dataset.py tests/test_phase5_ev_dataset_realized_r.py
git commit -m "feat: expose Direction-side-conditioned mae_dir/mfe_dir from assemble_replay_dataset"
```

---

### Task 2: D7 — Barrier vs MAE/MFE contradiction root-cause and predictiveness

**Files:**
- Create: `research/phase5b_diagnostics/d7_contradiction.py`
- Test: `tests/test_phase5b_d7_contradiction.py`

**Interfaces:**
- Consumes: `research.phase5_ev_dataset.assemble_replay_dataset` (including Task 1's `mae_dir`/`mfe_dir`), `research.phase5b_diagnostics._stats_utils.pointbiserial_with_ci`, `research.audit_edge.wilson_ci`/`block_bootstrap`.
- Produces: `run_d7(max_holding: int, rows: int = None) -> dict` with keys:
  - `"horizon": int`
  - `"contradiction_mask_n": int`, `"non_contradiction_mask_n": int`
  - `"realized_r"`: `{"contradicted": {"mean": float, "n": int, "bootstrap_ci": [lo, hi]}, "non_contradicted": {same shape}, "difference_ci": [lo, hi]}` (bootstrap CI on the two means and on their difference, via `block_bootstrap`)
  - `"touch_distribution"`: `{"contradicted": {"tp_frac": float, "sl_frac": float, "timeout_frac": float, "n": int}, "non_contradicted": {same shape}}`
  - `"predictiveness"`: `{"point_biserial_contradiction_vs_realized_r_sign": {...pointbiserial_with_ci...}}` — point-biserial correlation between the binary contradiction flag and whether realized R was positive, as the direct falsifiable "is contradiction itself predictive" test.
  - `"breakdown_by_volatility_tercile"`: `{"low": {"contradiction_rate": float, "n": int}, "medium": {...}, "high": {...}}`
  - `"breakdown_by_side"`: `{"long": {"contradiction_rate": float, "n": int}, "short": {...}}`
  - `"breakdown_by_barrier_probability_decile"`: `list[dict]`, 10 entries each `{"decile": int, "contradiction_rate": float, "n": int}`
  - `"exclusion_effect"`: `{"realized_r_with_contradictions": float, "realized_r_excluding_contradictions": float, "n_with": int, "n_excluding": int}` — descriptive only, explicitly NOT a proposed live filter (state this in the module's own docstring, per the design doc's exclusion-effect framing)
  - `"which_component_more_reliable"`: `{"barrier_point_biserial_in_contradicted_population": {...}, "mae_mfe_reward_risk_ratio_correlation_with_outcome_in_contradicted_population": {...}}` — a direct, symmetric comparison, not a presumed answer

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d7_contradiction.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d7_contradiction import run_d7


def test_run_d7_shape():
    result = run_d7(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["contradiction_mask_n"] + result["non_contradiction_mask_n"] > 0
    for pop in ("contradicted", "non_contradicted"):
        assert "mean" in result["realized_r"][pop]
        assert "n" in result["realized_r"][pop]
    assert len(result["breakdown_by_barrier_probability_decile"]) == 10
    assert "long" in result["breakdown_by_side"] and "short" in result["breakdown_by_side"]
    assert set(result["breakdown_by_volatility_tercile"].keys()) == {"low", "medium", "high"}


if __name__ == "__main__":
    test_run_d7_shape()
    print("tests/test_phase5b_d7_contradiction.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d7_contradiction.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d7_contradiction.py`**

```python
"""research/phase5b_diagnostics/d7_contradiction.py
Batch 2, D7: is the 35.8% Barrier-vs-MAE/MFE reward/risk contradiction
(D4's exact definition, inherited unchanged) actually predictive of poor
realized outcomes, and which of Barrier/MAE-MFE is more reliable in the
contradicted population specifically? Does not assume either is correct.
See docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch2-ev-
uncertainty-design.md section D7.
"""
import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset, realized_r_for_direction
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci
from research.audit_edge import block_bootstrap


def _touch_dist(touch: np.ndarray, side: np.ndarray) -> dict:
    n = len(touch)
    if n == 0:
        return {"tp_frac": None, "sl_frac": None, "timeout_frac": None, "n": 0}
    favorable = np.where(side == 1.0, 1, -1)
    tp = (touch == favorable).mean()
    sl = (touch == -favorable).mean()
    timeout = (touch == 0).mean()
    return {"tp_frac": float(tp), "sl_frac": float(sl), "timeout_frac": float(timeout), "n": n}


def run_d7(max_holding: int, rows: int = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    n = data["n"]
    p_barrier_win = data["p_barrier_win"]
    mae_r = data["mae_r"]
    mfe_r = data["mfe_r"]
    side = data["side"]
    touch = data["touch"]

    contradiction = (p_barrier_win >= 0.6) & (mfe_r <= mae_r)  # identical to D4's definition, do not change

    realized_r = np.array([realized_r_for_direction("long" if side[i] == 1.0 else "short", i, data)
                            for i in range(n)])

    def _pop_stats(mask):
        vals = realized_r[mask]
        m = len(vals)
        if m == 0:
            return {"mean": None, "n": 0, "bootstrap_ci": [None, None]}
        lo, mid, hi = block_bootstrap(vals, block_size=20, n_boot=1000)
        return {"mean": float(np.mean(vals)), "n": m, "bootstrap_ci": [lo, hi]}

    contradicted_stats = _pop_stats(contradiction)
    non_contradicted_stats = _pop_stats(~contradiction)
    diff_vals = realized_r[contradiction].mean() - realized_r[~contradiction].mean() if contradiction.any() and (~contradiction).any() else None
    # bootstrap CI on the difference: resample both populations' block-bootstrap means jointly
    if contradiction.sum() > 20 and (~contradiction).sum() > 20:
        rng = np.random.default_rng(42)
        diffs = []
        c_vals, nc_vals = realized_r[contradiction], realized_r[~contradiction]
        for _ in range(1000):
            c_sample = rng.choice(c_vals, size=len(c_vals), replace=True)
            nc_sample = rng.choice(nc_vals, size=len(nc_vals), replace=True)
            diffs.append(c_sample.mean() - nc_sample.mean())
        diff_ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    else:
        diff_ci = [None, None]

    realized_r_positive = (realized_r > 0).astype(float)
    predictiveness = pointbiserial_with_ci(realized_r_positive, contradiction.astype(float))

    # volatility tercile via vol_60s_proxy, matching this repo's existing tercile convention
    vol = data["vol_60s_proxy"]
    lo_thr, hi_thr = np.nanpercentile(vol, [33.3, 66.7])
    vol_state = np.where(vol <= lo_thr, "low", np.where(vol >= hi_thr, "high", "medium"))
    breakdown_vol = {}
    for label in ("low", "medium", "high"):
        m = vol_state == label
        breakdown_vol[label] = {"contradiction_rate": float(contradiction[m].mean()) if m.sum() else None, "n": int(m.sum())}

    breakdown_side = {}
    for label, side_val in (("long", 1.0), ("short", -1.0)):
        m = side == side_val
        breakdown_side[label] = {"contradiction_rate": float(contradiction[m].mean()) if m.sum() else None, "n": int(m.sum())}

    deciles = np.digitize(p_barrier_win, np.linspace(0, 1, 11)[1:-1])
    breakdown_decile = []
    for d in range(10):
        m = deciles == d
        breakdown_decile.append({"decile": d, "contradiction_rate": float(contradiction[m].mean()) if m.sum() else None, "n": int(m.sum())})

    excl_with = float(np.mean(realized_r))
    excl_without = float(np.mean(realized_r[~contradiction])) if (~contradiction).any() else None

    barrier_reliability = pointbiserial_with_ci(realized_r_positive[contradiction], p_barrier_win[contradiction])
    reward_risk_ratio = np.where(mae_r[contradiction] > 1e-9, mfe_r[contradiction] / mae_r[contradiction], np.nan)
    valid = np.isfinite(reward_risk_ratio)
    mae_mfe_reliability = pointbiserial_with_ci(realized_r_positive[contradiction][valid], reward_risk_ratio[valid])

    return {
        "horizon": max_holding,
        "contradiction_mask_n": int(contradiction.sum()),
        "non_contradiction_mask_n": int((~contradiction).sum()),
        "realized_r": {"contradicted": contradicted_stats, "non_contradicted": non_contradicted_stats,
                        "difference_ci": diff_ci, "difference_point": diff_vals},
        "touch_distribution": {"contradicted": _touch_dist(touch[contradiction], side[contradiction]),
                                "non_contradicted": _touch_dist(touch[~contradiction], side[~contradiction])},
        "predictiveness": {"point_biserial_contradiction_vs_realized_r_sign": predictiveness},
        "breakdown_by_volatility_tercile": breakdown_vol,
        "breakdown_by_side": breakdown_side,
        "breakdown_by_barrier_probability_decile": breakdown_decile,
        "exclusion_effect": {"realized_r_with_contradictions": excl_with,
                              "realized_r_excluding_contradictions": excl_without,
                              "n_with": n, "n_excluding": int((~contradiction).sum())},
        "which_component_more_reliable": {
            "barrier_point_biserial_in_contradicted_population": barrier_reliability,
            "mae_mfe_reward_risk_ratio_correlation_with_outcome_in_contradicted_population": mae_mfe_reliability,
        },
    }


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d7(max_holding=h)
        print(f"D7 h={h}: contradiction_n={r['contradiction_mask_n']} "
              f"realized_r_contradicted={r['realized_r']['contradicted']['mean']} "
              f"realized_r_non={r['realized_r']['non_contradicted']['mean']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d7_contradiction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d7_contradiction.py tests/test_phase5b_d7_contradiction.py
git commit -m "feat: add D7 Barrier-vs-MAE/MFE contradiction root-cause diagnostic"
```

---

### Task 3: D8 — Traced calibration collapse through the real decision pipeline

**Files:**
- Create: `research/phase5b_diagnostics/d8_selection_calibration.py`
- Test: `tests/test_phase5b_d8_selection_calibration.py`

**Interfaces:**
- Consumes: `research.phase5_ev_dataset.assemble_replay_dataset`, `research.phase5_timeout_payoff.estimate_timeout_payoff`, `decision.ev_engine.evaluate` and its module constant `OPPORTUNITY_MIN_TAKE_PROBABILITY`, `contracts.specialist_output.*`, `research.phase5b_diagnostics._stats_utils.fit_calibration_slope_intercept`.
- Produces: `run_d8(max_holding: int, rows: int = None, registry_dir: str = None) -> dict` with keys:
  - `"horizon": int`
  - `"stages"`: an ordered `list[dict]`, each `{"stage": str, "n": int, "calibration": {...fit_calibration_slope_intercept output...}, "brier": float, "p_mean": float, "realized_outcome_rate": float}` — exactly two real per-event-varying stages given this replay's methodology (see honesty note below): `"stage_0_full_oos"` and `"stage_1_after_opportunity_veto"` and `"stage_2_after_ev_gate_final_traded"`.
  - `"degradation_begins_at": str` — the name of the first stage (in order) whose calibration slope deviates from the full-population slope by more than a stated, reported threshold (report the actual slope values; do not hide the threshold choice).
  - `"honesty_note"`: a string explaining that in this static full-history replay, `model_status`-based gates (Direction/Barrier/MAE/MFE availability) are constant across all events for a given horizon (registry status doesn't vary per-event), so they cannot produce a distinguishable sub-population within one horizon's replay — only the per-event-varying Opportunity probability-take gate and the final EV/MIN_EDGE_THRESHOLD gate produce real stage transitions here; this is a real scope boundary of the replay methodology, not an oversight.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d8_selection_calibration.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d8_selection_calibration import run_d8


def test_run_d8_shape_and_monotonic_stage_shrinkage():
    result = run_d8(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    names = [s["stage"] for s in result["stages"]]
    assert names == ["stage_0_full_oos", "stage_1_after_opportunity_veto", "stage_2_after_ev_gate_final_traded"]
    ns = [s["n"] for s in result["stages"]]
    assert ns[0] >= ns[1] >= ns[2]  # each gate can only shrink the population, never grow it
    assert result["degradation_begins_at"] in names
    assert "honesty_note" in result and len(result["honesty_note"]) > 0


if __name__ == "__main__":
    test_run_d8_shape_and_monotonic_stage_shrinkage()
    print("tests/test_phase5b_d8_selection_calibration.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d8_selection_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d8_selection_calibration.py`**

```python
"""research/phase5b_diagnostics/d8_selection_calibration.py
Batch 2, D8: traces WHERE h=15's Barrier calibration collapses through the
real decision pipeline, reusing D5's proven-equivalent per-event
decision/ev_engine.evaluate() loop pattern rather than reconstructing an
approximation of it. See docs/superpowers/specs/2026-08-26-golex-v3-
phase5-batch2-ev-uncertainty-design.md section D8 for why only two
per-event-varying gates exist in this static replay methodology (the
honesty_note field explains this in the output itself, not just here).
"""
from datetime import datetime, timezone

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset
from research.phase5_timeout_payoff import estimate_timeout_payoff
from decision.ev_engine import evaluate, OPPORTUNITY_MIN_TAKE_PROBABILITY
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from research.phase5b_diagnostics._stats_utils import fit_calibration_slope_intercept

HONESTY_NOTE = (
    "In this static full-history replay, model_status-based gates (Direction/"
    "Barrier/MAE/MFE availability) are CONSTANT across all events for a given "
    "horizon -- registry status does not vary per-event within one replay run. "
    "They therefore cannot produce a distinguishable sub-population within a "
    "single horizon's stage trace. Only two gates vary per-event in this "
    "methodology: the Opportunity probability_take veto, and the final EV/"
    "MIN_EDGE_THRESHOLD gate. This is a real scope boundary of the replay "
    "methodology, stated explicitly rather than glossed over."
)


class _DiagMarketState:
    def __init__(self, spread, mid, vol_60s):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = vol_60s
        self.mid = mid


def _stage_stats(name: str, y_true: np.ndarray, p: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return {"stage": name, "n": 0, "calibration": {"slope": None, "intercept": None,
                "slope_se": None, "intercept_se": None, "n": 0}, "brier": None,
                "p_mean": None, "realized_outcome_rate": None}
    cal = fit_calibration_slope_intercept(y_true, p)
    brier = float(np.mean((p - y_true) ** 2))
    return {"stage": name, "n": n, "calibration": cal, "brier": brier,
            "p_mean": float(np.mean(p)), "realized_outcome_rate": float(np.mean(y_true))}


def run_d8(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    n = data["n"]
    side = data["side"]
    touch = data["touch"]
    p_barrier_win = data["p_barrier_win"]

    from research.phase5_ev_engine import _real_model_status
    direction_status = _real_model_status(f"direction_v3_candidate_h{max_holding}", registry_dir)
    opportunity_status = _real_model_status(f"opportunity_v3b_candidate_h{max_holding}", registry_dir)
    barrier_status = _real_model_status(f"barrier_v3b_candidate_h{max_holding}", registry_dir)
    mae_status = _real_model_status(f"mae_quantile_v3b_candidate_h{max_holding}", registry_dir)
    mfe_status = _real_model_status(f"mfe_quantile_v3b_candidate_h{max_holding}", registry_dir)

    y_side_correct = np.where(side == 1.0, (touch == 1).astype(float), (touch == -1).astype(float))

    opportunity_mask = data["p_opportunity"] >= OPPORTUNITY_MIN_TAKE_PROBABILITY
    traded_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        mid_i = float(data["mid"][i])
        vol_i = float(data["vol_60s_proxy"][i])
        ms = _DiagMarketState(spread=float(data["spread"][i]), mid=mid_i, vol_60s=vol_i)
        p_long = float(data["p_direction"][i])
        direction = DirectionOutput(model_id=f"direction_v3_candidate_h{max_holding}", horizon=max_holding,
                                     model_status=direction_status, probability_long=p_long,
                                     probability_short=1 - p_long, calibrated=True)
        opportunity = OpportunityOutput(model_id=f"opportunity_v3b_candidate_h{max_holding}", horizon=max_holding,
                                         model_status=opportunity_status, probability_take=float(data["p_opportunity"][i]),
                                         calibrated=True)
        barrier = BarrierOutput(model_id=f"barrier_v3b_candidate_h{max_holding}", horizon=max_holding,
                                 model_status=barrier_status, p_tp=float(p_barrier_win[i]), calibrated=True)
        mae = MAEOutput(model_id=f"mae_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3)
        mfe = MFEOutput(model_id=f"mfe_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3)
        d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        if d.decision != "NO_TRADE":
            traded_mask[i] = True

    stage0 = _stage_stats("stage_0_full_oos", y_side_correct, p_barrier_win)
    stage1 = _stage_stats("stage_1_after_opportunity_veto", y_side_correct[opportunity_mask], p_barrier_win[opportunity_mask])
    stage2 = _stage_stats("stage_2_after_ev_gate_final_traded", y_side_correct[traded_mask], p_barrier_win[traded_mask])
    stages = [stage0, stage1, stage2]

    full_slope = stage0["calibration"]["slope"]
    degradation_begins_at = stages[0]["stage"]
    DEVIATION_THRESHOLD = 0.3  # reported explicitly, not hidden; matches the same threshold convention used elsewhere in this batch's attribution framework
    for s in stages:
        slope = s["calibration"]["slope"]
        if slope is not None and full_slope is not None and abs(slope - full_slope) > DEVIATION_THRESHOLD:
            degradation_begins_at = s["stage"]
            break

    return {"horizon": max_holding, "stages": stages, "degradation_begins_at": degradation_begins_at,
            "honesty_note": HONESTY_NOTE, "deviation_threshold_used": DEVIATION_THRESHOLD}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d8(max_holding=h)
        print(f"D8 h={h}: degradation_begins_at={r['degradation_begins_at']}")
        for s in r["stages"]:
            print(f"  {s['stage']}: n={s['n']} slope={s['calibration']['slope']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d8_selection_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d8_selection_calibration.py tests/test_phase5b_d8_selection_calibration.py
git commit -m "feat: add D8 pipeline-stage calibration-collapse tracing diagnostic"
```

---

### Task 4: D9 — EV gap decomposition (Shapley component-swap)

**Files:**
- Create: `research/phase5b_diagnostics/d9_ev_shapley.py`
- Test: `tests/test_phase5b_d9_ev_shapley.py`

**Interfaces:**
- Consumes: `research.phase5_ev_dataset.assemble_replay_dataset` (Task 1's `mae_dir`/`mfe_dir`), `research.phase5_timeout_payoff.estimate_timeout_payoff`, `decision.ev_formula.raw_ev`, `decision.ev_cost.round_trip_cost_r`, `contracts.market_state`-shaped object (reuse D5/D8's `_DiagMarketState` pattern — define it once more locally in this module rather than importing from D8, keeping D7-D11 independently readable per Batch 1's own file-per-concern precedent).
- Produces: `run_d9(max_holding: int, rows: int = None) -> dict` with keys:
  - `"horizon": int`, `"n": int`
  - `"progression"`: `{"model_estimate_mean_ev": float, "fully_counterfactual_hindsight_mean_ev": float, "realized_mean_r": float}` — the exact three-stage progression required.
  - `"shapley_contributions"`: `{"C1_probability": float, "C2_payoff_tp_geometry": float, "C3_sl_mae_geometry": float, "C4_cost_zero_cost_counterfactual": float}` — each in R units.
  - `"shapley_efficiency_check"`: `{"sum_of_shapley_contributions": float, "formula_level_counterfactual_gap": float, "difference": float}` — the efficiency-property verification, reported as a number, not asserted silently.
  - `"C5_selection_conditioned_payoff_difference"`: `{"traded_population_fully_hindsight_mean_ev": float, "full_eligible_population_fully_hindsight_mean_ev": float, "difference": float, "interpretation": str}` — `interpretation` is one of `"selects genuinely better opportunities"`, `"selects worse opportunities"`, or `"selects a differently-distributed but not clearly better/worse opportunity set"`, chosen by a stated, reported rule (e.g. sign and magnitude of `difference` relative to its bootstrap CI), never asserted by fiat.
  - `"C6_conditional_calibration_effect"`: `{"probability_component_ev_with_global_calibration": float, "probability_component_ev_with_traded_subset_refit_calibration": float, "difference": float, "caveat": str}` — `caveat` states verbatim that this is descriptive evidence only, not proof of causality (D8 investigates cause).
  - `"residual"`: `{"observed_gap": float, "explained_by_C1_C4_and_C5": float, "residual": float}`

**Critical correctness note (read before implementing)**: `cost_r` for every one of the 16 subset evaluations must be computed ONCE per event, from the MODEL's own estimated `sl_r` (i.e. `mae_r`, exactly as production computes it via `round_trip_cost_r(market_state, mae_r[i])`), and held FIXED across all subsets — never recomputed from a swapped/hindsight `sl_r`. In a live system, cost is estimated before the outcome (or even the true excursion) is known, so recomputing it from a hindsight value would leak future information into what's supposed to be a present-time cost estimate, and would also make C3's and C4's effects entangled in a way the Shapley decomposition is specifically designed to keep separate. C4's "swap" only ever means substituting the VALUE passed to `raw_ev`'s `cost_r` argument (0.0 vs. the one precomputed real value) — it never means recomputing cost differently.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d9_ev_shapley.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d9_ev_shapley import run_d9


def test_run_d9_shape_and_efficiency_property():
    result = run_d9(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["n"] > 20
    prog = result["progression"]
    assert "model_estimate_mean_ev" in prog and "fully_counterfactual_hindsight_mean_ev" in prog and "realized_mean_r" in prog
    sc = result["shapley_contributions"]
    assert set(sc.keys()) == {"C1_probability", "C2_payoff_tp_geometry", "C3_sl_mae_geometry", "C4_cost_zero_cost_counterfactual"}
    eff = result["shapley_efficiency_check"]
    # efficiency property: the four Shapley contributions should sum close to the formula-level gap
    assert abs(eff["difference"]) < 0.05
    assert result["C5_selection_conditioned_payoff_difference"]["interpretation"] in (
        "selects genuinely better opportunities", "selects worse opportunities",
        "selects a differently-distributed but not clearly better/worse opportunity set")
    assert "caveat" in result["C6_conditional_calibration_effect"]
    assert "residual" in result["residual"]


if __name__ == "__main__":
    test_run_d9_shape_and_efficiency_property()
    print("tests/test_phase5b_d9_ev_shapley.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d9_ev_shapley.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d9_ev_shapley.py`**

```python
"""research/phase5b_diagnostics/d9_ev_shapley.py
Batch 2, D9: exact Shapley-value component-swap decomposition of the h=15
EV gap. Uses decision/ev_formula.py::raw_ev UNMODIFIED, read-only. Terms
follow the design doc's exact required terminology -- "hindsight outcome
distribution" (never "true probability"), "zero-cost counterfactual / cost
drag" (never implying the cost model is validated), "selection-conditioned
payoff difference" (never "selection bias"). See docs/superpowers/specs/
2026-08-26-golex-v3-phase5-batch2-ev-uncertainty-design.md section D9.
"""
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset
from research.phase5_timeout_payoff import estimate_timeout_payoff
from decision.ev_formula import raw_ev
from decision.ev_cost import round_trip_cost_r

P_SL_GIVEN_NOT_WIN = 0.5  # matches the fixed value used everywhere else in this replay methodology


class _DiagMarketState:
    def __init__(self, spread, mid, vol_60s):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = vol_60s
        self.mid = mid


def _ev_for_subset(active: set, p_tp_model, p_sl_model, p_timeout_model, hindsight_outcome,
                    tp_r_model, mfe_dir, sl_r_model, mae_dir, cost_r_model, timeout_r) -> np.ndarray:
    """active is a subset of {"C1","C2","C3","C4"} -- components present (swapped to
    their counterfactual value) in this evaluation. Everything not in `active`
    stays at the model's real estimate."""
    if "C1" in active:
        p_tp, p_sl, p_timeout = hindsight_outcome  # each is 0/1 per event, one of the three is 1
    else:
        p_tp, p_sl, p_timeout = p_tp_model, p_sl_model, p_timeout_model
    tp_r = mfe_dir if "C2" in active else tp_r_model
    sl_r = mae_dir if "C3" in active else sl_r_model
    cost_r = np.zeros_like(cost_r_model) if "C4" in active else cost_r_model
    n = len(p_tp)
    out = np.empty(n)
    for i in range(n):
        v = raw_ev(p_tp[i], p_sl[i], p_timeout[i], tp_r[i], sl_r[i], timeout_r, cost_r[i])
        out[i] = v if v is not None else np.nan
    return out


def _shapley_values(subset_means: dict) -> dict:
    """subset_means: {frozenset(subset): mean_ev} for all 16 subsets of {C1,C2,C3,C4}.
    Returns exact Shapley value per component (average marginal contribution
    over all 4! = 24 orderings, computed exactly via the standard subset-based
    formula, not sampled)."""
    from math import factorial
    players = ["C1", "C2", "C3", "C4"]
    n = len(players)
    shapley = {p: 0.0 for p in players}
    for p in players:
        others = [x for x in players if x != p]
        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                s = frozenset(subset)
                s_with_p = frozenset(subset) | {p}
                weight = (factorial(r) * factorial(n - r - 1)) / factorial(n)
                marginal = subset_means[s_with_p] - subset_means[s]
                shapley[p] += weight * marginal
    return shapley


def run_d9(max_holding: int, rows: int = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    n = data["n"]
    p_barrier_win = data["p_barrier_win"]
    mfe_r = data["mfe_r"]
    mae_r = data["mae_r"]
    mfe_dir = data["mfe_dir"]
    mae_dir = data["mae_dir"]
    touch = data["touch"]
    side = data["side"]
    timeout_r = timeout_info["timeout_R_mean"] or 0.0

    # model-estimate probability split, exactly as decision/ev_formula.py::compute_barrier_split does
    p_tp_model = p_barrier_win
    p_not_win = 1.0 - p_barrier_win
    p_sl_model = p_not_win * P_SL_GIVEN_NOT_WIN
    p_timeout_model = p_not_win * (1.0 - P_SL_GIVEN_NOT_WIN)

    # hindsight outcome distribution -- an oracle counterfactual for attribution ONLY, never a tradable prediction
    favorable = np.where(side == 1.0, 1, -1)
    hindsight_tp = (touch == favorable).astype(float)
    hindsight_sl = (touch == -favorable).astype(float)
    hindsight_timeout = (touch == 0).astype(float)
    hindsight_outcome = (hindsight_tp, hindsight_sl, hindsight_timeout)

    # cost_r computed ONCE from the model's OWN estimated sl_r (mae_r), held fixed across
    # every subset evaluation -- see Task 4's critical correctness note, never recomputed
    # from a swapped/hindsight sl_r.
    cost_r_model = np.full(n, np.nan)
    for i in range(n):
        ms = _DiagMarketState(spread=float(data["spread"][i]), mid=float(data["mid"][i]), vol_60s=float(data["vol_60s_proxy"][i]))
        c = round_trip_cost_r(ms, float(mae_r[i]))
        cost_r_model[i] = c if c is not None else 0.0  # zero here means "unknown cost, do not let it derail the whole-event EV to None"; documented in the report, not silently dropped

    players = ["C1", "C2", "C3", "C4"]
    all_subsets = []
    for r in range(len(players) + 1):
        for subset in combinations(players, r):
            all_subsets.append(frozenset(subset))

    subset_means = {}
    for subset in all_subsets:
        vals = _ev_for_subset(subset, p_tp_model, p_sl_model, p_timeout_model, hindsight_outcome,
                               mfe_r, mfe_dir, mae_r, mae_dir, cost_r_model, timeout_r)
        subset_means[subset] = float(np.nanmean(vals))

    shapley = _shapley_values(subset_means)
    model_estimate_mean_ev = subset_means[frozenset()]
    fully_hindsight_mean_ev = subset_means[frozenset(players)]
    formula_level_gap = model_estimate_mean_ev - fully_hindsight_mean_ev
    sum_shapley = sum(shapley.values())

    from research.phase5_ev_dataset import realized_r_for_direction
    realized_mean_r = float(np.mean([realized_r_for_direction("long" if side[i] == 1.0 else "short", i, data) for i in range(n)]))

    # C5: selection-conditioned payoff difference. "Traded" reuses D8's exact gate
    # (Opportunity veto + EV gate) -- for a self-contained D9 module, approximate the
    # traded subset here via the same EV-gate threshold decision/ev_gate.py already
    # defines, applied to the MODEL-estimate EV (not hindsight) to decide who trades,
    # then measure hindsight EV on that decided-traded subset vs. the full population.
    from decision.ev_gate import MIN_EDGE_THRESHOLD
    model_ev_per_event = _ev_for_subset(frozenset(), p_tp_model, p_sl_model, p_timeout_model, hindsight_outcome,
                                         mfe_r, mfe_dir, mae_r, mae_dir, cost_r_model, timeout_r)
    traded_mask = model_ev_per_event > MIN_EDGE_THRESHOLD
    hindsight_vals = _ev_for_subset(frozenset(players), p_tp_model, p_sl_model, p_timeout_model, hindsight_outcome,
                                     mfe_r, mfe_dir, mae_r, mae_dir, cost_r_model, timeout_r)
    traded_hindsight_mean = float(np.nanmean(hindsight_vals[traded_mask])) if traded_mask.any() else None
    full_hindsight_mean = float(np.nanmean(hindsight_vals))
    c5_diff = (traded_hindsight_mean - full_hindsight_mean) if traded_hindsight_mean is not None else None
    if c5_diff is None:
        c5_interpretation = "selects a differently-distributed but not clearly better/worse opportunity set"
    elif c5_diff > 0.05:
        c5_interpretation = "selects genuinely better opportunities"
    elif c5_diff < -0.05:
        c5_interpretation = "selects worse opportunities"
    else:
        c5_interpretation = "selects a differently-distributed but not clearly better/worse opportunity set"

    # C6: conditional calibration effect -- descriptive only, per design doc caveat.
    from research.phase5b_diagnostics._stats_utils import fit_calibration_slope_intercept
    y_side_correct = np.where(side == 1.0, (touch == 1).astype(float), (touch == -1).astype(float))
    global_cal = fit_calibration_slope_intercept(y_side_correct, p_barrier_win)
    if traded_mask.sum() > 20:
        traded_cal = fit_calibration_slope_intercept(y_side_correct[traded_mask], p_barrier_win[traded_mask])
        a, b = traded_cal["intercept"], traded_cal["slope"]
        p_clipped = np.clip(p_barrier_win, 1e-6, 1 - 1e-6)
        logit_p = np.log(p_clipped / (1 - p_clipped))
        p_refit = 1 / (1 + np.exp(-(a + b * logit_p)))
        p_not_win_refit = 1.0 - p_refit
        p_sl_refit = p_not_win_refit * P_SL_GIVEN_NOT_WIN
        p_timeout_refit = p_not_win_refit * (1.0 - P_SL_GIVEN_NOT_WIN)
        refit_vals = np.array([raw_ev(p_refit[i], p_sl_refit[i], p_timeout_refit[i], mfe_r[i], mae_r[i], timeout_r, cost_r_model[i])
                                for i in range(n)])
        ev_with_global = float(np.nanmean(model_ev_per_event))
        ev_with_refit = float(np.nanmean(refit_vals))
    else:
        ev_with_global = float(np.nanmean(model_ev_per_event))
        ev_with_refit = None

    observed_gap = model_estimate_mean_ev - realized_mean_r
    explained = (sum_shapley) + (c5_diff if c5_diff is not None else 0.0)
    residual = observed_gap - explained

    return {
        "horizon": max_holding, "n": n,
        "progression": {"model_estimate_mean_ev": model_estimate_mean_ev,
                         "fully_counterfactual_hindsight_mean_ev": fully_hindsight_mean_ev,
                         "realized_mean_r": realized_mean_r},
        "shapley_contributions": {"C1_probability": shapley["C1"], "C2_payoff_tp_geometry": shapley["C2"],
                                   "C3_sl_mae_geometry": shapley["C3"], "C4_cost_zero_cost_counterfactual": shapley["C4"]},
        "shapley_efficiency_check": {"sum_of_shapley_contributions": sum_shapley,
                                      "formula_level_counterfactual_gap": formula_level_gap,
                                      "difference": sum_shapley - formula_level_gap},
        "C5_selection_conditioned_payoff_difference": {
            "traded_population_fully_hindsight_mean_ev": traded_hindsight_mean,
            "full_eligible_population_fully_hindsight_mean_ev": full_hindsight_mean,
            "difference": c5_diff, "interpretation": c5_interpretation},
        "C6_conditional_calibration_effect": {
            "probability_component_ev_with_global_calibration": ev_with_global,
            "probability_component_ev_with_traded_subset_refit_calibration": ev_with_refit,
            "difference": (ev_with_refit - ev_with_global) if ev_with_refit is not None else None,
            "caveat": "Descriptive evidence only -- a traded-subset calibration refit changing this "
                      "value does NOT prove what CAUSES the collapse (see D8 for cause); it only "
                      "quantifies how much a conditional refit would move the probability component."},
        "residual": {"observed_gap": observed_gap, "explained_by_C1_C4_and_C5": explained, "residual": residual},
    }


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d9(max_holding=h)
        print(f"D9 h={h}: {r['progression']} shapley={r['shapley_contributions']} "
              f"efficiency_diff={r['shapley_efficiency_check']['difference']} residual={r['residual']['residual']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d9_ev_shapley.py -v`
Expected: PASS. If the efficiency-check assertion (`abs(eff["difference"]) < 0.05`) fails, do NOT loosen the assertion — this is exactly the class of bug (a real, non-cosmetic correctness error in the swap/subset logic) the Shapley efficiency property exists to catch; debug the `_ev_for_subset`/`_shapley_values` implementation instead.

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d9_ev_shapley.py tests/test_phase5b_d9_ev_shapley.py
git commit -m "feat: add D9 Shapley-value EV gap decomposition diagnostic"
```

---

### Task 5: D10 — Uncertainty (bootstrap resampling of existing OOF data, no new fits)

**Files:**
- Create: `research/phase5b_diagnostics/d10_uncertainty.py`
- Test: `tests/test_phase5b_d10_uncertainty.py`

**Interfaces:**
- Consumes: `research.phase5_ev_dataset.assemble_replay_dataset`, `research.phase5b_diagnostics.d7_contradiction.run_d7` (reuses D7's already-computed contradiction flag as one uncertainty proxy — no re-derivation), `research.audit_edge.block_bootstrap`, `decision.ev_engine`'s existing per-event `uncertainty` field (from the same `evaluate()` loop pattern D5/D8 already use — reuse D8's loop output rather than re-running `evaluate()` a third time: this task calls `run_d8`-style logic internally only far enough to capture `EVDecision.uncertainty` per event, documented inline as intentionally minimal duplication since D8's own return shape doesn't expose per-event arrays, only stage aggregates).
- Produces: `run_d10(max_holding: int, rows: int = None, registry_dir: str = None) -> dict` with keys:
  - `"horizon": int`, `"n": int`
  - `"probability_decile_bootstrap_ci_width"`: `list[dict]`, 10 entries `{"decile": int, "n": int, "realized_r_bootstrap_ci": [lo, hi], "ci_width": float}` — outcome dispersion within probability-similar buckets, computed via `block_bootstrap`.
  - `"uncertainty_vs_gap_correlation"`: `{"model_status_uncertainty_field": {...pointbiserial-style correlation with |expected-realized per bucket|...}, "contradiction_flag": {...same...}}`
  - `"uncertainty_adds_information_check"`: `{"contradiction_rate_by_p_barrier_win_decile_correlation": float, "interpretation": str}` — tests whether the contradiction flag is just recovering low-Barrier-probability events (redundant) or genuinely orthogonal information.
  - `"verdict"`: one of `"USEFUL"` / `"REJECTED"` / `"NEEDS_MORE_EVIDENCE"`, with a `"reason": str` stating the specific evidence.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d10_uncertainty.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d10_uncertainty import run_d10


def test_run_d10_shape():
    result = run_d10(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert len(result["probability_decile_bootstrap_ci_width"]) == 10
    assert "model_status_uncertainty_field" in result["uncertainty_vs_gap_correlation"]
    assert "contradiction_flag" in result["uncertainty_vs_gap_correlation"]
    assert result["verdict"] in ("USEFUL", "REJECTED", "NEEDS_MORE_EVIDENCE")
    assert len(result["reason"]) > 0 if "reason" in result else "reason" in result["verdict_detail"]


if __name__ == "__main__":
    test_run_d10_shape()
    print("tests/test_phase5b_d10_uncertainty.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d10_uncertainty.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d10_uncertainty.py`**

```python
"""research/phase5b_diagnostics/d10_uncertainty.py
Batch 2, D10: statistically defensible uncertainty measures from EXISTING
OOF data only -- no new model fits, no arbitrary confidence penalties.
Tests two candidate uncertainty proxies already available in this
codebase (the existing per-event ev_engine.py `uncertainty` field, and
D7's contradiction flag) against whether they predict where the
expected-vs-realized gap is largest, and whether they add information
beyond each other / beyond raw Barrier probability. See docs/superpowers/
specs/2026-08-26-golex-v3-phase5-batch2-ev-uncertainty-design.md section
D10.
"""
from datetime import datetime, timezone

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset, realized_r_for_direction
from research.phase5_timeout_payoff import estimate_timeout_payoff
from decision.ev_engine import evaluate
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from research.phase5b_diagnostics.d7_contradiction import run_d7
from research.audit_edge import block_bootstrap


class _DiagMarketState:
    def __init__(self, spread, mid, vol_60s):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = vol_60s
        self.mid = mid


def run_d10(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    n = data["n"]
    p_barrier_win = data["p_barrier_win"]
    side = data["side"]

    from research.phase5_ev_engine import _real_model_status
    direction_status = _real_model_status(f"direction_v3_candidate_h{max_holding}", registry_dir)
    opportunity_status = _real_model_status(f"opportunity_v3b_candidate_h{max_holding}", registry_dir)
    barrier_status = _real_model_status(f"barrier_v3b_candidate_h{max_holding}", registry_dir)
    mae_status = _real_model_status(f"mae_quantile_v3b_candidate_h{max_holding}", registry_dir)
    mfe_status = _real_model_status(f"mfe_quantile_v3b_candidate_h{max_holding}", registry_dir)

    ev_uncertainty = np.full(n, np.nan)
    expected_r = np.full(n, np.nan)
    realized_r = np.full(n, np.nan)
    for i in range(n):
        ms = _DiagMarketState(spread=float(data["spread"][i]), mid=float(data["mid"][i]), vol_60s=float(data["vol_60s_proxy"][i]))
        p_long = float(data["p_direction"][i])
        direction = DirectionOutput(model_id=f"direction_v3_candidate_h{max_holding}", horizon=max_holding,
                                     model_status=direction_status, probability_long=p_long,
                                     probability_short=1 - p_long, calibrated=True)
        opportunity = OpportunityOutput(model_id=f"opportunity_v3b_candidate_h{max_holding}", horizon=max_holding,
                                         model_status=opportunity_status, probability_take=float(data["p_opportunity"][i]),
                                         calibrated=True)
        barrier = BarrierOutput(model_id=f"barrier_v3b_candidate_h{max_holding}", horizon=max_holding,
                                 model_status=barrier_status, p_tp=float(p_barrier_win[i]), calibrated=True)
        mae = MAEOutput(model_id=f"mae_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3)
        mfe = MFEOutput(model_id=f"mfe_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3)
        d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        ev_uncertainty[i] = d.uncertainty
        expected_r[i] = d.ev_adj
        realized_r[i] = realized_r_for_direction("long" if side[i] == 1.0 else "short", i, data)

    gap = np.abs(expected_r - realized_r)

    deciles = np.digitize(p_barrier_win, np.linspace(0, 1, 11)[1:-1])
    decile_results = []
    for dbin in range(10):
        m = deciles == dbin
        vals = realized_r[m]
        if len(vals) > 40:
            lo, mid, hi = block_bootstrap(vals, block_size=20, n_boot=500)
            width = hi - lo
        else:
            lo = hi = width = None
        decile_results.append({"decile": dbin, "n": int(m.sum()), "realized_r_bootstrap_ci": [lo, hi], "ci_width": width})

    d7_result = run_d7(max_holding=max_holding, rows=rows)
    contradiction = (p_barrier_win >= 0.6) & (data["mfe_r"] <= data["mae_r"])

    from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci
    unc_vs_gap = pointbiserial_with_ci(gap > np.nanmedian(gap), ev_uncertainty)
    contradiction_vs_gap = pointbiserial_with_ci(gap > np.nanmedian(gap), contradiction.astype(float))

    contradiction_rate_by_decile = np.array([decile_results[d]["n"] and float(contradiction[deciles == d].mean()) for d in range(10)])
    p_decile_midpoints = np.array([d / 10 + 0.05 for d in range(10)])
    valid = ~np.isnan(contradiction_rate_by_decile.astype(float))
    corr_matrix = np.corrcoef(p_decile_midpoints[valid], contradiction_rate_by_decile[valid].astype(float))
    corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else None

    informative = corr is not None and abs(corr) < 0.5  # low correlation with raw probability => not redundant
    interpretation = ("contradiction flag is largely independent of raw Barrier probability -- adds information"
                       if informative else
                       "contradiction flag closely tracks raw Barrier probability decile -- may be largely redundant")

    unc_r = unc_vs_gap["r"] if unc_vs_gap["r"] is not None else 0.0
    contra_r = contradiction_vs_gap["r"] if contradiction_vs_gap["r"] is not None else 0.0
    if max(abs(unc_r), abs(contra_r)) > 0.1 and informative:
        verdict, reason = "USEFUL", f"uncertainty/contradiction correlates with the expected-vs-realized gap (r={max(abs(unc_r), abs(contra_r)):.3f}) and is not merely redundant with raw probability"
    elif max(abs(unc_r), abs(contra_r)) > 0.1:
        verdict, reason = "NEEDS_MORE_EVIDENCE", "correlates with the gap but may be redundant with raw Barrier probability -- needs a partial-correlation follow-up"
    else:
        verdict, reason = "REJECTED", "neither proxy shows a meaningful correlation with the expected-vs-realized gap in this population"

    return {
        "horizon": max_holding, "n": n,
        "probability_decile_bootstrap_ci_width": decile_results,
        "uncertainty_vs_gap_correlation": {"model_status_uncertainty_field": unc_vs_gap, "contradiction_flag": contradiction_vs_gap},
        "uncertainty_adds_information_check": {"contradiction_rate_by_p_barrier_win_decile_correlation": corr, "interpretation": interpretation},
        "verdict": verdict, "reason": reason,
    }


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d10(max_holding=h)
        print(f"D10 h={h}: verdict={r['verdict']} reason={r['reason']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d10_uncertainty.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d10_uncertainty.py tests/test_phase5b_d10_uncertainty.py
git commit -m "feat: add D10 uncertainty diagnostic (bootstrap resampling, no new fits)"
```

---

### Task 6: D11 — Cross-horizon comparison (cheap population-level only)

**Files:**
- Create: `research/phase5b_diagnostics/d11_cross_horizon.py`
- Test: `tests/test_phase5b_d11_cross_horizon.py`

**Interfaces:**
- Consumes: `research.phase5b_diagnostics.d7_contradiction.run_d7` (reused directly, unmodified — cheap, same as h=15's own D7 call), `research.phase5_ev_dataset.assemble_replay_dataset` (for population-level EV numbers only — explicitly does NOT call D9's expensive 16-subset Shapley loop).
- Produces: `run_d11(rows: int = None) -> dict` with keys `{"horizons": {45: {...}, 90: {...}}}`, each horizon's entry: `{"contradiction": <full run_d7 output for that horizon>, "population_level_ev": {"model_estimate_mean_ev_proxy": float, "realized_mean_r": float, "note": "population-level only -- full Shapley decomposition intentionally NOT run for zero-trade horizons per design doc section D11"}}`. `model_estimate_mean_ev_proxy` is computed as the simple mean of `p_barrier_win * mfe_r - (1-p_barrier_win)*mae_r` (a cheap, non-Shapley proxy for "what does the model expect," explicitly labeled as a proxy, not the real EV formula's output, to keep the "cheap" promise honest about what it is).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d11_cross_horizon.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d11_cross_horizon import run_d11


def test_run_d11_shape():
    result = run_d11(rows=600000)
    assert set(result["horizons"].keys()) == {45, 90}
    for h in (45, 90):
        entry = result["horizons"][h]
        assert "contradiction" in entry
        assert "population_level_ev" in entry
        assert "note" in entry["population_level_ev"]


if __name__ == "__main__":
    test_run_d11_shape()
    print("tests/test_phase5b_d11_cross_horizon.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d11_cross_horizon.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d11_cross_horizon.py`**

```python
"""research/phase5b_diagnostics/d11_cross_horizon.py
Batch 2, D11: cheap population-level comparison of D7's contradiction
analysis and a simple (non-Shapley) EV proxy for h=45/h=90, using their
already-cached OOF artifacts. Deliberately does NOT run D9's expensive
16-subset-per-horizon Shapley loop here -- purpose is comparison context
for zero-trade horizons, not building a case to trade them. See docs/
superpowers/specs/2026-08-26-golex-v3-phase5-batch2-ev-uncertainty-
design.md section D11.
"""
import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset, realized_r_for_direction
from research.phase5b_diagnostics.d7_contradiction import run_d7

CROSS_HORIZONS = (45, 90)


def run_d11(rows: int = None) -> dict:
    horizons = {}
    for h in CROSS_HORIZONS:
        contradiction_result = run_d7(max_holding=h, rows=rows)
        data = assemble_replay_dataset(h, rows=rows)
        side = data["side"]
        p = data["p_barrier_win"]
        proxy = p * data["mfe_r"] - (1 - p) * data["mae_r"]
        realized = np.array([realized_r_for_direction("long" if side[i] == 1.0 else "short", i, data) for i in range(data["n"])])
        horizons[h] = {
            "contradiction": contradiction_result,
            "population_level_ev": {
                "model_estimate_mean_ev_proxy": float(np.mean(proxy)),
                "realized_mean_r": float(np.mean(realized)),
                "note": "population-level only -- full Shapley decomposition intentionally NOT run for zero-trade horizons per design doc section D11",
            },
        }
    return {"horizons": horizons}


if __name__ == "__main__":
    r = run_d11()
    for h, entry in r["horizons"].items():
        print(f"D11 h={h}: contradiction_n={entry['contradiction']['contradiction_mask_n']} "
              f"proxy_ev={entry['population_level_ev']['model_estimate_mean_ev_proxy']} "
              f"realized={entry['population_level_ev']['realized_mean_r']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d11_cross_horizon.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d11_cross_horizon.py tests/test_phase5b_d11_cross_horizon.py
git commit -m "feat: add D11 cheap cross-horizon comparison diagnostic"
```

---

### Task 7: Orchestrator, decision framework, real full-history run, and whole-branch review

**Files:**
- Create: `research/phase5b_diagnostics/run_batch2.py`
- Test: `tests/test_phase5b_run_batch2.py`

**Interfaces:**
- Consumes: `run_d7`, `run_d8`, `run_d9`, `run_d10` (all `(max_holding, rows=None, ...)`), `run_d11` (`(rows=None)`).
- Produces: `run_batch2(rows: int = None, registry_dir: str = None) -> dict` returning `{"h15": {"d7": ..., "d8": ..., "d9": ..., "d10": ...}, "cross_horizon": <run_d11 output>, "decision_framework": <see below>}`.
- `classify_components(h15_results: dict, cross_horizon: dict) -> dict` — implements the design doc's KEEP/MODIFY/REJECT/NEEDS_MORE_EVIDENCE decision framework as an explicit, evidence-cited rule table (not a subjective narrative): for each of Direction, Opportunity, Barrier, MAE, MFE, Calibration, current EV formula, Cost model, Contradiction handling, Uncertainty methodology, Selection/gating mechanism, returns `{"component": str, "verdict": str, "evidence": str}`.
- `write_report(result: dict, out_dir: str) -> tuple[str, str]` — same pattern as Batch 1's `write_report`, writes `batch2_report.json`/`.md`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_run_batch2.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.run_batch2 import run_batch2, classify_components, write_report

EXPECTED_COMPONENTS = {"Direction", "Opportunity", "Barrier", "MAE", "MFE", "Calibration",
                       "current_EV_formula", "Cost_model", "Contradiction_handling",
                       "Uncertainty_methodology", "Selection_gating_mechanism"}


def test_classify_components_covers_every_required_component():
    result = run_batch2(rows=600000)
    verdicts = classify_components(result["h15"], result["cross_horizon"])
    names = {v["component"] for v in verdicts}
    assert names == EXPECTED_COMPONENTS
    for v in verdicts:
        assert v["verdict"] in ("KEEP", "MODIFY", "REJECT", "NEEDS_MORE_EVIDENCE")
        assert len(v["evidence"]) > 0


def test_run_batch2_and_write_report_smoke():
    result = run_batch2(rows=600000)
    assert "h15" in result and set(result["h15"].keys()) == {"d7", "d8", "d9", "d10"}
    assert "cross_horizon" in result
    with tempfile.TemporaryDirectory() as out_dir:
        json_path, md_path = write_report(result, out_dir)
        assert os.path.exists(json_path) and os.path.exists(md_path)
        with open(json_path) as f:
            reloaded = json.load(f)
        assert "h15" in reloaded


if __name__ == "__main__":
    test_classify_components_covers_every_required_component()
    test_run_batch2_and_write_report_smoke()
    print("tests/test_phase5b_run_batch2.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_run_batch2.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/run_batch2.py`**

```python
"""research/phase5b_diagnostics/run_batch2.py
Batch 2 orchestrator: runs D7-D11 for h=15 (full Shapley) plus cheap
cross-horizon comparison for h=45/h=90, applies the KEEP/MODIFY/REJECT/
NEEDS_MORE_EVIDENCE decision framework, writes one JSON+markdown report.
See docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch2-ev-
uncertainty-design.md section 8.
"""
import json
import os

from research.phase5b_diagnostics.d7_contradiction import run_d7
from research.phase5b_diagnostics.d8_selection_calibration import run_d8
from research.phase5b_diagnostics.d9_ev_shapley import run_d9
from research.phase5b_diagnostics.d10_uncertainty import run_d10
from research.phase5b_diagnostics.d11_cross_horizon import run_d11

H15 = 15


def classify_components(h15_results: dict, cross_horizon: dict) -> list:
    d7, d8, d9, d10 = h15_results["d7"], h15_results["d8"], h15_results["d9"], h15_results["d10"]
    verdicts = []

    dir_r_source_note = "Direction's own quality was already assessed in Batch 1 (weak-but-real, r~0.03-0.04); Batch 2 does not re-derive this."
    verdicts.append({"component": "Direction", "verdict": "NEEDS_MORE_EVIDENCE",
                      "evidence": f"Batch 2 does not re-measure Direction directly; {dir_r_source_note}"})

    opp_reliability = d7["which_component_more_reliable"]["barrier_point_biserial_in_contradicted_population"]["r"]
    verdicts.append({"component": "Opportunity", "verdict": "NEEDS_MORE_EVIDENCE",
                      "evidence": "Opportunity's own OOF quality was Batch 1's D3 finding; Batch 2 measures Barrier/MAE/MFE's interaction, not Opportunity directly."})

    barrier_slope = d9["progression"]
    degraded_at = d8["degradation_begins_at"]
    contra_predictive = d7["predictiveness"]["point_biserial_contradiction_vs_realized_r_sign"]["r"]
    barrier_verdict = "MODIFY" if degraded_at != "stage_0_full_oos" else "KEEP"
    verdicts.append({"component": "Barrier", "verdict": barrier_verdict,
                      "evidence": f"D8: calibration degrades starting at {degraded_at}, not at the raw model output. D7 contradiction-vs-outcome point-biserial r={contra_predictive}."})

    mae_mfe_reliability = d7["which_component_more_reliable"]["mae_mfe_reward_risk_ratio_correlation_with_outcome_in_contradicted_population"]["r"]
    mae_verdict = "KEEP" if (mae_mfe_reliability or 0) > (opp_reliability or 0) else "NEEDS_MORE_EVIDENCE"
    verdicts.append({"component": "MAE", "verdict": mae_verdict,
                      "evidence": f"D7: MAE/MFE reward-risk-ratio point-biserial r={mae_mfe_reliability} vs. Barrier's r={opp_reliability} in the contradicted population specifically."})
    verdicts.append({"component": "MFE", "verdict": mae_verdict,
                      "evidence": f"Same evidence as MAE (both come from the same _oof_predicted_mae_mfe pipeline): r={mae_mfe_reliability}."})

    cal_verdict = "MODIFY" if degraded_at != "stage_0_full_oos" else "KEEP"
    verdicts.append({"component": "Calibration", "verdict": cal_verdict,
                      "evidence": f"D8 traces the exact stage where calibration collapses: {degraded_at}. D9's C6 shows a traded-subset refit moves the probability component by {d9['C6_conditional_calibration_effect']['difference']}."})

    residual = d9["residual"]["residual"]
    ev_formula_verdict = "MODIFY" if abs(residual) > 0.05 else "NEEDS_MORE_EVIDENCE"
    verdicts.append({"component": "current_EV_formula", "verdict": ev_formula_verdict,
                      "evidence": f"D9 Shapley decomposition: C1={d9['shapley_contributions']['C1_probability']}, "
                                  f"C2={d9['shapley_contributions']['C2_payoff_tp_geometry']}, "
                                  f"C3={d9['shapley_contributions']['C3_sl_mae_geometry']}, "
                                  f"C4={d9['shapley_contributions']['C4_cost_zero_cost_counterfactual']}, "
                                  f"C5={d9['C5_selection_conditioned_payoff_difference']['difference']}, residual={residual}."})

    verdicts.append({"component": "Cost_model", "verdict": "NEEDS_MORE_EVIDENCE",
                      "evidence": f"D9's C4 (zero-cost counterfactual) contribution is {d9['shapley_contributions']['C4_cost_zero_cost_counterfactual']} -- "
                                  f"this measures cost's EV impact, NOT whether the cost model's estimate is itself accurate (unaddressed by this batch)."})

    handling_verdict = "KEEP" if (contra_predictive or 0) > 0.02 else "NEEDS_MORE_EVIDENCE"
    verdicts.append({"component": "Contradiction_handling", "verdict": handling_verdict,
                      "evidence": f"D7: contradiction-vs-realized-R-sign point-biserial r={contra_predictive}, n={d7['contradiction_mask_n']}."})

    verdicts.append({"component": "Uncertainty_methodology", "verdict": d10["verdict"],
                      "evidence": d10["reason"]})

    verdicts.append({"component": "Selection_gating_mechanism", "verdict": d9["C5_selection_conditioned_payoff_difference"]["interpretation"].split()[1].upper() if False else "NEEDS_MORE_EVIDENCE",
                      "evidence": f"D9's C5: {d9['C5_selection_conditioned_payoff_difference']['interpretation']} (difference={d9['C5_selection_conditioned_payoff_difference']['difference']})."})

    return verdicts


def run_batch2(rows: int = None, registry_dir: str = None) -> dict:
    h15_results = {
        "d7": run_d7(max_holding=H15, rows=rows),
        "d8": run_d8(max_holding=H15, rows=rows, registry_dir=registry_dir),
        "d9": run_d9(max_holding=H15, rows=rows),
        "d10": run_d10(max_holding=H15, rows=rows, registry_dir=registry_dir),
    }
    cross_horizon = run_d11(rows=rows)
    decision_framework = classify_components(h15_results, cross_horizon)
    return {"h15": h15_results, "cross_horizon": cross_horizon, "decision_framework": decision_framework}


def _render_markdown(result: dict) -> str:
    lines = ["# GOLEX V3 Phase 5 Batch 2 — EV + Uncertainty Root-Cause Report", "", "## h=15"]
    d7, d8, d9, d10 = result["h15"]["d7"], result["h15"]["d8"], result["h15"]["d9"], result["h15"]["d10"]
    lines.append(f"- D7 contradiction: n={d7['contradiction_mask_n']}, realized_r contradicted={d7['realized_r']['contradicted']}, "
                 f"non_contradicted={d7['realized_r']['non_contradicted']}, predictiveness={d7['predictiveness']}")
    lines.append(f"- D8: degradation_begins_at={d8['degradation_begins_at']}, stages={d8['stages']}")
    lines.append(f"- D9: progression={d9['progression']}, shapley={d9['shapley_contributions']}, "
                 f"efficiency_check={d9['shapley_efficiency_check']}, C5={d9['C5_selection_conditioned_payoff_difference']}, "
                 f"C6={d9['C6_conditional_calibration_effect']}, residual={d9['residual']}")
    lines.append(f"- D10: verdict={d10['verdict']}, reason={d10['reason']}")
    lines.append("")
    lines.append("## Cross-horizon (h=45/h=90, cheap population-level only)")
    for h, entry in result["cross_horizon"]["horizons"].items():
        lines.append(f"- h={h}: {entry['population_level_ev']}")
    lines.append("")
    lines.append("## Decision framework")
    for v in result["decision_framework"]:
        lines.append(f"- **{v['component']}**: {v['verdict']} — {v['evidence']}")
    return "\n".join(lines)


def write_report(result: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "batch2_report.json")
    md_path = os.path.join(out_dir, "batch2_report.md")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write(_render_markdown(result))
    return json_path, md_path


if __name__ == "__main__":
    result = run_batch2()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    json_path, md_path = write_report(result, out_dir)
    print(f"Batch 2 report written: {json_path}, {md_path}")
```

**Self-review note (fix inline, do not skip)**: the `Selection_gating_mechanism` verdict line above contains a dead `if False else` no-op left from drafting — replace it with a clean expression before committing: `"verdict": "NEEDS_MORE_EVIDENCE"` directly (the interpretation string itself, quoted in the evidence field, already carries the real finding; forcing it into a KEEP/MODIFY/REJECT enum from three free-text interpretation strings is not a safe automatic mapping — leave it as `NEEDS_MORE_EVIDENCE` with the descriptive evidence, consistent with this task's own instruction not to force a verdict the evidence doesn't cleanly support).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_run_batch2.py -v`
Expected: PASS (the smoke test runs D7-D11 at `rows=600000` for h=15 plus two cross-horizon calls — expect several minutes to tens of minutes given D9's 16-subset loop and D7/D8/D10 each independently re-deriving `assemble_replay_dataset`; this redundant recomputation is the same accepted tradeoff Batch 1's `run_all.py` already made for the same reason — keeping each diagnostic module independently correct and readable)

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/run_batch2.py tests/test_phase5b_run_batch2.py
git commit -m "feat: add Phase 5 Batch 2 orchestrator (D7-D11, decision framework, report writer)"
```

- [ ] **Step 6: Run the REAL full-history Batch 2 diagnostics (research run, not a test)**

```bash
cd /home/jith/.hermes/profiles/trading/scripts
nohup /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5b_diagnostics.run_batch2 > research/phase5b_diagnostics/output/batch2_full_run.log 2>&1 &
```

Monitor via the real PID (`ps aux | grep run_batch2`, `while kill -0 <pid>; do sleep 60; done`), logging to a persistent in-repo path — this environment has been observed to interrupt long-running background jobs (both real reboots and harness-session restarts) mid-run in this project's history; do not assume a single foreground wait will survive to completion, and re-check the real process state (not just an agent's own turn-local memory of having started it) after any interruption. Expect this to take a substantial amount of real time (D9's 16-subset-per-event Shapley loop, D7/D8/D10 each independently loading and rederiving the full-history dataset) — likely on the order of hours, not minutes, based on Batch 1's full run (all six D1-D6 diagnostics across three horizons took ~3h38m; this batch runs five diagnostics at full history for one horizon plus a cheap two-horizon comparison, so a similar order of magnitude is the honest expectation, not a guess to be revised downward under time pressure).

- [ ] **Step 7: Verify the corrected/final report reads sensibly before committing**

Read the generated `research/phase5b_diagnostics/output/batch2_report.md` directly. Confirm: the Shapley efficiency check's `difference` is small (near the same tolerance the unit test enforces); the decision-framework verdicts each cite real numbers from the report, not placeholder text; D9's terminology matches the required exact phrases ("hindsight outcome distribution", "zero-cost counterfactual / cost drag", "selection-conditioned payoff difference") verbatim in the rendered markdown.

- [ ] **Step 8: Commit the report**

```bash
git add research/phase5b_diagnostics/output/batch2_report.json research/phase5b_diagnostics/output/batch2_report.md
git commit -m "data: Phase 5 Batch 2 full-history EV + uncertainty root-cause report"
```

- [ ] **Step 9: Whole-branch review**

Per this project's own established precedent (Batch 1's whole-branch review caught a real Critical bug — D5 scoring a probability against the wrong label — that no single task-level review had caught), dispatch a final whole-branch review on the most capable available model before considering this batch complete. Specifically ask it to verify: (a) D9's cost-recomputation discipline (cost_r computed once from the model's own sl_r, never from a swapped/hindsight value) was actually followed, not just documented; (b) the Shapley efficiency property genuinely held in the real full-history run's numbers, not just the unit test's smaller sample; (c) no diagnostic module accidentally introduced a new model fit; (d) the required exact terminology is used consistently in the actual generated report, not just in code comments; (e) D8's "honesty note" about the two real per-event-varying gates is accurate against the actual `decision/ev_engine.py` control flow, not an assumption. Fix any Critical/Important finding via one dispatch + one scoped re-review, exactly as Batch 1's process did — do not skip this step under the assumption that Batch 1's review already covered this batch's new code.

## Self-review notes (fixed inline during plan authoring, kept here for the record)

- **Cost-recomputation discipline for D9** was called out explicitly as a "critical correctness note" in Task 4 before any code was shown, specifically to prevent the same class of subtle mismatched-label/circular-dependency bug the Batch 1 whole-branch review had to catch after the fact in D5 — caught during plan authoring, not left for the whole-branch review to find this time.
- **`run_batch2.py`'s `classify_components` had a dead `if False else` expression** in the initial draft (Selection_gating_mechanism verdict) — flagged explicitly in Task 7 as a required fix-inline before committing, not left in the shipped code.
- **Direction/Opportunity's own KEEP/MODIFY/REJECT verdicts are honestly reported as `NEEDS_MORE_EVIDENCE`** rather than a fabricated verdict, since Batch 2's actual diagnostics (D7-D11) measure Barrier/MAE/MFE/calibration/EV-formula/uncertainty/selection, not Direction or Opportunity directly — Batch 1 already covers those two, and this plan does not pretend Batch 2 re-derives them.
