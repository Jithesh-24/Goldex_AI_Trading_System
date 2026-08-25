# GOLEX V3 Phase 5 — Batch 1: Diagnostic Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer, with measurement alone, whether GOLEX's Phase 5A EV weakness (22x expected-vs-realized gap, extreme LONG/SHORT skew at h=15, zero trades at h=45/h=90) traces to the market/labels, Direction, the downstream specialists, calibration, or disagreement between specialists — across all three horizons, on real full-history data.

**Architecture:** Six independent, read-only diagnostic modules (D1-D6) under `research/phase5b_diagnostics/`, each computing new statistics from existing, already-corrected OOF infrastructure (`compute_direction_oof`, `_oof_for_opportunity`/`_oof_for_barrier`/`_oof_predicted_mae_mfe`, `assemble_replay_dataset`) — no new models, no refitting. A shared stats-utilities module provides confidence-interval helpers reused across D1/D3/D4/D5. An orchestrator (`run_all.py`) runs all six across all three horizons on full 6.7-year data, applies the design's attribution framework, and writes one JSON+markdown report.

**Tech Stack:** Python, numpy, pandas, scipy.stats (point-biserial, Fisher z), existing CatBoost/PurgedWalkForwardCV research infra (read-only reuse), pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch1-diagnostics-design.md`

## Global Constraints

- No new models, no refitting anything — every diagnostic reads existing OOF outputs; if a diagnostic needs a probability/prediction that doesn't already exist from a prior Phase 4/5A function, it derives a *statistic* from existing predictions, never fits a new predictor.
- No `models/registry/` writes — this batch produces no model.
- No production code (`decision/`, `app/`, `market/`) touched or modified.
- Full 6.7-year real data throughout — no `rows=` slicing in `run_all.py`'s actual diagnostic runs (unit tests for each module MAY use `rows=` slices for speed, per this repo's existing convention).
- All six diagnostics run for all three horizons (h=15, h=45, h=90) — no diagnostic is skipped, even when an earlier diagnostic looks decisive.
- D4's cross-specialist comparisons must use only arrays drawn from a single `assemble_replay_dataset(h)` call per horizon — never mix arrays from independent calls, which is exactly the same-event/same-side alignment bug Phase 5A fixed.
- D5's traded-subset statistics must be the literal string `"N/A (zero trades at this horizon)"` for h=45/h=90 — never `0`, never a silently omitted key.
- Every D1/D3/D5 statistic reports sample size `n` and identifies which population it was computed over (`oos`, `traded`, or `side_conditioned`), plus a confidence interval wherever practical (point-biserial via Fisher z, calibration slope via the fit's Hessian-derived standard error, D4's contradiction rates via `research.audit_edge.wilson_ci`).
- Stop-early findings are recorded as a `"decisive": true/false` flag with a one-line reason in the report — never used to skip a diagnostic or a horizon.
- Terminal deliverable: the real Batch 1 report (JSON + markdown), run once on full history and handed to the user. This plan does not scope Batch 2/3/4 content.

---

### Task 1: Shared statistics utilities

**Files:**
- Create: `research/phase5b_diagnostics/_stats_utils.py`
- Test: `tests/test_phase5b_stats_utils.py`

**Interfaces:**
- Produces:
  - `pointbiserial_with_ci(y_true: np.ndarray, p: np.ndarray) -> dict` with keys `{"r": float, "n": int, "ci_lo": float, "ci_hi": float}` (Fisher z-transform CI, `z=1.96`). Returns `{"r": None, "n": n, "ci_lo": None, "ci_hi": None}` if `n < 4` (Fisher z is undefined below that).
  - `fit_calibration_slope_intercept(y_true: np.ndarray, p_raw: np.ndarray) -> dict` with keys `{"intercept": float, "slope": float, "intercept_se": float, "slope_se": float, "n": int}` — logistic-on-logit(p) Newton fit (same math as `research/audit_edge.py`'s existing calibration block), extended to also return standard errors from the converged fit's Hessian.
  - `population_label(name: str, n: int) -> dict` returns `{"population": name, "n": n}` — a trivial helper so every statistic's population/`n` tagging is structurally identical across D1/D3/D5, not each hand-rolled per module.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_stats_utils.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics._stats_utils import (
    pointbiserial_with_ci, fit_calibration_slope_intercept, population_label,
)


def test_pointbiserial_with_ci_perfect_correlation():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=2000) >= 0.5).astype(int)
    p = np.where(y == 1, rng.uniform(0.6, 1.0, 2000), rng.uniform(0.0, 0.4, 2000))
    out = pointbiserial_with_ci(y, p)
    assert out["n"] == 2000
    assert out["r"] > 0.5
    assert out["ci_lo"] < out["r"] < out["ci_hi"]


def test_pointbiserial_with_ci_small_n_returns_none():
    out = pointbiserial_with_ci(np.array([1, 0]), np.array([0.6, 0.4]))
    assert out["r"] is None
    assert out["n"] == 2


def test_fit_calibration_slope_intercept_well_calibrated():
    rng = np.random.default_rng(1)
    n = 5000
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    out = fit_calibration_slope_intercept(y, p_true)
    assert out["n"] == n
    assert abs(out["slope"] - 1.0) < 0.15
    assert abs(out["intercept"]) < 0.15
    assert out["slope_se"] > 0
    assert out["intercept_se"] > 0


def test_population_label():
    assert population_label("oos", 12345) == {"population": "oos", "n": 12345}


if __name__ == "__main__":
    test_pointbiserial_with_ci_perfect_correlation()
    test_pointbiserial_with_ci_small_n_returns_none()
    test_fit_calibration_slope_intercept_well_calibrated()
    test_population_label()
    print("tests/test_phase5b_stats_utils.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_stats_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.phase5b_diagnostics'`

- [ ] **Step 3: Write `research/phase5b_diagnostics/__init__.py`** (empty file, makes this a package)

- [ ] **Step 4: Write `research/phase5b_diagnostics/_stats_utils.py`**

```python
"""research/phase5b_diagnostics/_stats_utils.py
Shared statistics helpers for Phase 5 Batch 1 diagnostics (D1-D6). Every
diagnostic that reports a point-biserial correlation, a calibration
slope/intercept, or a population's sample size uses these, so the
n/CI/population-tagging convention is identical across all six modules
instead of six independent reimplementations that could silently drift
apart (see docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch1-
diagnostics-design.md section 2a).
"""
import numpy as np
from scipy.stats import pointbiserialr


def pointbiserial_with_ci(y_true: np.ndarray, p: np.ndarray, z: float = 1.96) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = len(y_true)
    if n < 4:
        return {"r": None, "n": n, "ci_lo": None, "ci_hi": None}
    r, _ = pointbiserialr(y_true, p)
    r = float(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    z_r = np.arctanh(r)
    lo, hi = np.tanh(z_r - z * se), np.tanh(z_r + z * se)
    return {"r": r, "n": n, "ci_lo": float(lo), "ci_hi": float(hi)}


def fit_calibration_slope_intercept(y_true: np.ndarray, p_raw: np.ndarray) -> dict:
    y_c = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_raw, dtype=np.float64), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    n = len(y_c)
    a, b = 0.0, 1.0
    h_aa = h_bb = h_ab = -1.0
    for _ in range(50):
        z_lin = a + b * logit_p
        pr = 1 / (1 + np.exp(-z_lin))
        w = np.clip(pr * (1 - pr), 1e-6, None)
        grad_a = np.sum(y_c - pr)
        grad_b = np.sum((y_c - pr) * logit_p)
        h_aa = -np.sum(w)
        h_bb = -np.sum(w * logit_p ** 2)
        h_ab = -np.sum(w * logit_p)
        det = h_aa * h_bb - h_ab ** 2
        if abs(det) < 1e-12:
            break
        da = (grad_a * h_bb - grad_b * h_ab) / det
        db = (grad_b * h_aa - grad_a * h_ab) / det
        a -= da
        b -= db
    det = h_aa * h_bb - h_ab ** 2
    if abs(det) < 1e-12:
        intercept_se = slope_se = float("nan")
    else:
        cov_aa = -h_bb / det
        cov_bb = -h_aa / det
        intercept_se = float(np.sqrt(max(cov_aa, 0.0)))
        slope_se = float(np.sqrt(max(cov_bb, 0.0)))
    return {"intercept": float(a), "slope": float(b),
            "intercept_se": intercept_se, "slope_se": slope_se, "n": n}


def population_label(name: str, n: int) -> dict:
    return {"population": name, "n": n}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_phase5b_stats_utils.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add research/phase5b_diagnostics/__init__.py research/phase5b_diagnostics/_stats_utils.py tests/test_phase5b_stats_utils.py
git commit -m "feat: add shared stats utilities (point-biserial CI, calibration slope/SE) for Phase 5 Batch 1 diagnostics"
```

---

### Task 2: D1 — Direction quality

**Files:**
- Create: `research/phase5b_diagnostics/d1_direction_quality.py`
- Test: `tests/test_phase5b_d1_direction_quality.py`

**Interfaces:**
- Consumes: `research.direction_side.compute_direction_oof(max_holding, rows=None) -> dict` (keys: `t0_nz, feature_cols, p_direction_raw, p_direction_cal, side, has_oof, model_id, fold_metrics`), `research.phase5b_diagnostics._stats_utils.pointbiserial_with_ci`.
- Produces: `run_d1(max_holding: int, rows: int = None) -> dict` with keys:
  - `"horizon": int`
  - `"oos"`: `{"n": int, "point_biserial": {...pointbiserial_with_ci output...}, "p_direction_mean": float, "p_direction_std": float, "p_direction_deciles": list[float]}`
  - `"side_conditioned"`: `{"long": {same shape as "oos"}, "short": {same shape as "oos"}}`

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d1_direction_quality.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d1_direction_quality import run_d1


def test_run_d1_shape():
    result = run_d1(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["oos"]["n"] > 50
    assert result["oos"]["point_biserial"]["n"] == result["oos"]["n"]
    assert len(result["oos"]["p_direction_deciles"]) == 9
    assert "long" in result["side_conditioned"] and "short" in result["side_conditioned"]
    assert result["side_conditioned"]["long"]["n"] + result["side_conditioned"]["short"]["n"] == result["oos"]["n"]


if __name__ == "__main__":
    test_run_d1_shape()
    print("tests/test_phase5b_d1_direction_quality.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d1_direction_quality.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d1_direction_quality.py`**

```python
"""research/phase5b_diagnostics/d1_direction_quality.py
Batch 1, D1: Direction's OOF probability distribution and point-biserial
correlation against the true directional label, overall and split by
side. Closes the point-biserial measurement flagged as deferred in the
Phase 5A retrain-and-replay report. Read-only: computes statistics from
research.direction_side.compute_direction_oof's existing output, fits
nothing new. See docs/superpowers/specs/2026-08-26-golex-v3-phase5-
batch1-diagnostics-design.md section D1.
"""
import numpy as np
from research.direction_side import compute_direction_oof
from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci


def _population_stats(y_true, p_cal):
    n = len(y_true)
    pb = pointbiserial_with_ci(y_true, p_cal)
    deciles = list(np.percentile(p_cal, np.arange(10, 100, 10)).astype(float)) if n else []
    return {"n": n, "point_biserial": pb,
            "p_direction_mean": float(np.mean(p_cal)) if n else None,
            "p_direction_std": float(np.std(p_cal)) if n else None,
            "p_direction_deciles": deciles}


def run_d1(max_holding: int, rows: int = None) -> dict:
    oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz = ds["t0_idx"][nz]
    assert np.array_equal(t0_nz, oof["t0_nz"]), "direction_side event index mismatch"
    y_true_full = (y[nz] == 1).astype(float)

    has_oof = oof["has_oof"]
    y_true = y_true_full[has_oof]
    p_cal = oof["p_direction_cal"][has_oof]
    side = oof["side"][has_oof]

    oos = _population_stats(y_true, p_cal)
    long_mask = side == 1.0
    short_mask = side == -1.0
    side_conditioned = {
        "long": _population_stats(y_true[long_mask], p_cal[long_mask]),
        "short": _population_stats(y_true[short_mask], p_cal[short_mask]),
    }
    return {"horizon": max_holding, "oos": oos, "side_conditioned": side_conditioned}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d1(max_holding=h)
        print(f"D1 h={h}: n={r['oos']['n']} point_biserial={r['oos']['point_biserial']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d1_direction_quality.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d1_direction_quality.py tests/test_phase5b_d1_direction_quality.py
git commit -m "feat: add D1 Direction quality diagnostic (point-biserial + distribution, by side)"
```

---

### Task 3: D2 — Base-rate / directional skew audit (model-free)

**Files:**
- Create: `research/phase5b_diagnostics/d2_base_rate_audit.py`
- Test: `tests/test_phase5b_d2_base_rate_audit.py`

**Interfaces:**
- Consumes: `research.phase4_dataset.assemble_v3_dataset`, `features.labeling.TripleBarrierConfig`/`triple_barrier_labels`.
- Produces: `run_d2(max_holding: int, rows: int = None) -> dict` with keys:
  - `"horizon": int`
  - `"overall": {"n": int, "up_frac": float, "down_frac": float, "timeout_frac": float}`
  - `"by_year": list[dict]`, each `{"year": int, "n": int, "up_frac": float, "down_frac": float, "timeout_frac": float}`

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d2_base_rate_audit.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d2_base_rate_audit import run_d2


def test_run_d2_shape():
    result = run_d2(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    o = result["overall"]
    assert o["n"] > 50
    total = o["up_frac"] + o["down_frac"] + o["timeout_frac"]
    assert abs(total - 1.0) < 1e-6
    assert len(result["by_year"]) >= 1
    for row in result["by_year"]:
        assert "year" in row and "n" in row


if __name__ == "__main__":
    test_run_d2_shape()
    print("tests/test_phase5b_d2_base_rate_audit.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d2_base_rate_audit.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d2_base_rate_audit.py`**

```python
"""research/phase5b_diagnostics/d2_base_rate_audit.py
Batch 1, D2: model-free directional base rate from triple-barrier labels
alone (side=None, symmetric barriers), overall and by calendar year.
Answers whether h=15's 24-long/107,611-short replay skew could be
explained by the raw event population before any model exists. See
docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch1-diagnostics-
design.md section D2.
"""
import numpy as np
import pandas as pd
from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels


def _fracs(touch: np.ndarray) -> dict:
    n = len(touch)
    if n == 0:
        return {"n": 0, "up_frac": None, "down_frac": None, "timeout_frac": None}
    return {"n": n,
            "up_frac": float((touch == 1).mean()),
            "down_frac": float((touch == -1).mean()),
            "timeout_frac": float((touch == 0).mean())}


def run_d2(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    touch = labels["touch"].to_numpy()
    times = pd.to_datetime(ds["feat_v3"]["time"].to_numpy())[ds["t0_idx"]]
    years = times.year

    overall = _fracs(touch)
    by_year = []
    for yr in sorted(set(years.tolist())):
        m = years == yr
        row = _fracs(touch[m])
        row["year"] = int(yr)
        by_year.append(row)

    return {"horizon": max_holding, "overall": overall, "by_year": by_year}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d2(max_holding=h)
        print(f"D2 h={h}: {r['overall']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d2_base_rate_audit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d2_base_rate_audit.py tests/test_phase5b_d2_base_rate_audit.py
git commit -m "feat: add D2 model-free base-rate/directional-skew audit diagnostic"
```

---

### Task 4: D3 — Opportunity/Barrier/MAE/MFE OOF quality

**Files:**
- Create: `research/phase5b_diagnostics/d3_specialist_oof_quality.py`
- Test: `tests/test_phase5b_d3_specialist_oof_quality.py`

**Interfaces:**
- Consumes: `research.phase5_calibration._oof_for_opportunity(max_holding, rows=None) -> (t0_nz, y_full, p_full, mask)`, `research.phase5_calibration._oof_for_barrier` (same signature, aliases `_oof_for_opportunity`), `research.phase5_calibration._oof_predicted_mae_mfe(max_holding, rows=None) -> (t0_nz, mae_full, mfe_full, mask)`, `research.direction_side.compute_direction_oof` (for the shared `side` array — same event index as the calibration functions per their own internal alignment asserts), `research.phase5b_diagnostics._stats_utils.pointbiserial_with_ci`/`fit_calibration_slope_intercept`.
- Produces: `run_d3(max_holding: int, rows: int = None) -> dict` with keys:
  - `"horizon": int`
  - `"opportunity": {"n": int, "point_biserial": {...}, "win_rate": float, "baseline_win_rate": 0.4887, "calibration": {...fit_calibration_slope_intercept output...}}`
  - `"barrier"`: same shape as `"opportunity"` (computed independently even though it's the same underlying function today — keeps D3's output shape stable if Barrier's own OOF source ever diverges from Opportunity's)
  - `"mae_mfe"`: `{"mae_coverage_by_side": {"long": {"q75_coverage": float, "n": int}, "short": {...}}, "mfe_coverage_by_side": {"long": {...}, "short": {...}}}`

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d3_specialist_oof_quality.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d3_specialist_oof_quality import run_d3


def test_run_d3_shape():
    result = run_d3(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    for role in ("opportunity", "barrier"):
        r = result[role]
        assert r["n"] > 20
        assert 0.0 <= r["win_rate"] <= 1.0
        assert r["baseline_win_rate"] == 0.4887
        assert "slope" in r["calibration"]
    mm = result["mae_mfe"]
    assert "long" in mm["mae_coverage_by_side"] and "short" in mm["mae_coverage_by_side"]
    assert "long" in mm["mfe_coverage_by_side"] and "short" in mm["mfe_coverage_by_side"]


if __name__ == "__main__":
    test_run_d3_shape()
    print("tests/test_phase5b_d3_specialist_oof_quality.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d3_specialist_oof_quality.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d3_specialist_oof_quality.py`**

```python
"""research/phase5b_diagnostics/d3_specialist_oof_quality.py
Batch 1, D3: Opportunity/Barrier point-biserial correlation, win rate vs.
the existing 0.4887 baseline, and calibration slope/intercept, computed
on the FULL OOF population -- independent of whether the final EV gate
ever allows a trade. MAE/MFE quantile coverage broken down by side (new;
the existing v3b registry entries only report global/per-vol-regime
coverage from training time). See docs/superpowers/specs/2026-08-26-
golex-v3-phase5-batch1-diagnostics-design.md section D3.
"""
import numpy as np
from research.phase5_calibration import _oof_for_opportunity, _oof_for_barrier, _oof_predicted_mae_mfe
from research.direction_side import compute_direction_oof
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci, fit_calibration_slope_intercept

BASELINE_WIN_RATE = 0.4887


def _role_stats(t0_nz, y_full, p_full, mask):
    y_true = y_full[mask]
    p = p_full[mask]
    n = len(y_true)
    pb = pointbiserial_with_ci(y_true, p)
    win_rate = float(y_true.mean()) if n else None
    cal = fit_calibration_slope_intercept(y_true, p) if n else {"intercept": None, "slope": None,
                                                                   "intercept_se": None, "slope_se": None, "n": 0}
    return {"n": n, "point_biserial": pb, "win_rate": win_rate,
            "baseline_win_rate": BASELINE_WIN_RATE, "calibration": cal}


def run_d3(max_holding: int, rows: int = None) -> dict:
    t0_o, y_o, p_o, m_o = _oof_for_opportunity(max_holding, rows=rows)
    t0_b, y_b, p_b, m_b = _oof_for_barrier(max_holding, rows=rows)
    t0_mm, mae_full, mfe_full, m_mm = _oof_predicted_mae_mfe(max_holding, rows=rows)
    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    assert np.array_equal(t0_o, dir_oof["t0_nz"]), "event index mismatch: opportunity vs direction_side"
    assert np.array_equal(t0_mm, dir_oof["t0_nz"]), "event index mismatch: mae/mfe vs direction_side"

    opportunity = _role_stats(t0_o, y_o, p_o, m_o)
    barrier = _role_stats(t0_b, y_b, p_b, m_b)

    side = dir_oof["side"]
    QUANTILE = 0.75
    mae_mfe = {"mae_coverage_by_side": {}, "mfe_coverage_by_side": {}}
    for label, side_val in (("long", 1.0), ("short", -1.0)):
        smask = m_mm & (side == side_val)
        n = int(smask.sum())
        if n > 0:
            mae_cov = float((mae_full[smask] <= np.nan_to_num(mae_full[smask], nan=np.inf)).mean()) if False else None
        # coverage = fraction of true excursions <= the OOF-predicted q75 value is not
        # computable from mae_full/mfe_full alone (those ARE the q75 predictions, not
        # paired with a separate true-excursion array) -- report n and the predicted
        # q75 distribution's own mean/std by side instead, which IS available here.
        mae_vals = mae_full[smask]
        mfe_vals = mfe_full[smask]
        mae_mfe["mae_coverage_by_side"][label] = {"n": n, "q75_pred_mean": float(np.mean(mae_vals)) if n else None}
        mae_mfe["mfe_coverage_by_side"][label] = {"n": n, "q75_pred_mean": float(np.mean(mfe_vals)) if n else None}

    return {"horizon": max_holding, "opportunity": opportunity, "barrier": barrier, "mae_mfe": mae_mfe}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d3(max_holding=h)
        print(f"D3 h={h}: opportunity_n={r['opportunity']['n']} barrier_n={r['barrier']['n']}")
```

**Note on the `mae_mfe` field** (read before implementing, do not skip): `_oof_predicted_mae_mfe`'s `mae_full`/`mfe_full` arrays ARE the OOF-predicted q75 values themselves — there is no separate "true realized excursion" array paired with them inside `_oof_for_predicted_mae_mfe`'s return shape, so a true "coverage" statistic (fraction of true excursions below the predicted q75) is NOT computable from this function's output alone within this task's scope. Report the predicted q75 distribution's own mean by side instead (as shown above) — this still answers "does the predicted excursion differ meaningfully by side," which is D3's actual purpose here, without overclaiming a coverage statistic this task's inputs cannot support. Name the field `q75_pred_mean`, not `q75_coverage`, to avoid implying something false.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d3_specialist_oof_quality.py -v`
Expected: PASS (update the test's assertion to check for `"q75_pred_mean"` instead of `"q75_coverage"` if you wrote the test before reading this note — the test shown in Step 1 already only checks for `"long"`/`"short"` keys, which is compatible either way)

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d3_specialist_oof_quality.py tests/test_phase5b_d3_specialist_oof_quality.py
git commit -m "feat: add D3 Opportunity/Barrier/MAE/MFE OOF quality diagnostic"
```

---

### Task 5: D4 — Cross-specialist consistency (strict same-event alignment)

**Files:**
- Create: `research/phase5b_diagnostics/d4_cross_specialist_consistency.py`
- Test: `tests/test_phase5b_d4_cross_specialist_consistency.py`

**Interfaces:**
- Consumes: `research.phase5_ev_dataset.assemble_replay_dataset(max_holding, rows=None) -> dict` (keys used here: `n, p_opportunity, p_barrier_win, mae_r, mfe_r, side, direction_model_id`), `research.audit_edge.wilson_ci(k, n, z=1.96) -> (lo, hi)`.
- Produces: `run_d4(max_holding: int, rows: int = None) -> dict` with keys:
  - `"horizon": int`
  - `"n": int`
  - `"contradiction_barrier_vs_reward_risk": {"rate": float, "k": int, "n": int, "ci_lo": float, "ci_hi": float}`
  - `"contradiction_opportunity_vs_barrier": {"rate": float, "k": int, "n": int, "ci_lo": float, "ci_hi": float}`

**Hard constraint** (per design doc, restated so this task's implementer cannot miss it): every array used in this module's comparisons (`p_opportunity`, `p_barrier_win`, `mae_r`, `mfe_r`) must come from the SAME `assemble_replay_dataset(max_holding, rows=rows)` call — never call it twice and mix results, never substitute an array from `_oof_for_opportunity`/`_oof_for_barrier` directly (those have a different, unfiltered index space than `assemble_replay_dataset`'s combined-mask-aligned arrays).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d4_cross_specialist_consistency.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d4_cross_specialist_consistency import run_d4


def test_run_d4_shape():
    result = run_d4(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["n"] > 20
    for key in ("contradiction_barrier_vs_reward_risk", "contradiction_opportunity_vs_barrier"):
        c = result[key]
        assert 0.0 <= c["rate"] <= 1.0
        assert c["k"] <= c["n"]
        assert c["ci_lo"] <= c["rate"] <= c["ci_hi"]


if __name__ == "__main__":
    test_run_d4_shape()
    print("tests/test_phase5b_d4_cross_specialist_consistency.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d4_cross_specialist_consistency.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d4_cross_specialist_consistency.py`**

```python
"""research/phase5b_diagnostics/d4_cross_specialist_consistency.py
Batch 1, D4: mechanical, non-subjective contradiction rates between
Barrier/Opportunity/MAE/MFE, with every comparison keyed to the SAME
event/side/horizon/TP-SL-definition via a single assemble_replay_dataset
call -- the exact discipline Phase 5A's build_meta fix exists to enforce.
Production sets sl_r, tp_r = mae.q75, mfe.q75 directly (decision/ev_cost.py
::candidate_sl_tp) -- comparing mfe_r against itself would be circular, so
the reward-to-risk check compares Barrier's independently-fit p_barrier_win
against the independently-fit mae_r/mfe_r RATIO instead. See docs/
superpowers/specs/2026-08-26-golex-v3-phase5-batch1-diagnostics-design.md
section D4.
"""
import numpy as np
from research.phase5_ev_dataset import assemble_replay_dataset
from research.audit_edge import wilson_ci


def _rate_with_ci(mask: np.ndarray) -> dict:
    n = len(mask)
    k = int(mask.sum())
    rate = float(k / n) if n else None
    lo, hi = wilson_ci(k, n) if n else (None, None)
    return {"rate": rate, "k": k, "n": n, "ci_lo": lo, "ci_hi": hi}


def run_d4(max_holding: int, rows: int = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    n = data["n"]
    p_opportunity = data["p_opportunity"]
    p_barrier_win = data["p_barrier_win"]
    mae_r = data["mae_r"]
    mfe_r = data["mfe_r"]

    barrier_vs_reward_risk = (p_barrier_win >= 0.6) & (mfe_r <= mae_r)
    opportunity_vs_barrier = (p_opportunity >= 0.5) & (p_barrier_win < 0.5)

    return {"horizon": max_holding, "n": n,
            "contradiction_barrier_vs_reward_risk": _rate_with_ci(barrier_vs_reward_risk),
            "contradiction_opportunity_vs_barrier": _rate_with_ci(opportunity_vs_barrier)}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d4(max_holding=h)
        print(f"D4 h={h}: n={r['n']} "
              f"barrier_vs_reward_risk={r['contradiction_barrier_vs_reward_risk']['rate']:.4f} "
              f"opportunity_vs_barrier={r['contradiction_opportunity_vs_barrier']['rate']:.4f}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d4_cross_specialist_consistency.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d4_cross_specialist_consistency.py tests/test_phase5b_d4_cross_specialist_consistency.py
git commit -m "feat: add D4 cross-specialist consistency diagnostic (strict same-event alignment)"
```

---

### Task 6: D5 — Reliability / calibration (global, by side, traded-subset-where-it-exists)

**Files:**
- Create: `research/phase5b_diagnostics/d5_calibration_reliability.py`
- Test: `tests/test_phase5b_d5_calibration_reliability.py`

**Interfaces:**
- Consumes: `research.phase5_ev_dataset.assemble_replay_dataset`, `research.phase5_timeout_payoff.estimate_timeout_payoff`, `decision.ev_engine.evaluate`, `contracts.specialist_output.{DirectionOutput,OpportunityOutput,BarrierOutput,MAEOutput,MFEOutput}`, `research.phase5b_diagnostics._stats_utils.fit_calibration_slope_intercept`. This task does NOT modify or call into `research/phase5_ev_engine.py`'s `replay_and_validate` (that function only returns aggregate counters, not the per-event decision/probability arrays a reliability curve needs) — instead it re-runs the SAME per-event `evaluate()` loop pattern directly inside this diagnostic module, read-only, to capture per-event outputs. This is not a change to any production or Phase 5A file.
- Produces: `run_d5(max_holding: int, rows: int = None, registry_dir: str = None) -> dict` with keys:
  - `"horizon": int`
  - `"global"`: `{"n": int, "brier": float, "ece": float, "calibration": {...}, "reliability_bins": list[dict]}`
  - `"by_side"`: `{"long": {same shape as "global"}, "short": {same shape as "global"}}`
  - `"traded_subset"`: same shape as `"global"` for h=15, OR the literal string `"N/A (zero trades at this horizon)"` for h=45/h=90 — check `n_traded == 0` after running the per-event loop, do not assume by horizon number (a future retrain could change which horizons trade).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d5_calibration_reliability.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d5_calibration_reliability import run_d5


def test_run_d5_shape_h15_has_traded_subset():
    result = run_d5(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    assert result["global"]["n"] > 20
    assert "long" in result["by_side"] and "short" in result["by_side"]
    assert isinstance(result["traded_subset"], (dict, str))


def test_run_d5_reliability_bins_sum_to_global_n():
    result = run_d5(max_holding=15, rows=600000)
    total_binned = sum(b["n"] for b in result["global"]["reliability_bins"])
    assert total_binned == result["global"]["n"]


if __name__ == "__main__":
    test_run_d5_shape_h15_has_traded_subset()
    test_run_d5_reliability_bins_sum_to_global_n()
    print("tests/test_phase5b_d5_calibration_reliability.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d5_calibration_reliability.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d5_calibration_reliability.py`**

```python
"""research/phase5b_diagnostics/d5_calibration_reliability.py
Batch 1, D5: reliability curves, Brier score, ECE, and calibration
slope/intercept -- global, by long/short side, and in the traded subset
ONLY where n_traded > 0 (per design: never fabricate a traded-subset
statistic for a zero-trade horizon; report the literal N/A string
instead). Re-runs the same per-event decision/ev_engine.evaluate() loop
research/phase5_ev_engine.py::replay_and_validate already uses, but
captures per-event probability/outcome/decision arrays that function
doesn't expose, rather than modifying that (production-adjacent, Phase
5A-reviewed) file. See docs/superpowers/specs/2026-08-26-golex-v3-
phase5-batch1-diagnostics-design.md section D5.
"""
from datetime import datetime, timezone

import numpy as np

from research.phase5_ev_dataset import assemble_replay_dataset, realized_r_for_direction
from research.phase5_timeout_payoff import estimate_timeout_payoff
from decision.ev_engine import evaluate
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput

N_BINS = 10


class _DiagMarketState:
    def __init__(self, spread, mid, vol_60s):
        self.spread = spread
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = vol_60s
        self.mid = mid


def _reliability_and_scores(y_true: np.ndarray, p: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None, "calibration": {"intercept": None, "slope": None,
                "intercept_se": None, "slope_se": None, "n": 0}, "reliability_bins": []}
    from research.phase5b_diagnostics._stats_utils import fit_calibration_slope_intercept
    brier = float(np.mean((p - y_true) ** 2))
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
    bins = []
    ece = 0.0
    for b in range(N_BINS):
        m = bin_idx == b
        bn = int(m.sum())
        if bn == 0:
            bins.append({"bin": b, "n": 0, "mean_predicted": None, "observed_rate": None})
            continue
        mean_p = float(p[m].mean())
        obs = float(y_true[m].mean())
        bins.append({"bin": b, "n": bn, "mean_predicted": mean_p, "observed_rate": obs})
        ece += (bn / n) * abs(obs - mean_p)
    cal = fit_calibration_slope_intercept(y_true, p)
    return {"n": n, "brier": brier, "ece": float(ece), "calibration": cal, "reliability_bins": bins}


def run_d5(max_holding: int, rows: int = None, registry_dir: str = None) -> dict:
    data = assemble_replay_dataset(max_holding, rows=rows)
    timeout_info = estimate_timeout_payoff(max_holding, rows=rows)
    n = data["n"]
    side = data["side"]

    from research.phase5_ev_engine import _real_model_status
    direction_status = _real_model_status(f"direction_v3_candidate_h{max_holding}", registry_dir)
    opportunity_status = _real_model_status(f"opportunity_v3b_candidate_h{max_holding}", registry_dir)
    barrier_status = _real_model_status(f"barrier_v3b_candidate_h{max_holding}", registry_dir)
    mae_status = _real_model_status(f"mae_quantile_v3b_candidate_h{max_holding}", registry_dir)
    mfe_status = _real_model_status(f"mfe_quantile_v3b_candidate_h{max_holding}", registry_dir)

    p_used = np.full(n, np.nan)   # the probability that drove each event's decision (Barrier's p_tp)
    y_outcome = np.full(n, np.nan)  # 1 if the traded/proposed side's touch matched, else 0 -- NaN if NO_TRADE
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
                                 model_status=barrier_status, p_tp=float(data["p_barrier_win"][i]), calibrated=True)
        mae = MAEOutput(model_id=f"mae_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mae_status,
                         q50=float(data["mae_r"][i]) * 0.7, q75=float(data["mae_r"][i]), q90=float(data["mae_r"][i]) * 1.3)
        mfe = MFEOutput(model_id=f"mfe_quantile_v3b_candidate_h{max_holding}", horizon=max_holding, model_status=mfe_status,
                         q50=float(data["mfe_r"][i]) * 0.7, q75=float(data["mfe_r"][i]), q90=float(data["mfe_r"][i]) * 1.3)

        d = evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe,
                      timeout_r=timeout_info["timeout_R_mean"] or 0.0,
                      timeout_r_provisional_proxy=timeout_info["provisional_proxy"])
        p_used[i] = float(data["p_barrier_win"][i])
        if d.decision != "NO_TRADE":
            traded_mask[i] = True
            touch = data["touch"][i]
            y_outcome[i] = 1.0 if ((d.direction == "long" and touch == 1) or (d.direction == "short" and touch == -1)) else 0.0

    # "global" calibration uses touch-derived correctness for Barrier's OWN proposed
    # side (side array), not just traded events -- this answers "is p_barrier_win calibrated
    # against ITS side's real touch outcome", independent of whether the EV gate traded it.
    # (p_used/traded_mask, populated in the loop above, are used below for traded_subset.)
    touch_all = data["touch"]
    y_side_correct = np.where(side == 1.0, (touch_all == 1).astype(float), (touch_all == -1).astype(float))
    global_stats = _reliability_and_scores(y_side_correct, data["p_barrier_win"])

    long_mask = side == 1.0
    short_mask = side == -1.0
    by_side = {
        "long": _reliability_and_scores(y_side_correct[long_mask], data["p_barrier_win"][long_mask]),
        "short": _reliability_and_scores(y_side_correct[short_mask], data["p_barrier_win"][short_mask]),
    }

    n_traded = int(traded_mask.sum())
    if n_traded == 0:
        traded_subset = "N/A (zero trades at this horizon)"
    else:
        traded_subset = _reliability_and_scores(y_outcome[traded_mask], p_used[traded_mask])

    return {"horizon": max_holding, "global": global_stats, "by_side": by_side, "traded_subset": traded_subset}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d5(max_holding=h)
        print(f"D5 h={h}: global_n={r['global']['n']} brier={r['global']['brier']} "
              f"traded_subset={'N/A' if isinstance(r['traded_subset'], str) else r['traded_subset']['n']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d5_calibration_reliability.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d5_calibration_reliability.py tests/test_phase5b_d5_calibration_reliability.py
git commit -m "feat: add D5 calibration/reliability diagnostic (global, by side, traded-subset-where-it-exists)"
```

---

### Task 7: D6 — Long/short conditioning behavior

**Files:**
- Create: `research/phase5b_diagnostics/d6_long_short_conditioning.py`
- Test: `tests/test_phase5b_d6_long_short_conditioning.py`

**Interfaces:**
- Consumes: `research.phase5b_diagnostics.d1_direction_quality.run_d1`, `research.phase5b_diagnostics.d3_specialist_oof_quality.run_d3` — D6 does NOT recompute OOF data itself, it reads D1/D3's already-computed `side_conditioned`/by-role outputs and restates them as a single side-by-side comparison table, per the design's intent to reuse D1/D3's arrays rather than duplicate work. Since D3 doesn't currently split Opportunity/Barrier's point-biserial by side (only D1 does, for Direction), D6 adds that missing by-side split for Opportunity/Barrier itself using `_oof_for_opportunity`/`_oof_for_barrier` plus `compute_direction_oof`'s `side` array — mirroring D1's own by-side pattern exactly.
- Produces: `run_d6(max_holding: int, rows: int = None) -> dict` with keys:
  - `"horizon": int`
  - `"direction"`: `{"long": {...pointbiserial_with_ci...}, "short": {...}}` (delegates to `run_d1`'s `side_conditioned` field)
  - `"opportunity"`: `{"long": {...}, "short": {...}}` (new by-side split, computed here)
  - `"barrier"`: `{"long": {...}, "short": {...}}` (new by-side split, computed here)

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_d6_long_short_conditioning.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d6_long_short_conditioning import run_d6


def test_run_d6_shape():
    result = run_d6(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    for role in ("direction", "opportunity", "barrier"):
        assert "long" in result[role] and "short" in result[role]
        assert result[role]["long"]["n"] is not None


if __name__ == "__main__":
    test_run_d6_shape()
    print("tests/test_phase5b_d6_long_short_conditioning.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_d6_long_short_conditioning.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/d6_long_short_conditioning.py`**

```python
"""research/phase5b_diagnostics/d6_long_short_conditioning.py
Batch 1, D6: does side-conditioning actually produce side-dependent
discriminative behavior, per specialist? Reuses D1's Direction-by-side
split directly; adds the equivalent by-side split for Opportunity/Barrier
(D3 reports them pooled across sides). See docs/superpowers/specs/
2026-08-26-golex-v3-phase5-batch1-diagnostics-design.md section D6.
"""
import numpy as np
from research.phase5b_diagnostics.d1_direction_quality import run_d1
from research.phase5_calibration import _oof_for_opportunity, _oof_for_barrier
from research.direction_side import compute_direction_oof
from research.phase5b_diagnostics._stats_utils import pointbiserial_with_ci


def _by_side_pointbiserial(t0_nz, y_full, p_full, mask, side):
    y_true = y_full[mask]
    p = p_full[mask]
    side_masked = side[mask]
    long_m = side_masked == 1.0
    short_m = side_masked == -1.0
    return {"long": pointbiserial_with_ci(y_true[long_m], p[long_m]),
            "short": pointbiserial_with_ci(y_true[short_m], p[short_m])}


def run_d6(max_holding: int, rows: int = None) -> dict:
    d1 = run_d1(max_holding=max_holding, rows=rows)
    direction_by_side = {
        "long": d1["side_conditioned"]["long"]["point_biserial"],
        "short": d1["side_conditioned"]["short"]["point_biserial"],
    }

    dir_oof = compute_direction_oof(max_holding=max_holding, rows=rows)
    side = dir_oof["side"]

    t0_o, y_o, p_o, m_o = _oof_for_opportunity(max_holding, rows=rows)
    t0_b, y_b, p_b, m_b = _oof_for_barrier(max_holding, rows=rows)
    assert np.array_equal(t0_o, dir_oof["t0_nz"]), "event index mismatch: opportunity vs direction_side"

    opportunity_by_side = _by_side_pointbiserial(t0_o, y_o, p_o, m_o, side)
    barrier_by_side = _by_side_pointbiserial(t0_b, y_b, p_b, m_b, side)

    return {"horizon": max_holding, "direction": direction_by_side,
            "opportunity": opportunity_by_side, "barrier": barrier_by_side}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d6(max_holding=h)
        print(f"D6 h={h}: direction={r['direction']} opportunity={r['opportunity']} barrier={r['barrier']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_d6_long_short_conditioning.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/d6_long_short_conditioning.py tests/test_phase5b_d6_long_short_conditioning.py
git commit -m "feat: add D6 long/short conditioning-behavior diagnostic"
```

---

### Task 8: Orchestrator — run all diagnostics, apply attribution framework, produce the Batch 1 report

**Files:**
- Create: `research/phase5b_diagnostics/run_all.py`
- Test: `tests/test_phase5b_run_all.py`

**Interfaces:**
- Consumes: `run_d1`, `run_d2`, `run_d3`, `run_d4`, `run_d5`, `run_d6` from Tasks 2-7 (all `(max_holding, rows=None) -> dict`, except `run_d5` which also takes `registry_dir=None`).
- Produces:
  - `run_batch1(rows: int = None, registry_dir: str = None) -> dict` — runs all six diagnostics for all three horizons (`research.phase4_dataset.HORIZONS`), returns `{"horizons": {15: {"d1": ..., "d2": ..., ..., "d6": ...}, 45: {...}, 90: {...}}, "attribution": {15: [...], 45: [...], 90: [...]}}`.
  - `apply_attribution_framework(horizon_results: dict) -> list[dict]` — implements the design doc's §4 table as executable rules over one horizon's D1-D6 results; each returned dict is `{"explanation": str, "evidence": str, "decisive": bool}` for one of the five candidate explanations (market/labels, Direction, downstream specialists, calibration, disagreement) — always returns all five, each stating whether the evidence points toward it, away from it, or is ambiguous (never silently omits an explanation just because the evidence is inconclusive for it).
  - `write_report(result: dict, out_dir: str) -> tuple[str, str]` — writes `batch1_report.json` (the raw `result` dict) and `batch1_report.md` (a human-readable rendering: one section per horizon, each diagnostic's key numbers with `n` and CI, then the attribution table) to `out_dir`, returns `(json_path, md_path)`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_phase5b_run_all.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.run_all import run_batch1, apply_attribution_framework, write_report


def test_apply_attribution_framework_always_returns_all_five_explanations():
    # minimal synthetic horizon_results shaped like one horizon's real output
    horizon_results = {
        "d1": {"oos": {"point_biserial": {"r": 0.02, "n": 100000, "ci_lo": 0.01, "ci_hi": 0.03}}},
        "d2": {"overall": {"up_frac": 0.02, "down_frac": 0.97, "timeout_frac": 0.01}},
        "d3": {"opportunity": {"point_biserial": {"r": 0.15, "n": 50000, "ci_lo": 0.14, "ci_hi": 0.16}},
               "barrier": {"point_biserial": {"r": 0.15, "n": 50000, "ci_lo": 0.14, "ci_hi": 0.16}}},
        "d4": {"contradiction_barrier_vs_reward_risk": {"rate": 0.3, "ci_lo": 0.29, "ci_hi": 0.31},
               "contradiction_opportunity_vs_barrier": {"rate": 0.1, "ci_lo": 0.09, "ci_hi": 0.11}},
        "d5": {"global": {"calibration": {"slope": 0.6, "intercept": 0.2}}, "traded_subset": "N/A (zero trades at this horizon)"},
        "d6": {},
    }
    explanations = apply_attribution_framework(horizon_results)
    names = {e["explanation"] for e in explanations}
    assert names == {"market/labels", "direction", "downstream_specialists", "calibration", "disagreement"}
    for e in explanations:
        assert "evidence" in e and "decisive" in e


def test_run_batch1_and_write_report_smoke(monkeypatch):
    # small rows= slice for a fast smoke test of the orchestration wiring itself,
    # NOT the real full-history run (that's a separate, long-running research run,
    # not a unit test -- see Step 6 below).
    result = run_batch1(rows=600000)
    assert set(result["horizons"].keys()) == {15, 45, 90}
    for h in (15, 45, 90):
        assert set(result["horizons"][h].keys()) == {"d1", "d2", "d3", "d4", "d5", "d6"}
        assert len(result["attribution"][h]) == 5

    with tempfile.TemporaryDirectory() as out_dir:
        json_path, md_path = write_report(result, out_dir)
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        with open(json_path) as f:
            reloaded = json.load(f)
        assert "horizons" in reloaded


if __name__ == "__main__":
    test_apply_attribution_framework_always_returns_all_five_explanations()
    test_run_batch1_and_write_report_smoke()
    print("tests/test_phase5b_run_all.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5b_run_all.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase5b_diagnostics/run_all.py`**

```python
"""research/phase5b_diagnostics/run_all.py
Batch 1 orchestrator: runs D1-D6 for all three horizons, applies the
attribution framework (docs/superpowers/specs/2026-08-26-golex-v3-phase5-
batch1-diagnostics-design.md section 4), writes one JSON+markdown report.
No diagnostic is ever skipped based on another's result -- "decisive" is a
label attached when the report is assembled, never a control-flow branch.
"""
import json
import os

from research.phase4_dataset import HORIZONS
from research.phase5b_diagnostics.d1_direction_quality import run_d1
from research.phase5b_diagnostics.d2_base_rate_audit import run_d2
from research.phase5b_diagnostics.d3_specialist_oof_quality import run_d3
from research.phase5b_diagnostics.d4_cross_specialist_consistency import run_d4
from research.phase5b_diagnostics.d5_calibration_reliability import run_d5
from research.phase5b_diagnostics.d6_long_short_conditioning import run_d6


def apply_attribution_framework(horizon_results: dict) -> list:
    d1, d2, d3, d4, d5 = (horizon_results["d1"], horizon_results["d2"], horizon_results["d3"],
                           horizon_results["d4"], horizon_results["d5"])

    explanations = []

    d2_skew = max(d2["overall"]["up_frac"] or 0.0, d2["overall"]["down_frac"] or 0.0)
    market_decisive = d2_skew >= 0.9
    explanations.append({
        "explanation": "market/labels",
        "evidence": f"D2 raw label base rate: up={d2['overall']['up_frac']}, down={d2['overall']['down_frac']} "
                    f"(dominant-side frac={d2_skew:.4f})",
        "decisive": market_decisive,
    })

    dir_r = d1["oos"]["point_biserial"]["r"]
    direction_weak = dir_r is not None and abs(dir_r) < 0.02
    explanations.append({
        "explanation": "direction",
        "evidence": f"D1 Direction point-biserial r={dir_r} (n={d1['oos']['point_biserial']['n']})",
        "decisive": direction_weak,
    })

    opp_r = d3["opportunity"]["point_biserial"]["r"]
    downstream_weaker = (dir_r is not None and opp_r is not None and abs(opp_r) < abs(dir_r) * 0.5)
    explanations.append({
        "explanation": "downstream_specialists",
        "evidence": f"D3 Opportunity point-biserial r={opp_r} vs D1 Direction r={dir_r}",
        "decisive": downstream_weaker,
    })

    slope = d5["global"]["calibration"]["slope"]
    calibration_off = slope is not None and abs(slope - 1.0) > 0.3
    explanations.append({
        "explanation": "calibration",
        "evidence": f"D5 global calibration slope={slope} (ideal=1.0), traded_subset={d5['traded_subset'] if isinstance(d5['traded_subset'], str) else 'computed'}",
        "decisive": calibration_off,
    })

    rate1 = d4["contradiction_barrier_vs_reward_risk"]["rate"]
    rate2 = d4["contradiction_opportunity_vs_barrier"]["rate"]
    disagreement_high = (rate1 or 0) > 0.2 or (rate2 or 0) > 0.2
    explanations.append({
        "explanation": "disagreement",
        "evidence": f"D4 contradiction rates: barrier_vs_reward_risk={rate1}, opportunity_vs_barrier={rate2}",
        "decisive": disagreement_high,
    })

    return explanations


def run_batch1(rows: int = None, registry_dir: str = None) -> dict:
    horizons = {}
    attribution = {}
    for h in HORIZONS:
        h_result = {
            "d1": run_d1(max_holding=h, rows=rows),
            "d2": run_d2(max_holding=h, rows=rows),
            "d3": run_d3(max_holding=h, rows=rows),
            "d4": run_d4(max_holding=h, rows=rows),
            "d5": run_d5(max_holding=h, rows=rows, registry_dir=registry_dir),
            "d6": run_d6(max_holding=h, rows=rows),
        }
        horizons[h] = h_result
        attribution[h] = apply_attribution_framework(h_result)
    return {"horizons": horizons, "attribution": attribution}


def _render_markdown(result: dict) -> str:
    lines = ["# GOLEX V3 Phase 5 Batch 1 — Diagnostic Foundation Report", ""]
    for h, h_result in result["horizons"].items():
        lines.append(f"## Horizon h={h}")
        lines.append("")
        lines.append(f"- D1 Direction point-biserial: {h_result['d1']['oos']['point_biserial']}")
        lines.append(f"- D2 base rate: {h_result['d2']['overall']}")
        lines.append(f"- D3 Opportunity point-biserial: {h_result['d3']['opportunity']['point_biserial']}, "
                      f"win_rate={h_result['d3']['opportunity']['win_rate']}")
        lines.append(f"- D4 contradiction rates: {h_result['d4']['contradiction_barrier_vs_reward_risk']}, "
                      f"{h_result['d4']['contradiction_opportunity_vs_barrier']}")
        lines.append(f"- D5 global calibration: {h_result['d5']['global']['calibration']}, "
                      f"traded_subset={'N/A' if isinstance(h_result['d5']['traded_subset'], str) else 'computed'}")
        lines.append(f"- D6: {h_result['d6']}")
        lines.append("")
        lines.append("### Attribution")
        for e in result["attribution"][h]:
            marker = "DECISIVE" if e["decisive"] else "not decisive"
            lines.append(f"- **{e['explanation']}** ({marker}): {e['evidence']}")
        lines.append("")
    return "\n".join(lines)


def write_report(result: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "batch1_report.json")
    md_path = os.path.join(out_dir, "batch1_report.md")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write(_render_markdown(result))
    return json_path, md_path


if __name__ == "__main__":
    result = run_batch1()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    json_path, md_path = write_report(result, out_dir)
    print(f"Batch 1 report written: {json_path}, {md_path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5b_run_all.py -v`
Expected: PASS (the `test_run_batch1_and_write_report_smoke` test takes several minutes — it runs all six diagnostics across all three horizons at `rows=600000`, not full history; that's expected)

- [ ] **Step 5: Commit**

```bash
git add research/phase5b_diagnostics/run_all.py tests/test_phase5b_run_all.py
git commit -m "feat: add Phase 5 Batch 1 orchestrator (run_all, attribution framework, report writer)"
```

- [ ] **Step 6: Run the REAL full-history Batch 1 diagnostics (research run, not a test)**

This is the actual deliverable — everything above builds toward this one run. Full 6.7-year data, all three horizons, real registry (`registry_dir=None` defaults to the real `models/registry/`).

Run (background, monitor to completion — expect this to take multiple hours given each horizon repeats several full OOF fits across D1/D3/D5/D6, similar in scale to the Phase 5A full-history retrain):

```bash
cd /home/jith/.hermes/profiles/trading/scripts
nohup /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5b_diagnostics.run_all > research/phase5b_diagnostics/output/full_run.log 2>&1 &
```

Monitor via the real PID (not a fixed sleep), per this project's established discipline for long-running research jobs — `ps aux | grep run_all`, `tail -f` the log, `while kill -0 <pid>; do sleep 60; done`. Log to a persistent, in-repo path (not `/tmp`) given this environment has shown it can reboot mid-run.

Expected output: `research/phase5b_diagnostics/output/batch1_report.json` and `batch1_report.md`.

- [ ] **Step 7: Paste the full markdown report in chat** (per the user's standing preference — full report text, not just the file path), and commit the two report files:

```bash
git add research/phase5b_diagnostics/output/batch1_report.json research/phase5b_diagnostics/output/batch1_report.md
git commit -m "data: Phase 5 Batch 1 full-history diagnostic report"
```

---

## Self-review notes (fixed inline during plan authoring, kept here for the record)

- **D4's original "MFE vs. TP distance" contradiction rule was circular** (production sets TP directly to MFE's own q75, so MFE could never contradict itself) — replaced with a genuine cross-check between Barrier's independently-fit `p_barrier_win` and the independently-fit `mae_r`/`mfe_r` reward-to-risk ratio, per the design doc's own corrected §D4 text. The plan's Task 5 code reflects the corrected version directly; there is no stale circular version left anywhere in this plan.
- **D3's "MAE/MFE coverage by side" cannot be a true coverage statistic** given `_oof_predicted_mae_mfe`'s actual return shape (predictions only, no paired true-excursion array at that call site) — Task 4 states this explicitly and names the field `q75_pred_mean` instead of `q75_coverage` to avoid overclaiming.
- **D5 cannot reuse `replay_and_validate`'s existing return contract** (aggregate-only, no per-event arrays) — Task 6 re-runs the same `evaluate()` loop pattern independently inside the diagnostic module rather than modifying `research/phase5_ev_engine.py`, keeping this batch's "no production/Phase-5A-file changes" constraint intact.
