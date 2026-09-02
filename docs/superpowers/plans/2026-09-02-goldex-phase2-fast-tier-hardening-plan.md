# GOLDEX Phase 2 Fast Tier — Hardening Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two explicitly unresolved readiness gaps from the Phase 2 Fast Tier report (latency 10-20x over budget; context-bucket calibration effectively non-functional), then adversarially test the seven core intelligence properties, and verify simulator-scale + MT5-interface compatibility — without touching architecture, Phase 1, or scope boundaries.

**Architecture:** No redesign. Same `intelligence/` package, same `ToolTrust`/`context_bucket`/`FastTierReasoner`/`FastTierDecisionEngine` components. Latency fix = bound + dedupe + extend the *existing* refit-cache mechanism (already used for GARCH/Kalman) to more sources, plus a bounded look-back window on `closes_so_far`. Calibration fix = recenter/rescale the two existing constants (`MAGNITUDE_LOG_CENTER`, `MAGNITUDE_LOG_SCALE`) from a real measured distribution instead of a hand-guess — no new stateful component, no quantile-tracking machinery (rejected as scope expansion; see Task 4 rationale).

**Tech Stack:** Python, numpy, scipy, pytest (matches Phase 2 Fast Tier).

**Spec:** This plan is governed directly by the user's "GOLDEX — PHASE 2 FAST-TIER HARDENING" mandate (verbatim in conversation, not a separate file) plus the existing `docs/superpowers/reports/2026-08-29-goldex-phase2-fast-tier-report.md` ("What is NOT ready" section, which this plan closes) and `docs/superpowers/specs/2026-08-29-goldex-phase2-final-architecture-decision.md` (architecture invariants, unchanged).

## Global Constraints

- Do NOT add new trading strategies, predictors, model families, or evidence sources.
- Do NOT build the Slow Tier.
- Do NOT connect XM/MT5 live execution or place any order.
- Do NOT change Phase 1 `simulator/`, `market/`, `contracts/` semantics — read-only verification only, unless a genuinely required compatibility fix is found (must be justified explicitly if so).
- Do NOT resurrect any V3/V4 component or import path.
- Do NOT optimize against the protected OOS split (`EnvironmentTag.SIMULATED_OOS_TEST`) — all calibration/measurement in this plan uses training-partition synthetic data only, same partition discipline as the Phase 2 plan.
- Do NOT introduce a profitability target or claim.
- Do NOT reduce correctness to hit a latency number, and do NOT change directional/trust/credit semantics to make calibration "look better."
- Every numeric constant this plan changes must be justified by a measured distribution recorded in the task's own commit/report, not guessed.
- One final whole-branch review (most capable model), ONE fix wave, no second fix wave — residual findings get parked with an explicit ruling, same process as the Phase 2 plan.

---

### Task 1: Kalman dedup — one filter run per evidence pass, not two

**Files:**
- Modify: `intelligence/evidence_sources.py:131-158` (`_make_kalman_velocity_compute`, `_make_kalman_innovation_compute`)
- Test: `tests/intelligence/test_evidence_sources.py`

**Interfaces:**
- Consumes: `kalman_level_trend_filter(closes) -> (levels, velocities, innovations)` from `research.phase4_kalman_trend_mechanism` (unchanged).
- Produces: `kalman_filtered_velocity` and `kalman_innovation` `EvidenceSourceSpec.compute` callables with identical return values as before (this is a pure performance change — output must be byte-identical to today).

Currently `kalman_filtered_velocity` and `kalman_innovation` each independently call `kalman_level_trend_filter(closes)` — the same O(n) recursive filter run twice per evidence pass for no reason (confirmed by the investigation fork: ~17ms wasted per pass). Fix: give both wrapper closures a shared, request-scoped cache keyed on `id(closes_so_far)` combined with `len(closes_so_far)`, OR (simpler, and what to actually implement) have `build_default_registry()` construct one shared mutable single-slot cache object and close both wrappers over it, invalidated whenever the input array differs from what's cached.

- [ ] **Step 1: Write the failing test proving double-computation today**

```python
def test_kalman_velocity_and_innovation_share_one_filter_run(monkeypatch):
    import intelligence.evidence_sources as es
    calls = []
    original = es.kalman_level_trend_filter

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(es, "kalman_level_trend_filter", counting)
    registry = es.build_default_registry()
    closes = np.cumsum(np.random.default_rng(0).normal(0, 1, 200)) + 2000.0
    registry.specs()["kalman_filtered_velocity"].compute(closes)
    registry.specs()["kalman_innovation"].compute(closes)
    assert len(calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/intelligence/test_evidence_sources.py::test_kalman_velocity_and_innovation_share_one_filter_run -v`
Expected: FAIL, `len(calls) == 2`.

- [ ] **Step 3: Implement the shared-cache fix**

```python
def _make_kalman_pair_computes():
    """Both kalman_filtered_velocity and kalman_innovation come from one
    kalman_level_trend_filter run. This shared, closure-scoped single-slot
    cache avoids running the O(n) recursive filter twice per evidence pass
    -- pure performance, zero output change (see
    test_kalman_velocity_and_innovation_share_one_filter_run)."""
    cache: dict = {"key": None, "result": None}

    def _filtered(closes: np.ndarray):
        key = (closes.shape, float(closes[-1]) if len(closes) else None, len(closes))
        if cache["key"] != key:
            cache["result"] = kalman_level_trend_filter(closes)
            cache["key"] = key
        return cache["result"]

    def compute_velocity(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "kalman_filtered_velocity"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < 2:
            return EvidenceValue(None, 0.0, name)
        _levels, velocities, _innovations = _filtered(closes)
        value = _last_finite(velocities)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    def compute_innovation(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "kalman_innovation"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < 2:
            return EvidenceValue(None, 0.0, name)
        _levels, _velocities, innovations = _filtered(closes)
        value = _last_finite(innovations)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute_velocity, compute_innovation
```

Replace `_make_kalman_velocity_compute()` and `_make_kalman_innovation_compute()` calls in `build_default_registry()` with:

```python
_kalman_velocity_compute, _kalman_innovation_compute = _make_kalman_pair_computes()
```

and register `compute=_kalman_velocity_compute` / `compute=_kalman_innovation_compute` respectively. Delete the two old single-purpose factory functions.

Caution: the cache key must include enough of `closes_so_far` to distinguish two *different* arrays of the same length within one process lifetime (e.g. across two different replay runs sharing a reasoner). Using `(shape, last_value, len)` is a cheap approximate key — good enough because within one `_compute_evidence` call both wrappers are invoked back-to-back on the literal same array object; state must NOT persist stale results across different arrays. Add a second test for this:

```python
def test_kalman_pair_cache_does_not_leak_across_different_arrays():
    import intelligence.evidence_sources as es
    compute_v, compute_i = es._make_kalman_pair_computes()
    closes_a = np.cumsum(np.random.default_rng(1).normal(0, 1, 100)) + 1000.0
    closes_b = np.cumsum(np.random.default_rng(2).normal(0, 1, 100)) + 3000.0
    v_a = compute_v(closes_a).value
    v_b = compute_v(closes_b).value
    assert v_a != v_b
    # Cross-check against uncached ground truth
    _levels, vel_b, _inn = kalman_level_trend_filter(closes_b)
    assert abs(v_b - _last_finite(vel_b)) < 1e-12
```

- [ ] **Step 4: Run both tests to verify they pass**

Run: `.venv/bin/pytest tests/intelligence/test_evidence_sources.py -k kalman -v`
Expected: PASS (both new tests, plus all pre-existing Kalman tests in the file still pass unchanged).

- [ ] **Step 5: Run the full evidence_sources test file to confirm no regression**

Run: `.venv/bin/pytest tests/intelligence/test_evidence_sources.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add intelligence/evidence_sources.py tests/intelligence/test_evidence_sources.py
git commit -m "perf: share one Kalman filter run between velocity and innovation sources"
```

---

### Task 2: Extend refit-caching to the 4 non-directional context-inert sources

**Files:**
- Modify: `intelligence/fast_tier.py:40-44` (`EXPENSIVE_SOURCE_NAMES`)
- Test: `tests/intelligence/test_fast_tier.py`

**Interfaces:**
- Consumes: `FastTierReasoner._compute_evidence` (unchanged mechanism — already refit-caches by `bar - cached_bar_index < refit_interval`).
- Produces: same `_compute_evidence` behavior, now caching 6 of 9 sources instead of 3.

The investigation fork measured `multiscale_vol_ratio` (33ms), `vol_regime_transition` (36ms), `rolling_skew` (21ms), `rolling_excess_kurtosis` (21ms) — roughly 110ms/pass — as recomputed on every single call despite being non-directional (excluded from votes/credit per the C1 fix) and, for 2 of the 4, not even feeding `context_bucket()`. The refit-cache mechanism (`EXPENSIVE_SOURCE_NAMES` + `refit_interval`) already exists and is already an accepted "stale between refits" tradeoff for GARCH/Kalman — extending it to these 4 is the same tradeoff, not a new one, and changes no directional/trust/credit semantics (they don't participate in any of those).

- [ ] **Step 1: Write the failing test**

```python
def test_all_six_expensive_sources_are_refit_cached():
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner, EXPENSIVE_SOURCE_NAMES
    assert EXPENSIVE_SOURCE_NAMES == frozenset({
        "garch_conditional_variance",
        "kalman_filtered_velocity",
        "kalman_innovation",
        "multiscale_vol_ratio",
        "vol_regime_transition",
        "rolling_skew",
        "rolling_excess_kurtosis",
    })

def test_non_directional_sources_reuse_cached_value_between_refits(monkeypatch):
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry, refit_interval=50)
    closes = np.cumsum(np.random.default_rng(3).normal(0, 1, 200)) + 2000.0
    ev1 = reasoner._compute_evidence(closes[:120])
    ev2 = reasoner._compute_evidence(closes[:121])  # 1 bar later, well within refit_interval
    assert ev1["rolling_skew"].value == ev2["rolling_skew"].value
    assert ev1["multiscale_vol_ratio"].value == ev2["multiscale_vol_ratio"].value
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier.py -k "expensive_sources or reuse_cached_value" -v`
Expected: FAIL (set mismatch on the first test; second test currently passes trivially only if unrelated — check it actually exercises new behavior by first confirming values *differ* today before the fix, via a throwaway run, not committed).

- [ ] **Step 3: Implement**

```python
EXPENSIVE_SOURCE_NAMES = frozenset({
    "garch_conditional_variance",
    "kalman_filtered_velocity",
    "kalman_innovation",
    "multiscale_vol_ratio",
    "vol_regime_transition",
    "rolling_skew",
    "rolling_excess_kurtosis",
})
```

Update the module docstring comment above it (lines 34-39) to state the set now covers 6 of 9 sources and why (all are pure functions of `closes_so_far` with no side effects, so caching-between-refits is a uniform, safe tradeoff regardless of directionality; only `momentum_scalar` and `path_pca_projection` remain always-fresh because they are already cheap enough — ~20us and ~45ms — that caching them buys negligible latency for a real staleness cost).

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add intelligence/fast_tier.py tests/intelligence/test_fast_tier.py
git commit -m "perf: extend refit-caching to the 4 non-directional context-inert sources"
```

---

### Task 3: Bound `closes_so_far` to a fixed look-back window before compute

**Files:**
- Modify: `intelligence/fast_tier.py` (`FastTierReasoner.__init__`, `_compute_evidence`)
- Test: `tests/intelligence/test_fast_tier.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `FastTierReasoner(registry, refit_interval=50, load_bearing_floor=0.05, max_history_window=2000)` — new constructor parameter, default `2000`. All evidence sources now see at most the last `max_history_window` closes, never the full unbounded replay history.

This is a genuine, disclosed behavior change (not hidden as "pure performance"): the report and investigation fork both note `closes_so_far` grows unboundedly across a real replay (years of M1 bars), so every O(n) source's cost — and now, after Task 1/2, specifically GARCH's cost — grows without bound over a long run even with refit-caching, because each refit still refits over the *entire* history. Capping the window makes GARCH/Kalman fit over "the most recent N bars" instead of "all bars ever seen," which is a defensible and common choice (conditional variance and trend velocity are dominated by recent dynamics anyway — old GARCH-style models are rarely fit over years of 1-minute data in practice) but is NOT numerically identical to the old unbounded behavior. Document this explicitly in the module docstring and the final report; do not claim byte-identical output for this task, unlike Task 1.

- [ ] **Step 1: Write the failing test**

```python
def test_max_history_window_bounds_what_sources_see(monkeypatch):
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry, max_history_window=100)
    seen_lengths = []
    original_specs = registry.specs()
    for name, spec in original_specs.items():
        orig_compute = spec.compute
        def wrapped(closes_so_far, orig_compute=orig_compute):
            seen_lengths.append(len(closes_so_far))
            return orig_compute(closes_so_far)
        spec.compute = wrapped
    closes = np.cumsum(np.random.default_rng(4).normal(0, 1, 5000)) + 2000.0
    reasoner._compute_evidence(closes)
    assert max(seen_lengths) <= 100
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier.py::test_max_history_window_bounds_what_sources_see -v`
Expected: FAIL, `max(seen_lengths) == 5000`.

- [ ] **Step 3: Implement**

In `__init__`, add `max_history_window: int = 2000` and store `self.max_history_window = max_history_window`. In `_compute_evidence`, before the loop:

```python
def _compute_evidence(self, closes_so_far: np.ndarray) -> dict[str, EvidenceValue]:
    if len(closes_so_far) > self.max_history_window:
        closes_so_far = closes_so_far[-self.max_history_window:]
    bar = len(closes_so_far)
    ...
```

Note: `bar` (used as the refit-cache bar index) must be computed from the *windowed* length so the refit-cache invalidation logic stays correct relative to what's actually being fit — do not use the original unbounded length here.

Add a docstring note directly above `max_history_window`'s definition:

```python
# Sources see at most the most recent `max_history_window` closes, never
# the full unbounded replay history. This bounds worst-case per-decision
# cost independent of how long a replay run has been going (previously,
# GARCH/Kalman refit over the ENTIRE history every refit_interval bars, so
# cost grew without bound over a multi-year replay even with caching).
# This is a genuine behavior change, not just a performance one: GARCH and
# Kalman now fit "the last N bars" rather than "everything ever observed."
# Documented, disclosed, not hidden -- see the hardening report for the
# before/after distributional comparison that justifies 2000 as a default
# (long enough to keep GARCH's likelihood well-conditioned, short enough to
# bound worst-case latency).
```

- [ ] **Step 4: Add a distributional sanity test** — confirm windowing doesn't wildly change GARCH/Kalman output relative to full-history on realistic data, so the disclosed semantic change is small in practice, not just declared small:

```python
def test_windowed_garch_output_close_to_full_history_output():
    from research.phase4_garch_volatility_mechanism import fit_garch11
    closes = np.cumsum(np.random.default_rng(5).normal(0, 1, 3000)) + 2000.0
    returns_full = np.diff(closes, prepend=closes[0])
    returns_windowed = np.diff(closes[-2000:], prepend=closes[-2000])
    _params_full, sigma2_full = fit_garch11(returns_full)
    _params_win, sigma2_win = fit_garch11(returns_windowed)
    # Compare the LAST (most recent, decision-relevant) conditional variance --
    # not the whole series, which legitimately differs by construction.
    rel_diff = abs(sigma2_full[-1] - sigma2_win[-1]) / max(abs(sigma2_full[-1]), 1e-12)
    assert rel_diff < 0.5  # same order of magnitude, not identical
```

Run it once and record the actual `rel_diff` value in the task's commit message or a code comment — this is the evidence backing the "small in practice" claim, not a guess.

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add intelligence/fast_tier.py tests/intelligence/test_fast_tier.py
git commit -m "perf: bound closes_so_far to a fixed look-back window, documented as a disclosed behavior change"
```

---

### Task 4: Recalibrate `context_bucket` from a real measured magnitude distribution

**Files:**
- Create: `scripts/calibrate_context_bucket.py` (throwaway-but-committed calibration script, not a test)
- Modify: `intelligence/fast_tier.py:62-81` (`MAGNITUDE_LOG_CENTER`, `MAGNITUDE_LOG_SCALE`)
- Test: `tests/intelligence/test_fast_tier.py`

**Interfaces:**
- Consumes: `build_default_registry()`, `FastTierReasoner`, `context_bucket()` (unchanged signatures).
- Produces: recalibrated `MAGNITUDE_LOG_CENTER`/`MAGNITUDE_LOG_SCALE` module constants; no new stateful component.

**Root cause (confirmed by investigation fork, empirical histogram over 220 sampled decision points on a 1200-bar synthetic random walk):** magnitude median ≈ 0.037 vs. the assumed `MAGNITUDE_LOG_CENTER = log(0.15)` — the constant was hand-guessed ("documented round numbers... measured empirically" per the old docstring, but not actually fit to the registry's real output) and is ~4x off, so nearly every real decision lands with `z < 0`, concentrating in buckets 0-1 (measured 96% in the original report's larger sample; 220/220 in buckets {0,1} in the fork's sample). `ToolTrust`'s Beta update itself has no bug — the sparsity is a consequence of bucket concentration, not a separate defect.

**Why recentering constants, not a stateful quantile tracker:** the mandate's Section 3 forbids scope expansion; a rolling-quantile bucketer is a new stateful component with its own causal-boundary surface (Task 14's truncation tests would need to be re-derived for it) for a problem that a two-constant recalibration fully explains and fixes. If a future pass finds the *distribution itself* drifts enough over time that a fixed center goes stale again (e.g. a different instrument, a different price scale), that is the trigger to revisit — not this pass.

- [ ] **Step 1: Write the calibration script** (not a pytest test — a one-off, but committed so the numbers below are reproducible and auditable, not asserted from nowhere)

```python
"""Calibrates MAGNITUDE_LOG_CENTER / MAGNITUDE_LOG_SCALE in intelligence/fast_tier.py
against the real evidence registry's output distribution on TRAINING-partition
synthetic data only (never protected OOS -- see mandate Section on calibration).
Run manually; not part of the test suite. Prints the values to hand-copy into
fast_tier.py along with the sample statistics that justify them.
"""
import math
import numpy as np

from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import FastTierReasoner, _evidence_scalar

def main():
    rng = np.random.default_rng(42)
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry)
    closes = np.cumsum(rng.normal(0, 0.15, 6000)) + 2000.0  # gold-tick-like noise scale
    magnitudes = []
    for bar in range(200, len(closes), 5):
        evidence = reasoner._compute_evidence(closes[:bar])
        sigma2 = _evidence_scalar(evidence, "garch_conditional_variance")
        velocity = _evidence_scalar(evidence, "kalman_filtered_velocity")
        if sigma2 is None and velocity is None:
            continue
        sigma2 = max(sigma2, 0.0) if sigma2 is not None else 0.0
        velocity = abs(velocity) if velocity is not None else 0.0
        magnitude = math.log1p(sigma2) + math.log1p(velocity)
        if magnitude > 0.0:
            magnitudes.append(magnitude)
    arr = np.array(magnitudes)
    median = float(np.median(arr))
    p10, p90 = float(np.percentile(arr, 10)), float(np.percentile(arr, 90))
    center = math.log(median)
    coverage_zspan = 3.0  # span the p10-p90 range over roughly a +/-1.5 sigmoid-input range
    scale = math.log(p90 / p10) / coverage_zspan if p90 > p10 else 1.0
    print(f"n={len(arr)} median={median:.5f} p10={p10:.5f} p90={p90:.5f}")
    print(f"MAGNITUDE_LOG_CENTER = math.log({median:.5f})  # = {center:.5f}")
    print(f"MAGNITUDE_LOG_SCALE = {scale:.5f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the output**

Run: `.venv/bin/python scripts/calibrate_context_bucket.py`
Record the printed `median`, `p10`, `p90`, `MAGNITUDE_LOG_CENTER`, `MAGNITUDE_LOG_SCALE` values in the commit message for this task verbatim.

- [ ] **Step 3: Write the failing test** — asserts buckets are actually spread across the range on realistic data, not concentrated:

```python
def test_context_bucket_spreads_across_range_on_realistic_data():
    from collections import Counter
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner, context_bucket
    rng = np.random.default_rng(7)  # different seed from calibration -- held-out check
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry)
    closes = np.cumsum(rng.normal(0, 0.15, 3000)) + 2000.0
    buckets = []
    for bar in range(200, len(closes), 5):
        evidence = reasoner._compute_evidence(closes[:bar])
        buckets.append(context_bucket(evidence))
    counts = Counter(buckets)
    live_buckets = {b for b in counts if b >= 0}
    # Not asserting a specific distribution shape (that would be fitting to
    # this exact data) -- only that calibration is no longer collapsed to
    # 1-2 buckets out of 5.
    assert len(live_buckets) >= 3, f"only reached buckets {sorted(counts)}"
    largest_share = max(counts[b] for b in live_buckets) / len(buckets)
    assert largest_share < 0.70, f"one bucket still holds {largest_share:.0%} of decisions"
```

- [ ] **Step 4: Run to verify it fails against the current (uncalibrated) constants**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier.py::test_context_bucket_spreads_across_range_on_realistic_data -v`
Expected: FAIL (matches the 96%-in-one-bucket finding).

- [ ] **Step 5: Implement** — replace the constants with the calibration script's output and rewrite the surrounding docstring comment (lines 62-78) to state the real provenance:

```python
# Re-centering constants for the context magnitude (see context_bucket()).
# CALIBRATED (not hand-guessed) from scripts/calibrate_context_bucket.py
# run against training-partition synthetic data (see that script's docstring
# for why never the protected OOS split): n=<N> samples, median=<MEDIAN>,
# p10=<P10>, p90=<P90>. The original hand-guessed center (log(0.15)) was
# ~4x off the real measured median (~0.037), which collapsed 96% of real
# decisions into a single bucket -- see the Phase 2 hardening report for
# before/after histograms.
MAGNITUDE_LOG_CENTER = math.log(<MEDIAN>)
MAGNITUDE_LOG_SCALE = <SCALE>
```

(Fill `<N>`, `<MEDIAN>`, `<P10>`, `<P90>`, `<SCALE>` with the actual Step 2 output — do not invent numbers.)

- [ ] **Step 6: Run tests, verify pass**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier.py -v`
Expected: all PASS, including the pre-existing `context_bucket` reachability tests from the Phase 2 fix wave (must not regress those).

- [ ] **Step 7: Commit**

```bash
git add scripts/calibrate_context_bucket.py intelligence/fast_tier.py tests/intelligence/test_fast_tier.py
git commit -m "fix: recalibrate context_bucket constants from measured evidence distribution, not a guess"
```

---

### Task 5: Latency report — rerun perf tests, tighten bounds, before/after numbers

**Files:**
- Modify: `tests/intelligence/test_fast_tier_performance.py` (tighten bounds to reflect the Task 1-3 fixes)
- Create: `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-latency-report.md`

**Interfaces:**
- Consumes: the existing perf test harness (unchanged fixtures/timing methodology).
- Produces: a standalone latency report with measured before/after numbers per the mandate's Section 7 acceptance criterion 2.

- [ ] **Step 1: Run the existing perf test suite and record current (post Task 1-4) numbers**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier_performance.py -v -s`
Record the printed mean/p50/p99 for `compute_all`, cached `hypothesis()`, refit-triggering `hypothesis()`, and `decide()`.

- [ ] **Step 2: Tighten the bound assertions to reflect the real improvement, with headroom, not a razor edge**

Update each `assert ... < BOUND_MS` in the file to roughly 1.5-2x the Step 1 measured value (same margin discipline as the Phase 2 whole-branch review's own bound-tightening — enough to catch a real regression, not so tight it flakes). Do not simply copy the exact measured number as the bound.

- [ ] **Step 3: Run again to confirm the tightened bounds hold**

Run: `.venv/bin/pytest tests/intelligence/test_fast_tier_performance.py -v -s`
Expected: PASS with real margin.

- [ ] **Step 4: Write the latency report**

Must include, at minimum: a table of before (from the Phase 2 report: `decide()` p99 ≈ 40-60ms, `compute_all()` p99 ≈ 440-475ms) vs. after (Step 1's numbers); the per-source cost breakdown from the investigation (GARCH ~220ms/54% dominant, Kalman dedup savings, the 4 newly-cached non-directional sources); an explicit statement of what is now bounded (worst-case cost independent of replay length, via Task 3's window) vs. what remains architecturally unbounded if anything; and an honest statement of whether the Phase 1 ~2ms/bar budget is now met (it will very likely still not be, even after these fixes — GARCH's Python-loop MLE alone at any bounded window is still tens of ms; say so plainly, do not round up).

- [ ] **Step 5: Commit**

```bash
git add tests/intelligence/test_fast_tier_performance.py docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-latency-report.md
git commit -m "perf: measure and report before/after latency, tighten perf test bounds"
```

---

### Task 6: Adversarial test — conditional trust distinguishes useful vs. unreliable evidence by context

**Files:**
- Create: `tests/intelligence/test_adversarial_conditional_trust.py`

**Interfaces:**
- Consumes: `ToolTrust`, `context_bucket`, `FastTierReasoner.hypothesis` (unchanged).

Construct two synthetic evidence sources by directly manipulating `ToolTrust` state (not the real registry, to get deterministic ground truth): source A is genuinely predictive in context bucket 2 (train it with mostly `agreed=True` updates in bucket 2, mostly `agreed=False` in bucket 0) and source B is the reverse. Verify `posterior_mean` reflects this asymmetry per-bucket, and that a hand-built `Hypothesis`-equivalent weighting (reuse the actual weighting formula from `FastTierReasoner.hypothesis`, not a reimplementation) upweights the source that's trustworthy in the current bucket.

- [ ] **Step 1: Write the test**

```python
def test_trust_is_conditioned_on_context_not_global():
    from intelligence.fast_tier import ToolTrust
    trust = ToolTrust()
    for _ in range(30):
        trust.update("source_a", context_bucket=2, agreed=True)
        trust.update("source_a", context_bucket=0, agreed=False)
        trust.update("source_b", context_bucket=0, agreed=True)
        trust.update("source_b", context_bucket=2, agreed=False)

    assert trust.posterior_mean("source_a", 2) > 0.9
    assert trust.posterior_mean("source_a", 0) < 0.1
    assert trust.posterior_mean("source_b", 0) > 0.9
    assert trust.posterior_mean("source_b", 2) < 0.1
    # The SAME source must be judged differently depending on context --
    # this is the entire point of conditioning on context_bucket instead
    # of a single global trust score.
    assert trust.posterior_mean("source_a", 2) - trust.posterior_mean("source_a", 0) > 0.7
```

- [ ] **Step 2: Run, verify it passes against current `ToolTrust`** (this is a characterization test of existing correct behavior, not a bug fix — expect PASS on first run)

Run: `.venv/bin/pytest tests/intelligence/test_adversarial_conditional_trust.py -v`
Expected: PASS. If it fails, STOP and treat as a real defect — do not weaken the assertion to make it pass.

- [ ] **Step 3: Commit**

```bash
git add tests/intelligence/test_adversarial_conditional_trust.py
git commit -m "test: adversarial coverage for context-conditional trust"
```

---

### Task 7: Adversarial test — contradiction does not average into a false-confidence trade

**Files:**
- Create: `tests/intelligence/test_adversarial_contradiction.py`

**Interfaces:**
- Consumes: `FastTierReasoner.hypothesis`, `FastTierDecisionEngine.decide` (unchanged).

Build a registry of stub `EvidenceSourceSpec`s (real spec objects, fake `compute` callables returning fixed `EvidenceValue`s) where half vote strongly LONG and half vote strongly SHORT with equal confidence and equal trust. Verify `aggregate_uncertainty` is high (near 1.0, per the `disagreement` term) and that `FastTierDecisionEngine.decide` returns `NO_TRADE` despite `net_directional_belief` potentially being non-trivial in magnitude — i.e. uncertainty, not just belief magnitude, must gate the decision.

- [ ] **Step 1: Write the test**

```python
def test_contradictory_evidence_produces_high_uncertainty_not_averaged_confidence():
    from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
    from intelligence.fast_tier import FastTierReasoner, ToolTrust

    registry = EvidenceRegistry()
    for i in range(3):
        registry.register(EvidenceSourceSpec(
            name=f"long_source_{i}", mathematical_formulation="stub", required_inputs=[],
            assumptions="stub", known_failure_conditions="none",
            compute=lambda closes: EvidenceValue(1.0, 1.0, "long_source"),
            is_directional=True, computational_cost_hint="stub",
        ))
        registry.register(EvidenceSourceSpec(
            name=f"short_source_{i}", mathematical_formulation="stub", required_inputs=[],
            assumptions="stub", known_failure_conditions="none",
            compute=lambda closes: EvidenceValue(-1.0, 1.0, "short_source"),
            is_directional=True, computational_cost_hint="stub",
        ))

    reasoner = FastTierReasoner(registry)
    trust = ToolTrust()  # uninformative prior for all -- equal trust
    closes = np.linspace(2000.0, 2010.0, 50)
    hyp = reasoner.hypothesis(closes, market_state=None, trust=trust)

    assert hyp.aggregate_uncertainty > 0.9
    assert abs(hyp.net_directional_belief) < 0.1  # equal-strength opposition nets near zero
```

- [ ] **Step 2: Write a second test through the actual decision engine** to prove NO_TRADE, not just a high uncertainty number in isolation:

```python
def test_decision_engine_abstains_under_contradiction():
    from intelligence.decision_engine import FastTierDecisionEngine
    from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
    from intelligence.fast_tier import FastTierReasoner, ToolTrust
    from intelligence.bootstrap import analytical_sltp_bootstrap, analytical_sizing_bootstrap
    # (import whatever real constructor signature FastTierDecisionEngine takes --
    # match Task 6/8's actual constructor from intelligence/decision_engine.py)
    ...
    action, sl, tp, size = engine.decide(market_state)
    assert action == "NO_TRADE"
```

(Implementer: read `intelligence/decision_engine.py`'s actual `FastTierDecisionEngine.__init__` and `MarketState` construction from `tests/intelligence/test_full_fast_tier_integration.py` for the exact real construction pattern — do not guess constructor args.)

- [ ] **Step 3: Run both, verify PASS** (again, characterization of already-correct behavior per the Phase 2 whole-branch review's contradiction test — if either fails, treat as a real defect, do not adjust the assertion)

Run: `.venv/bin/pytest tests/intelligence/test_adversarial_contradiction.py -v`

- [ ] **Step 4: Commit**

```bash
git add tests/intelligence/test_adversarial_contradiction.py
git commit -m "test: adversarial coverage proving contradiction produces abstention, not averaged confidence"
```

---

### Task 8: Adversarial test — abstention, direction neutrality, credit assignment, thesis invalidation, reassessment loop, causality

**Files:**
- Create: `tests/intelligence/test_adversarial_abstention_and_neutrality.py`
- Create: `tests/intelligence/test_adversarial_credit_and_reassessment.py`

**Interfaces:**
- Consumes: `FastTierReasoner`, `FastTierDecisionEngine`, `credit_assignment.assign_trade_credit`, `Thesis`, `context_bucket`, `ToolTrust` — all unchanged.

This task bundles the remaining five mandate-required adversarial properties not yet independently stress-tested (several are already covered by Phase 2's own suite — Task 5's `test_dissenting_load_bearing_source_gets_opposite_credit` for credit assignment, Task 14's causal-truncation test for causality, `manage()`'s existing tests for reassessment — this task adds targeted adversarial cases the Phase 2 suite's normal-path tests didn't specifically target, and a single cross-reference note for each already-covered property rather than duplicating it).

- [ ] **Step 1: Abstention under genuine uncertainty (new case: all applicable but very low confidence)**

```python
def test_low_confidence_evidence_produces_no_trade():
    from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
    from intelligence.fast_tier import FastTierReasoner, ToolTrust
    registry = EvidenceRegistry()
    registry.register(EvidenceSourceSpec(
        name="weak_source", mathematical_formulation="stub", required_inputs=[],
        assumptions="stub", known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(1.0, 0.02, "weak_source"),  # confidence near zero
        is_directional=True, computational_cost_hint="stub",
    ))
    reasoner = FastTierReasoner(registry)
    hyp = reasoner.hypothesis(np.linspace(2000, 2001, 50), market_state=None, trust=ToolTrust())
    assert abs(hyp.net_directional_belief) < 0.05 or hyp.aggregate_uncertainty > 0.5
```

- [ ] **Step 2: Direction neutrality regression guard** — reference the Phase 2 fix-wave test by name (do not duplicate its exact assertion) and add ONE new adversarial case: a market with a real (non-synthetic-driftless) upward trend, verifying LONG bias tracks the trend's sign rather than existing unconditionally:

```python
def test_direction_neutrality_tracks_real_trend_not_a_permanent_bias():
    """Complements test_directional_belief_unbiased_on_symmetric_data (Phase 2
    fix wave, tests/intelligence/test_fast_tier.py) which proves no bias on
    DRIFTLESS data. This proves the belief correctly FLIPS sign when the
    underlying trend flips -- a permanently-biased system could coincidentally
    pass the driftless test while still being unable to track a real reversal."""
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner, ToolTrust
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry)
    trust = ToolTrust()
    up = np.linspace(2000.0, 2100.0, 400)
    down = np.linspace(2100.0, 2000.0, 400)
    hyp_up = reasoner.hypothesis(up, market_state=None, trust=trust)
    hyp_down = reasoner.hypothesis(down, market_state=None, trust=trust)
    assert hyp_up.net_directional_belief > 0
    assert hyp_down.net_directional_belief < 0
```

- [ ] **Step 3: Credit assignment — cross-reference, no new test needed**

Add a one-line comment in this file (no test code) pointing to `tests/intelligence/test_credit_assignment.py::test_dissenting_load_bearing_source_gets_opposite_credit` and `test_short_trade_credit_is_relative_to_the_short_direction` (Phase 2 fix wave) as already satisfying this mandate item. Confirm both still pass:

Run: `.venv/bin/pytest tests/intelligence/test_credit_assignment.py -v`
Expected: PASS (no changes needed — this is a verification step, not new code).

- [ ] **Step 4: Thesis invalidation — new adversarial case: invalidation on a sign flip mid-hold**

```python
def test_thesis_invalidates_on_belief_sign_flip_during_hold():
    from intelligence.decision_engine import FastTierDecisionEngine
    # Implementer: construct via the real pattern in
    # tests/intelligence/test_full_fast_tier_integration.py. Enter a LONG
    # position with a synthetic uptrend, then feed a sharp reversal and
    # confirm manage() returns "EXIT" (POLICY_EXIT) even though price has
    # not yet hit SL or TP.
    ...
```

(Implementer: this mirrors `test_full_fast_tier_integration.py`'s existing `POLICY_EXIT` assertion from the Phase 2 fix wave — read that test first, then write a MORE adversarial variant: a reversal sharp enough to flip `net_directional_belief`'s sign but NOT sharp enough to hit SL, isolating thesis-invalidation-driven exit from stop-loss-driven exit, which the existing test may not cleanly isolate.)

- [ ] **Step 5: Continuous reassessment loop — new case: multiple HOLD cycles before EXIT, not just one**

```python
def test_manage_holds_repeatedly_across_multiple_bars_before_exit():
    # Assert manage() is called and returns "HOLD" for several consecutive
    # bars while the thesis remains valid, THEN returns "EXIT" once
    # invalidated -- proving no fixed holding horizon (mandate requirement:
    # "works repeatedly without requiring a fixed holding horizon").
    ...
```

- [ ] **Step 6: Causality — cross-reference, no new test needed**

Add a one-line comment pointing to `tests/intelligence/test_causal_memory_boundaries.py::test_credit_assignment_truncated_vs_full_stream_identical_for_common_prefix` (Task 14 fix) as already satisfying this. Confirm it still passes:

Run: `.venv/bin/pytest tests/intelligence/test_causal_memory_boundaries.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full new test files, verify all pass**

Run: `.venv/bin/pytest tests/intelligence/test_adversarial_abstention_and_neutrality.py tests/intelligence/test_adversarial_credit_and_reassessment.py -v`
Expected: all PASS. Any failure here is a real defect — investigate via `superpowers:systematic-debugging`, do not weaken assertions.

- [ ] **Step 8: Commit**

```bash
git add tests/intelligence/test_adversarial_abstention_and_neutrality.py tests/intelligence/test_adversarial_credit_and_reassessment.py
git commit -m "test: adversarial coverage for abstention, direction neutrality, thesis invalidation, reassessment"
```

---

### Task 9: Simulator-scale chronological run + health-metrics report

**Files:**
- Create: `scripts/run_fast_tier_health_check.py`
- Create: `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-simulator-run-report.md`

**Interfaces:**
- Consumes: `simulator.replay.run_replay` (or whatever the real Phase 1 entry point is — read `tests/intelligence/test_full_fast_tier_integration.py` for the exact call signature already in use), `FastTierDecisionEngine`, `ExperienceStore`.

Not the final six-year training run — mandate is explicit ("This is NOT the final six-year training run"). A moderate-scale run (e.g. one to a few years of synthetic chronological data, or whatever realistic-scale fixture the Phase 2 integration test already uses, extended in length) verifying `MarketState -> Fast Tier -> Action -> Execution -> Experience` works correctly at scale, with health metrics recorded, not profitability.

- [ ] **Step 1: Write the script** — reuse the exact composition pattern from `tests/intelligence/test_full_fast_tier_integration.py` (real registry, real trust, real reasoner, real decision engine, real `run_replay`, real `ExperienceStore`), but run it over a longer synthetic chronological series and record: total decisions made, NO_TRADE rate, LONG/SHORT split, average trade duration in bars, number of `POLICY_EXIT`s vs. SL/TP/liquidation exits, per-source applicability-gate rate (how often each source was gated out), context bucket distribution over the run (should now be spread, per Task 4), any exceptions/crashes (must be zero), wall-clock total runtime.

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/run_fast_tier_health_check.py`
Must complete without exception. Record all Step 1 metrics in the report.

- [ ] **Step 3: Write the report** — explicitly state this is a system-health/integration check, NOT a profitability or backtest result; no P&L claim, no Sharpe ratio, no win-rate-as-success-metric framing. Frame every number as "did the pipeline behave correctly" (bounded NO_TRADE rate, no crashes, spread context buckets, reasonable trade durations) not "did it make money."

- [ ] **Step 4: Commit**

```bash
git add scripts/run_fast_tier_health_check.py docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-simulator-run-report.md
git commit -m "test: moderate-scale chronological simulator health check for hardened Fast Tier"
```

---

### Task 10: MT5 interface compatibility verification (no live orders)

**Files:**
- Create: `tests/intelligence/test_mt5_interface_compatibility.py`
- Create: `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-mt5-compatibility-report.md`

**Interfaces:**
- Consumes: whatever the real MT5-normalized `MarketState` construction path is — search the repo first (`grep -rl "MT5\|mt5" --include=*.py .` starting from `contracts/` and `market/`) for the actual normalization function/class; read it before writing this task's code. Do NOT invent an MT5 adapter that doesn't exist — if none exists yet, this task's deliverable is a report stating that explicitly (a real, honest limitation) rather than fabricated compatibility code.

**This task's scope is READ-ONLY VERIFICATION.** No order placement, no live connection, no new MT5 integration code beyond what's needed to construct a `MarketState` instance shaped like real MT5 output would produce.

- [ ] **Step 1: Locate the existing MT5 `MarketState` construction path** (investigation step, not code) — read `contracts/market_state.py` for the `MarketState` dataclass fields, then search for any existing MT5 normalization code from Phase 1.

- [ ] **Step 2: Write a test constructing a `MarketState` the way MT5-sourced data would produce it** (same field values/types/ranges a live MT5 tick-to-bar normalizer would emit — timezone-aware timestamps, real bid/ask spread fields, `data_quality`, etc. — matching whatever Phase 1 already validated for this), and feed it through `FastTierDecisionEngine.decide()`/`.manage()` unchanged:

```python
def test_fast_tier_consumes_mt5_shaped_market_state_without_modification():
    # Construct MarketState exactly as Phase 1's MT5 path would (see
    # contracts/market_state.py and Phase 1's existing MT5 normalization
    # code, if any -- cite the file read in Step 1 here).
    ...
    action, sl, tp, size = engine.decide(mt5_shaped_state)
    assert action in ("NO_TRADE", "LONG", "SHORT")
```

- [ ] **Step 3: Run, verify pass**

Run: `.venv/bin/pytest tests/intelligence/test_mt5_interface_compatibility.py -v`

- [ ] **Step 4: Write the compatibility report** — state plainly whether the same `FastTierDecisionEngine` interface handles MT5-shaped `MarketState` with zero Fast-Tier-side branching (expected: yes, since `decide`/`manage` only ever consumed the `MarketState` contract, never a data-source-specific type) — or, if Step 1 found no existing MT5 normalization path in this repo yet, state that plainly as the actual finding instead of fabricating one.

- [ ] **Step 5: Commit**

```bash
git add tests/intelligence/test_mt5_interface_compatibility.py docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-mt5-compatibility-report.md
git commit -m "test: verify Fast Tier consumes MT5-shaped MarketState without architectural modification"
```

---

### Task 11: Whole-branch review + final hardening report

**Files:**
- Create: `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-final-report.md`

Standard SDD final step: one whole-branch review (most capable model) over this plan's own commit range, one fix wave, one scoped re-review, no second fix wave (park + rule on anything residual). The final report must consolidate: latency before/after (Task 5), calibration root cause + fix + before/after bucket distribution (Task 4), adversarial test results (Tasks 6-8) — pass/fail per each of the 7 mandate-listed properties by name, simulator-scale health-check results (Task 9), MT5 compatibility finding (Task 10), remaining limitations (be exhaustive and honest — do not let this pass declare victory if e.g. the ~2ms/bar budget is still not met), and an explicit recommendation for the next phase per the mandate's Section 7/8 (what, if anything, should happen next — and an explicit statement that no further work should proceed without new authorization, matching "Then wait for authorization").

- [ ] **Step 1**: Follow `superpowers:subagent-driven-development`'s whole-branch review process exactly as used for the Phase 2 plan (dispatch, review scoped to this plan's commit range, one fix wave, one re-review).
- [ ] **Step 2**: Write and commit the final report per the content list above.
- [ ] **Step 3**: Delete this plan's SDD workspace directory once review is clean per the skill's Finish instructions.
- [ ] **Step 4**: Use `superpowers:finishing-a-development-branch`.

---
