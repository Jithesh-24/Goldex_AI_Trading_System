# GOLDEX Phase 2 — Fast-Tier Hardening: Final Report

**Date:** 2026-09-02
**Branch:** `goldex-genesis-event-time-test`
**Commit range:** `0c78322`..`6faa6b1` (15 commits, including the whole-branch review fix round)
**Mandate:** "GOLDEX — PHASE 2 FAST-TIER HARDENING" (this session's authorizing directive)

This is the consolidated deliverable required by the mandate's Section 8. It reports
what was found, what was fixed, what was left as an honestly-disclosed limitation, and
an explicit recommendation for what should happen next.

---

## Issue A: Latency

### Findings

Root causes of the original latency gap (Task 5 measurement, before optimization):
Kalman velocity and innovation sources ran two independent filter passes over
identical data; only 3 of 9 evidence sources had refit-caching; `closes_so_far` grew
unbounded across a replay, so every GARCH/Kalman fit re-processed the entire history;
`context_bucket`'s constants were hand-guessed, not measured.

### Fixes applied

- **Task 1:** Kalman velocity/innovation now share one filter run via a request-scoped
  cache (output verified byte-identical to before).
- **Task 2:** Refit-caching extended from 3 to all 7 `EXPENSIVE_SOURCE_NAMES`.
- **Task 3:** `closes_so_far` bounded to `max_history_window` (default 2000 bars) —
  disclosed as a genuine semantic change (GARCH/Kalman now fit "last N bars," not full
  history; measured rel_diff vs. full-history = 0.083742).
- **Task 11 fix round:** the refit-cache's fingerprint (added in Task 8 to fix a
  stale-cache leak) was itself found, by this hardening pass's own simulator
  health-check, to silently defeat the refit cache for any replay longer than
  `max_history_window` — the fingerprint was drawn from the truncated window's first
  element, which slides every bar once truncation is active. Fixed by drawing the
  fingerprint from the original, pre-truncation array's first element instead (stable
  for one continuing replay's whole life, still distinguishes different series).

### Before / after

| Call | Before hardening | After Tasks 1-4 (Task 5 report) | After Task 11 fix round |
|---|---|---|---|
| Cached `hypothesis()` | not separately measured | p99≈17-18ms | mean=12.8ms, p99=19.0ms |
| Refit-triggering `hypothesis()` | not separately measured | p99≈125-141ms | mean=89.5ms, p99=148.1ms |
| Uncached `compute_all()` | not separately measured | p99≈402-447ms | unchanged (this is the real 9-source cost floor) |
| `decide()` | ~40-60ms | p99≈36-40ms | not re-measured post-Task-11 (decide() itself untouched) |

### Was the ~2ms/bar budget met?

**No — explicitly not met**, and this is stated plainly rather than glossed over. Even
the best case (fully cached `hypothesis()`, ~13-19ms) is roughly 6.5-9.5x the ~2ms/bar
target; the common refit-triggering case is 45-75x over. The architecture's GARCH/Kalman
refit cost is fundamentally the bottleneck, not caching overhead or Python/serialization
overhead — caching (Tasks 1, 2, 8, and the Task 11 fix) closes the gap between
"how often you pay the expensive cost" and "how expensive that cost is," but does not
change the cost itself. **Minimum architectural change required to hit ~2ms/bar:**
either a fundamentally cheaper (non-GARCH) volatility/velocity estimator, or moving the
refit off the hot decision path entirely (e.g. an async/background refit thread feeding
a bounded-staleness cache) — both are architecture changes, correctly out of this
hardening pass's authorized scope (no redesign).

Full detail: `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-latency-report.md`.

---

## Issue B: Context-bucket calibration

### Root cause

The original `MAGNITUDE_LOG_CENTER`/`MAGNITUDE_LOG_SCALE` constants were hand-guessed
(`log(0.15)`, `1.0`), not measured. When first recalibrated against real evidence (Task
4), the calibration script's own synthetic data generator (a single constant-sigma
random walk) caused GARCH conditional variance to converge to a near-flat steady state,
starving the p10/p90-derived scale of genuine spread (p50≈p75≈p90 in the raw sample) —
an implementation defect in the *calibration methodology*, not evidence of insufficient
underlying signal.

### Fix

Rewrote the calibration script's (and, separately, the reachability test's) synthetic
data generator to alternate quiet/normal/turbulent noise-scale sub-windows instead of
one constant-sigma walk, restoring genuine variance spread. Final calibrated constants,
from n=1160 real samples: `MAGNITUDE_LOG_CENTER = math.log(0.22554)`,
`MAGNITUDE_LOG_SCALE = 1.19983` (median=0.22554, p10=0.00617, p90=0.22554).

**Not** fit against the same evidence used to evaluate the system — the calibration
script's data and the reachability test's held-out data use deliberately different
segment lengths, sigma cycles, and seeds, for genuine train/held-out separation.

### Before / after

- Before: 2 tests failing (bucket 3 unreachable; 84% of decisions concentrated in one
  bucket vs. required <70%).
- After: 19/19 `test_fast_tier.py` tests passing; Task 9's simulator health-check
  independently confirms 3 distinct context buckets occupied at moderate scale (not
  collapsed to one).

---

## Adversarial testing — all 7 mandate-required properties

| Property | Status | Evidence |
|---|---|---|
| Conditional trust | PASS | `test_adversarial_conditional_trust.py` — posterior_mean asymmetry AND the real `hypothesis()` weighting formula both verified to upweight the trustworthy source per context bucket |
| Contradiction | PASS | `test_adversarial_contradiction.py` — contradictory evidence produces high uncertainty, not averaged confidence; real `FastTierDecisionEngine` abstains (`NO_TRADE`) under contradiction |
| Abstention | PASS (after fix) | `test_adversarial_abstention_and_neutrality.py` — a near-zero-confidence lone source originally still produced full-magnitude belief (Critical bug, found by this test); fixed by folding `(1 - confidence)` into per-source uncertainty |
| Direction neutrality | PASS (after fix) | same file — a perfectly unambiguous downtrend originally produced a false 2-vs-2 vote tie and `aggregate_uncertainty=1.0` (Critical bug: a reasoner-instance stale-cache leak, not a Kalman sign error as first suspected — root-caused and fixed independently by the controller) |
| Credit assignment | PASS | `test_credit_assignment.py` (12/12, pre-existing, cross-referenced) — dissenting sources get opposite credit, short-trade credit is direction-relative |
| Thesis tracking | PASS | `test_adversarial_credit_and_reassessment.py` — belief sign-flip mid-hold correctly triggers `POLICY_EXIT`, isolated from SL/TP-driven exits |
| Reassessment | PASS | same file — `manage()` holds repeatedly across multiple bars before exiting, no fixed holding horizon |
| Causality | PASS | `test_causal_memory_boundaries.py` (6/6, pre-existing, cross-referenced) — truncated vs. full stream produce identical credit assignment for the common prefix (no future leakage) |

Two genuine Critical production bugs were found and fixed by this adversarial testing
(abstention gap, direction-neutrality stale-cache leak) — exactly the outcome this
section of the mandate exists to produce. Full `tests/intelligence/` suite: 148/148
passing as of the final commit (1 known-unrelated perf-benchmark flake investigated and
confirmed unrelated — see Task 11 fix-round verification below).

---

## Simulator integration (Task 9)

Ran the hardened Fast Tier through Phase 1's real, unmodified `run_replay` at 4,000 bars
(reduced from a planned 20,000 after a 50+-minute abort — disclosed, not concealed; 4,000
bars is still ~13x the integration test's fixture). Zero exceptions across 2,249
decisions. NO_TRADE rate 91.6% (bounded, not degenerate). All four exit paths
(TP/SL/POLICY_EXIT/forced-close) fired. 3 distinct context buckets occupied. Per-source
gate rates all in a narrow 3.4%-7.0% band (no dead or rubber-stamp sources).

This run's own timing (~390ms/reasoner-call average) is what surfaced the
cache-fingerprint-defeat bug described above under Latency — found, root-caused, and
fixed within this hardening pass.

Full detail: `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-simulator-run-report.md`.

---

## MT5 real-time compatibility (Task 10)

A real, active Phase 1 MT5 normalization pipeline already exists
(`market/mt5_feed.py` → `market/feed_listener.py` → `market/state_engine.py`'s
`StateEngine.on_tick()` → `MarketState(source="mt5_live", ...)`) — not fabricated.
Confirmed by test and by grep that `FastTierDecisionEngine.decide()`/`.manage()` consume
MT5-normalized `MarketState` with **zero source-specific branching**
(`MarketState.source` is never referenced anywhere in `intelligence/decision_engine.py`
or `intelligence/fast_tier.py`). No live orders were placed; no new MT5 integration code
was written beyond the read-only verification test.

Full detail: `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-mt5-compatibility-report.md`.

---

## Whole-branch review outcome

One whole-branch review (sonnet, per this project's established review cadence) over
the full plan's commit range (`0c78322`..`af6077c`, 13 commits). Result: 0 Critical
findings, 1 Important finding (the cache-fingerprint defeat, described above — reviewer
independently re-derived and confirmed the root cause before recommending a fix), 1
minor finding (`load_bearing_floor=0.05` makes `load_bearing_sources` "effectively every
applicable source, not a meaningfully discriminating subset" — this is already honestly
disclosed in the constructor's own docstring; not fixed, since real calibration of that
threshold is out of this pass's scope, but flagged here prominently per the reviewer's
request). No scope creep found: diff touches only `intelligence/`, `scripts/`, `tests/`,
`docs/`, matching the plan's authorized file list exactly. One fix round applied for the
Important finding; scoped re-review confirmed it addressed with no new breakage (one
unrelated pre-existing perf-benchmark flake, independently verified by the controller to
call `EvidenceRegistry.compute_all()` directly, bypassing `FastTierReasoner` entirely —
not caused by the fix).

---

## Remaining limitations (exhaustive, honest)

1. **The ~2ms/bar Phase 1 latency budget is still not met.** Best case ~13-19ms
   (6.5-9.5x over), common case ~90-150ms (45-75x over), worst case ~400-450ms
   (200-225x over). This is a real architectural ceiling from GARCH/Kalman refit cost,
   not a caching/overhead problem — closing it further requires either a cheaper
   estimator or moving refits off the hot path (architecture changes, out of scope here).
2. **`load_bearing_floor=0.05` does not meaningfully discriminate** which sources count
   as "load-bearing" for credit assignment and thesis tracking — nearly every applicable
   source clears it. Credit-assignment and thesis-tracking correctness were adversarially
   verified to work (see table above), but the "which sources actually mattered" signal
   this threshold is meant to produce is weak. Needs real calibration in a future phase.
3. **Task 3's `max_history_window` bounding is a genuine semantic change**, not just a
   performance optimization: GARCH/Kalman now fit the last 2000 bars only, not full
   history (measured rel_diff = 0.083742 vs. full-history). This is disclosed, not a bug,
   but changes what "the model has learned" means for any single reasoner instance
   across a very long replay.
4. **Task 9's simulator run was at 4,000 bars, not the mandate's full six-year scale.**
   The pipeline composes correctly at this moderate scale; genuine six-year-scale
   behavior (memory growth, long-run drift, rare-event handling) remains unverified.
5. **The context-bucket calibration is derived from synthetic regime-varying data**, not
   real historical market data. It is honestly separated from its own held-out test set,
   but "genuinely resembles real market regime transitions" was not independently
   verified against real historical price series in this pass.

---

## Explicit recommendation for next phase

Per the mandate's Section 7/8, this hardening pass is complete: all 16 acceptance
criteria are addressed (latency bottlenecks identified and materially improved where
architecturally possible, with the remaining gap explicitly quantified rather than
hidden; calibration root-caused and fixed; all 7 adversarial properties tested, with 2
genuine Critical bugs found and fixed; causality intact; Phase 1 unchanged except the
disclosed, justified MT5-compatibility read-only test; no V3/V4 return; no scope creep;
full test suite passing; whole-branch review passed with its one fix round closed).

**Recommendation:** before any large-scale historical experience or Slow Tier work,
address limitation #1 (the latency ceiling) and #2 (load-bearing-floor calibration) as
dedicated follow-up work, since both are load-bearing for what a Slow Tier would build
on top of. Limitation #4 (six-year-scale verification) should happen once, deliberately,
as its own gated step — not folded into further hardening.

**This hardening pass does not authorize any further work.** Per the mandate's Section
8: STOP here. Do not begin Slow Tier construction, do not begin live MT5 trading, do not
start another research cycle. Wait for explicit authorization before proceeding.
