# GOLDEX Phase 2 Fast Tier Hardening — Latency Report

Date: 2026-09-02
Branch: `goldex-genesis-event-time-test`
Scope: Tasks 1-4 of the Phase 2 Fast Tier hardening plan (Kalman dedup,
extended refit-caching, bounded `closes_so_far` window, recalibrated
`context_bucket` constants) — this report measures their combined effect
on latency and states plainly whether the Phase 1 ~2ms/bar budget is met.

Machine: same host used for the original Phase 2 report and this task's
own measurements (no cross-machine comparison — before/after numbers below
come from the same perf-test harness, `tests/intelligence/test_fast_tier_performance.py`,
run before and after this branch's changes).

## 1. Before vs. after

"Before" is the pre-hardening baseline reported in the Phase 2 final
report (`docs/superpowers/reports/2026-08-29-goldex-phase2-fast-tier-report.md`).
"After" is this task's measurement, taken over three consecutive runs of
the perf suite post Task 1-4 (base commit `bc230c5` plus this branch's
changes); the range shown is min-max p99 across those runs.

| Metric | Before (pre-hardening) | After (post Task 1-4) | Change |
|---|---|---|---|
| `EvidenceRegistry.compute_all()` p99 | ~440-475ms | ~402-447ms | modest improvement, same order of magnitude |
| `FastTierReasoner.hypothesis()` — cached call p99 | not separately reported (rolled into overall hypothesis cost) | ~17-18ms | — |
| `FastTierReasoner.hypothesis()` — refit-triggering call p99 | not separately reported | ~125-141ms | — |
| `FastTierDecisionEngine.decide()` p99 | ~40-60ms | ~36-40ms | modest improvement, same order of magnitude |

Mean figures from the after-measurement (representative run):
`compute_all` mean ≈ 302-329ms, cached `hypothesis()` mean ≈ 11-13ms,
refit `hypothesis()` mean ≈ 77-87ms, `decide()` mean ≈ 7.3ms.

**Read honestly:** `compute_all()` and `decide()` p99 did **not** drop by
an order of magnitude. They moved modestly within the same order of
magnitude as the pre-hardening numbers. The one large, clearly-attributable
win is on the *cached* `hypothesis()` path specifically (Task 2's four
newly-cached sources), which is now materially cheaper on the ~49-out-of-50
calls between refits — but that saving is largely invisible in `decide()`'s
p99 because `decide()`'s p99 is dominated by whichever call in its
250-call run happened to be a refit-triggering call (same GARCH/Kalman
cost as before, since Task 1-4 did not speed up GARCH's MLE itself, only
deduplicated Kalman and bounded the window it runs over).

## 2. Per-source cost breakdown (from the Task 1-4 investigation)

- **GARCH (`garch_conditional_variance`)**: ~220ms per fresh fit, ~54% of
  `compute_all()`'s total cost — the single dominant cost in the whole
  Fast Tier evidence stack. This is the Python-loop MLE fit; Task 1-4 did
  not touch its algorithm, only when/how often it runs (refit-caching,
  already in place before this branch) and how much history it sees
  (Task 3's bounded window, new in this branch).
- **Kalman dedup (Task 1)**: the velocity and innovation evidence sources
  were independently running a full Kalman filter pass each, i.e. the
  filter ran twice per evidence pass for no semantic reason. Task 1 shares
  one filter run between both sources, saving one redundant Kalman pass
  per refit. This is a real but secondary saving relative to GARCH — it
  only fires on refit-triggering calls, not every call.
- **4 newly-cached non-directional sources (Task 2)**: `multiscale_vol_ratio`
  (~33ms), `vol_regime_transition` (~36ms), `rolling_skew` (~21ms),
  `rolling_excess_kurtosis` (~21ms) — roughly 110ms/pass combined — were
  previously recomputed on *every* call despite being non-directional
  (excluded from votes/credit) and, for two of the four, not even feeding
  `context_bucket()`. Task 2 extended the existing `EXPENSIVE_SOURCE_NAMES`
  / `refit_interval` caching mechanism (already accepted for GARCH/Kalman)
  to these four. This is the change most responsible for cached
  `hypothesis()` being cheap (~17ms p99) relative to a refit-triggering
  call (~130ms p99) — roughly a 7x cached/refit ratio measured in this
  task's runs.
- **Bounded `closes_so_far` window (Task 3)**: caps history fed to sources
  at `max_history_window=2000` bars (see
  `intelligence/fast_tier.py:228`). Does not change per-call latency on
  short synthetic runs (this perf suite uses 1,000-1,350 closes, under
  the cap), but bounds *worst-case* cost independent of replay length —
  see Section 3.

## 3. What is bounded vs. what remains unbounded

**Now bounded (Task 3):** worst-case per-refit cost of GARCH/Kalman is
capped, because each refit fits over at most the most recent 2,000 bars
instead of the entire replay history seen so far. Before Task 3, a
multi-year M1 replay would make every refit progressively more expensive
as `closes_so_far` grew without bound; after Task 3, refit cost plateaus
once the replay passes 2,000 bars. This is a disclosed, non-numerically-identical
behavior change (documented in `intelligence/fast_tier.py`'s module
docstring per the plan) — GARCH/Kalman are now fit over "the most recent
2,000 bars" rather than "all bars ever seen."

**Still architecturally unbounded / unaddressed by this hardening pass:**
- GARCH's per-fit cost itself (~220ms at full window, still the majority
  of `compute_all()`'s cost even after Task 3's window cap) is not
  algorithmically improved — it is a Python-loop MLE. Bounding the window
  bounds the *ceiling* GARCH can grow to over a long replay, but does not
  lower the *floor* cost of any single refit at any window size up to
  2,000 bars. A vectorized/compiled GARCH implementation (numpy-vectorized
  likelihood, Numba, or a C/Rust extension) was out of scope for this
  hardening pass and remains the largest lever if the ~2ms/bar budget is
  ever actually required.
- `decide()`'s worst-case latency (a refit-triggering call) is therefore
  also still dominated by GARCH and is not meaningfully reduced by this
  branch — Task 1-4 make the *common case* (cached calls, ~49/50) cheap,
  not the *worst case* (refit calls, ~1/50).

## 4. Does this meet the Phase 1 ~2ms/bar budget?

**No.** Plainly: it does not, and it was not expected to.

- `decide()` p99 measured ~36-40ms after Task 1-4, against a Phase 1
  `build_snapshot()` p99 reference of ~2ms — roughly **18-20x** over
  budget, essentially unchanged in order of magnitude from the
  pre-hardening ~40-60ms (which was ~20-30x over budget).
- The mean case is much closer to budget (`decide()` mean ≈ 7.3ms, ~3.7x
  the 2ms reference) thanks to Task 2's caching making 49-out-of-50 calls
  cheap, but the p99 — which is what the budget statement is about, since
  a live replay pays the refit cost on schedule every `refit_interval`
  bars, not probabilistically — is still far over.
- The root cause is unchanged: GARCH's Python-loop MLE, at any window
  size the refit-caching mechanism is willing to skip, costs tens of
  milliseconds on its own, which alone exceeds the entire 2ms/bar budget
  by more than an order of magnitude. Bounding the window (Task 3) caps
  how bad this gets over a long replay; it does not make any single
  refit fast enough to fit the budget.
- Task 1-4 were latency and calibration *hardening*, not a redesign of
  the Fast Tier's cost profile. They deliver real, measurable
  improvements (see Section 1-2) but do not and were never claimed to
  close an ~18-20x gap. Closing that gap would require either replacing
  GARCH's fit algorithm, moving it off the hot path (e.g. async/background
  refit with a staleness tolerance beyond what refit-caching already
  provides), or revisiting whether GARCH belongs in the Fast Tier's
  every-refit path at all.

## 5. Test changes

`tests/intelligence/test_fast_tier_performance.py` bounds were tightened
from the pre-hardening measurements (compute_all p99 ~460ms → bound
1,000,000us; cached hypothesis p99 ~56ms → bound 150,000us; refit
hypothesis p99 ~134ms → bound 350,000us; decide p99 ~40ms → bound
120,000us — all ~2.5-3x margins) to the post-Task-1-4 measurements taken
in this task (compute_all p99 ~402-447ms → bound 750,000us, ~1.7-1.9x;
cached hypothesis p99 ~17-18ms → bound 30,000us, ~1.7x; refit hypothesis
p99 ~125-141ms → bound 250,000us, ~1.8-2.0x; decide p99 ~36-40ms → bound
65,000us, ~1.6-1.8x). All three tests pass with real margin post-tightening
(confirmed by a third consecutive run: compute_all 446.8ms/750ms bound,
cached hypothesis 17.7ms/30ms bound, refit hypothesis 126.3ms/250ms
bound, decide 39.5ms/65ms bound).
