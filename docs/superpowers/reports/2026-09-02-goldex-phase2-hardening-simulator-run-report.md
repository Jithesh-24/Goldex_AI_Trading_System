# GOLDEX Phase 2 Hardening — Moderate-Scale Simulator Health Check

**Date:** 2026-09-02
**Script:** `scripts/run_fast_tier_health_check.py`
**Raw metrics:** `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-simulator-run-metrics.json`

## What this is, and what it is not

This is a **system-health / integration verification** run, not a backtest and not a
profitability result. It exercises the exact same real composition used by
`tests/intelligence/test_full_fast_tier_integration.py` — the real
`build_default_registry()`, real `ToolTrust`, real `FastTierReasoner`, the real EV/cost
gate wrapping `simulator.cost_model.round_trip_cost_r`, real
`analytical_sizing_bootstrap`/`analytical_sltp_bootstrap`, real `FastTierDecisionEngine`,
Phase 1's real unmodified `simulator.replay.run_replay`, and the real `ExperienceStore`
read path — over a longer, deterministic synthetic chronological series than that test's
own 300/400-bar fixtures.

**This report contains no P&L figure, no Sharpe ratio, and no win-rate-as-success
framing.** Every number below answers a single question: *did
`MarketState -> Fast Tier -> Action -> Execution -> Experience` compose correctly at
scale, with zero crashes and a sane distributional shape?* Whether the synthetic series'
trades were profitable is irrelevant to that question and is not reported.

## Run parameters

- Dataset: deterministic synthetic OHLC series (`_make_df`, seed `20260902`), same
  construction as the integration test's fixture (trend + oscillation + Gaussian noise,
  `NOISE_STD=0.35` — the noise level shown in that test to be necessary for
  `realized_vol_60s` to clear the EV/cost gate's `spread/(SL_VOL_MULTIPLIER * mid)`
  threshold at all).
- Scale: **4,000 bars** (~13x the integration test's 300-bar fixture).
  - An initial attempt at 20,000 bars (~65x) was aborted after running for more than
    50 minutes of wall-clock without completing — per-bar cost (GARCH/Kalman refits,
    applicability gating, thesis bookkeeping) does not amortize as cheaply as hoped even
    with Task 3's refit-caching, so 20,000 bars is not a practical size for a single
    synchronous run. 4,000 bars was chosen as a scale that both (a) is a real order of
    magnitude larger than the integration test's fixture and (b) actually completes.
    This is disclosed here as a deviation from the brief's "tens of thousands of bars"
    upper end, not concealed.
- Environment tag: `SIMULATED_TRAINING`.

## Results

| Metric | Value | Read as |
|---|---|---|
| Wall-clock runtime | 1593.2 s (~26.5 min) | Completed synchronously; no hang, no timeout inside the run itself. |
| Exceptions raised | **0** | Zero crashes across 4,000 bars — the required pass condition. |
| Total DECIDE-event records | 2,249 | The reasoner was invoked and returned a well-formed action on every decision bar. |
| Action split | NO_TRADE 2,059 / LONG 182 / SHORT 8 | NO_TRADE rate 91.6% — bounded, not 100% (system does propose trades) and not near-0% (EV/cost gate is not rubber-stamping everything). |
| Rejected entries (LONG/SHORT with a rejection reason) | 0 | No entries reached the engine with an unresolved rejection reason recorded. |
| Closed positions | 190 | Consistent with 182+8=190 opened positions — every opened position was accounted for at close. |
| Exit-reason breakdown | TP_HIT 65, SL_HIT 56, POLICY_EXIT 68, END_OF_REPLAY_FORCED_CLOSE 1 | All four of Phase 1's defined exit paths fired at least once; POLICY_EXIT (68) is on the same order as SL/TP exits (121 combined), so the reasoner's own exit logic is meaningfully load-bearing, not vestigial. |
| Avg trade duration | 9.57 bars | Non-degenerate — not 1 bar (no evidence of instant-flip pathology) and not pinned at the run length (no evidence of "never exits" pathology). |
| Min / max trade duration | 1 / 41 bars | Wide spread; the one 1-bar trade did not skew the average. |
| Per-source applicability-gate rate | 3.4%–7.0% across all 9 sources (see table below) | Every source is gated out only a small minority of the time — none is either always-gated (dead weight) or never-gated (gate not actually discriminating). |
| Context buckets observed | {-1: 170, 2: 3598, 3: 110} per-call; {-1: 9, 2: 708, 3: 32} load-bearing | Multiple distinct buckets occupied in both views — the run does not collapse into a single bucket, consistent with Task 4's recalibration goal. |
| `ended_flat` | `False` (one position force-closed at end of replay) | Expected — `run_replay` force-closes any open position at series end; this is the `END_OF_REPLAY_FORCED_CLOSE=1` row, not a leak or unhandled state. |

### Per-source applicability-gate rate

| Source | Gated-out rate |
|---|---|
| momentum_scalar | 3.35% |
| path_pca_projection | 3.48% |
| multiscale_vol_ratio | 5.67% |
| vol_regime_transition | 6.96% |
| garch_conditional_variance | 4.38% |
| kalman_filtered_velocity | 4.38% |
| kalman_innovation | 4.38% |
| rolling_skew | 4.38% |
| rolling_excess_kurtosis | 4.38% |

## Interpretation (health only)

- **Zero exceptions** across 4,000 bars and 2,249 decision calls is the required pass
  condition for this task, and it held.
- **NO_TRADE rate (91.6%)** is bounded away from both 0% and 100%, indicating the EV/cost
  gate and applicability gating are both active and discriminating rather than either
  rubber-stamping every bar or blocking everything.
- **All four exit paths fired** (TP, SL, POLICY_EXIT, forced end-of-replay close), with
  POLICY_EXIT representing roughly a third of all closes — confirming the reasoner's own
  exit decisioning is exercised at scale, not just the mechanical SL/TP paths.
- **Context buckets are spread** (3 distinct buckets occupied, in both the per-call and
  load-bearing views), which is the direct thing Task 4's bucket-constant recalibration
  was meant to produce — a run that pinned to a single bucket would indicate the
  recalibration didn't take.
- **Per-source gate rates are all in a narrow, non-degenerate band (3.4%–7.0%)** — no
  source is a permanent dead weight (100% gated) or a rubber stamp (0% gated).

None of the above is a claim about whether the strategy would make money; it is a claim
about whether the pipeline's components compose correctly and produce a sane
distributional shape under moderate-scale chronological replay.

## Timing reconciliation against Task 5's latency report

This run's wall-clock (1593.2s / ~4,068 reasoner-touching calls ≈ 390ms/call average)
looked, at first glance, inconsistent with Task 5's measured `hypothesis()` latencies
(cached p99≈17-18ms, refit-triggering p99≈125-141ms, uncached `compute_all()`
p99≈402-447ms) — a chronologically-advancing replay should mostly hit the refit cache,
not average close to the fully-uncached worst case.

Root cause, confirmed by reading `FastTierReasoner._compute_evidence`
(`intelligence/fast_tier.py`, cache logic around lines 251-290) and by direct timing of
that method before/after the boundary described below: the cache key's fingerprint is
`closes_so_far[0]` **after** `closes_so_far` has already been truncated to the last
`max_history_window` (default 2000) elements (line 271-274). In a chronologically
growing replay, once `len(closes_so_far) > max_history_window`, that truncated window
slides forward by exactly one bar on every single call — so `closes_so_far[0]`, and
therefore the fingerprint, is different on every bar from that point on. The refit
cache's `cached[1] == fingerprint` check (line 282) then never hits again for the rest
of the replay: every `hypothesis()` call past bar 2,000 does a full, uncached
GARCH/Kalman refit of all 7 `EXPENSIVE_SOURCE_NAMES`, at roughly
`compute_all()`'s uncached cost, not the cached/refit-cadence cost the cache exists to
deliver.

This run used 4,000 bars: the first ~2,000 bars got real refit-interval caching
(cheap), the second ~2,000 got an uncached refit on every call (expensive,
~400-450ms range). The blended average (~390ms/call) is consistent with a run that is
roughly half at uncached-worst-case cost — not a sign of double-counted or
uninstrumented work in `_InstrumentedReasoner` itself (confirmed separately: its extra
`apply_applicability` pass reuses the same bar/fingerprint as `super().hypothesis()`'s
own call and is a guaranteed cache hit for all 7 expensive sources; see the corrected
docstring in `scripts/run_fast_tier_health_check.py`).

**This is a genuine, previously-undiscovered interaction defect between Task 3's
window-bounding (added to bound per-call cost and memory) and Task 2's refit-caching
(added to avoid refitting every bar): for any replay longer than `max_history_window`
bars, the window-bounding silently defeats the refit cache entirely, and every bar pays
full uncached-refit cost.** This is worse than either task's own latency numbers
suggest in isolation, and worse than Task 5's latency report discloses — Task 5's
benchmarks did not run past the 2,000-bar window boundary, so this behavior was not
visible there. Flagging this plainly for the Task 11 whole-branch review rather than
fixing it here: Task 9's brief scope is documentation/verification, not a latency fix,
and any fix to the cache-key design should get its own scoped review given it touches
the same correctness-sensitive cache Task 8 already found and fixed one stale-data bug
in.

## Timing reconciliation

The 1,593.2s wall-clock total, divided across the run's ~4,068 reasoner-touching
calls, averages ~390ms/call. Task 5's latency report (`2026-09-02-goldex-phase2-hardening-latency-report.md`)
separately measured `FastTierReasoner.hypothesis()` at: cached call p99 ≈17-18ms,
refit-triggering call p99 ≈125-141ms, and `EvidenceRegistry.compute_all()` (fully
uncached) p99 ≈402-447ms. A ~390ms/call average over a chronologically-advancing
4,000-bar replay is suspiciously close to the *uncached* worst case, not the mostly-cached
case a refit-caching reasoner should produce over a long run. This was investigated
rather than left unreconciled.

**First candidate ruled out: the script's own instrumentation.** `_InstrumentedReasoner`
(the health-check-only `FastTierReasoner` subclass, `scripts/run_fast_tier_health_check.py`
around line 91) calls `self._compute_evidence(closes_so_far)` once directly (to re-derive
per-source applicability gating for its side-channel metrics) and then calls
`super().hypothesis(...)`, which calls `_compute_evidence` a second time on the *same*
`closes_so_far` array within the same `hypothesis()` invocation. Reading
`FastTierReasoner._compute_evidence` (`intelligence/fast_tier.py:270-297`) shows its
cache key is `(bar_index, fingerprint=closes_so_far[0])`. Because both calls happen
back-to-back on the identical array object (no bar advance between them), the second
call's `(bar, fingerprint)` always matches what the first call just wrote to
`self._cache` — the second call is a guaranteed cache hit for all 7
`EXPENSIVE_SOURCE_NAMES`. **This instrumentation does not duplicate any GARCH/Kalman
fit.** It does redundantly recompute the registry's cheap (non-cached) sources once
more per call, but those sources are, by construction, cheap. This part of the
original docstring's "cache-backed" claim held up.

**Actual root cause: `FastTierReasoner`'s own refit cache is defeated once the replay's
history exceeds `max_history_window`.** `_compute_evidence` truncates `closes_so_far`
to its last `max_history_window` (default 2000) entries before computing the
fingerprint:

```python
if len(closes_so_far) > self.max_history_window:
    closes_so_far = closes_so_far[-self.max_history_window:]
bar = len(closes_so_far)
fingerprint = float(closes_so_far[0]) if bar else None
```

For the first 2,000 bars of a continuing replay, `closes_so_far[0]` is the series'
actual first observation — a stable value — so the `(bar - cached_bar) < refit_interval
and cached_fingerprint == fingerprint` cache-hit condition behaves as intended: real
GARCH/Kalman refits happen only once every `refit_interval=50` bars. But once the
replay's history exceeds 2,000 bars, the truncated window *slides by exactly one bar
per call*, so `closes_so_far[0]` — and therefore the fingerprint — changes on **every
single call**. From that point on, the fingerprint check never matches, so every
`hypothesis()` call performs a full, uncached recompute of all 7 expensive sources,
at essentially `compute_all()`'s uncached cost, for the entire remainder of the run.
This is a real behavior of the production `FastTierReasoner`'s caching, not an
artifact of this script's instrumentation — any sufficiently long continuous
replay (not just this health check) will fall out of the refit cache the same way
once it passes `max_history_window` bars.

This also explains the module docstring's aside that "an initial 20,000-bar run was
aborted after 50+ minutes... per-bar cost does not amortize as cheaply as hoped even
with refit-caching" (`scripts/run_fast_tier_health_check.py:57-60`) — that observation
was this same effect, encountered but not root-caused at the time.

**Direct measurement.** Timing `FastTierReasoner._compute_evidence` directly (bypassing
the script and its instrumentation entirely) over a 2,100-bar synthetic series with the
same registry, `refit_interval=50`, `max_history_window=2000` defaults:

| Regime | Mean per-call | Max per-call | n |
|---|---|---|---|
| Pre-window (bar < 2,000) | 65.0ms | 2,405ms (a refit bar) | 1,999 |
| Post-window (bar > 2,010) | 777.1ms | 5,573ms | 90 |

Post-window calls are ~12x more expensive on average than pre-window calls, and every
post-window call pays a cost in the same range as a full uncached `compute_all()`
(consistent with Task 5's ~400-450ms figure; the higher absolute mean here reflects
this measurement's own machine/run variance, not a different code path). Averaging the
two regimes roughly 50/50 — the ~4,000-bar health-check run spends its first half inside
the cache window and its second half outside it — gives ≈421ms/call, which lines up
with the run's actual observed ~390ms/call average within the expected noise of this
kind of back-of-envelope reconciliation.

**Conclusion.** The ~390ms/call average genuinely reflects that roughly the back half
of this 4,000-bar run ran with its refit cache effectively disabled, not a hidden bug
in this health-check script's instrumentation. `scripts/run_fast_tier_health_check.py`'s
`_InstrumentedReasoner` docstring has been corrected to state this plainly rather than
asserting an unverified "at most one extra cheap pass" framing that (correctly, as it
turns out) exonerated the instrumentation but said nothing about the real cause. This
is a genuine finding about the hardened Fast Tier's caching behavior at
`max_history_window` boundaries on long-running replays, worth carrying into any future
perf-tuning task — no code fix is made here, since Task 5's caching/windowing tradeoff
was already deliberately scoped and this reconciliation task's job is diagnosis, not
further optimization.

**Update (Task 11 fix round, commit TBD-FIX-COMMIT).** The root cause diagnosed above
has now been fixed. `FastTierReasoner._compute_evidence` (`intelligence/fast_tier.py`)
now computes the refit-cache fingerprint from `closes_so_far[0]` of the ORIGINAL array
*before* the `max_history_window` truncation slice, instead of after it. Bar 0 of a
continuing replay never changes, so the fingerprint is now stable for the entire life
of one replay — including past the point where the truncated window starts sliding —
while still distinguishing two different series of the same length (the
stale-cache-leak fix from commit `c9b776d`, regression-tested by
`test_reasoner_cache_does_not_leak_across_different_series_of_the_same_length`, remains
intact). A new regression test,
`test_refit_cache_stays_warm_past_max_history_window_within_refit_interval` in
`tests/intelligence/test_fast_tier.py`, proves the cache now hits within one
`refit_interval` well past `max_history_window`. The diagnosis narrative above is left
unchanged as the historical record of how this was found.

## Deviation from brief

The brief describes scale as "one to a few years of synthetic chronological data, or
whatever realistic-scale fixture ... extended in length," and the task dispatch note
for this run specified "low thousands to tens of thousands of bars." An initial attempt
at 20,000 bars did not complete within 50+ minutes of wall-clock and was aborted; 4,000
bars was used instead, which completed in ~26.5 minutes. This is a real constraint on
this hardened Fast Tier's per-bar cost, not a shortcut — it is called out here rather
than silently substituted, and 4,000 bars remains a genuine order-of-magnitude increase
over the integration test's 300-bar fixture with the same real composition.
