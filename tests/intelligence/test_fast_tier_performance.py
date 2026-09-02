"""tests/intelligence/test_fast_tier_performance.py -- Task 12: REAL
latency measurement for the Fast Tier (mandate Section 8/12 item 12: "do
not claim [millisecond-capable] without measurement"). Follows
tests/simulator/test_replay_performance.py's exact convention: every
number printed via time.perf_counter, not just asserted below some
threshold -- the printed number IS the deliverable. Loose sanity bounds
sit alongside each print, same as that file.

Three things measured, using the REAL 9-source default registry (not a
trivial stub), so the numbers reflect actual production cost:

  1. EvidenceRegistry.compute_all() latency -- all 9 wrapped sources
     computed fresh, no refit caching (worst case).
  2. FastTierReasoner.hypothesis() latency -- across 250 successive
     decision points (> refit_interval=50, so the refit-caching mechanism
     actually kicks in several times), reporting cached-call and
     refit-triggering-call latency SEPARATELY.
  3. FastTierDecisionEngine.decide() end-to-end latency (evidence +
     reasoning + EV/cost gate + Task 11's real analytical SL/TP/size
     bootstrap combined), also across enough calls for refit caching to
     matter.

BOUNDS: every assert below is set to roughly 1.5-2x the ACTUAL measured p99
on this machine AFTER the Task 1-4 hardening fixes (compute_all ~420ms,
cached hypothesis ~17ms, refit hypothesis ~141ms, decide ~36ms), not an
arbitrary round number -- re-measured and re-tightened by Task 5 (see
docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-latency-report.md).
A prior version of this comment described a pre-hardening baseline
(compute_all ~460ms, cached hypothesis ~56ms, refit hypothesis ~134ms,
decide ~40ms) with looser 2-3x bounds; those numbers are now stale --
Task 1-4 changed what "cached hypothesis" costs (down ~3x, from the 4
newly-cached non-directional sources) even though compute_all and
refit-triggering calls (which still pay full source cost) did not move
much, since GARCH's Python-loop MLE dominates both and was not itself
optimized. They are still multiples, not exact ceilings, so ordinary
machine-load variance does not make them flaky.

WHAT decide() LATENCY DOES *NOT* COVER: `decide()` is the FLAT-position
path. `FastTierDecisionEngine.manage()` (Task 8) also calls
`reasoner.hypothesis(...)` on essentially every bar a position is HELD, so
the reasoning cost below is paid on every bar of a replay session, not only
on flat bars. The true per-bar cost of the composed system during a held
position is approximately the same order of magnitude as the `decide()`
number measured here -- not zero. See intelligence/decision_engine.py's
module docstring.

Dataset size: matches Task 9's own judgment call -- 1,000-point closes
history (comparable order of magnitude to test_replay_performance.py's
N_MICROBENCH_ITERS=1000), enough for the recursive GARCH/Kalman sources
to do real O(n) work without slowing the suite down.
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from contracts.market_state import MarketState, M1BarState, DataQuality, FeedHealthState
from intelligence.bootstrap import analytical_sizing_bootstrap, analytical_sltp_bootstrap
from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import FastTierReasoner, ToolTrust
from simulator.contracts import AccountState, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r

N_CLOSES_FOR_COMPUTE_ALL = 1000  # matches Task 9's N_MICROBENCH_ITERS order of magnitude
N_DECISION_POINTS = 250  # > refit_interval=50, so refit caching triggers multiple times
REFIT_INTERVAL = 50


def _percentiles(values_us):
    values_us = sorted(values_us)
    n = len(values_us)
    return {
        "mean": sum(values_us) / n,
        "p50": values_us[n // 2],
        "p99": values_us[int(n * 0.99)],
    }


def _print_stats(label, unit, stats, n):
    print(f"[SYNTHETIC][Task 12] {label}: n={n} mean={stats['mean']:.2f}{unit} "
          f"p50={stats['p50']:.2f}{unit} p99={stats['p99']:.2f}{unit}")


def _make_closes(n, start_price=1900.0):
    # Gentle deterministic drift + oscillation -- enough numeric texture
    # for GARCH/Kalman/PCA-slope/kurtosis sources to compute real (non-
    # degenerate) values, matching test_replay_performance.py's synthetic
    # price convention.
    return np.array(
        [start_price + 0.05 * i + 0.3 * np.sin(i / 7.0) for i in range(n)],
        dtype=np.float64,
    )


def _sample_market_state(mid, ts):
    m1 = M1BarState(open=mid - 0.1, high=mid + 0.3, low=mid - 0.3, close=mid,
                     tick_count=12, start_time=ts, end_time=ts, complete=True)
    return MarketState(
        symbol="XAUUSD", source="synthetic_replay", state_version="v1", sequence=0,
        market_timestamp=ts, ingestion_timestamp=ts, processing_timestamp=ts,
        bid=mid - 0.1, ask=mid + 0.1, mid=mid, spread=0.22, last=mid,
        last_quality=DataQuality.VALID,
        tick_count_60s=60, tick_count_300s=300, tick_rate_per_sec=1.0,
        current_m1=m1, completed_m1=m1,
        realized_vol_60s=0.0008, spread_mean_60s=0.22, spread_std_60s=0.01,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.0,
        feed_latency_sec=0.0, state_update_latency_sec=0.0,
    )


def _cheap_cost_gate(market_state, candidate_sl_distance_r):
    return round_trip_cost_r(market_state, candidate_sl_distance_r, max_staleness_seconds=float("inf"))


def test_evidence_registry_compute_all_latency():
    """1. EvidenceRegistry.compute_all() latency -- all 9 sources computed
    fresh (no refit caching applied -- that caching lives in
    FastTierReasoner, not the registry itself), worst-case per-call cost."""
    registry = build_default_registry()
    assert len(registry.names()) == 9, f"expected 9 registered sources, got {len(registry.names())}: {registry.names()}"

    closes = _make_closes(N_CLOSES_FOR_COMPUTE_ALL)
    per_call_us = []
    n_iters = 200
    for i in range(n_iters):
        window = closes[: N_CLOSES_FOR_COMPUTE_ALL - (i % 50)]  # vary window slightly, avoid any accidental memoization
        t0 = time.perf_counter()
        registry.compute_all(window)
        per_call_us.append((time.perf_counter() - t0) * 1e6)

    stats = _percentiles(per_call_us)
    _print_stats("EvidenceRegistry.compute_all (9 sources, fresh, no caching)", "us", stats, n_iters)
    # Bound tightened by Task 5 against the post-Task-1-4 measured baseline
    # (see the module docstring's BOUNDS note): measured p99 ~= 402-426ms
    # here (two runs), so ~1.75x the higher of those.
    assert stats["p99"] < 750_000, "compute_all p99 regressed past ~1.75x its measured baseline (~426ms)"


def test_fast_tier_reasoner_hypothesis_latency():
    """2. FastTierReasoner.hypothesis() latency across N_DECISION_POINTS
    successive calls (closes buffer grows each call, mirroring
    FastTierDecisionEngine's real accumulation pattern) -- enough calls for
    the refit_interval=50 caching to trigger multiple times. Cached-call and
    refit-triggering-call latency reported separately."""
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry, refit_interval=REFIT_INTERVAL)
    trust = ToolTrust()

    base_ts = datetime(2020, 1, 6, 10, 0, 0, tzinfo=timezone.utc)
    full_closes = _make_closes(N_DECISION_POINTS + 100)  # +100 warmup so early calls aren't degenerate-short

    cached_call_us = []
    refit_call_us = []
    for i in range(N_DECISION_POINTS):
        n_closes = 100 + i
        closes_so_far = full_closes[:n_closes]
        market_state = _sample_market_state(float(closes_so_far[-1]), base_ts + timedelta(minutes=i))

        t0 = time.perf_counter()
        reasoner.hypothesis(closes_so_far, market_state, trust)
        elapsed_us = (time.perf_counter() - t0) * 1e6

        # A refit is triggered on this call iff any EXPENSIVE_SOURCE's cache
        # is empty or its distance since last refit >= refit_interval --
        # first call and every REFIT_INTERVAL-th call thereafter.
        if i == 0 or i % REFIT_INTERVAL == 0:
            refit_call_us.append(elapsed_us)
        else:
            cached_call_us.append(elapsed_us)

    cached_stats = _percentiles(cached_call_us)
    refit_stats = _percentiles(refit_call_us)
    _print_stats("FastTierReasoner.hypothesis (cached GARCH/Kalman)", "us", cached_stats, len(cached_call_us))
    _print_stats("FastTierReasoner.hypothesis (refit-triggering call)", "us", refit_stats, len(refit_call_us))

    # ~1.7-1.8x the post-Task-1-4 measured baselines (cached p99 ~= 17ms,
    # refit p99 ~= 141ms) -- tight enough that a real regression fails,
    # loose enough to tolerate ordinary machine-load variance. See the
    # BOUNDS note above.
    assert cached_stats["p99"] < 30_000, "cached hypothesis() p99 regressed past ~1.75x its measured baseline (~17ms)"
    assert refit_stats["p99"] < 250_000, "refit hypothesis() p99 regressed past ~1.75x its measured baseline (~141ms)"
    # The whole point of the refit-caching mechanism (Task 5): a
    # refit-triggering call should be meaningfully more expensive than a
    # cached one, or the caching isn't buying anything. Report, don't hide,
    # if this ever stops being true.
    print(f"[SYNTHETIC][Task 12] refit/cached mean ratio: {refit_stats['mean'] / max(cached_stats['mean'], 1e-9):.1f}x")


def test_fast_tier_decision_engine_decide_latency():
    """3. FastTierDecisionEngine.decide() end-to-end latency -- evidence +
    reasoning + EV/cost gate + Task 11's real analytical SL/TP/size
    bootstrap, across N_DECISION_POINTS calls (grown closes buffer, refit
    caching active) using the REAL production wiring (build_default_registry,
    analytical_sltp_bootstrap, analytical_sizing_bootstrap, round_trip_cost_r)
    -- this is the number that matters for "does the Fast Tier fit inside
    Phase 1's existing per-bar latency budget (build_snapshot p99 ~2ms)."""
    registry = build_default_registry()
    trust = ToolTrust()
    reasoner = FastTierReasoner(registry, refit_interval=REFIT_INTERVAL)
    config = SimulatedExecutionConfig()
    engine = FastTierDecisionEngine(
        registry=registry,
        trust=trust,
        reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate,
        sizing_bootstrap=analytical_sizing_bootstrap(config),
        sltp_bootstrap=analytical_sltp_bootstrap,
    )

    base_ts = datetime(2020, 1, 6, 10, 0, 0, tzinfo=timezone.utc)
    account = AccountState.initial(config, base_ts)
    full_closes = _make_closes(N_DECISION_POINTS + 100)

    per_call_us = []
    for i in range(N_DECISION_POINTS):
        market_state = _sample_market_state(float(full_closes[100 + i]), base_ts + timedelta(minutes=i))
        t0 = time.perf_counter()
        engine.decide(market_state, account)
        per_call_us.append((time.perf_counter() - t0) * 1e6)
        # decide() only ever runs while flat in real usage (simulator.replay
        # enforces this) -- clear any thesis so every call in this loop
        # exercises the same decide() code path rather than skewing toward
        # manage()'s cheaper branches, matching engine's actual DecideFn use.
        engine.clear_thesis()

    stats = _percentiles(per_call_us)
    _print_stats("FastTierDecisionEngine.decide (evidence+reasoning+gate+bootstrap, end-to-end)", "us", stats,
                 N_DECISION_POINTS)
    # ~1.8x the post-Task-1-4 measured baseline (p99 ~= 36ms). See the
    # BOUNDS note above.
    assert stats["p99"] < 65_000, "decide() p99 regressed past ~1.8x its measured baseline (~36ms)"

    # Phase 1's own measured per-bar budget reference point (see
    # tests/simulator/test_replay_performance.py):
    #   build_snapshot construction latency: p99 ~= 2ms (2000us)
    # Printed for direct side-by-side comparison -- not asserted against,
    # since decide() legitimately does much more work than build_snapshot
    # (9 evidence sources including GARCH/Kalman refits vs. a single bar
    # parse), but the controller needs this number to judge fit-for-budget.
    phase1_build_snapshot_p99_us = 2000.0
    ratio = stats["p99"] / phase1_build_snapshot_p99_us
    print(f"[SYNTHETIC][Task 12] decide() p99 vs Phase 1 build_snapshot p99 (~{phase1_build_snapshot_p99_us:.0f}us): "
          f"{ratio:.1f}x")


if __name__ == "__main__":
    test_evidence_registry_compute_all_latency()
    test_fast_tier_reasoner_hypothesis_latency()
    test_fast_tier_decision_engine_decide_latency()
    print("tests/intelligence/test_fast_tier_performance.py: OK")
