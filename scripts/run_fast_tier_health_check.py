"""scripts/run_fast_tier_health_check.py -- Task 9: moderate-scale
chronological simulator health check for the hardened Fast Tier.

Composes the SAME real, undoubled system as
`tests/intelligence/test_full_fast_tier_integration.py` (real
`build_default_registry()`, real `ToolTrust`, real `FastTierReasoner`, the
real EV/cost gate wrapping `simulator.cost_model.round_trip_cost_r`, real
`analytical_sizing_bootstrap`/`analytical_sltp_bootstrap`, and Phase 1's
real, unmodified `simulator.replay.run_replay`) and runs it over a longer
synthetic chronological dataset than that test's own 300/400-bar fixtures,
recording system-HEALTH metrics.

This is explicitly NOT the final six-year training run, and this script
produces NO profitability claim of any kind: no P&L, no Sharpe ratio, no
win-rate framing. It only asks "did MarketState -> Fast Tier -> Action ->
Execution -> Experience compose correctly at a larger scale, with zero
crashes and sane distributional shape (bounded NO_TRADE rate, spread
context buckets, non-degenerate trade durations)."

Dataset: reuses the integration test's exact synthetic-series construction
(_make_df) and NOISE_STD=0.35, extended from 300 to N_BARS bars -- that
noise level is what pushes realized_vol_60s above the
spread/(SL_VOL_MULTIPLIER * mid) threshold so the EV/cost gate can pass at
all (see that test's module docstring "INTEGRATION FINDING" for the full
derivation); a lower-noise, "gentle drift" series was shown there to
produce zero trades over 2,000 bars for a real, load-bearing reason, not a
test artifact.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from intelligence.applicability import apply_applicability
from intelligence.bootstrap import analytical_sizing_bootstrap, analytical_sltp_bootstrap
from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.experience_store import ExperienceStore
from intelligence.fast_tier import FastTierReasoner, ToolTrust, context_bucket
from simulator.contracts import EnvironmentTag, PositionOutcome, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r
from simulator.experience import ExperienceRecorder
from simulator.replay import run_replay

# Moderate scale, NOT the final six-year training run. 4,000 bars is ~13x
# the integration test's 300-bar fixture -- large enough to exercise the
# system at real scale (refit-caching cycling many times, context buckets
# spreading, many trade lifecycles) while staying comfortably inside a
# single synchronous run's wall-clock budget. (An initial 20,000-bar run was
# aborted after 50+ minutes of wall-clock without completing -- per-bar cost
# does not amortize as cheaply as hoped even with refit-caching, so this
# script uses a scale that actually completes synchronously.)
N_BARS = 4_000
NOISE_STD = 0.35


def _make_df(n=N_BARS, start_price=1900.0):
    """Same construction as the integration test's `_make_df`, extended in
    length only -- deterministic synthetic OHLC series with a trend
    component (so directional evidence sources have real signal), plus
    oscillation and per-bar noise large enough to clear the EV/cost gate."""
    rng = np.random.default_rng(20260902)
    times = pd.date_range("2020-01-06 00:00:00", periods=n, freq="1min")
    trend = 0.03 * np.arange(n)
    wave = 1.5 * np.sin(np.arange(n) / 40.0)
    noise = rng.normal(0.0, NOISE_STD, size=n)
    closes = start_price + trend + wave + noise
    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) + 0.15
    lows = np.minimum(opens, closes) - 0.15
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows,
        "close": closes, "tick_volume": [15] * n, "spread": [20.0] * n,
    })


def _cheap_cost_gate(market_state, candidate_sl_distance_r):
    return round_trip_cost_r(market_state, candidate_sl_distance_r, max_staleness_seconds=float("inf"))


class _InstrumentedReasoner(FastTierReasoner):
    """Same real FastTierReasoner, plus a health-check-only side channel
    recording, per hypothesis() call, which sources got applicability-gated
    to zero confidence and which context bucket the call landed in. Adds no
    new evidence/decision logic -- purely observational bookkeeping on top
    of the real computation path, and re-derives gating via the exact same
    `apply_applicability` call the real `hypothesis()` makes. This extra pass
    is genuinely cache-backed and cheap: it calls `_compute_evidence` on the
    exact same `closes_so_far` array `super().hypothesis()` then calls again
    for the same bar, so for the 7 EXPENSIVE_SOURCE_NAMES the second call is
    always a guaranteed cache hit (identical bar index and fingerprint) --
    this instrumentation does NOT double any GARCH/Kalman fit. It does
    recompute the registry's cheap (non-cached) sources twice per call, but
    those are, per their name, cheap.

    IMPORTANT -- this instrumentation is NOT the explanation for this
    script's observed ~390ms/reasoner-call average (see the "Timing
    reconciliation" section of
    docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-simulator-run-report.md).
    That average is a real property of `FastTierReasoner`'s own refit cache:
    `_compute_evidence`'s cache key includes `closes_so_far[0]` as a
    fingerprint, and once the growing history exceeds `max_history_window`
    (2000 bars by default), the window slides by one bar every call, so
    `closes_so_far[0]` -- and therefore the fingerprint -- changes on EVERY
    bar. From that point on, every single hypothesis() call is a full,
    uncached GARCH/Kalman refit across all 7 expensive sources, at
    roughly `EvidenceRegistry.compute_all()`'s uncached cost (~400-450ms
    p99 per Task 5's latency report), not the ~17-141ms cached/refit-cadence
    cost the refit cache is designed to deliver. This is a real caching
    behavior of the production `FastTierReasoner`, independent of this
    script's instrumentation -- confirmed by direct timing of
    `FastTierReasoner._compute_evidence` before/after the 2,000-bar
    boundary (see the report)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_calls: Counter = Counter()
        self.gate_gated_out: Counter = Counter()
        self.bucket_counts: Counter = Counter()

    def hypothesis(self, closes_so_far, market_state, trust):
        raw_evidence = self._compute_evidence(closes_so_far)
        gated_evidence = {}
        for name, ev in raw_evidence.items():
            gated = apply_applicability(name, ev, closes_so_far, market_state)
            gated_evidence[name] = gated
            self.gate_calls[name] += 1
            if gated.confidence <= 0.0:
                self.gate_gated_out[name] += 1
        self.bucket_counts[context_bucket(gated_evidence)] += 1
        return super().hypothesis(closes_so_far, market_state, trust)


class _ThesisCapturingEngine(FastTierDecisionEngine):
    """Same pattern as the integration test's `_ThesisCapturingEngine`:
    snapshots this engine's Thesis at the moment of each LONG/SHORT
    decide() call, in call order, for later matching against the
    decision_ids `run_replay` mints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_theses_in_order = []

    def decide(self, market_state, account):
        result = super().decide(market_state, account)
        action = result[0]
        if action in ("LONG", "SHORT"):
            self.entry_theses_in_order.append(self.open_thesis)
        return result


def _make_real_engine(config: SimulatedExecutionConfig, reasoner_cls=_InstrumentedReasoner):
    registry = build_default_registry()
    trust = ToolTrust()
    reasoner = reasoner_cls(registry)
    return FastTierDecisionEngine(
        registry=registry,
        trust=trust,
        reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate,
        sizing_bootstrap=analytical_sizing_bootstrap(config),
        sltp_bootstrap=analytical_sltp_bootstrap,
    )


def main():
    t_start = time.monotonic()
    exceptions: list[str] = []

    df = _make_df()
    config = SimulatedExecutionConfig()

    base_engine = _make_real_engine(config)
    engine = _ThesisCapturingEngine(
        registry=base_engine.registry, trust=base_engine.trust, reasoner=base_engine.reasoner,
        ev_cost_gate=base_engine.ev_cost_gate, sizing_bootstrap=base_engine.sizing_bootstrap,
        sltp_bootstrap=base_engine.sltp_bootstrap,
    )

    try:
        recorder = run_replay(df, engine.decide, engine.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    except Exception as exc:  # noqa: BLE001 -- this script's whole job is to prove zero exceptions
        exceptions.append(f"run_replay crashed: {exc!r}")
        recorder = ExperienceRecorder()

    t_replay_done = time.monotonic()

    # Real ExperienceStore read path, per the brief's interface list --
    # confirms the store composes over this run's own recorded partition.
    try:
        store = ExperienceStore(recorder, EnvironmentTag.SIMULATED_TRAINING)
        records = store.records()
    except Exception as exc:  # noqa: BLE001
        exceptions.append(f"ExperienceStore.records() crashed: {exc!r}")
        records = recorder.all_records()

    # ---- Metric computation (all read-only over the real record stream) ----
    decide_records = [r for r in records if r.event_type == "DECIDE"]
    manage_records = [r for r in records if r.event_type == "MANAGE"]
    closed_records = [r for r in records if r.event_type == "POSITION_CLOSED"]

    n_decide = len(decide_records)
    action_counts = Counter(r.action for r in decide_records)
    n_no_trade = action_counts.get("NO_TRADE", 0)
    n_long = action_counts.get("LONG", 0)
    n_short = action_counts.get("SHORT", 0)
    no_trade_rate = n_no_trade / n_decide if n_decide else float("nan")

    n_rejected = sum(1 for r in decide_records if r.action in ("LONG", "SHORT") and r.rejection_reason is not None)

    outcome_counts = Counter(r.outcome for r in closed_records if r.outcome is not None)

    # Trade duration in bars: for each POSITION_CLOSED, count MANAGE records
    # sharing its decision_id, plus 1 for the opening DECIDE bar itself.
    # This is exact under run_replay's semantics (one MANAGE per bar a
    # position stays open, none on the same bar it's opened, none on the
    # bar it closes unless manage_fn itself triggered the close).
    manage_counts_by_decision_id: dict[str, int] = defaultdict(int)
    for r in manage_records:
        if r.decision_id is not None:
            manage_counts_by_decision_id[r.decision_id] += 1

    durations = []
    for r in closed_records:
        if r.decision_id is None:
            continue
        durations.append(1 + manage_counts_by_decision_id.get(r.decision_id, 0))
    avg_duration = (sum(durations) / len(durations)) if durations else float("nan")

    # Per-source applicability-gate rate.
    reasoner = engine.reasoner
    gate_rates = {}
    for name, calls in reasoner.gate_calls.items():
        gated_out = reasoner.gate_gated_out.get(name, 0)
        gate_rates[name] = gated_out / calls if calls else float("nan")

    # Per-hypothesis()-call context bucket distribution (every DECIDE and
    # MANAGE bar), the direct measurement of Task 4's recalibration.
    bucket_dist_per_call = {str(k): v for k, v in sorted(reasoner.bucket_counts.items())}

    # Cross-check: bucket distribution restricted to buckets that actually
    # became load-bearing on an entered trade's thesis -- the subset
    # credit assignment/thesis memory actually key on.
    bucket_counts_from_theses = Counter()
    for thesis in engine.entry_theses_in_order:
        if thesis is None:
            continue
        for _source_name, bucket, _contribution in thesis.load_bearing_sources:
            bucket_counts_from_theses[bucket] += 1

    wall_clock_total_s = time.monotonic() - t_start
    wall_clock_replay_s = t_replay_done - t_start

    report = {
        "n_bars": N_BARS,
        "noise_std": NOISE_STD,
        "wall_clock_total_seconds": wall_clock_total_s,
        "wall_clock_replay_seconds": wall_clock_replay_s,
        "exceptions": exceptions,
        "n_exceptions": len(exceptions),
        "total_records": len(records),
        "total_decisions": n_decide,
        "action_counts": dict(action_counts),
        "no_trade_rate": no_trade_rate,
        "n_long": n_long,
        "n_short": n_short,
        "n_rejected_entries": n_rejected,
        "n_closed_positions": len(closed_records),
        "outcome_counts": {str(k): v for k, v in outcome_counts.items()},
        "n_policy_exit": outcome_counts.get(PositionOutcome.POLICY_EXIT, 0),
        "n_sl_tp_liquidation_exit": sum(
            v for k, v in outcome_counts.items()
            if k in (PositionOutcome.SL_HIT, PositionOutcome.TP_HIT, PositionOutcome.LIQUIDATION)
        ),
        "n_forced_close_end_of_replay": outcome_counts.get(PositionOutcome.END_OF_REPLAY_FORCED_CLOSE, 0),
        "n_trade_durations_sampled": len(durations),
        "avg_trade_duration_bars": avg_duration,
        "min_trade_duration_bars": min(durations) if durations else None,
        "max_trade_duration_bars": max(durations) if durations else None,
        "per_source_applicability_gate_rate": gate_rates,
        "context_bucket_distribution_per_hypothesis_call": bucket_dist_per_call,
        "n_distinct_context_buckets_observed_per_call": len(bucket_dist_per_call),
        "context_bucket_distribution_from_load_bearing_sources": {
            str(k): v for k, v in sorted(bucket_counts_from_theses.items())
        },
        "n_distinct_context_buckets_observed_load_bearing": len(bucket_counts_from_theses),
        "ended_flat": engine.open_thesis is None,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "superpowers", "reports")
    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    json_path = os.path.join(out_path, "2026-09-02-goldex-phase2-hardening-simulator-run-metrics.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nMetrics JSON written to: {json_path}")

    if exceptions:
        print(f"\nFAILED: {len(exceptions)} exception(s) recorded during the run.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nOK: run completed with zero exceptions.")


if __name__ == "__main__":
    main()
