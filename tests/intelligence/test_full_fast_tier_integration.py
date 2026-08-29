"""tests/intelligence/test_full_fast_tier_integration.py -- Task 13: the
"does the whole thing work together for real" test.

Composes a fully-real FastTierDecisionEngine -- build_default_registry()
(all 9 wrapped evidence sources), a real ToolTrust(), a real
FastTierReasoner(registry), a real EV/cost gate wrapping
simulator.cost_model.round_trip_cost_r, and Task 11's REAL
analytical_sizing_bootstrap/analytical_sltp_bootstrap -- no test doubles
anywhere -- and runs it through Phase 1's real, unmodified
simulator.replay.run_replay on a longer synthetic dataset than any single
prior task's unit tests used, following the general shape of
tests/intelligence/test_decision_engine.py (Task 6) and
tests/simulator/test_no_leakage.py's truncate-and-recompute equivalence
pattern, now applied to the full composed system rather than one module.

DATASET SIZE / RUNTIME NOTE: an initial version of this test used 2,000
bars with low per-bar noise (matching test_fast_tier_performance.py's
gentle-drift convention). That combination genuinely never produced a
single LONG/SHORT trade in ~570s of real wall-clock replay time -- see the
"INTEGRATION FINDING" note below for why (a real, load-bearing property of
the composed EV/cost gate, not a test bug) -- and Task 12 already
documented real per-call latency (order ~0.1-1ms/call at these history
lengths, compounding to minutes over thousands of calls when the reasoner's
refit/registry-compute-all path runs on every DECIDE/MANAGE bar). This
version uses a deliberately noisier synthetic series (large enough
per-bar noise to push realized_vol_60s up into the regime where the EV/cost
gate can actually pass -- see below) and a much smaller bar count (300 for
the main/rejection runs, 400/200 for the truncation check) to keep total
runtime well under Task 12's observed latency ceiling while still
exercising the composed system for real, end to end.

INTEGRATION FINDING (genuine, discovered while building this test, not
worked around in production code, corrected after a review pass caught an
error in an earlier draft of this note -- see below): `intelligence/
bootstrap.py`'s `analytical_sltp_bootstrap` sets `sl_distance =
SL_VOL_MULTIPLIER * vol * mid`, and `intelligence/decision_engine.py`'s
`decide()` immediately re-normalizes that same distance back into an
R-multiple as `sl_distance_r = sl_distance_price / (vol * mid)`. `vol`
cancels out of THAT ratio, so `sl_distance_r` is always exactly
`SL_VOL_MULTIPLIER` (2.0) -- this part is correct and expected, not a bug:
an R-multiple is by construction dimensionless in vol (a "2-sigma stop" is
always "2R" by definition, regardless of what sigma numerically is).

What does NOT cancel is `vol` out of `cost_r` itself.
`simulator.cost_model.round_trip_cost_r` computes
`cost_r = (spread * 2) / (sl_distance_r * vol * mid)`, i.e., substituting
the constant above, `cost_r = spread / (SL_VOL_MULTIPLIER/2 * vol * mid)`
-- `vol` sits in the denominator here and does NOT cancel, so `cost_r`
correctly falls as `vol` rises: vol-scaling the stop is exactly the
mechanism that makes a wider (higher-vol) stop absorb a fixed spread cost
more easily. `decide()` rejects a candidate when `edge_proxy_r <= cost_r`
where `edge_proxy_r = abs(belief) * (1 - uncertainty)` is capped at 1.0.
`cost_r > 1.0` (making the trade un-clearable at ANY belief) happens
precisely when `vol < spread / (SL_VOL_MULTIPLIER * mid)` -- i.e. when the
round-trip spread cost, measured against a 2-sigma stop, would eat more
than the entire available risk budget. Refusing to trade in that regime is
CORRECT financial behavior, not a defect: it is the EV/cost gate doing
exactly what it is meant to do (never trade when the edge can't plausibly
clear round-trip cost).

This was directly observed: the original 2,000-bar, noise-std=0.05 dataset
produced realized_vol_60s around 4e-5 (below that spread/(SL_VOL_MULTIPLIER
* mid) ~5e-5 threshold at this test's spread=0.2/mid=1900 scale) throughout,
so `cost_r` sat around 2.4-2.9 the entire run and correctly rejected all
2000 bars. Raising per-bar noise (std 0.35 below, giving realized_vol_60s
around 2.5e-4-3e-4, comfortably above the threshold) pushes cost_r down into
the 0.35-0.5 range, and trades occur normally once genuine edge is present.

The real, narrower, legitimate concern this exercise surfaced: `edge_proxy_r`
is a dimensionless, ad hoc confidence score (`|belief| * (1-uncertainty)`,
both already normalized quantities from the reasoner) being compared
directly against `cost_r`, which is a genuine R-multiple in the market's own
risk units -- nothing establishes that these two quantities are on a
comparable numeric scale (e.g. that an `edge_proxy_r` of 0.5 actually
corresponds to anything like "0.5R of real expected edge"). `decide()`'s own
module docstring already flags `edge_proxy_r` as "a crude, dimensionless
in-[0, 1] stand-in for real expected R" -- this test's finding is not that
the gate is broken, but that its threshold interacts with real market
regimes exactly the way the vol-vs-spread math predicts, and that
`edge_proxy_r`'s calibration against `cost_r`'s R-multiple scale is still an
open item (consistent with Task 12's already-documented deferred-tuning
scope), worth a deliberate look before this gate's specific pass/reject
boundary is trusted at any particular belief threshold. Reported here per
this task's brief ("if you discover ANY integration mismatch ... that's a
genuine finding to report prominently, not something to silently work around
by changing your test to avoid the mismatch") -- this test documents the
finding and chooses a noisier-but-plausible dataset (tradeable vol regime)
to still exercise the trading path end to end, rather than treating the
low-vol NO_TRADE behavior as a bug to route around.

Four sub-checks, all against ONE composed run's actual ExperienceRecord
stream (plus a second, freshly-constructed engine/run for the no-look-ahead
truncation check):

  a) no-look-ahead: truncate-and-recompute equivalence at the full
     composed-system level.
  b) rejection handling: at least one rejected entry occurs (engineered via
     a tiny starting balance) and assign_replay_credit correctly excludes
     it from credit.
  c) thesis lifecycle: at least one real LONG/SHORT trade occurs with a
     populated thesis at entry (captured via engine.open_thesis
     immediately after decide()), correctly cleared (engine.open_thesis is
     None) once that position has closed.
  d) credit assignment: assign_replay_credit against the real run's
     records moves at least one non-rejected trade's ToolTrust posterior
     away from the Beta(1,1) prior for its actual load-bearing sources.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from intelligence.bootstrap import analytical_sizing_bootstrap, analytical_sltp_bootstrap
from intelligence.credit_assignment import assign_replay_credit
from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import FastTierReasoner, ToolTrust
from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r
from simulator.replay import run_replay

N_BARS = 300

# Per-bar noise std: deliberately larger than a "gentle drift" convention
# would use (see the module docstring's INTEGRATION FINDING note) -- this
# pushes realized_vol_60s above the spread/(SL_VOL_MULTIPLIER * mid)
# threshold so the EV/cost gate can actually pass at all. Below that
# threshold the round-trip spread cost exceeds the entire 2-sigma risk
# budget, so the composed system correctly refuses to trade at any belief.
NOISE_STD = 0.35


def _make_df(n=N_BARS, start_price=1900.0):
    """Deterministic synthetic OHLC series with a clear trend component
    (so the real evidence sources -- momentum, PCA-slope, Kalman velocity
    -- have real directional signal to pick up, not just noise) plus
    oscillation and per-bar noise for numeric texture (GARCH/kurtosis
    sources) large enough to clear the EV/cost gate -- see NOISE_STD and
    the module docstring's INTEGRATION FINDING note."""
    rng = np.random.default_rng(20260829)
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


def _make_real_engine(config: SimulatedExecutionConfig) -> FastTierDecisionEngine:
    """Constructs a FastTierDecisionEngine with every slot wired to a REAL
    implementation from Tasks 2-11 -- no test doubles anywhere."""
    registry = build_default_registry()
    trust = ToolTrust()
    reasoner = FastTierReasoner(registry)
    return FastTierDecisionEngine(
        registry=registry,
        trust=trust,
        reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate,
        sizing_bootstrap=analytical_sizing_bootstrap(config),
        sltp_bootstrap=analytical_sltp_bootstrap,
    )


class _ThesisCapturingEngine(FastTierDecisionEngine):
    """Thin subclass that snapshots this engine's Thesis at the moment of
    each LONG/SHORT decide() call, keyed by the decision_id that
    simulator.replay.run_replay is about to generate for it. Task 7's
    design deliberately never persists Thesis onto an ExperienceRecord
    (module docstring of intelligence/credit_assignment.py), so an offline
    harness must capture it at decide()-time -- this mirrors exactly what
    that module's own docstring says any caller of assign_replay_credit
    must do. Captured by matching sequence order against decision_ids
    afterward, since decide() itself doesn't see the decision_id
    run_replay is about to mint for it (run_replay only mints it after
    decide() returns)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_theses_in_order = []  # list of Thesis, one per LONG/SHORT decide()

    def decide(self, market_state, account):
        result = super().decide(market_state, account)
        action = result[0]
        if action in ("LONG", "SHORT"):
            self.entry_theses_in_order.append(self.open_thesis)
        return result


def test_full_composed_fast_tier_end_to_end_through_real_replay():
    df = _make_df()
    config = SimulatedExecutionConfig()
    engine = _make_real_engine(config)
    engine = _ThesisCapturingEngine(
        registry=engine.registry, trust=engine.trust, reasoner=engine.reasoner,
        ev_cost_gate=engine.ev_cost_gate, sizing_bootstrap=engine.sizing_bootstrap,
        sltp_bootstrap=engine.sltp_bootstrap,
    )

    recorder = run_replay(df, engine.decide, engine.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    # run_replay calls decide_fn or manage_fn exactly once per bar (DECIDE
    # while flat, MANAGE while a position is open), and additionally writes
    # a POSITION_CLOSED record on whichever bar actually closes a position
    # (same-bar SL/TP/liquidation resolution can close a position without
    # manage_fn ever being invoked that bar -- see simulator/replay.py) --
    # so len(records) is bounded between N_BARS (nothing ever closed) and
    # 2 * N_BARS (a POSITION_CLOSED on every single bar), never less.
    assert N_BARS <= len(records) <= 2 * N_BARS
    n_closed = sum(1 for r in records if r.event_type == "POSITION_CLOSED")
    assert n_closed >= 1, "expected at least one closed position over this run"

    decide_records = [r for r in records if r.event_type == "DECIDE"]
    actions = {r.action for r in decide_records}
    assert actions <= {"NO_TRADE", "LONG", "SHORT"}

    # --- (c) thesis lifecycle -------------------------------------------
    entry_decides = [r for r in decide_records if r.action in ("LONG", "SHORT")]
    assert len(entry_decides) >= 1, (
        "expected at least one real LONG/SHORT trade against the real registry/reasoner "
        "on this trending synthetic dataset -- if this fails, the composed system's "
        "directional/uncertainty/EV gates never actually fire together in practice"
    )
    assert len(engine.entry_theses_in_order) == len(entry_decides)
    for thesis in engine.entry_theses_in_order:
        assert thesis is not None
        assert len(thesis.load_bearing_sources) > 0

    # Thesis correctly cleared at exit. The real, non-vacuous check: whether
    # the run ended FLAT (no position open on the final bar) is determined
    # directly from the last DECIDE/MANAGE record's event_type -- MANAGE
    # only ever fires while a position is open (simulator/replay.py only
    # calls manage_fn while `position is not None`), so if the last such
    # record is a DECIDE, the run ended flat; if it's a MANAGE, a position
    # was still open when the dataset ran out.
    last_decide_or_manage = [r for r in records if r.event_type in ("DECIDE", "MANAGE")][-1]
    if last_decide_or_manage.event_type == "DECIDE":
        # Ended flat: engine.open_thesis must genuinely be None -- this is
        # the actual clearing behavior under test (decision_engine.py's
        # decide() unconditionally clears any leftover thesis the moment
        # it is entered while flat, and the run's last decide() call did
        # exactly that).
        assert engine.open_thesis is None, (
            "run ended flat but engine.open_thesis was not cleared -- thesis leaked past its "
            "position's close"
        )
    else:
        # Ended with a position still open (dataset ran out mid-trade): the
        # thesis for that still-open position should still be present, not
        # spuriously cleared.
        assert engine.open_thesis is not None, (
            "run ended with a position still open but engine.open_thesis was already cleared"
        )

    # --- (b) rejection handling (engineered via tiny starting balance) ---
    # NOTE: a tiny starting_balance ALONE does not force a rejection here --
    # analytical_sizing_bootstrap's real formula is
    # `account.equity * config.risk_fraction_of_equity` (Task 11), so size
    # and margin required both scale down proportionally with equity and
    # INSUFFICIENT_MARGIN never triggers on balance alone. To genuinely
    # engineer a rejection (per this task's brief: "if your synthetic data
    # doesn't naturally produce one, engineer a scenario that does, e.g. a
    # tiny starting balance"), this also overrides sizing_bootstrap to a
    # fixed oversized value on this ONE rejection-scenario engine instance
    # -- deliberately, to force the edge case -- while every other slot
    # (registry, trust, reasoner, ev_cost_gate, sltp_bootstrap) and the
    # MAIN run above stay fully real, undoubled.
    tiny_config = SimulatedExecutionConfig(starting_balance=5.0)
    rejection_engine = _make_real_engine(tiny_config)
    rejection_engine = _ThesisCapturingEngine(
        registry=rejection_engine.registry, trust=rejection_engine.trust, reasoner=rejection_engine.reasoner,
        ev_cost_gate=rejection_engine.ev_cost_gate,
        sizing_bootstrap=lambda hyp, account: 1_000_000.0,
        sltp_bootstrap=rejection_engine.sltp_bootstrap,
    )
    rejection_recorder = run_replay(
        df, rejection_engine.decide, rejection_engine.manage, tiny_config, EnvironmentTag.SIMULATED_TRAINING
    )
    rejection_records = rejection_recorder.all_records()
    rejected = [
        r for r in rejection_records
        if r.event_type == "DECIDE" and r.action in ("LONG", "SHORT") and r.rejection_reason is not None
    ]
    assert len(rejected) >= 1, (
        "expected at least one rejected entry against a $5 starting balance on this dataset"
    )
    rejected_decision_ids = {r.decision_id for r in rejected}
    # No POSITION_CLOSED record should ever reference a rejected decision_id.
    assert not any(
        r.event_type == "POSITION_CLOSED" and r.decision_id in rejected_decision_ids
        for r in rejection_records
    )

    rejection_entries = [
        r for r in rejection_records if r.event_type == "DECIDE" and r.action in ("LONG", "SHORT")
    ]
    assert len(rejection_engine.entry_theses_in_order) == len(rejection_entries)
    theses_by_decision_id = {
        entry.decision_id: thesis
        for entry, thesis in zip(rejection_entries, rejection_engine.entry_theses_in_order)
    }
    rejection_trust = ToolTrust()
    credited_count = assign_replay_credit(rejection_records, theses_by_decision_id, rejection_trust)
    # Every rejected entry's thesis must contribute zero credit: confirm no
    # rejected decision_id's load-bearing sources moved off the Beta(1,1)
    # prior via this specific trust instance (which saw ONLY this run's
    # records, so any movement can only have come from these trades).
    for decision_id in rejected_decision_ids:
        thesis = theses_by_decision_id.get(decision_id)
        if thesis is None:
            continue
        for source_name, context_bucket, _ in thesis.load_bearing_sources:
            assert rejection_trust.posterior_mean(source_name, context_bucket) == 0.5, (
                f"rejected trade {decision_id} contaminated posterior for "
                f"({source_name}, {context_bucket})"
            )

    # --- (d) credit assignment on the MAIN (non-tiny-balance) run --------
    theses_by_decision_id_main = {
        entry.decision_id: thesis
        for entry, thesis in zip(entry_decides, engine.entry_theses_in_order)
    }
    trust_for_credit = ToolTrust()
    credited_count_main = assign_replay_credit(records, theses_by_decision_id_main, trust_for_credit)
    assert credited_count_main >= 1, (
        "expected at least one non-rejected, closed trade to be creditable in the main run"
    )
    moved = False
    for thesis in theses_by_decision_id_main.values():
        for source_name, context_bucket, _ in thesis.load_bearing_sources:
            if trust_for_credit.posterior_mean(source_name, context_bucket) != 0.5:
                moved = True
                break
        if moved:
            break
    assert moved, "expected at least one ToolTrust posterior to move away from the Beta(1,1) prior"


def test_no_look_ahead_full_composed_system_truncate_and_recompute():
    """(a) no-look-ahead at the FULL composed-system level: a freshly
    constructed engine run on a truncated prefix of the dataset must
    produce DECIDE/MANAGE records identical (up to the truncation point) to
    a freshly constructed engine run on the full dataset -- mirroring
    tests/simulator/test_no_leakage.py's
    test_replay_records_identical_snapshots_regardless_of_unreached_future,
    now applied to the whole FastTierDecisionEngine rather than a
    NO_TRADE-only stub decide_fn. Uses two FRESH engine instances (not one
    reused across both runs) since ToolTrust/thesis state is itself
    stateful across decide()/manage() calls and would otherwise contaminate
    the comparison."""
    df = _make_df(n=400)
    config = SimulatedExecutionConfig()

    engine_full = _make_real_engine(config)
    recorder_full = run_replay(df, engine_full.decide, engine_full.manage, config, EnvironmentTag.SIMULATED_TRAINING)

    truncated = df.iloc[: len(df) // 2].copy()
    engine_truncated = _make_real_engine(config)
    recorder_truncated = run_replay(
        truncated, engine_truncated.decide, engine_truncated.manage, config, EnvironmentTag.SIMULATED_TRAINING
    )

    full_records = recorder_full.all_records()
    truncated_records = recorder_truncated.all_records()
    n_common = len(truncated_records)
    assert n_common > 0
    assert n_common < len(full_records)

    def _snapshots_equal(snap_a, snap_b):
        if snap_a.keys() != snap_b.keys():
            return False
        for key in snap_a:
            va, vb = snap_a[key], snap_b[key]
            if isinstance(va, float) and isinstance(vb, float) and pd.isna(va) and pd.isna(vb):
                continue
            if va != vb:
                return False
        return True

    for i, (a, b) in enumerate(zip(full_records[:n_common], truncated_records)):
        assert a.event_type == b.event_type, f"record {i}: event_type diverged ({a.event_type} vs {b.event_type})"
        assert a.action == b.action, f"record {i}: action diverged ({a.action} vs {b.action})"
        assert _snapshots_equal(a.market_state_snapshot, b.market_state_snapshot), (
            f"record {i}: market_state_snapshot diverged -- possible look-ahead leakage in the "
            f"composed system"
        )
        assert a.rejection_reason == b.rejection_reason, f"record {i}: rejection_reason diverged"
