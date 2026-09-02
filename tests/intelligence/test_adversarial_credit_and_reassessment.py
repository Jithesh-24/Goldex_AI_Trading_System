"""tests/intelligence/test_adversarial_credit_and_reassessment.py -- Task 8
(part 2 of 2): adversarial coverage for thesis invalidation and continuous
reassessment, plus two cross-reference notes (no new test code) for the
credit-assignment and causality mandate items already covered elsewhere.
See tests/intelligence/test_adversarial_abstention_and_neutrality.py for
abstention and direction neutrality.

Construction pattern follows tests/intelligence/test_full_fast_tier_
integration.py's `_make_real_engine`/`run_replay` composition (real
registry, real ToolTrust, real FastTierReasoner, the same cheap
round_trip_cost_r-backed EV/cost gate, and Task 11's real analytical
sizing/SL-TP bootstraps -- no test doubles), reusing that file's
noisier-dataset convention (NOISE_STD=0.35) since low-noise data never
clears the EV/cost gate (see that file's INTEGRATION FINDING docstring
note).

Isolation note for the sign-flip test: `simulator.replay.run_replay` only
calls `manage_fn` on a bar AFTER `resolve_same_bar_ambiguity` (SL/TP) has
already run and found the position still open that bar (simulator/
replay.py). So a POSITION_CLOSED record whose outcome is specifically
POLICY_EXIT (rather than SL_HIT/TP_HIT) is, by construction, a close that
`manage()` itself decided -- not one price forced via the static SL/TP --
which is exactly the isolation the brief asks for, achieved structurally by
run_replay's own ordering rather than by hand-tuning a reversal to just
barely miss the stop."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from intelligence.bootstrap import analytical_sizing_bootstrap, analytical_sltp_bootstrap
from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import FastTierReasoner, ToolTrust
from simulator.contracts import EnvironmentTag, PositionOutcome, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r
from simulator.replay import run_replay

NOISE_STD = 0.35  # see module docstring; must clear the EV/cost gate's vol threshold


def _make_reversal_df(n_up=250, n_down=250, start_price=1900.0):
    """A clean uptrend long enough to let the real registry/reasoner build
    conviction and open a LONG, followed by a sharp, sustained downtrend
    steep enough to flip net_directional_belief's sign well before the
    dataset ends."""
    rng = np.random.default_rng(20260902)
    n = n_up + n_down
    times = pd.date_range("2020-01-06 00:00:00", periods=n, freq="1min")
    trend = np.empty(n)
    trend[:n_up] = 0.03 * np.arange(n_up)
    up_end = trend[n_up - 1] if n_up > 0 else 0.0
    # Steep reversal -- much larger per-bar slope than the uptrend, chosen to
    # flip belief sign, not merely dampen it.
    trend[n_up:] = up_end - 0.20 * np.arange(1, n_down + 1)
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
    """Same composition as test_full_fast_tier_integration.py's
    `_make_real_engine` -- every slot wired to a real Tasks 2-11
    implementation, no test doubles."""
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


def test_thesis_invalidates_on_belief_sign_flip_during_hold():
    """Mirrors test_full_fast_tier_integration.py's existing POLICY_EXIT
    assertion (Phase 2 fix wave) but on a dataset built SPECIFICALLY to
    isolate thesis-invalidation-driven exit from stop-loss-driven exit: a
    clean uptrend (to open a LONG) followed by a sharp, sustained reversal
    (to flip net_directional_belief's sign mid-hold). Because
    run_replay only calls manage_fn after SL/TP have already been checked
    and found not-yet-hit that bar (see module docstring), any POLICY_EXIT
    outcome recorded here is necessarily a manage()-driven exit, not a
    coincidental SL/TP hit."""
    df = _make_reversal_df()
    config = SimulatedExecutionConfig()
    engine = _make_real_engine(config)

    recorder = run_replay(df, engine.decide, engine.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    policy_exits = [
        r for r in records
        if r.event_type == "POSITION_CLOSED" and r.outcome == PositionOutcome.POLICY_EXIT
    ]
    assert len(policy_exits) >= 1, (
        "expected at least one manage()-driven POLICY_EXIT once the trend reverses sharply -- "
        f"outcomes seen: {sorted({str(r.outcome) for r in records if r.event_type == 'POSITION_CLOSED'})}"
    )
    # Every POLICY_EXIT must trace back to a real, non-rejected LONG/SHORT entry.
    entry_ids = {
        r.decision_id for r in records
        if r.event_type == "DECIDE" and r.action in ("LONG", "SHORT") and r.rejection_reason is None
    }
    for r in policy_exits:
        assert r.decision_id in entry_ids

    # At least one entry was actually a LONG (the uptrend segment should
    # produce one) -- confirms this test exercised the intended direction,
    # not just an incidental SHORT opened somewhere in the noise.
    entry_decides = [
        r for r in records
        if r.event_type == "DECIDE" and r.action in ("LONG", "SHORT") and r.rejection_reason is None
    ]
    assert any(r.action == "LONG" for r in entry_decides), (
        "expected at least one LONG entry against the uptrend segment of this dataset"
    )


def test_manage_holds_repeatedly_across_multiple_bars_before_exit():
    """Proves manage() is called and returns HOLD for several consecutive
    bars while the thesis remains valid, then returns EXIT once
    invalidated -- i.e. no fixed holding horizon is hardcoded. Groups MANAGE
    records by decision_id (all MANAGE calls for one still-open position
    share the same decision_id, per simulator/replay.py) and requires at
    least one position lifecycle with >=2 HOLD manage() calls followed by a
    POLICY_EXIT close."""
    df = _make_reversal_df()
    config = SimulatedExecutionConfig()
    engine = _make_real_engine(config)

    recorder = run_replay(df, engine.decide, engine.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    manage_records = [r for r in records if r.event_type == "MANAGE"]
    close_records_by_id = {
        r.decision_id: r for r in records if r.event_type == "POSITION_CLOSED"
    }

    found_multi_hold_then_exit = False
    by_decision_id: dict = {}
    for r in manage_records:
        by_decision_id.setdefault(r.decision_id, []).append(r.action)

    for decision_id, actions in by_decision_id.items():
        close = close_records_by_id.get(decision_id)
        if close is None or close.outcome != PositionOutcome.POLICY_EXIT:
            continue
        # actions is the ordered sequence of manage() calls for this
        # position's lifecycle; the final call must be the one that decided
        # EXIT (recorded on the same bar as the POSITION_CLOSED record).
        n_holds_before_exit = sum(1 for a in actions if a == "HOLD")
        if n_holds_before_exit >= 2 and actions[-1] == "EXIT":
            found_multi_hold_then_exit = True
            break

    assert found_multi_hold_then_exit, (
        "expected at least one position lifecycle with >=2 consecutive HOLD manage() "
        "calls followed by an EXIT-driven POLICY_EXIT close -- if this fails either the "
        "reassessment loop exits too eagerly (fixed short horizon) or the reversal in "
        "this dataset never gives the thesis a chance to survive multiple bars first"
    )


# --- Cross-reference notes (mandate items already covered, no new test code) --
#
# Credit assignment: tests/intelligence/test_credit_assignment.py::
# test_dissenting_load_bearing_source_gets_opposite_credit and
# ::test_short_trade_credit_is_relative_to_the_short_direction (Phase 2 fix
# wave) already exercise this adversarial property (a dissenting load-bearing
# source receives credit of the opposite sign; short-trade credit is relative
# to the short direction, not naively long-biased). Confirmed passing via:
#   .venv/bin/pytest tests/intelligence/test_credit_assignment.py -v
#
# Causality: tests/intelligence/test_causal_memory_boundaries.py::
# test_credit_assignment_truncated_vs_full_stream_identical_for_common_prefix
# (Task 14 fix) already proves credit assignment never uses information from
# beyond the truncation point (no look-ahead / no future leakage into past
# credit). Confirmed passing via:
#   .venv/bin/pytest tests/intelligence/test_causal_memory_boundaries.py -v
