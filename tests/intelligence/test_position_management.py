"""tests/intelligence/test_position_management.py -- Task 8: continuous
position reassessment (FastTierDecisionEngine.manage()). Runs through
simulator.replay.run_replay end-to-end (not unit-level fakes) so the
resulting ExperienceRecord/PositionOutcome is the real one Phase 1's engine
produces -- confirms manage()'s "EXIT" return actually causes an early
POLICY_EXIT close, and that a thesis which never invalidates holds through
to a normal forced/SL/TP-driven close.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timezone

import pandas as pd

from contracts.market_state import DataQuality, FeedHealthState, MarketState
from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import Hypothesis, ToolTrust
from simulator.contracts import AccountState, EnvironmentTag, PositionOutcome, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r
from simulator.replay import run_replay


def _make_df(n=20, start_price=1500.0, step=0.0):
    """Flat-ish synthetic price path -- deliberately never travels far
    enough on its own to hit a wide SL/TP within `n` bars, so any early
    close in these tests is attributable to manage()'s "EXIT", not price
    action hitting the static SL/TP."""
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [start_price + i * step for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.2 for p in prices], "low": [p - 0.2 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [0.2] * n,
    })


def _wide_sltp(offset=50.0):
    """SL/TP wide enough that this synthetic flat-ish price path will never
    hit either within the test's bar count -- isolates manage()'s "EXIT"
    as the only possible early-close mechanism in these tests."""
    def sltp_bootstrap(hyp, market_state):
        mid = market_state.mid
        if hyp.net_directional_belief >= 0:
            return mid - offset, mid + offset
        return mid + offset, mid - offset
    return sltp_bootstrap


def _fixed_sizing(size=0.01):
    return lambda hyp, account: size


def _cheap_cost_gate(market_state, candidate_sl_distance_r):
    return round_trip_cost_r(market_state, candidate_sl_distance_r, max_staleness_seconds=float("inf"))


class _SequenceReasoner:
    """Test double: returns a fixed "entry" Hypothesis until told the
    position has actually opened (`self.entered` flipped by
    `_decide_and_flip` below), then a fixed "manage" Hypothesis for every
    call after that. Deliberately NOT keyed on a raw call count -- early
    bars in a fresh replay legitimately produce a few NO_TRADE decide()
    calls first (e.g. `market_state.realized_vol_60s` isn't available on
    bar 0, so `decide()`'s EV/cost gate can't compute a cost yet and
    conservatively returns NO_TRADE -- see decide()'s docstring), so a raw
    "1st call == entry" assumption is wrong: it would use the "manage" hyp
    for the actual entry and never trigger the invalidation this test
    wants to exercise. Keying off the real DECIDE outcome instead makes
    this robust to however many NO_TRADE warm-up bars the real registry
    integration produces."""

    def __init__(self, entry_hyp: Hypothesis, manage_hyp: Hypothesis):
        self.entry_hyp = entry_hyp
        self.manage_hyp = manage_hyp
        self.entered = False

    def hypothesis(self, closes_so_far, market_state, trust):
        return self.manage_hyp if self.entered else self.entry_hyp


def _decide_and_flip(engine, reasoner):
    """Wraps engine.decide so that, the moment it actually opens a
    position (a non-NO_TRADE action), the reasoner test double switches
    from its "entry" hypothesis to its "manage" hypothesis for every
    subsequent call -- i.e. every MANAGE-step reassessment sees the
    "manage" hyp, regardless of how many NO_TRADE warm-up DECIDE calls
    preceded the real entry."""
    def decide_fn(market_state, account):
        result = engine.decide(market_state, account)
        if result[0] != "NO_TRADE":
            reasoner.entered = True
        return result
    return decide_fn


def _make_engine(reasoner, sltp=None):
    registry = build_default_registry()
    trust = ToolTrust()
    return FastTierDecisionEngine(
        registry=registry,
        trust=trust,
        reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate,
        sizing_bootstrap=_fixed_sizing(),
        sltp_bootstrap=sltp or _wide_sltp(),
    )


def _find_position_closed(records):
    closed = [r for r in records if r.event_type == "POSITION_CLOSED"]
    assert len(closed) == 1, f"expected exactly one POSITION_CLOSED record, got {len(closed)}"
    return closed[0]


def test_thesis_reversal_triggers_exit_before_static_sltp():
    """Entry belief is strongly LONG; on the very next MANAGE step the
    reasoner's fresh belief has flipped to strongly SHORT (a clean sign
    flip). With a wide SL/TP that this flat synthetic price path would
    never otherwise hit, the only possible close mechanism here is
    manage()'s "EXIT" -- confirm it fires, clears the thesis immediately,
    and the resulting close is POLICY_EXIT well before end-of-replay."""
    df = _make_df(n=30)
    entry_hyp = Hypothesis(0.8, 0.05, [("momentum_scalar", 0, 0.8)])
    reversed_hyp = Hypothesis(-0.8, 0.05, [("momentum_scalar", 0, -0.8)])
    reasoner = _SequenceReasoner(entry_hyp, reversed_hyp)
    engine = _make_engine(reasoner)

    recorder = run_replay(df, _decide_and_flip(engine, reasoner), engine.manage, SimulatedExecutionConfig(),
                           EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    close_record = _find_position_closed(records)
    assert close_record.outcome == PositionOutcome.POLICY_EXIT

    manage_records = [r for r in records if r.event_type == "MANAGE"]
    assert len(manage_records) >= 1
    assert manage_records[0].action == "EXIT"
    # Only one MANAGE record: the position closed on the very first
    # reassessment, well before the n=30-bar replay's forced close.
    assert len(manage_records) == 1

    # Thesis must be cleared the moment EXIT is decided, not deferred.
    assert engine.open_thesis is None


def test_thesis_magnitude_collapse_triggers_exit():
    """Entry belief strongly LONG; fresh belief stays the same sign but its
    magnitude collapses below half of entry conviction -- confirm this also
    invalidates the thesis (the second, independent invalidation
    condition), not just an outright sign flip."""
    df = _make_df(n=30)
    entry_hyp = Hypothesis(0.8, 0.05, [("momentum_scalar", 0, 0.8)])
    collapsed_hyp = Hypothesis(0.1, 0.05, [("momentum_scalar", 0, 0.1)])  # 0.1 < 0.5 * 0.8
    reasoner = _SequenceReasoner(entry_hyp, collapsed_hyp)
    engine = _make_engine(reasoner)

    recorder = run_replay(df, _decide_and_flip(engine, reasoner), engine.manage, SimulatedExecutionConfig(),
                           EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    close_record = _find_position_closed(records)
    assert close_record.outcome == PositionOutcome.POLICY_EXIT
    assert engine.open_thesis is None


def test_consistent_thesis_holds_through_to_forced_close():
    """Entry belief and every subsequent reassessed belief stay strongly
    LONG and consistent (same sign, no magnitude collapse) -- manage() must
    return "HOLD" every MANAGE step, and with a static SL/TP wide enough
    to never be hit on this flat synthetic path, the position survives to
    the replay's forced end-of-data close, never a POLICY_EXIT."""
    df = _make_df(n=15)
    entry_hyp = Hypothesis(0.8, 0.05, [("momentum_scalar", 0, 0.8)])
    consistent_hyp = Hypothesis(0.75, 0.05, [("momentum_scalar", 0, 0.75)])  # same sign, > half magnitude
    reasoner = _SequenceReasoner(entry_hyp, consistent_hyp)
    engine = _make_engine(reasoner)

    recorder = run_replay(df, _decide_and_flip(engine, reasoner), engine.manage, SimulatedExecutionConfig(),
                           EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    close_record = _find_position_closed(records)
    assert close_record.outcome == PositionOutcome.END_OF_REPLAY_FORCED_CLOSE

    manage_records = [r for r in records if r.event_type == "MANAGE"]
    assert len(manage_records) > 0
    for r in manage_records:
        assert r.action == "HOLD"


def test_manage_returns_hold_when_no_thesis_on_file():
    """Defensive fallback: manage() must never fabricate an EXIT when it
    has no thesis to compare against."""
    registry = build_default_registry()
    engine = FastTierDecisionEngine(
        registry=registry, trust=ToolTrust(),
        reasoner=_SequenceReasoner(Hypothesis(0.8, 0.05, []), Hypothesis(-0.8, 0.05, [])),
        ev_cost_gate=_cheap_cost_gate, sizing_bootstrap=_fixed_sizing(), sltp_bootstrap=_wide_sltp(),
    )
    now = datetime.now(timezone.utc)
    ms = MarketState(
        symbol="XAUUSD", source="synthetic_replay", sequence=1,
        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
        bid=1999.9, ask=2000.1, mid=2000.0, spread=0.2,
        data_quality=DataQuality.VALID, tick_count_60s=10, tick_count_300s=50,
        tick_rate_per_sec=1.0, market_closed=False, feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5, realized_vol_60s=0.0005,
    )
    config = SimulatedExecutionConfig()
    account = AccountState.initial(config, now)
    assert engine.open_thesis is None
    assert engine.manage(ms, None, account) == "HOLD"


if __name__ == "__main__":
    test_thesis_reversal_triggers_exit_before_static_sltp()
    test_thesis_magnitude_collapse_triggers_exit()
    test_consistent_thesis_holds_through_to_forced_close()
    test_manage_returns_hold_when_no_thesis_on_file()
    print("tests/intelligence/test_position_management.py: OK")
