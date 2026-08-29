"""tests/intelligence/test_thesis.py -- Task 7: thesis memory lifecycle,
exercised through an actual simulator.replay.run_replay open -> hold ->
close sequence (not a unit test in isolation), since the real question is
whether FastTierDecisionEngine's self._open_thesis attribute is correctly
wired into Phase 1's actual replay loop:

  1. `engine.open_thesis` is None before any DECIDE call ever happens.
  2. It is populated with exactly the Hypothesis's load_bearing_sources the
     instant a DECIDE call returns LONG/SHORT.
  3. It stays populated, unchanged, across every MANAGE ("HOLD") call while
     the position remains open.
  4. Once that position closes (here: a TP hit resolved by
     simulator.replay's own same-bar-ambiguity check, a path
     FastTierDecisionEngine's decide()/manage() never see directly), the
     stale thesis is discarded by the next DECIDE call -- the one place
     `simulator.replay.run_replay` is structurally guaranteed to invoke
     while flat (simulator/replay.py:93-94) -- confirming the "no leakage
     across positions" guarantee holds for a real replay pass.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.fast_tier import Hypothesis
from intelligence.thesis import Thesis
from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.replay import run_replay
from tests.intelligence.test_decision_engine import (
    _StubReasoner, _cheap_cost_gate, _fixed_sizing, _fixed_sltp,
)


def _make_df_with_tp_hit_at(tp_hit_index, n=8, base_price=1500.0):
    """A small upward drift (needed so `market_state_builder.build_snapshot`
    can compute a nonzero `realized_vol_60s` from row 2 onward --
    `decide()` refuses to act on a candidate SL distance while that's None
    or <= 0) everywhere except a spike in row `tp_hit_index`'s `high` large
    enough to clear `_fixed_sltp(mid_offset=1.0)`'s TP for a LONG opened
    around bar 2's mid. Every other row's high/low stays tight around its
    own price so no earlier bar can accidentally trip SL/TP first."""
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [base_price + i * 0.05 for i in range(n)]
    highs = [p + 0.2 for p in prices]
    lows = [p - 0.2 for p in prices]
    highs[tp_hit_index] = prices[tp_hit_index] + 5.0  # comfortably clears TP
    return pd.DataFrame({
        "time": times, "open": prices, "high": highs, "low": lows,
        "close": [p + 0.02 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def _make_engine_with_load_bearing():
    load_bearing = [("momentum_scalar", 2, 0.42), ("garch_conditional_variance", 1, -0.11)]
    hyp = Hypothesis(0.8, 0.05, load_bearing)
    reasoner = _StubReasoner(hyp)
    engine = FastTierDecisionEngine(
        registry=None, trust=None, reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate, sizing_bootstrap=_fixed_sizing(),
        sltp_bootstrap=_fixed_sltp(),
    )
    return engine, load_bearing


def test_thesis_is_none_before_any_decision():
    engine, _ = _make_engine_with_load_bearing()
    assert engine.open_thesis is None


def test_thesis_lifecycle_through_real_replay_open_hold_close():
    engine, load_bearing = _make_engine_with_load_bearing()
    assert engine.open_thesis is None

    # Rows 0-1: DECIDE is called but stays flat/NO_TRADE -- realized_vol_60s
    # needs at least 2 prior bars to be computable at all (see
    # market_state_builder.build_snapshot's VOL_LOOKBACK_BARS window), so
    # decide()'s own EV/cost gate can't act yet. Row 2 opens LONG. Rows 3-4
    # are genuine HOLD bars (manage() called with the position still open).
    # Row 5's high spike closes it via TP_HIT before manage() is ever
    # invoked for that bar. Row 6 is the next flat DECIDE call that must
    # observe the thesis already cleared (and reopens a fresh LONG). Row 7
    # is the trailing last bar, closing row 6's position via an ordinary
    # end-of-replay forced close.
    df = _make_df_with_tp_hit_at(tp_hit_index=5, n=8)

    observed_at_manage = []  # snapshot of engine.open_thesis on every MANAGE call
    observed_at_decide_entry = []  # snapshot of engine.open_thesis at the START of every decide() call

    orig_decide = engine.decide
    orig_manage = engine.manage
    orig_clear_thesis = engine.clear_thesis
    clear_thesis_calls = []

    def decide_spy(market_state, account):
        observed_at_decide_entry.append(engine.open_thesis)
        return orig_decide(market_state, account)

    def manage_spy(market_state, position_view, account):
        observed_at_manage.append(engine.open_thesis)
        return orig_manage(market_state, position_view, account)

    def clear_thesis_spy():
        clear_thesis_calls.append(True)
        return orig_clear_thesis()

    engine.clear_thesis = clear_thesis_spy

    config = SimulatedExecutionConfig()
    recorder = run_replay(df, decide_spy, manage_spy, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    # Sanity: the sequence actually exercised what this test claims.
    decide_records = [r for r in records if r.event_type == "DECIDE"]
    manage_records = [r for r in records if r.event_type == "MANAGE"]
    closed_records = [r for r in records if r.event_type == "POSITION_CLOSED"]
    assert [r.action for r in decide_records] == ["NO_TRADE", "NO_TRADE", "LONG", "LONG"]
    assert len(manage_records) == 2  # rows 3 and 4 -- genuine HOLD bars
    assert len(closed_records) == 2  # TP_HIT at row 5, forced close at end of replay

    # decide() calls #1-3 (rows 0, 1, 2): all entered flat (None). Rows 0-1
    # stay NO_TRADE; row 2 is the one that opens LONG.
    assert observed_at_decide_entry[0] is None
    assert observed_at_decide_entry[1] is None
    assert observed_at_decide_entry[2] is None

    # manage() calls (rows 3, 4): thesis stays populated and unchanged,
    # with exactly the entry Hypothesis's load_bearing_sources.
    for snap in observed_at_manage:
        assert isinstance(snap, Thesis)
        assert snap.load_bearing_sources == load_bearing
        assert snap.entry_belief == 0.8

    # decide() call #4 (row 6): entered flat again after the row-5 TP_HIT,
    # which this engine's decide()/manage() seam never directly observed --
    # so the spy still sees the stale row-2 thesis at the very top of the
    # call, before decide()'s own self-heal clears it a few lines in.
    assert observed_at_decide_entry[3] is not None
    assert observed_at_decide_entry[3].load_bearing_sources == load_bearing
    assert observed_at_decide_entry[3].entry_timestamp.replace(tzinfo=None) == df.iloc[2]["time"].to_pydatetime()
    # And clear_thesis() was actually invoked exactly once to do it (proving
    # the self-heal branch ran, not just that a later assignment happened
    # to overwrite the stale value).
    assert len(clear_thesis_calls) == 1

    # And after that decide() opens a fresh LONG, the *new* thesis reflects
    # this position's own entry, not a leftover from the first.
    assert engine.open_thesis is not None
    assert engine.open_thesis.load_bearing_sources == load_bearing
    assert engine.open_thesis.entry_timestamp is not None
    assert engine.open_thesis.entry_timestamp.replace(tzinfo=None) == df.iloc[6]["time"].to_pydatetime()


def test_clear_thesis_is_idempotent_and_public():
    engine, _ = _make_engine_with_load_bearing()
    engine.clear_thesis()  # no-op while already None -- must not raise
    assert engine.open_thesis is None
