"""tests/simulator/test_replay.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from simulator.contracts import AccountState, EnvironmentTag, SimulatedExecutionConfig, PositionOutcome
from simulator.replay import run_replay


def _make_df(n=20, start_price=1500.0):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [start_price + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.2 for p in prices], "low": [p - 0.2 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def _always_no_trade(market_state, account):
    return ("NO_TRADE", None, None)


def test_run_replay_all_no_trade_never_opens_position():
    df = _make_df()
    config = SimulatedExecutionConfig()
    recorder = run_replay(df, _always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    assert len(records) > 0
    assert all(r.event_type == "DECIDE" for r in records)
    assert all(r.action == "NO_TRADE" for r in records)


def _open_one_long_then_hold_forever():
    state = {"opened": False}

    def decide(market_state, account):
        if not state["opened"]:
            state["opened"] = True
            return ("LONG", None, None)
        return ("NO_TRADE", None, None)

    def manage(market_state, position_view, account):
        return "HOLD"

    return decide, manage


def test_run_replay_opens_and_force_closes_at_end_of_data():
    df = _make_df()
    config = SimulatedExecutionConfig()
    decide, manage = _open_one_long_then_hold_forever()
    recorder = run_replay(df, decide, manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    closed = [r for r in records if r.event_type == "POSITION_CLOSED"]
    assert len(closed) == 1
    assert closed[0].outcome == PositionOutcome.END_OF_REPLAY_FORCED_CLOSE


def test_run_replay_reopens_immediately_after_exit_no_cooldown():
    df = _make_df(n=10)
    config = SimulatedExecutionConfig()
    call_count = {"decides_while_flat": 0}

    def decide(market_state, account):
        call_count["decides_while_flat"] += 1
        if call_count["decides_while_flat"] in (1, 3):
            return ("LONG", None, None)
        return ("NO_TRADE", None, None)

    def manage(market_state, position_view, account):
        return "EXIT"  # exit on the very next bar after opening

    recorder = run_replay(df, decide, manage, config, EnvironmentTag.SIMULATED_TRAINING)
    closed = [r for r in recorder.all_records() if r.event_type == "POSITION_CLOSED"]
    assert len(closed) >= 2  # opened, exited, reopened, exited again -- with no forced gap


def test_run_replay_force_closes_a_position_opened_on_the_very_last_bar():
    """Regression: the in-loop END_OF_REPLAY_FORCED_CLOSE branch only covered
    positions already open at bar n-1. A position opened BY decide_fn on bar
    n-1 was left open forever with no close record."""
    df = _make_df(n=6)
    config = SimulatedExecutionConfig()
    calls = {"n": 0}

    def decide(market_state, account):
        calls["n"] += 1
        return ("LONG", None, None) if calls["n"] == len(df) else ("NO_TRADE", None, None)

    recorder = run_replay(df, decide, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    closed = [r for r in recorder.all_records() if r.event_type == "POSITION_CLOSED"]
    assert len(closed) == 1
    assert closed[0].outcome == PositionOutcome.END_OF_REPLAY_FORCED_CLOSE
    assert closed[0].account_state["open_position_id"] is None
    assert closed[0].account_state["margin_used"] == 0.0
    assert closed[0].account_state["exposure"] == 0.0


def test_run_replay_liquidates_an_unstopped_position_on_a_catastrophic_move():
    """Regression: equity was frozen while a position was open, so the
    LIQUIDATION safety net could never fire for a policy running with no SL."""
    n = 8
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    # Flat, then a collapse far beyond the account's equity.
    prices = [1500.0] * 3 + [200.0] * (n - 3)
    df = pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.2 for p in prices], "low": [p - 0.2 for p in prices],
        "close": prices, "tick_volume": [10] * n, "spread": [20.0] * n,
    })
    config = SimulatedExecutionConfig(risk_fraction_of_equity=1.0)  # size = equity units
    calls = {"n": 0}

    def decide(market_state, account):
        calls["n"] += 1
        return ("LONG", None, None) if calls["n"] == 1 else ("NO_TRADE", None, None)

    recorder = run_replay(df, decide, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    closed = [r for r in recorder.all_records() if r.event_type == "POSITION_CLOSED"]
    assert len(closed) == 1
    assert closed[0].outcome == PositionOutcome.LIQUIDATION


def test_run_replay_data_gap_blocks_new_entries():
    """Data gap (mid-week 3+ hour gap) must block policy-requested LONG/SHORT.
    The decision_fn asks to go LONG at the bar immediately following the gap,
    but the action is forced to NO_TRADE, and gap_type is tagged DATA_GAP."""
    from datetime import datetime, timedelta, timezone

    # Build two normal bars, then a 3-hour gap (10800 seconds, way over 90-second tolerance),
    # then one more bar. The gap occurs between index 1 and 2.
    times = [
        datetime(2020, 1, 6, 10, 0, 0, tzinfo=timezone.utc),  # Monday 10:00
        datetime(2020, 1, 6, 10, 1, 0, tzinfo=timezone.utc),  # Monday 10:01
        datetime(2020, 1, 6, 13, 1, 0, tzinfo=timezone.utc),  # Monday 13:01 (3 hours later, mid-week)
        datetime(2020, 1, 6, 13, 2, 0, tzinfo=timezone.utc),  # Monday 13:02
    ]
    prices = [1500.0, 1500.1, 1500.2, 1500.3]
    df = pd.DataFrame({
        "time": times,
        "open": prices,
        "high": [p + 0.2 for p in prices],
        "low": [p - 0.2 for p in prices],
        "close": [p + 0.05 for p in prices],
        "tick_volume": [10] * 4,
        "spread": [20.0] * 4,
    })

    def decide_always_long(market_state, account):
        # Always request LONG
        return ("LONG", None, None)

    def manage_exit_immediately(market_state, position_view, account):
        # Exit on the very next bar, so we can test DECIDE records for each bar
        return "EXIT"

    config = SimulatedExecutionConfig()
    recorder = run_replay(df, decide_always_long, manage_exit_immediately, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    decide_records = [r for r in records if r.event_type == "DECIDE"]

    # We should have 3 DECIDE records:
    # - Bar 0: DECIDE, LONG opens, then MANAGE EXIT closes it
    # - Bar 1: MANAGE EXIT (no DECIDE since position was open)
    # - Bar 2: DECIDE, DATA_GAP blocks LONG -> NO_TRADE
    # - Bar 3: DECIDE, LONG allowed
    assert len(decide_records) == 3

    # Bar 0: LONG allowed (i == 0, gap_type == NORMAL)
    assert decide_records[0].gap_type == "NORMAL"
    assert decide_records[0].action == "LONG"

    # Bar 2 (13:01): LONG blocked because of DATA_GAP (3-hour gap from bar 1)
    # This is decide_records[1] because bar 1 has no DECIDE (position was still open)
    assert decide_records[1].gap_type == "DATA_GAP"
    assert decide_records[1].action == "NO_TRADE"  # forced NO_TRADE despite decision_fn returning LONG

    # Bar 3 (13:02): LONG allowed again (small 1-minute gap after data gap)
    assert decide_records[2].gap_type == "NORMAL"
    assert decide_records[2].action == "LONG"


def test_run_replay_uses_caller_supplied_size_when_decide_fn_returns_a_4_tuple():
    """DecideFn may optionally return a 4th tuple element, size. When present
    (and not None), the opened position must use it instead of the engine's
    equity * risk_fraction_of_equity default."""
    df = _make_df(n=6)
    config = SimulatedExecutionConfig(risk_fraction_of_equity=0.5)  # default would be equity * 0.5
    explicit_size = 3.25
    calls = {"n": 0}

    def decide(market_state, account):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("LONG", None, None, explicit_size)
        return ("NO_TRADE", None, None)

    opened_position_views = []

    def manage(market_state, position_view, account):
        opened_position_views.append(position_view)
        return "HOLD"

    recorder = run_replay(df, decide, manage, config, EnvironmentTag.SIMULATED_TRAINING)
    assert len(opened_position_views) > 0
    assert opened_position_views[0].size == explicit_size
    # Sanity: the risk-fraction default (equity * 0.5, equity starts >> 3.25)
    # would clearly differ from explicit_size, so this is a discriminating test.
    account = AccountState.initial(config, df.iloc[0]["time"].to_pydatetime())
    assert explicit_size != account.equity * config.risk_fraction_of_equity


if __name__ == "__main__":
    test_run_replay_all_no_trade_never_opens_position()
    test_run_replay_force_closes_a_position_opened_on_the_very_last_bar()
    test_run_replay_liquidates_an_unstopped_position_on_a_catastrophic_move()
    test_run_replay_opens_and_force_closes_at_end_of_data()
    test_run_replay_reopens_immediately_after_exit_no_cooldown()
    test_run_replay_data_gap_blocks_new_entries()
    test_run_replay_uses_caller_supplied_size_when_decide_fn_returns_a_4_tuple()
    print("tests/simulator/test_replay.py: OK")
