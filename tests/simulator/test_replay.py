"""tests/simulator/test_replay.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig, PositionOutcome
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


if __name__ == "__main__":
    test_run_replay_all_no_trade_never_opens_position()
    test_run_replay_force_closes_a_position_opened_on_the_very_last_bar()
    test_run_replay_liquidates_an_unstopped_position_on_a_catastrophic_move()
    test_run_replay_opens_and_force_closes_at_end_of_data()
    test_run_replay_reopens_immediately_after_exit_no_cooldown()
    print("tests/simulator/test_replay.py: OK")
