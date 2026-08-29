"""tests/simulator/test_experience.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import pytest

from contracts.market_state import MarketState
from simulator.contracts import EnvironmentTag, PositionOutcome, SimulatedExecutionConfig
from simulator.experience import ExperienceRecord, ExperienceRecorder, write_tag_guard
from simulator.replay import run_replay


def _record(tag=EnvironmentTag.SIMULATED_TRAINING):
    return ExperienceRecord(
        environment_tag=tag, timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc), event_type="DECIDE",
        market_state_snapshot={"mid": 1500.0}, position_view=None, action="NO_TRADE",
        account_state={"balance": 10000.0}, realized_pnl=None, cost_amount=None, outcome=None,
        gap_type="NORMAL",
    )


def test_recorder_stores_records_in_order():
    recorder = ExperienceRecorder()
    r1 = _record()
    r2 = _record()
    recorder.record(r1)
    recorder.record(r2)
    assert recorder.all_records() == [r1, r2]


def test_write_tag_guard_allows_matching_tag():
    write_tag_guard(EnvironmentTag.SIMULATED_TRAINING, _record(EnvironmentTag.SIMULATED_TRAINING))


def test_write_tag_guard_rejects_mismatched_tag():
    with pytest.raises(ValueError):
        write_tag_guard(EnvironmentTag.SIMULATED_OOS_TEST, _record(EnvironmentTag.SIMULATED_TRAINING))


def test_experience_record_captures_position_closed_event():
    record = ExperienceRecord(
        environment_tag=EnvironmentTag.SIMULATED_TRAINING, timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_type="POSITION_CLOSED", market_state_snapshot={"mid": 1510.0},
        position_view={"position_id": "p1"}, action=None, account_state={"balance": 10005.0},
        realized_pnl=15.0, cost_amount=2.0, outcome=PositionOutcome.POLICY_EXIT, gap_type="NORMAL",
    )
    assert record.outcome == PositionOutcome.POLICY_EXIT
    assert record.realized_pnl == 15.0


def _make_df(n=10):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def test_market_state_snapshot_contains_full_market_state_field_set():
    df = _make_df()
    config = SimulatedExecutionConfig()

    def always_no_trade(market_state, account):
        return ("NO_TRADE", None, None)

    recorder = run_replay(df, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    assert records, "expected at least one recorded record"
    expected_fields = set(MarketState.model_fields.keys())
    for record in records:
        assert set(record.market_state_snapshot.keys()) == expected_fields, (
            f"market_state_snapshot fields {set(record.market_state_snapshot.keys())} "
            f"do not match full MarketState field set {expected_fields}"
        )
    # Sanity: not just mid/spread -- e.g. identity/feed-health fields are present too.
    assert "symbol" in records[0].market_state_snapshot
    assert "feed_health" in records[0].market_state_snapshot


def test_account_state_dict_includes_realized_pnl_drawdown_currency():
    df = _make_df()
    config = SimulatedExecutionConfig()

    def always_no_trade(market_state, account):
        return ("NO_TRADE", None, None)

    recorder = run_replay(df, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    assert records, "expected at least one recorded record"
    for record in records:
        assert "realized_pnl_total" in record.account_state
        assert "drawdown" in record.account_state
        assert "currency" in record.account_state


if __name__ == "__main__":
    test_recorder_stores_records_in_order()
    test_write_tag_guard_allows_matching_tag()
    test_write_tag_guard_rejects_mismatched_tag()
    test_experience_record_captures_position_closed_event()
    test_market_state_snapshot_contains_full_market_state_field_set()
    test_account_state_dict_includes_realized_pnl_drawdown_currency()
    print("tests/simulator/test_experience.py: OK")
