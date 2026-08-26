"""tests/simulator/test_experience.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from simulator.contracts import EnvironmentTag, PositionOutcome
from simulator.experience import ExperienceRecord, ExperienceRecorder, write_tag_guard


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


if __name__ == "__main__":
    test_recorder_stores_records_in_order()
    test_write_tag_guard_allows_matching_tag()
    test_write_tag_guard_rejects_mismatched_tag()
    test_experience_record_captures_position_closed_event()
    print("tests/simulator/test_experience.py: OK")
