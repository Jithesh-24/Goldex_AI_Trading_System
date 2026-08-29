"""Tests for intelligence/experience_store.py -- the read-access guard over
Phase 1's ExperienceRecorder, partitioned by environment_tag."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from intelligence.experience_store import ExperienceStore, ProtectedPartitionError
from simulator.contracts import EnvironmentTag, PositionOutcome
from simulator.experience import ExperienceRecord, ExperienceRecorder


def _record(tag, decision_id, event_type="DECIDE"):
    return ExperienceRecord(
        environment_tag=tag,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        market_state_snapshot={},
        position_view=None,
        action="LONG",
        account_state={},
        realized_pnl=None,
        cost_amount=None,
        outcome=None,
        gap_type="NORMAL",
        decision_id=decision_id,
    )


def test_construction_against_protected_oos_partition_raises_immediately():
    recorder = ExperienceRecorder()
    recorder.record(_record(EnvironmentTag.SIMULATED_OOS_TEST, "d1"))

    with pytest.raises(ProtectedPartitionError):
        ExperienceStore(recorder, EnvironmentTag.SIMULATED_OOS_TEST)


def test_construction_against_protected_partition_raises_even_if_empty():
    recorder = ExperienceRecorder()

    with pytest.raises(ProtectedPartitionError):
        ExperienceStore(recorder, EnvironmentTag.SIMULATED_OOS_TEST)


def test_normal_partition_reads_records_in_decision_id_order():
    recorder = ExperienceRecorder()
    recorder.record(_record(EnvironmentTag.SIMULATED_TRAINING, "d3"))
    recorder.record(_record(EnvironmentTag.SIMULATED_TRAINING, "d1"))
    recorder.record(_record(EnvironmentTag.SIMULATED_TRAINING, "d2"))
    # A different partition's record must not leak in.
    recorder.record(_record(EnvironmentTag.SIMULATED_VALIDATION, "d0"))

    store = ExperienceStore(recorder, EnvironmentTag.SIMULATED_TRAINING)
    records = store.records()

    assert [r.decision_id for r in records] == ["d1", "d2", "d3"]
    assert all(r.environment_tag == EnvironmentTag.SIMULATED_TRAINING for r in records)


def test_records_with_none_decision_id_sort_first_and_are_included():
    recorder = ExperienceRecorder()
    recorder.record(_record(EnvironmentTag.SIMULATED_VALIDATION, "d2"))
    recorder.record(_record(EnvironmentTag.SIMULATED_VALIDATION, None))
    recorder.record(_record(EnvironmentTag.SIMULATED_VALIDATION, "d1"))

    store = ExperienceStore(recorder, EnvironmentTag.SIMULATED_VALIDATION)
    records = store.records()

    assert [r.decision_id for r in records] == [None, "d1", "d2"]
