"""tests/research/test_phase4_trajectory_assembly.py
Synthetic-data unit test for research/phase4_trajectory_assembly.py. Builds
ExperienceRecord lists by hand (not via run_replay) so the assembly logic
itself is tested in isolation: right observations, right order, right
terminal outcome, and no leakage of one trajectory's records into another's
row when two trajectories' records are interleaved in the record stream
(as they would be for two sequential trades in a real replay)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta

from simulator.contracts import EnvironmentTag, PositionOutcome
from simulator.experience import ExperienceRecord
from research.phase4_trajectory_assembly import assemble_trajectories, trajectories_to_rows

TAG = EnvironmentTag.SIMULATED_TRAINING
T0 = datetime(2020, 1, 6, 10, 0, 0)


def _rec(event_type, ts, decision_id=None, observation_features=None, action=None,
         realized_pnl=None, outcome=None, cost_r=None):
    return ExperienceRecord(
        environment_tag=TAG, timestamp=ts, event_type=event_type,
        market_state_snapshot={"mid": 1500.0, "spread": 20.0}, position_view=None, action=action,
        account_state={}, realized_pnl=realized_pnl, cost_amount=None, outcome=outcome,
        gap_type="NORMAL", cost_r=cost_r, observation_features=observation_features,
        decision_id=decision_id,
    )


def test_single_trajectory_assembled_in_order_with_terminal_outcome():
    records = [
        _rec("DECIDE", T0, decision_id="d1", observation_features={"f": 1}, action="LONG"),
        _rec("MANAGE", T0 + timedelta(minutes=1), decision_id="d1", observation_features={"f": 2}),
        _rec("MANAGE", T0 + timedelta(minutes=2), decision_id="d1", observation_features={"f": 3}),
        _rec("POSITION_CLOSED", T0 + timedelta(minutes=3), decision_id="d1",
             realized_pnl=12.5, outcome=PositionOutcome.TP_HIT, cost_r=0.3),
    ]
    trajectories = assemble_trajectories(records)
    assert len(trajectories) == 1
    t = trajectories[0]
    assert t.decision_id == "d1"
    assert t.decide_observation_features == {"f": 1}
    assert t.manage_observation_sequence == [{"f": 2}, {"f": 3}]
    assert t.full_observation_sequence() == [{"f": 1}, {"f": 2}, {"f": 3}]
    assert t.realized_pnl == 12.5
    assert t.outcome == PositionOutcome.TP_HIT
    assert t.cost_r == 0.3


def test_no_trade_decide_records_are_skipped():
    records = [
        _rec("DECIDE", T0, decision_id=None, observation_features={"f": 99}, action="NO_TRADE"),
    ]
    assert assemble_trajectories(records) == []


def test_two_interleaved_trajectories_do_not_leak_into_each_other():
    # Simulate two sequential trades whose records could, in a denser replay,
    # be interleaved with unrelated NO_TRADE DECIDE records between them --
    # here we directly interleave trajectory-A and trajectory-B records to
    # stress-test that bucketing by decision_id (not by list position)
    # correctly isolates each trajectory.
    records = [
        _rec("DECIDE", T0, decision_id="dA", observation_features={"f": "A0"}, action="LONG"),
        _rec("MANAGE", T0 + timedelta(minutes=1), decision_id="dA", observation_features={"f": "A1"}),
        _rec("POSITION_CLOSED", T0 + timedelta(minutes=2), decision_id="dA",
             realized_pnl=-3.0, outcome=PositionOutcome.SL_HIT, cost_r=-0.1),
        _rec("DECIDE", T0 + timedelta(minutes=3), decision_id="dB", observation_features={"f": "B0"}, action="SHORT"),
        _rec("MANAGE", T0 + timedelta(minutes=4), decision_id="dB", observation_features={"f": "B1"}),
        _rec("MANAGE", T0 + timedelta(minutes=5), decision_id="dB", observation_features={"f": "B2"}),
        _rec("POSITION_CLOSED", T0 + timedelta(minutes=6), decision_id="dB",
             realized_pnl=7.0, outcome=PositionOutcome.TP_HIT, cost_r=0.2),
    ]
    trajectories = assemble_trajectories(records)
    assert len(trajectories) == 2
    traj_by_id = {t.decision_id: t for t in trajectories}

    a = traj_by_id["dA"]
    assert a.decide_observation_features == {"f": "A0"}
    assert a.manage_observation_sequence == [{"f": "A1"}]
    assert a.realized_pnl == -3.0
    assert a.outcome == PositionOutcome.SL_HIT
    # Nothing from trajectory B leaked into A's sequence.
    assert all("B" not in str(v) for step in a.full_observation_sequence() for v in step.values())

    b = traj_by_id["dB"]
    assert b.decide_observation_features == {"f": "B0"}
    assert b.manage_observation_sequence == [{"f": "B1"}, {"f": "B2"}]
    assert b.realized_pnl == 7.0
    assert b.outcome == PositionOutcome.TP_HIT
    assert all("A" not in str(v) for step in b.full_observation_sequence() for v in step.values())


def test_trajectory_with_no_manage_steps_before_close():
    records = [
        _rec("DECIDE", T0, decision_id="d1", observation_features={"f": 1}, action="LONG"),
        _rec("POSITION_CLOSED", T0 + timedelta(minutes=1), decision_id="d1",
             realized_pnl=1.0, outcome=PositionOutcome.SL_HIT, cost_r=-0.05),
    ]
    trajectories = assemble_trajectories(records)
    assert len(trajectories) == 1
    assert trajectories[0].manage_observation_sequence == []
    assert trajectories[0].full_observation_sequence() == [{"f": 1}]


def test_trajectories_to_rows_flattens_cleanly():
    records = [
        _rec("DECIDE", T0, decision_id="d1", observation_features={"f": 1}, action="LONG"),
        _rec("MANAGE", T0 + timedelta(minutes=1), decision_id="d1", observation_features={"f": 2}),
        _rec("POSITION_CLOSED", T0 + timedelta(minutes=2), decision_id="d1",
             realized_pnl=5.0, outcome=PositionOutcome.TP_HIT, cost_r=0.1),
    ]
    rows = trajectories_to_rows(assemble_trajectories(records))
    assert len(rows) == 1
    assert rows[0]["n_manage_steps"] == 1
    assert rows[0]["realized_pnl"] == 5.0


if __name__ == "__main__":
    test_single_trajectory_assembled_in_order_with_terminal_outcome()
    test_no_trade_decide_records_are_skipped()
    test_two_interleaved_trajectories_do_not_leak_into_each_other()
    test_trajectory_with_no_manage_steps_before_close()
    test_trajectories_to_rows_flattens_cleanly()
    print("tests/research/test_phase4_trajectory_assembly.py: OK")
