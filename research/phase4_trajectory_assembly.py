"""research/phase4_trajectory_assembly.py
Phase 4 Section 10/26(b): a NEW, additive orchestration module above
run_replay(). Given a completed replay's ExperienceRecorder (or its
all_records() list), assembles full trajectories keyed by the Phase 4
decision_id link: for each decision that opened a position, join

  - the originating DECIDE record's observation_features (Phase 3A field),
  - the ordered sequence of MANAGE records recorded while that position was
    open (each with its own observation_features, in chronological order),
  - the eventual POSITION_CLOSED record's realized_pnl / outcome / cost_r.

into one Trajectory object. This module does not modify simulator/engine.py,
simulator/replay.py, research/phase2_tournament.py, or research/
phase3_tournament.py -- it is a pure read-only consumer of
ExperienceRecorder.all_records().

DECIDE records that don't open a position (decision_id is None) are not
trajectories -- they carry no eventual outcome to join against, and are
skipped.
"""
from dataclasses import dataclass, field
from typing import Optional

from simulator.experience import ExperienceRecord


@dataclass
class Trajectory:
    decision_id: str
    decide_timestamp: object
    decide_observation_features: Optional[dict]
    decide_action: str
    # Chronologically ordered MANAGE records' observation_features (may be
    # empty if the position closed on the very next bar via SL/TP/liquidation
    # before any MANAGE record was generated).
    manage_observation_sequence: list = field(default_factory=list)
    close_timestamp: object = None
    realized_pnl: Optional[float] = None
    outcome: object = None
    cost_r: Optional[float] = None

    def full_observation_sequence(self) -> list:
        """The complete ordered sequence of observation_features dicts from
        open to close: the decide-time snapshot followed by every manage-time
        snapshot, in chronological order. None entries (a step that didn't
        expose observation_features) are preserved positionally rather than
        dropped, so callers can decide how to handle missing steps."""
        return [self.decide_observation_features] + list(self.manage_observation_sequence)


def assemble_trajectories(records: list) -> list:
    """Build one Trajectory per decision_id that opened a position, in the
    order those decisions were made. Read-only over `records` (typically
    recorder.all_records()) -- never mutates them."""
    if records and isinstance(records[0], ExperienceRecord):
        pass  # type hint only; duck-typed below so plain dict-like records also work

    by_decision: dict = {}
    order: list = []

    for r in records:
        decision_id = getattr(r, "decision_id", None)
        if decision_id is None:
            continue
        if decision_id not in by_decision:
            by_decision[decision_id] = {
                "decide": None, "manage": [], "close": None,
            }
            order.append(decision_id)
        bucket = by_decision[decision_id]
        if r.event_type == "DECIDE":
            bucket["decide"] = r
        elif r.event_type == "MANAGE":
            bucket["manage"].append(r)
        elif r.event_type == "POSITION_CLOSED":
            bucket["close"] = r

    trajectories = []
    for decision_id in order:
        bucket = by_decision[decision_id]
        decide = bucket["decide"]
        close = bucket["close"]
        if decide is None or close is None:
            # Incomplete linkage (should not happen for a well-formed replay,
            # but skip rather than fabricate data if it does).
            continue
        trajectories.append(Trajectory(
            decision_id=decision_id,
            decide_timestamp=decide.timestamp,
            decide_observation_features=decide.observation_features,
            decide_action=decide.action,
            manage_observation_sequence=[m.observation_features for m in bucket["manage"]],
            close_timestamp=close.timestamp,
            realized_pnl=close.realized_pnl,
            outcome=close.outcome,
            cost_r=close.cost_r,
        ))
    return trajectories


def trajectories_to_rows(trajectories: list) -> list:
    """Flatten Trajectory objects into plain dicts (one per trajectory) --
    convenient for building a pandas DataFrame without importing pandas into
    this module."""
    rows = []
    for t in trajectories:
        rows.append({
            "decision_id": t.decision_id,
            "decide_timestamp": t.decide_timestamp,
            "decide_action": t.decide_action,
            "decide_observation_features": t.decide_observation_features,
            "manage_observation_sequence": t.manage_observation_sequence,
            "n_manage_steps": len(t.manage_observation_sequence),
            "close_timestamp": t.close_timestamp,
            "realized_pnl": t.realized_pnl,
            "outcome": t.outcome,
            "cost_r": t.cost_r,
        })
    return rows
