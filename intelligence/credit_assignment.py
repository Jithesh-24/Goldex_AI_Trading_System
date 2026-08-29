"""intelligence/credit_assignment.py -- Task 9: trade credit assignment
(mandate Section 7).

The mandate flags a "Phase 3's previous credit-assignment bug" by name.
The controller could not find any such bug documented anywhere in this
repo's git history or docs, and is not fabricating one to design against.
Instead this module is built to the mandate's general correctness
requirement stated alongside that reference: never credit an outcome to a
tool/decision that did not cause it. Concretely that means:

  1. A trade's outcome must only update `ToolTrust` for the exact
     `(source_name, context_bucket)` pairs that were load-bearing for THAT
     trade's entry thesis (Task 7's `Thesis.load_bearing_sources`) -- never
     for some other source that happened to be active (computed,
     non-zero-confidence) at the same time but wasn't part of this trade's
     reasoning, and never for a source that was load-bearing for a
     DIFFERENT, nearby trade.
  2. A rejected entry (Task 6's `rejection_reason`) contributes zero credit,
     to zero sources, full stop.

RECONSTRUCTING ONE TRADE'S LIFECYCLE FROM ExperienceRecords
-------------------------------------------------------------
Phase 1's `simulator/replay.py` (confirmed by reading it in full) is the
single producer of `ExperienceRecord`s and establishes the linking
convention this module relies on:

  - `decision_id` is a fresh `uuid.uuid4()` string, generated ONLY on a
    DECIDE record whose action is LONG/SHORT (i.e. only when a position
    might open). It is `None` on every NO_TRADE DECIDE record -- there is
    nothing to link.
  - If `open_position()` rejects that entry, `replay.py` stamps
    `rejection_reason` onto that same DECIDE record but resets its local
    `current_decision_id` to `None` before continuing -- so no MANAGE or
    POSITION_CLOSED record ever carries that decision_id. A rejected
    entry's `decision_id` therefore appears on exactly one record in the
    whole stream: itself.
  - If the entry is accepted, every subsequent MANAGE record (while that
    position stays open) and the eventual POSITION_CLOSED record that
    closes it are stamped with that SAME `decision_id`, until the position
    closes and `current_decision_id` resets to `None` again.

So `decision_id` is exactly the grouping key for "one trade's lifecycle":
grouping all records (excluding `decision_id is None`) by `decision_id`
recovers, for each key, precisely {the DECIDE that opened it (or attempted
to and was rejected), zero or more MANAGE records, and at most one
POSITION_CLOSED record} -- never records from a different, unrelated trade,
even when two trades' lifecycles are adjacent or interleaved in time,
because each decision_id is a fresh random UUID scoped by construction to
one open_position()/close_position() pair.

WHY A Thesis MUST BE SUPPLIED EXTERNALLY
-------------------------------------------------------------
`Thesis` (intelligence/thesis.py) is deliberately NOT persisted onto any
ExperienceRecord -- Task 7's whole design is that it exists only transiently
on `FastTierDecisionEngine._open_thesis`, discarded the moment a position
closes, specifically so it can never leak across positions. That means the
load-bearing-sources information this module needs cannot be recovered from
the ExperienceRecord stream alone; whatever offline harness runs credit
assignment after a replay must have captured each trade's Thesis at entry
time (e.g. by reading `FastTierDecisionEngine.open_thesis` right after a
LONG/SHORT `decide()` call, before the next `decide()` clears it) and keep
it keyed by that entry's `decision_id`. This module's entry points therefore
take the Thesis (or a `{decision_id: Thesis}` map) as an explicit
parameter -- they never try to reconstruct it from records.

WHAT "AGREED" MEANS
-------------------------------------------------------------
`POSITION_CLOSED.realized_pnl` (`simulator/engine.py:close_position`,
confirmed by reading it) is ALREADY net of round-trip execution cost --
spread and slippage are embedded in `entry_fill_price()`/`exit_fill_price()`
before `realized_pnl` is computed, and `cost_amount` is a separate reporting
figure, not a second deduction. So "did the thesis's direction turn out
correct, net of cost" is exactly `realized_pnl > 0.0`; no separate
`execution_cost_total` subtraction is needed or correct (subtracting it
again would double-charge cost, the same class of bug `engine.py`'s own
whole-branch-review bugfix comment warns about). A realized_pnl of exactly
0.0 counts as `agreed=False` (the thesis did not produce a positive
realized outcome) -- ties are rare (never for a real fill) and this keeps
the definition a strict "outcome favored the thesis's direction," not an
"at least broke even" standard.

SCOPE: entry-decision credit only. The brief calls out exit decisions
(Task 8's POLICY_EXIT reassessment signal) as "credited separately from
entry" per the plan, but this task's brief text asks only for entry-decision
credit assignment. Crediting the reassessment signal that produced a
POLICY_EXIT (e.g. whether re-evaluating and exiting early was itself a good
call, as distinct from whether the original entry thesis was right) would
need its own outcome definition (right relative to what counterfactual?)
that isn't specified here -- built now it would be speculative scope, not a
natural reading of this task. `assign_trade_credit` below always credits
against the ENTRY thesis, function of POLICY_EXIT or SL/TP/liquidation
alike; a later task can add exit-signal credit as a genuine extension.
"""
from __future__ import annotations

from typing import Iterable, Optional

from intelligence.fast_tier import ToolTrust
from intelligence.thesis import Thesis
from simulator.experience import ExperienceRecord


def group_by_decision(
    records: Iterable[ExperienceRecord],
) -> dict[str, list[ExperienceRecord]]:
    """Groups a flat ExperienceRecord stream into one list per decision_id,
    in the order encountered. Records with `decision_id is None` (NO_TRADE
    DECIDE records) are dropped -- there is nothing to link them to."""
    groups: dict[str, list[ExperienceRecord]] = {}
    for record in records:
        if record.decision_id is None:
            continue
        groups.setdefault(record.decision_id, []).append(record)
    return groups


def assign_trade_credit(
    records: Iterable[ExperienceRecord],
    thesis: Optional[Thesis],
    trust: ToolTrust,
) -> bool:
    """Credits (or withholds credit for) exactly one trade's lifecycle --
    the DECIDE record that opened it (or attempted to), any MANAGE records,
    and its POSITION_CLOSED record, all sharing one `decision_id` -- see
    module docstring for why these must already be grouped this way and why
    `thesis` must be supplied by the caller.

    Returns True iff at least one `ToolTrust.update()` call was made (for
    logging/testing convenience); the caller is not required to use it.

    Withholds ALL credit (zero update() calls) when:
      - the DECIDE record for this trade has `rejection_reason is not None`
        (the entry never actually opened a position -- Task 6's rejection
        path), or
      - no POSITION_CLOSED record is present for this group (the position
        hadn't closed yet when this ran -- nothing realized to credit), or
      - `thesis` is None or has no load_bearing_sources (nothing to credit).
    """
    records = list(records)
    decide_record = next(
        (r for r in records if r.event_type == "DECIDE"), None
    )
    if decide_record is None or decide_record.rejection_reason is not None:
        return False

    close_record = next(
        (r for r in records if r.event_type == "POSITION_CLOSED"), None
    )
    if close_record is None or close_record.realized_pnl is None:
        return False

    if thesis is None or not thesis.load_bearing_sources:
        return False

    agreed = close_record.realized_pnl > 0.0

    credited = False
    for source_name, context_bucket, _contribution in thesis.load_bearing_sources:
        trust.update(source_name, context_bucket, agreed)
        credited = True
    return credited


def assign_replay_credit(
    records: Iterable[ExperienceRecord],
    theses_by_decision_id: dict[str, Thesis],
    trust: ToolTrust,
) -> int:
    """Batch entry point: groups a full replay's ExperienceRecord stream by
    decision_id (see `group_by_decision`) and calls `assign_trade_credit`
    once per group, looking up that group's Thesis by its decision_id.
    Groups whose decision_id has no matching entry in
    `theses_by_decision_id` are skipped -- there is nothing to credit
    against, and this is never treated as an error since it's the expected
    shape for a rejected-entry group.

    Returns the number of trades actually credited (>= 0 ToolTrust.update()
    calls each), for logging/testing convenience.
    """
    groups = group_by_decision(records)
    credited_count = 0
    for decision_id, group_records in groups.items():
        thesis = theses_by_decision_id.get(decision_id)
        if assign_trade_credit(group_records, thesis, trust):
            credited_count += 1
    return credited_count
