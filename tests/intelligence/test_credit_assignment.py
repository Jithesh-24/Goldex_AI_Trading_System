"""tests/intelligence/test_credit_assignment.py -- Task 9: trade credit
assignment. Builds synthetic ExperienceRecord sequences directly (rather
than running a full replay) so each trade's ground-truth outcome and
load-bearing sources are fully controlled, and specifically constructs two
trades close together in time with OVERLAPPING but DISTINCT load-bearing
source sets -- exactly the scenario where getting attribution wrong (e.g.
crediting every source "active" near a trade rather than only the ones its
own thesis actually used) would be an easy mistake to make silently.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta

from intelligence.credit_assignment import (
    assign_replay_credit,
    assign_trade_credit,
    group_by_decision,
)
from intelligence.fast_tier import ToolTrust
from intelligence.thesis import Thesis
from simulator.contracts import EnvironmentTag, PositionOutcome
from simulator.experience import ExperienceRecord

TAG = EnvironmentTag.SIMULATED_TRAINING
T0 = datetime(2020, 1, 6, 10, 0, 0)


def _decide(decision_id, t, action="LONG", rejection_reason=None):
    return ExperienceRecord(
        environment_tag=TAG, timestamp=t, event_type="DECIDE",
        market_state_snapshot={}, position_view=None, action=action,
        account_state={}, realized_pnl=None, cost_amount=None, outcome=None,
        gap_type="NORMAL", decision_id=decision_id, rejection_reason=rejection_reason,
    )


def _manage(decision_id, t, action="HOLD"):
    return ExperienceRecord(
        environment_tag=TAG, timestamp=t, event_type="MANAGE",
        market_state_snapshot={}, position_view={}, action=action,
        account_state={}, realized_pnl=None, cost_amount=None, outcome=None,
        gap_type="NORMAL", decision_id=decision_id,
    )


def _closed(decision_id, t, realized_pnl, outcome=PositionOutcome.TP_HIT, cost_amount=1.0):
    return ExperienceRecord(
        environment_tag=TAG, timestamp=t, event_type="POSITION_CLOSED",
        market_state_snapshot={}, position_view={}, action=None,
        account_state={}, realized_pnl=realized_pnl, cost_amount=cost_amount,
        outcome=outcome, gap_type="NORMAL", decision_id=decision_id,
    )


# --- single-trade unit tests -------------------------------------------

def test_winning_trade_credits_exactly_its_load_bearing_sources_as_agreed():
    trust = ToolTrust()
    # Both sources voted LONG (positive contribution), matching the LONG
    # trade -- so a win validates both. (A source whose contribution
    # DISAGREED with the trade's direction is covered by
    # test_dissenting_load_bearing_source_gets_opposite_credit below.)
    thesis = Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4), ("kalman_innovation", 1, 0.1)])
    records = [
        _decide("d1", T0, "LONG"),
        _manage("d1", T0 + timedelta(minutes=1)),
        _closed("d1", T0 + timedelta(minutes=2), realized_pnl=25.0),
    ]
    credited = assign_trade_credit(records, thesis, trust)
    assert credited is True
    assert trust.posterior_mean("momentum_scalar", 2) > 0.5
    assert trust.posterior_mean("kalman_innovation", 1) > 0.5
    # Untouched source stays at the Beta(1,1) prior mean.
    assert trust.posterior_mean("kalman_filtered_velocity", 0) == 0.5


def test_losing_trade_credits_load_bearing_sources_as_disagreed():
    trust = ToolTrust()
    thesis = Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4)])
    records = [
        _decide("d1", T0, "LONG"),
        _closed("d1", T0 + timedelta(minutes=1), realized_pnl=-12.0),
    ]
    assign_trade_credit(records, thesis, trust)
    assert trust.posterior_mean("momentum_scalar", 2) < 0.5


def test_zero_pnl_counts_as_disagreed_not_a_free_pass():
    trust = ToolTrust()
    thesis = Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4)])
    records = [_decide("d1", T0, "LONG"), _closed("d1", T0, realized_pnl=0.0)]
    assign_trade_credit(records, thesis, trust)
    assert trust.posterior_mean("momentum_scalar", 2) < 0.5


def test_rejected_entry_contributes_zero_credit_anywhere():
    trust = ToolTrust()
    thesis = Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4), ("kalman_filtered_velocity", 0, 0.2)])
    records = [_decide("d1", T0, "LONG", rejection_reason="insufficient_margin")]
    credited = assign_trade_credit(records, thesis, trust)
    assert credited is False
    # Every source's posterior must remain the untouched Beta(1,1) prior
    # (mean 0.5, variance 1/12) -- proving no update() call happened at all.
    for name, bucket in [("momentum_scalar", 2), ("kalman_filtered_velocity", 0)]:
        assert trust.posterior_mean(name, bucket) == 0.5
        assert abs(trust.posterior_uncertainty(name, bucket) - (1.0 / 12.0)) < 1e-9


def test_no_close_record_yet_withholds_credit():
    trust = ToolTrust()
    thesis = Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4)])
    records = [_decide("d1", T0, "LONG"), _manage("d1", T0 + timedelta(minutes=1))]
    assert assign_trade_credit(records, thesis, trust) is False
    assert trust.posterior_mean("momentum_scalar", 2) == 0.5


def test_missing_thesis_withholds_credit():
    trust = ToolTrust()
    records = [_decide("d1", T0, "LONG"), _closed("d1", T0, realized_pnl=10.0)]
    assert assign_trade_credit(records, None, trust) is False


# --- the wrong-attribution regression test ------------------------------

def test_two_nearby_trades_with_overlapping_source_sets_credit_separately():
    """Trade A (decision_id 'a') is a WINNER whose thesis is load-bearing on
    momentum_scalar + garch_conditional_variance. Trade B (decision_id 'b')
    opens right after A closes -- close in time -- and is a LOSER whose
    thesis is load-bearing on momentum_scalar (shared with A) +
    kalman_filtered_velocity (which was also "active"/computed near trade A
    but was NOT part of A's thesis). An incorrect implementation that
    credits every source seen active near a trade (rather than only that
    trade's own thesis.load_bearing_sources) would either:
      - credit kalman_filtered_velocity for trade A's win (wrong: it wasn't
        part of A's thesis), or
      - credit momentum_scalar only once combining both outcomes instead of
        once per trade with each trade's own agreed/disagreed sign, or
      - fail to give momentum_scalar the DISAGREE update from B on top of
        the AGREE update it already got from A.
    This test pins the exact correct posterior for every source, which only
    holds if each trade's credit lands on exactly its own load-bearing set.
    """
    trust = ToolTrust()

    # Contribution signs match each trade's own direction (A LONG -> both
    # positive; B SHORT -> both negative), so this test isolates per-trade
    # ATTRIBUTION without also entangling the per-source dissent logic --
    # that is tested separately below.
    thesis_a = Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4), ("kalman_innovation", 1, 0.1)])
    thesis_b = Thesis(load_bearing_sources=[("momentum_scalar", 2, -0.3), ("kalman_filtered_velocity", 0, -0.2)])

    records = [
        _decide("a", T0, "LONG"),
        _manage("a", T0 + timedelta(minutes=1)),
        _closed("a", T0 + timedelta(minutes=2), realized_pnl=50.0),  # A wins
        _decide("b", T0 + timedelta(minutes=2), "SHORT"),
        _closed("b", T0 + timedelta(minutes=3), realized_pnl=-30.0),  # B loses
    ]

    theses_by_decision_id = {"a": thesis_a, "b": thesis_b}
    credited_count = assign_replay_credit(records, theses_by_decision_id, trust)
    assert credited_count == 2

    # momentum_scalar (bucket 2) got ONE agree (from A) and ONE disagree
    # (from B): Beta(1+1, 1+1) -> mean exactly 0.5, but NOT the untouched
    # prior -- both updates must have actually landed (alpha=2, beta=2).
    a, b = trust._get("momentum_scalar", 2)
    assert (a, b) == (2.0, 2.0)

    # kalman_innovation (bucket 1): only A's thesis used it, and A
    # won -> exactly one agree update, never touched by B.
    a, b = trust._get("kalman_innovation", 1)
    assert (a, b) == (2.0, 1.0)

    # kalman_filtered_velocity (bucket 0): only B's thesis used it, and B
    # lost -> exactly one disagree update, never touched by A even though it
    # was temporally adjacent to A's trade.
    a, b = trust._get("kalman_filtered_velocity", 0)
    assert (a, b) == (1.0, 2.0)


def test_rejected_entry_amid_real_trades_does_not_pollute_others():
    trust = ToolTrust()
    thesis_win = Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4)])

    records = [
        _decide("rejected", T0, "LONG", rejection_reason="invalid_sl"),
        _decide("win", T0 + timedelta(minutes=1), "LONG"),
        _closed("win", T0 + timedelta(minutes=2), realized_pnl=10.0),
    ]
    theses = {"rejected": Thesis(load_bearing_sources=[("momentum_scalar", 2, 0.4)]), "win": thesis_win}
    credited_count = assign_replay_credit(records, theses, trust)
    assert credited_count == 1
    a, b = trust._get("momentum_scalar", 2)
    assert (a, b) == (2.0, 1.0)  # exactly one agree update, from "win" only


# --- group_by_decision ----------------------------------------------------

def test_group_by_decision_drops_no_trade_records_and_groups_correctly():
    no_trade = _decide(None, T0, action="NO_TRADE")
    records = [
        no_trade,
        _decide("a", T0 + timedelta(minutes=1), "LONG"),
        _manage("a", T0 + timedelta(minutes=2)),
        _closed("a", T0 + timedelta(minutes=3), realized_pnl=5.0),
        _decide("b", T0 + timedelta(minutes=4), "SHORT"),
        _closed("b", T0 + timedelta(minutes=5), realized_pnl=-5.0),
    ]
    groups = group_by_decision(records)
    assert set(groups.keys()) == {"a", "b"}
    assert len(groups["a"]) == 3
    assert len(groups["b"]) == 2


# --- C2 regression: signed per-source credit ---------------------------

def test_dissenting_load_bearing_source_gets_opposite_credit():
    """Whole-branch-review C2 regression. `Thesis.load_bearing_sources` is
    sign-agnostic (a source qualifies on |contribution| >= floor), so it can
    contain a source that voted OPPOSITE to the direction the trade was
    actually taken in. Crediting the trade-level `realized_pnl > 0` flag
    uniformly to every load-bearing source rewarded such a dissenter for an
    outcome it had argued against.

    Here the trade goes LONG and WINS. `momentum_scalar` voted LONG (+0.4)
    and must be credited as agreed; `kalman_innovation` voted SHORT (-0.35)
    and must be credited as NOT agreed. The two must move in OPPOSITE
    directions -- that is the whole point.
    """
    trust = ToolTrust()
    thesis = Thesis(load_bearing_sources=[
        ("momentum_scalar", 2, 0.4),      # agrees with the LONG that was taken
        ("kalman_innovation", 2, -0.35),  # dissents: argued SHORT
    ])
    records = [
        _decide("d1", T0, "LONG"),
        _closed("d1", T0 + timedelta(minutes=2), realized_pnl=25.0),
    ]
    assert assign_trade_credit(records, thesis, trust) is True

    agreeing = trust.posterior_mean("momentum_scalar", 2)
    dissenting = trust.posterior_mean("kalman_innovation", 2)
    assert agreeing > 0.5, "source that voted with the winning direction must be credited as agreed"
    assert dissenting < 0.5, (
        "source that voted AGAINST the direction actually taken must NOT be credited as agreed "
        "just because the trade it argued against happened to profit"
    )
    assert (agreeing > 0.5) != (dissenting > 0.5), "agreeing and dissenting sources must move oppositely"


def test_dissenting_source_is_vindicated_when_the_trade_loses():
    """The symmetric half of the same rule: the trade goes LONG and LOSES,
    so the source that argued SHORT was RIGHT (agreed=True) and the source
    that argued LONG was wrong (agreed=False)."""
    trust = ToolTrust()
    thesis = Thesis(load_bearing_sources=[
        ("momentum_scalar", 3, 0.4),
        ("kalman_innovation", 3, -0.35),
    ])
    records = [
        _decide("d1", T0, "LONG"),
        _closed("d1", T0 + timedelta(minutes=2), realized_pnl=-25.0),
    ]
    assert assign_trade_credit(records, thesis, trust) is True
    assert trust.posterior_mean("momentum_scalar", 3) < 0.5
    assert trust.posterior_mean("kalman_innovation", 3) > 0.5


def test_short_trade_credit_is_relative_to_the_short_direction():
    """Direction is taken from the DECIDE record's action, not from a
    hardcoded LONG assumption: on a WINNING SHORT, the SHORT-voting
    (negative-contribution) source is the one validated."""
    trust = ToolTrust()
    thesis = Thesis(load_bearing_sources=[
        ("momentum_scalar", 1, -0.4),   # voted SHORT, matching the SHORT taken
        ("kalman_innovation", 1, 0.3),  # voted LONG, dissenting
    ])
    records = [
        _decide("d1", T0, "SHORT"),
        _closed("d1", T0 + timedelta(minutes=2), realized_pnl=40.0),
    ]
    assert assign_trade_credit(records, thesis, trust) is True
    assert trust.posterior_mean("momentum_scalar", 1) > 0.5
    assert trust.posterior_mean("kalman_innovation", 1) < 0.5
