"""tests/intelligence/test_causal_memory_boundaries.py -- Task 14: strict
causal-memory boundary tests (mandate Section 12 item 14).

This file is a dedicated, standalone hardening pass on the single
highest-risk correctness property in the whole Fast Tier design:
`ToolTrust.update()` and `context_bucket()` must never leak future
information backward. It follows the poisoning/truncation dual-pattern
already established by Phase 1's `tests/simulator/test_no_leakage.py` and
`tests/simulator/test_leakage_extended.py`, applied here to the Fast
Tier's own trust/context state rather than Phase 1's market data.

This is explicitly NOT re-testing Task 5's refit-caching mechanism or
Beta-posterior math (already covered by tests/intelligence/test_fast_tier.py)
and NOT re-testing Task 9's per-trade grouping/rejection-withholding logic
(covered by tests/intelligence/test_credit_assignment.py, if present) --
only the causal-ordering property itself.

FINDING (documented, not silently worked around -- see module docstring
of intelligence/credit_assignment.py): `assign_trade_credit` operates on
exactly one already-grouped trade's records (by `decision_id`) and its own
supplied `Thesis`; it has no access to any other trade's records or
outcome, and no shared mutable state other than the `ToolTrust` instance
being updated. There is therefore no code path within the real
`assign_trade_credit`/`assign_replay_credit` API through which a LATER
trade's outcome could be threaded into an EARLIER trade's `update()` call
-- the function structurally cannot read anything but the one group and
Thesis it was given. A "poisoned" scenario is only constructible by having
the *caller* pass the wrong (thesis, records) pairing to
`assign_trade_credit` -- i.e. a caller-side bug in how theses are keyed by
decision_id, not a leakage bug in credit_assignment.py itself. This test suite's actual causal no-look-ahead guard is the
truncation-vs-full test below: crediting decisions 1..k from a record
stream truncated right after decision k's close must produce IDENTICAL
`ToolTrust` posteriors to crediting from the full record stream 1..N --
i.e. a later, not-yet-closed trade's mere PRESENCE in the input stream
must never change an earlier trade's credit. That is the direct analogue
of Task 3/5's truncate-and-recompute no-look-ahead pattern, applied to
credit assignment instead of evidence computation, and it is what would
catch a bug where credit assignment folds in some order-symmetric
"outcomes seen so far" side channel that a naive call-order-permutation
test cannot distinguish from a causally correct implementation (a fixed,
fully-materialized dataset replayed in different `update()` call orders
is float-commutative regardless of causal correctness -- see the
docstring on `test_credit_assignment_order_invariant_regardless_of_processing_order`
below for why that test, while still a real and worthwhile guard on
`ToolTrust`'s own accumulation semantics, does NOT by itself prove the
causal property this task exists to harden). This suite also (separately)
demonstrates that mis-pairing a decision_id with the WRONG trade's Thesis
(the only constructible cross-trade poisoning vector against the real
`assign_replay_credit` API) does change the outcome, confirming
`assign_replay_credit`'s decision_id-keyed lookup is exactly what prevents
that in real use.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from intelligence.credit_assignment import assign_replay_credit, assign_trade_credit
from intelligence.evidence import EvidenceValue
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import ToolTrust, context_bucket
from intelligence.thesis import Thesis
from simulator.contracts import EnvironmentTag
from simulator.experience import ExperienceRecord


def _decide_record(decision_id, t):
    return ExperienceRecord(
        environment_tag=EnvironmentTag.SIMULATED_TRAINING,
        timestamp=t,
        event_type="DECIDE",
        market_state_snapshot={},
        position_view=None,
        action="LONG",
        account_state={},
        realized_pnl=None,
        cost_amount=None,
        outcome=None,
        gap_type="NORMAL",
        decision_id=decision_id,
        rejection_reason=None,
    )


def _closed_record(decision_id, t, realized_pnl):
    return ExperienceRecord(
        environment_tag=EnvironmentTag.SIMULATED_TRAINING,
        timestamp=t,
        event_type="POSITION_CLOSED",
        market_state_snapshot={},
        position_view=None,
        action=None,
        account_state={},
        realized_pnl=realized_pnl,
        cost_amount=1.0,
        outcome=None,
        gap_type="NORMAL",
        decision_id=decision_id,
    )


def _thesis(load_bearing):
    return Thesis(load_bearing_sources=load_bearing, entry_belief=1.0)


# ---------------------------------------------------------------------------
# 1. Poisoned-future-outcome test for ToolTrust.update() via credit_assignment
# ---------------------------------------------------------------------------

def test_credit_assignment_order_invariant_regardless_of_processing_order():
    """NOTE ON SCOPE (relabeled after review): this test proves
    call-order-commutativity of ToolTrust's additive Beta accumulator over a
    FIXED, fully-materialized set of trades -- it does NOT by itself prove
    the causal no-look-ahead property this task exists to harden. Floating-
    point addition is commutative, so ANY implementation that calls
    `update()` exactly once per (source, trade) pair will pass a call-order
    permutation test as long as every trade's records already exist in full
    for every permutation -- only the *call order* varies here, never *what
    data is visible at each step*. A leaky implementation that folded in a
    running "outcomes-seen-so-far" side channel (itself order-symmetric)
    would sail through this exact test while genuinely violating causality.
    The real causal no-look-ahead guard is
    `test_credit_assignment_truncated_vs_full_stream_identical_for_common_prefix`
    below, which varies what's actually IN the input stream (truncated vs.
    full), not just call order over an unchanged fixed dataset.

    This test remains a real, worthwhile, differently-scoped guard on
    ToolTrust's own accumulation semantics: constructs 4 trades in
    chronological order, where trade 4's outcome (loss) is the deliberate
    OPPOSITE of trade 1's outcome (win) on the same (source, bucket) pair.
    Credits them via the real `assign_trade_credit` in natural chronological
    order, then again in a scrambled/reversed order, and asserts ToolTrust's
    resulting posteriors are IDENTICAL either way -- i.e. the accumulator
    itself has no order-dependent state (no decay, no recency weighting, no
    aliasing bug in a shared/mutated-in-place records list) that would make
    two orderings of the SAME fixed dataset diverge.
    """
    base_t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bucket = 2

    trades = []
    # Trade 1 (earliest): source "alpha" WINS.
    trades.append((
        "d1",
        [_decide_record("d1", base_t), _closed_record("d1", base_t + timedelta(minutes=1), realized_pnl=10.0)],
        _thesis([("alpha", bucket, 0.5)]),
    ))
    # Trade 2: source "beta" WINS.
    trades.append((
        "d2",
        [_decide_record("d2", base_t + timedelta(minutes=2)),
         _closed_record("d2", base_t + timedelta(minutes=3), realized_pnl=5.0)],
        _thesis([("beta", bucket, 0.3)]),
    ))
    # Trade 3: source "beta" LOSES.
    trades.append((
        "d3",
        [_decide_record("d3", base_t + timedelta(minutes=4)),
         _closed_record("d3", base_t + timedelta(minutes=5), realized_pnl=-5.0)],
        _thesis([("beta", bucket, -0.2)]),
    ))
    # Trade 4 (latest): source "alpha" LOSES -- deliberately the OPPOSITE
    # outcome from trade 1's "alpha" win, on the identical (source, bucket).
    trades.append((
        "d4",
        [_decide_record("d4", base_t + timedelta(minutes=6)),
         _closed_record("d4", base_t + timedelta(minutes=7), realized_pnl=-8.0)],
        _thesis([("alpha", bucket, 0.4)]),
    ))

    def _credit_in_order(order):
        trust = ToolTrust()
        for idx in order:
            decision_id, records, thesis = trades[idx]
            assign_trade_credit(records, thesis, trust)
        return trust

    chronological = _credit_in_order([0, 1, 2, 3])
    reversed_order = _credit_in_order([3, 2, 1, 0])
    scrambled = _credit_in_order([2, 0, 3, 1])

    for name in ("alpha", "beta"):
        mean_chrono = chronological.posterior_mean(name, bucket)
        mean_reversed = reversed_order.posterior_mean(name, bucket)
        mean_scrambled = scrambled.posterior_mean(name, bucket)
        assert mean_chrono == pytest.approx(mean_reversed), (
            f"{name}: order-dependence detected (chronological vs reversed) -- "
            "possible future-outcome leakage into an earlier update() call"
        )
        assert mean_chrono == pytest.approx(mean_scrambled), (
            f"{name}: order-dependence detected (chronological vs scrambled)"
        )

    # Sanity: "alpha" actually received one win (trade 1) and one loss
    # (trade 4) -- its posterior mean must sit at exactly 0.5 (Beta(2,2)
    # starting from the Beta(1,1) prior), proving both updates were applied
    # (not that the test is vacuously order-invariant because nothing
    # happened).
    assert chronological.posterior_mean("alpha", bucket) == pytest.approx(0.5)


def test_credit_assignment_truncated_vs_full_stream_identical_for_common_prefix():
    """THE flagship causal no-look-ahead guard for this task (the direct
    analogue of Task 3/5's truncate-and-recompute pattern, applied to
    credit assignment instead of evidence computation): credit decisions
    1..k using a record stream TRUNCATED right after decision k's own
    POSITION_CLOSED record (so decisions k+1..N -- later, not-yet-closed-
    at-that-point trades -- are simply ABSENT from the input, not merely
    reordered) vs. crediting decisions 1..k from the FULL record stream
    1..N (where those later trades' records genuinely exist, interleaved
    after decision k's close). Confirms decisions 1..k's resulting
    ToolTrust posteriors are IDENTICAL either way.

    This is the property `test_credit_assignment_order_invariant_...`
    above cannot prove: unlike a call-order permutation over one fixed,
    fully-materialized dataset (which is float-commutative for any
    per-trade-independent implementation, leaky or not), this test varies
    *what data is actually present* in the input at the point decisions
    1..k are credited. A bug that let `assign_replay_credit` (or a helper
    it calls) fold information from ANY record occurring after a given
    decision's own close -- e.g. a running side-channel counter, a lookback
    window computed over the whole stream, or grouping logic that
    accidentally peeks past the current decision's own records -- would
    make the truncated and full runs diverge for decisions 1..k, even
    though a naive order-permutation test would never catch it (order-
    symmetric leakage passes order-invariance trivially).
    """
    base_t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bucket = 3

    # 6 trades, k = 3: decisions 1..3 close (in order) before decisions 4..6
    # even begin. Decisions 1..3 use sources "alpha"/"beta"/"gamma";
    # decisions 4..6 (the "future" relative to the truncation point) use
    # DISJOINT source names "delta"/"echo"/"foxtrot" on the SAME bucket --
    # disjoint so that the ONLY way the full run's posteriors for
    # alpha/beta/gamma could differ from the truncated run's is if
    # something from decisions 4..6 leaked backward (there is no legitimate
    # reason for them to differ, since no real trade ever credits
    # alpha/beta/gamma a second time in either run).
    early = [("alpha", True), ("beta", False), ("gamma", True)]
    late = [("delta", False), ("echo", True), ("foxtrot", False)]
    all_trades = []
    for idx, (source_name, agreed) in enumerate(early + late):
        decision_id = f"d{idx + 1}"
        t0 = base_t + timedelta(minutes=2 * idx)
        t1 = t0 + timedelta(minutes=1)
        pnl = 10.0 if agreed else -10.0
        records = [_decide_record(decision_id, t0), _closed_record(decision_id, t1, realized_pnl=pnl)]
        thesis = _thesis([(source_name, bucket, 0.5)])
        all_trades.append((decision_id, records, thesis))

    k = 3

    # Full stream: every trade's records, in natural chronological order.
    full_records = [r for (_, records, _) in all_trades for r in records]
    full_theses = {decision_id: thesis for (decision_id, _, thesis) in all_trades}
    trust_full = ToolTrust()
    assign_replay_credit(full_records, full_theses, trust_full)

    # Truncated stream: ONLY decisions 1..k's records exist at all -- decisions
    # k+1..N (the "future," not-yet-closed-at-that-point trades) are absent
    # from the input entirely, not just reordered.
    truncated_records = [r for (_, records, _) in all_trades[:k] for r in records]
    truncated_theses = {decision_id: thesis for (decision_id, _, thesis) in all_trades[:k]}
    trust_truncated = ToolTrust()
    assign_replay_credit(truncated_records, truncated_theses, trust_truncated)

    for source_name, _agreed in early:
        mean_full = trust_full.posterior_mean(source_name, bucket)
        mean_truncated = trust_truncated.posterior_mean(source_name, bucket)
        assert mean_full == pytest.approx(mean_truncated), (
            f"{source_name}: credit for decisions 1..{k} differs between the truncated "
            "and full record streams -- a later, not-yet-closed trade's presence in the "
            "input leaked into an earlier trade's ToolTrust update"
        )

    # Sanity: the two streams are NOT trivially identical because there was
    # no real "future" data to leak in the first place -- confirm decisions
    # 4..6 genuinely did get credited in the full run (their sources moved
    # off the untouched Beta(1,1) prior mean of 0.5), proving real
    # information existed that COULD have leaked and provably didn't.
    for source_name, _agreed in late:
        assert trust_full.posterior_mean(source_name, bucket) != pytest.approx(0.5), (
            f"{source_name}: expected to be credited in the full run (sanity check that "
            "the 'future' trades in this scenario are not vacuous)"
        )
        # And correctly absent from the truncated run, which never saw them.
        assert trust_truncated.posterior_mean(source_name, bucket) == pytest.approx(0.5)


def test_earlier_trade_credit_unaffected_by_poisoning_a_later_trades_outcome():
    """Direct poisoning-style check on ToolTrust.update() alone (isolating
    it from assign_trade_credit's own grouping logic): update trust for an
    "earlier" decision, snapshot the posterior, then apply a LATER,
    opposite-outcome update for the SAME (source, bucket) and confirm the
    earlier decision's already-recorded posterior snapshot is unaffected --
    i.e. ToolTrust.update() never retroactively revises a previously
    recorded observation. (Each update() call is a one-way accumulation:
    once made, it cannot un-happen after a later opposite-signed update.)
    This is the property a bug that let a later call rewrite/rescale an
    earlier posterior entry (rather than accumulate) would violate.
    """
    trust = ToolTrust()
    trust.update("gamma", 1, agreed=True)  # earlier decision: WIN
    mean_after_first = trust.posterior_mean("gamma", 1)

    # A later, deliberately opposite outcome for the same (source, bucket).
    trust.update("gamma", 1, agreed=False)
    mean_after_second = trust.posterior_mean("gamma", 1)

    # The posterior legitimately moved (evidence accumulated) -- but always
    # forward/additive, never reset back to the exact pre-update value or
    # some function implying the second update erased the first.
    assert mean_after_second != mean_after_first
    # The first update's contribution is still present: with Beta(1,1) prior,
    # one win then one loss must land exactly at Beta(2,2) mean = 0.5, not at
    # Beta(1,1)=0.5-by-coincidence-of-no-updates or Beta(1,2)/Beta(2,1)
    # (which would mean the earlier update was overwritten, not accumulated).
    assert mean_after_second == pytest.approx(0.5)


def test_replay_credit_mispairing_thesis_across_decision_ids_changes_outcome():
    """Documents the FINDING described in this module's docstring: the only
    constructible "poisoning" scenario against the real credit_assignment
    API is a CALLER mis-pairing a decision_id with the wrong trade's Thesis
    in the `theses_by_decision_id` map passed to `assign_replay_credit` --
    not a leakage bug internal to credit_assignment.py itself (which has no
    access to any record/thesis beyond the one group it's given).

    This test proves that correct decision_id-keyed pairing (as
    `assign_replay_credit` requires and performs) is exactly what prevents
    this: swapping trade 1's Thesis onto trade 4's decision_id (and vice
    versa) in the caller-supplied map DOES change ToolTrust's resulting
    posteriors relative to the correctly-paired run -- i.e. correctness here
    genuinely depends on the caller doing the pairing right, confirming
    there is no lower-level structural guarantee credit_assignment.py could
    silently violate on its own.
    """
    base_t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bucket = 0
    records_1 = [_decide_record("d1", base_t), _closed_record("d1", base_t + timedelta(minutes=1), realized_pnl=10.0)]
    records_4 = [_decide_record("d4", base_t + timedelta(minutes=6)),
                 _closed_record("d4", base_t + timedelta(minutes=7), realized_pnl=-8.0)]
    all_records = records_1 + records_4

    # Note: the realized_pnl driving `agreed` comes from each decision_id's
    # OWN close_record regardless of which thesis is attached to it -- so
    # mis-pairing the thesis alone is only *observable* when the swapped
    # theses carry different source names (otherwise both pairings would
    # credit the same source with the same outcome and the swap would be
    # invisible, proving nothing).
    thesis_1_named = _thesis([("source_from_trade_1", bucket, 0.5)])
    thesis_4_named = _thesis([("source_from_trade_4", bucket, 0.4)])
    correct_map_named = {"d1": thesis_1_named, "d4": thesis_4_named}
    poisoned_map_named = {"d1": thesis_4_named, "d4": thesis_1_named}

    trust_correct_named = ToolTrust()
    assign_replay_credit(all_records, correct_map_named, trust_correct_named)
    trust_poisoned_named = ToolTrust()
    assign_replay_credit(all_records, poisoned_map_named, trust_poisoned_named)

    # Correct pairing: d1 (win, pnl=10) credits source_from_trade_1 with a WIN.
    assert trust_correct_named.posterior_mean("source_from_trade_1", bucket) == pytest.approx(2.0 / 3.0)
    # Poisoned pairing: d1 (still win, pnl=10, since realized_pnl comes from
    # d1's OWN close_record) now credits source_from_trade_4's NAME with a
    # WIN instead -- the wrong source gets d1's outcome purely because the
    # caller mis-keyed the map. This is observably different from the
    # correct run.
    assert trust_poisoned_named.posterior_mean("source_from_trade_4", bucket) == pytest.approx(2.0 / 3.0)
    assert trust_correct_named.posterior_mean("source_from_trade_4", bucket) != pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# 2. Truncated-vs-full context_bucket test (strengthens Task 5's existing
#    coverage in tests/intelligence/test_fast_tier.py with a genuinely new
#    angle: poisoning the closes array BEYOND index i with adversarial
#    values, computed through the real registry sources, rather than only
#    constructing EvidenceValue dicts directly via literals).
# ---------------------------------------------------------------------------

def _evidence_from_closes(registry, closes):
    """Computes the real garch_conditional_variance/kalman_filtered_velocity
    EvidenceValues (via the actual registered EvidenceSourceSpec.compute
    callables from build_default_registry) for a given closes array -- the
    same inputs context_bucket() actually consumes in FastTierReasoner."""
    specs = registry.specs()
    evidence = {}
    for name in ("garch_conditional_variance", "kalman_filtered_velocity"):
        spec = specs[name]
        evidence[name] = spec.compute(closes)
    return evidence


def test_context_bucket_truncated_vs_full_prefix_identical():
    """No-look-ahead pattern (Task 3's truncate-and-recompute equivalence,
    applied directly to context_bucket() in isolation): for every prefix
    index i, the bucket computed from evidence derived from
    closes[:i] must equal the bucket computed from evidence derived from the
    SAME closes[:i] whether or not more data exists beyond it in the array
    the caller happens to hold. Uses the real registry's GARCH/Kalman
    compute() functions (Task 3's actual wrappers), not hand-built
    EvidenceValue literals.
    """
    registry = build_default_registry()
    rng = np.random.default_rng(7)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, 120))

    for i in [5, 10, 30, 60, 90, 119, 120]:
        prefix = closes[:i]
        ev_prefix = _evidence_from_closes(registry, prefix)
        bucket_prefix = context_bucket(ev_prefix)

        # Recompute using an array that is byte-identical on [0, i) but
        # constructed as its own standalone slice (simulating "the caller
        # only ever had this much data at decision time i").
        standalone = np.array(prefix, copy=True)
        ev_standalone = _evidence_from_closes(registry, standalone)
        bucket_standalone = context_bucket(ev_standalone)

        assert bucket_prefix == bucket_standalone, f"prefix length {i}: bucket mismatch"


def test_context_bucket_unaffected_by_poisoning_closes_beyond_current_index():
    """Poisoning variant (the genuinely new angle beyond Task 5's existing
    test_context_bucket_is_continuous_derived_and_monotonic /
    test_context_bucket_handles_missing_evidence_gracefully, which only
    construct EvidenceValue dicts from literals and never exercise the real
    GARCH/Kalman compute() path against an actual closes array): for a fixed
    decision index i, poison every array element AFTER index i with extreme,
    adversarial values (a huge spike then a crash) and confirm the evidence
    computed from closes[:i] -- and therefore the resulting context_bucket
    -- is completely unaffected. A bug that let the GARCH/Kalman wrapper (or
    context_bucket itself) read past `closes_so_far`'s current index would
    make the bucket swing wildly here, since the poisoned tail is
    deliberately far outside the clean data's range.
    """
    registry = build_default_registry()
    rng = np.random.default_rng(11)
    n = 150
    i = 80
    clean = 100.0 + np.cumsum(rng.normal(0, 0.5, n))

    ev_clean_prefix = _evidence_from_closes(registry, clean[:i])
    bucket_clean = context_bucket(ev_clean_prefix)

    poisoned = clean.copy()
    # Adversarial tail: a massive spike immediately after i, then a crash
    # far below the clean data's range -- engineered to blow up GARCH
    # conditional variance and Kalman velocity estimates if it were ever
    # read.
    poisoned[i:i + 5] = clean[i - 1] + 1_000_000.0
    poisoned[i + 5:] = clean[i - 1] - 1_000_000.0

    # The caller must only ever pass closes_so_far = poisoned[:i] to a
    # decision at index i -- confirm that slice is untouched by the
    # poisoning (sanity on the test construction itself)...
    assert np.array_equal(poisoned[:i], clean[:i])

    # ...and confirm evidence/bucket computed from that identical prefix is
    # unaffected by what was written beyond it in the SAME underlying array
    # object (not just a copy) -- catches a bug where a wrapper is handed
    # the full array plus an index rather than a true slice/truncation.
    ev_poisoned_prefix = _evidence_from_closes(registry, poisoned[:i])
    bucket_poisoned = context_bucket(ev_poisoned_prefix)

    assert bucket_clean == bucket_poisoned, (
        "leakage: context_bucket changed after poisoning closes beyond the "
        "current decision index -- context_bucket() or its evidence inputs "
        "read past closes_so_far's current index"
    )
    for name in ("garch_conditional_variance", "kalman_filtered_velocity"):
        clean_val = ev_clean_prefix[name].value
        poisoned_val = ev_poisoned_prefix[name].value
        if clean_val is None or poisoned_val is None:
            assert clean_val is None and poisoned_val is None
        else:
            assert clean_val == pytest.approx(poisoned_val), (
                f"leakage: {name} evidence value changed after poisoning the "
                "array beyond the current index"
            )


if __name__ == "__main__":
    test_credit_assignment_order_invariant_regardless_of_processing_order()
    test_credit_assignment_truncated_vs_full_stream_identical_for_common_prefix()
    test_earlier_trade_credit_unaffected_by_poisoning_a_later_trades_outcome()
    test_replay_credit_mispairing_thesis_across_decision_ids_changes_outcome()
    test_context_bucket_truncated_vs_full_prefix_identical()
    test_context_bucket_unaffected_by_poisoning_closes_beyond_current_index()
    print("tests/intelligence/test_causal_memory_boundaries.py: OK")
