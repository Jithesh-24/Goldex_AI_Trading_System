"""Tests for intelligence/fast_tier.py -- the Bayesian adaptive-trust
mechanism at the core of the Fast Tier."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from contracts.market_state import DataQuality, FeedHealthState, MarketState
from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import (
    GATED_OUT_CONTEXT_BUCKET,
    EXPENSIVE_SOURCE_NAMES,
    FastTierReasoner,
    Hypothesis,
    N_CONTEXT_BUCKETS,
    ToolTrust,
    context_bucket,
)


def _synthetic_closes(n=500, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.5, n)
    return 100.0 + np.cumsum(returns)


def _market_state(*, data_quality=DataQuality.VALID, market_closed=False):
    now = datetime.now(timezone.utc)
    return MarketState(
        symbol="XAUUSD",
        source="synthetic_replay",
        sequence=1,
        market_timestamp=now,
        ingestion_timestamp=now,
        processing_timestamp=now,
        bid=2000.0,
        ask=2000.2,
        mid=2000.1,
        spread=0.2,
        data_quality=data_quality,
        tick_count_60s=10,
        tick_count_300s=50,
        tick_rate_per_sec=1.0,
        market_closed=market_closed,
        feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5,
    )


# ---------------------------------------------------------------------------
# Property 1: conditional usefulness, not a fixed weight
# ---------------------------------------------------------------------------

def test_conditional_usefulness_not_fixed_weight():
    trust = ToolTrust()
    bucket_a, bucket_b = 0, 1

    for _ in range(30):
        trust.update("momentum_scalar", bucket_a, agreed=True)
    for _ in range(30):
        trust.update("momentum_scalar", bucket_b, agreed=False)

    mean_a = trust.posterior_mean("momentum_scalar", bucket_a)
    mean_b = trust.posterior_mean("momentum_scalar", bucket_b)

    assert mean_a > 0.9
    assert mean_b < 0.1
    assert mean_a > mean_b  # the direct proof: same source, context-dependent trust

    # A third, never-seen bucket must sit at the uninformative Beta(1,1)
    # mean (0.5) -- distinct from both learned buckets.
    mean_c = trust.posterior_mean("momentum_scalar", 2)
    assert mean_c == pytest.approx(0.5)


def _registry_with_context_sources(sigma2, velocity, test_value=1.0, test_conf=1.0):
    """A registry carrying the two real state-space sources context_bucket()
    reads (garch_conditional_variance, kalman_filtered_velocity) plus one
    directional "test_source" -- lets a test drive FastTierReasoner.hypothesis()
    end-to-end and land in a real, context_bucket()-computed bucket rather than
    an integer literal."""
    registry = EvidenceRegistry()
    registry.register(EvidenceSourceSpec(
        name="garch_conditional_variance", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(sigma2, 1.0, "garch_conditional_variance"),
        is_directional=False,  # mirrors production: a variance is not a price-direction vote
    ))
    registry.register(EvidenceSourceSpec(
        name="kalman_filtered_velocity", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(velocity, 1.0, "kalman_filtered_velocity"),
        is_directional=True,
    ))
    registry.register(EvidenceSourceSpec(
        name="test_source", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(test_value, test_conf, "test_source"),
        is_directional=True,
    ))
    return registry


def test_conditional_usefulness_end_to_end_via_real_context_bucket():
    """Integration-level version of property 1: drives hypothesis() twice
    with evidence that lands in two genuinely different real buckets (via
    context_bucket() itself, not hardcoded integers), trains "test_source"'s
    trust asymmetrically per the actual buckets those scenarios produce, and
    shows the same source's contribution differs between the two calls."""
    closes = _synthetic_closes(n=200)
    market_state = _market_state()

    low_sigma2, low_velocity = 0.0, 0.0
    high_sigma2, high_velocity = 1e8, 100.0

    # Determine the real buckets these two scenarios land in via the actual
    # context_bucket() function -- not asserted/assumed integers.
    bucket_low = context_bucket({
        "garch_conditional_variance": EvidenceValue(low_sigma2, 1.0, "garch_conditional_variance"),
        "kalman_filtered_velocity": EvidenceValue(low_velocity, 1.0, "kalman_filtered_velocity"),
    })
    bucket_high = context_bucket({
        "garch_conditional_variance": EvidenceValue(high_sigma2, 1.0, "garch_conditional_variance"),
        "kalman_filtered_velocity": EvidenceValue(high_velocity, 1.0, "kalman_filtered_velocity"),
    })
    assert bucket_low != bucket_high  # precondition: the two scenarios are genuinely distinct contexts

    trust = ToolTrust()
    # test_source is trained to be reliable in the low-magnitude bucket and
    # unreliable in the high-magnitude bucket.
    for _ in range(30):
        trust.update("test_source", bucket_low, agreed=True)
    for _ in range(30):
        trust.update("test_source", bucket_high, agreed=False)

    reasoner_low = FastTierReasoner(
        _registry_with_context_sources(low_sigma2, low_velocity), refit_interval=50,
    )
    hyp_low = reasoner_low.hypothesis(closes, market_state, trust)

    reasoner_high = FastTierReasoner(
        _registry_with_context_sources(high_sigma2, high_velocity), refit_interval=50,
    )
    hyp_high = reasoner_high.hypothesis(closes, market_state, trust)

    contrib_low = next(c for (n, b, c) in hyp_low.load_bearing_sources if n == "test_source")
    # test_source's weight collapsed in the high bucket, so it may or may not
    # clear the load-bearing floor there -- check its raw posterior mean
    # directly for the definitive comparison, and its Hypothesis-level
    # presence/contribution as the end-to-end confirmation.
    mean_low = trust.posterior_mean("test_source", bucket_low)
    mean_high = trust.posterior_mean("test_source", bucket_high)
    assert mean_low > mean_high

    high_load_bearing = [c for (n, b, c) in hyp_high.load_bearing_sources if n == "test_source"]
    # End-to-end proof: the same source, same directional evidence value,
    # driven purely by real context_bucket() output, is load-bearing (net
    # trusted contribution above the floor) in the bucket it was trained
    # reliable in, and is NOT load-bearing in the bucket it was trained
    # unreliable in -- genuinely conditional usefulness, observed through
    # the full hypothesis() pipeline rather than asserted via literals.
    assert contrib_low > 0.0
    assert high_load_bearing == []


def test_posterior_uncertainty_shrinks_with_more_observations():
    trust = ToolTrust()
    unc_prior = trust.posterior_uncertainty("rolling_skew", 0)
    for _ in range(50):
        trust.update("rolling_skew", 0, agreed=True)
    unc_after = trust.posterior_uncertainty("rolling_skew", 0)
    assert unc_after < unc_prior


# ---------------------------------------------------------------------------
# Property 2: contradiction produces elevated uncertainty, not silent averaging
# ---------------------------------------------------------------------------

def _fixed_registry(value_a, value_b, conf=1.0):
    """A minimal 2-source registry so the contradiction test is not
    entangled with the real 9-source registry's behavior."""
    registry = EvidenceRegistry()
    registry.register(EvidenceSourceSpec(
        name="source_a",
        mathematical_formulation="constant test stub",
        required_inputs=["closes"],
        assumptions="test stub",
        known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(value_a, conf, "source_a"),
        is_directional=True,
    ))
    registry.register(EvidenceSourceSpec(
        name="source_b",
        mathematical_formulation="constant test stub",
        required_inputs=["closes"],
        assumptions="test stub",
        known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(value_b, conf, "source_b"),
        is_directional=True,
    ))
    return registry


def _well_trusted(trust, name, bucket, n=20):
    for _ in range(n):
        trust.update(name, bucket, agreed=True)


def test_contradiction_elevates_uncertainty_vs_agreement():
    closes = _synthetic_closes(n=200)
    market_state = _market_state()

    # Agreement scenario: both sources strongly bullish.
    registry_agree = _fixed_registry(value_a=1.0, value_b=1.0)
    reasoner_agree = FastTierReasoner(registry_agree, refit_interval=50)
    trust_agree = ToolTrust()
    bucket = context_bucket({})  # no GARCH/Kalman evidence -> deterministic mid bucket
    _well_trusted(trust_agree, "source_a", bucket)
    _well_trusted(trust_agree, "source_b", bucket)
    hyp_agree = reasoner_agree.hypothesis(closes, market_state, trust_agree)

    # Contradiction scenario: source_a strongly bullish, source_b strongly bearish.
    registry_conflict = _fixed_registry(value_a=1.0, value_b=-1.0)
    reasoner_conflict = FastTierReasoner(registry_conflict, refit_interval=50)
    trust_conflict = ToolTrust()
    _well_trusted(trust_conflict, "source_a", bucket)
    _well_trusted(trust_conflict, "source_b", bucket)
    hyp_conflict = reasoner_conflict.hypothesis(closes, market_state, trust_conflict)

    assert hyp_conflict.aggregate_uncertainty > hyp_agree.aggregate_uncertainty
    # The net belief in the conflict case must not resolve to a confident
    # midpoint -- it should collapse toward 0, not just "small."
    assert abs(hyp_conflict.net_directional_belief) < abs(hyp_agree.net_directional_belief)


def test_contradiction_graded_disagreement_is_continuous_not_capped_at_tie():
    """Exercises the graded opposing_weight/total_weight path directly (an
    unequal weight split), not just the exact-tie special case above. Also
    guards the specific bug the review caught: disagreement must not be
    capped at 0.5 except at an exact tie -- a near-tie split should already
    read as strongly elevated, continuous with (not discontinuous from) the
    exact-tie case."""
    closes = _synthetic_closes(n=200)
    market_state = _market_state()
    bucket = context_bucket({})

    # A 3-vs-1 trust split: source_a earns far more agreement (higher
    # posterior mean -> higher weight) than source_b, but they still point
    # in opposite directions. This produces an *unequal* weight split, so
    # opposing_weight/total_weight lands strictly between 0 and 0.5 -- the
    # exact segment of the formula finding 1 flagged as buggy.
    registry = _fixed_registry(value_a=1.0, value_b=-1.0)
    reasoner = FastTierReasoner(registry, refit_interval=50)
    trust = ToolTrust()
    _well_trusted(trust, "source_a", bucket, n=30)
    _well_trusted(trust, "source_b", bucket, n=3)
    hyp_unequal = reasoner.hypothesis(closes, market_state, trust)

    # A near-50/50 split (both sources very similarly, highly trusted).
    registry_tie_ish = _fixed_registry(value_a=1.0, value_b=-1.0)
    reasoner_tie_ish = FastTierReasoner(registry_tie_ish, refit_interval=50)
    trust_tie_ish = ToolTrust()
    _well_trusted(trust_tie_ish, "source_a", bucket, n=30)
    _well_trusted(trust_tie_ish, "source_b", bucket, n=29)
    hyp_near_tie = reasoner_tie_ish.hypothesis(closes, market_state, trust_tie_ish)

    # Both scenarios contradict, so both must show real elevated
    # uncertainty (the graded path fires, not just "==1.0 at an exact tie").
    assert hyp_unequal.aggregate_uncertainty > 0.0
    assert hyp_near_tie.aggregate_uncertainty > 0.0
    # A more unequal split must show LESS disagreement-driven uncertainty
    # than a near-tie split -- proving the term is graded/continuous, not a
    # step function that only distinguishes "tie" from "everything else."
    assert hyp_near_tie.aggregate_uncertainty > hyp_unequal.aggregate_uncertainty
    # Sanity: with a fixed variance component in both scenarios (same trust
    # mechanics, just different sample counts), a bare average of the raw
    # (uncapped) opposing/total ratio would top out at 0.5 for the near-tie
    # case; the doubled, continuous formula must clear that.
    assert hyp_near_tie.aggregate_uncertainty > 0.5


# ---------------------------------------------------------------------------
# Property 3: genuine abstention when every source is inapplicable
# ---------------------------------------------------------------------------

def test_abstention_when_all_sources_inapplicable():
    # Too little history -> every real source's own applicability gate
    # (Task 4, mirroring Task 3's internal gates) fails, including Kalman's
    # minimal 2-close requirement.
    closes = _synthetic_closes(n=1)
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry, refit_interval=50)
    trust = ToolTrust()

    hyp = reasoner.hypothesis(closes, _market_state(), trust)

    assert isinstance(hyp, Hypothesis)
    assert hyp.aggregate_uncertainty >= 0.99
    assert hyp.load_bearing_sources == []
    assert abs(hyp.net_directional_belief) < 1e-9


def test_abstention_when_market_state_invalid():
    # Ample history, but MarketState flags data as invalid -- applicability
    # zeroes every source's confidence regardless of computed value.
    closes = _synthetic_closes(n=500)
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry, refit_interval=50)
    trust = ToolTrust()

    hyp = reasoner.hypothesis(closes, _market_state(data_quality=DataQuality.INVALID), trust)

    assert hyp.aggregate_uncertainty >= 0.99
    assert hyp.load_bearing_sources == []


# ---------------------------------------------------------------------------
# Refit-caching mechanism
# ---------------------------------------------------------------------------

def test_expensive_sources_not_recomputed_every_bar():
    call_counts = {"garch": 0, "kalman": 0}

    def garch_compute(closes):
        call_counts["garch"] += 1
        return EvidenceValue(0.5, 1.0, "garch_conditional_variance")

    def kalman_v_compute(closes):
        call_counts["kalman"] += 1
        return EvidenceValue(0.1, 1.0, "kalman_filtered_velocity")

    def cheap_compute(closes):
        return EvidenceValue(0.2, 1.0, "momentum_scalar")

    registry = EvidenceRegistry()
    registry.register(EvidenceSourceSpec(
        name="garch_conditional_variance", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=garch_compute,
        is_directional=False,
    ))
    registry.register(EvidenceSourceSpec(
        name="kalman_filtered_velocity", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=kalman_v_compute,
        is_directional=True,
    ))
    registry.register(EvidenceSourceSpec(
        name="momentum_scalar", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=cheap_compute,
        is_directional=True,
    ))

    reasoner = FastTierReasoner(registry, refit_interval=50)
    trust = ToolTrust()
    market_state = _market_state()

    n_decision_points = 200
    for bar in range(1, n_decision_points + 1):
        closes = _synthetic_closes(n=bar, seed=bar)
        reasoner.hypothesis(closes, market_state, trust)

    # Expected refits roughly every 50 bars over 200 decision points -> ~4-5
    # calls, nowhere close to 200. Assert meaningfully fewer than N.
    assert call_counts["garch"] < n_decision_points / 10
    assert call_counts["kalman"] < n_decision_points / 10
    assert call_counts["garch"] >= 1
    assert call_counts["kalman"] >= 1


def test_all_six_expensive_sources_are_refit_cached():
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner, EXPENSIVE_SOURCE_NAMES
    assert EXPENSIVE_SOURCE_NAMES == frozenset({
        "garch_conditional_variance",
        "kalman_filtered_velocity",
        "kalman_innovation",
        "multiscale_vol_ratio",
        "vol_regime_transition",
        "rolling_skew",
        "rolling_excess_kurtosis",
    })


def test_non_directional_sources_reuse_cached_value_between_refits(monkeypatch):
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry, refit_interval=50)
    closes = np.cumsum(np.random.default_rng(3).normal(0, 1, 200)) + 2000.0
    ev1 = reasoner._compute_evidence(closes[:120])
    ev2 = reasoner._compute_evidence(closes[:121])  # 1 bar later, well within refit_interval
    assert ev1["rolling_skew"].value == ev2["rolling_skew"].value
    assert ev1["multiscale_vol_ratio"].value == ev2["multiscale_vol_ratio"].value


# ---------------------------------------------------------------------------
# context_bucket derives from continuous values, not hardcoded categories
# ---------------------------------------------------------------------------

def test_context_bucket_is_continuous_derived_and_monotonic():
    sigma2_values = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e5, 1e8]
    buckets = []
    for sigma2 in sigma2_values:
        evidence = {
            "garch_conditional_variance": EvidenceValue(sigma2, 1.0, "garch_conditional_variance"),
            "kalman_filtered_velocity": EvidenceValue(0.0, 1.0, "kalman_filtered_velocity"),
        }
        buckets.append(context_bucket(evidence))

    # Monotonically non-decreasing as the input magnitude grows.
    for lo, hi in zip(buckets, buckets[1:]):
        assert hi >= lo

    # Spans more than a single bucket across this wide a range, and stays
    # within the declared small fixed number of buckets.
    assert len(set(buckets)) > 1
    assert all(0 <= b < N_CONTEXT_BUCKETS for b in buckets)

    # Same property for the Kalman-velocity axis in isolation.
    velocity_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    vel_buckets = []
    for v in velocity_values:
        evidence = {
            "garch_conditional_variance": EvidenceValue(0.0, 1.0, "garch_conditional_variance"),
            "kalman_filtered_velocity": EvidenceValue(v, 1.0, "kalman_filtered_velocity"),
        }
        vel_buckets.append(context_bucket(evidence))
    for lo, hi in zip(vel_buckets, vel_buckets[1:]):
        assert hi >= lo
    assert len(set(vel_buckets)) > 1


def test_context_bucket_handles_missing_evidence_gracefully():
    # No exception. With NO usable context evidence at all, the result is the
    # explicit GATED_OUT_CONTEXT_BUCKET sentinel -- deliberately outside the
    # genuine bucket range, so "we know nothing about this regime" never
    # pools with a real measurement (see below).
    bucket = context_bucket({})
    assert bucket == GATED_OUT_CONTEXT_BUCKET
    assert not (0 <= bucket < N_CONTEXT_BUCKETS)

    bucket_none_value = context_bucket({
        "garch_conditional_variance": EvidenceValue(None, 0.0, "garch_conditional_variance"),
        "kalman_filtered_velocity": EvidenceValue(None, 0.0, "kalman_filtered_velocity"),
    })
    assert bucket_none_value == bucket

    # A zero-CONFIDENCE (applicability-gated) reading is also "no usable
    # evidence", even though the values themselves are finite.
    assert context_bucket({
        "garch_conditional_variance": EvidenceValue(1.0, 0.0, "garch_conditional_variance"),
        "kalman_filtered_velocity": EvidenceValue(0.5, 0.0, "kalman_filtered_velocity"),
    }) == GATED_OUT_CONTEXT_BUCKET


def test_gated_out_sentinel_is_distinguishable_from_a_genuine_low_reading():
    """Whole-branch-review regression: the fully-gated-out case must NOT
    share a bucket with any value a genuine reading can produce -- otherwise
    one Beta posterior is trained on two categorically different situations
    ("no information" vs "a real, quiet market")."""
    gated = context_bucket({})
    genuine_buckets = set()
    for sigma2 in (0.0, 1e-9, 1e-6, 1e-3, 0.05, 0.5, 5.0, 50.0):
        for velocity in (0.0, 1e-9, 1e-4, 0.01, 0.3, 3.0, 30.0):
            genuine_buckets.add(context_bucket({
                "garch_conditional_variance": EvidenceValue(sigma2, 1.0, "garch_conditional_variance"),
                "kalman_filtered_velocity": EvidenceValue(velocity, 1.0, "kalman_filtered_velocity"),
            }))
    assert gated not in genuine_buckets, (
        "the gated-out sentinel collides with a bucket a genuine reading can produce"
    )


def test_every_context_bucket_is_reachable_over_realistic_inputs():
    """Whole-branch-review regression: the magnitude formula used to be
    non-negative by construction, so the logistic squash never went below
    0.5 and buckets 0 and 1 were STRUCTURALLY unreachable (only 3 of the 5
    buckets were ever live). Over a realistic quiet-to-turbulent span of
    GARCH variance and Kalman velocity, every bucket must be reachable."""
    observed = set()
    for sigma2 in (1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.15, 0.5, 1.5, 5.0, 20.0):
        for velocity in (1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.2, 1.0, 5.0):
            observed.add(context_bucket({
                "garch_conditional_variance": EvidenceValue(sigma2, 1.0, "garch_conditional_variance"),
                "kalman_filtered_velocity": EvidenceValue(velocity, 1.0, "kalman_filtered_velocity"),
            }))
    assert observed == set(range(N_CONTEXT_BUCKETS)), (
        f"only buckets {sorted(observed)} reachable; expected all of "
        f"{list(range(N_CONTEXT_BUCKETS))}"
    )


# --- C1 regression: no structural directional bias ----------------------

def test_net_directional_belief_is_unbiased_on_symmetric_driftless_data():
    """Whole-branch-review C1 regression.

    `hypothesis()` used to fold EVERY source's `copysign(1, value)` into
    `net_directional_belief`. But `multiscale_vol_ratio` and
    `garch_conditional_variance` are non-negative BY CONSTRUCTION, so they
    voted LONG on literally every sample, and no Beta posterior mean (which
    lives strictly in (0, 1)) could ever zero or flip that vote. The measured
    consequence on driftless random walks was a mean
    `net_directional_belief` of about +0.32 and a ~78%/7% LONG/SHORT split --
    a hard long-only bias living in the aggregation math itself.

    On truly symmetric, zero-drift data, the expected net belief is ~0: for
    every up-path there is an equally likely mirrored down-path. This test
    runs the REAL 9-source registry over many independent driftless random
    walks and pins the mean near zero.
    """
    registry = build_default_registry()
    n_paths = 30
    beliefs = []
    for seed in range(n_paths):
        rng = np.random.default_rng(1000 + seed)
        closes = 1900.0 + np.cumsum(rng.normal(0.0, 0.35, size=260))
        reasoner = FastTierReasoner(registry, refit_interval=50)
        trust = ToolTrust()
        for i in range(200, 260, 6):
            hyp = reasoner.hypothesis(closes[:i], None, trust)
            beliefs.append(hyp.net_directional_belief)

    beliefs = np.asarray(beliefs, dtype=np.float64)
    mean_belief = float(beliefs.mean())
    n_long = int((beliefs > 0).sum())
    n_short = int((beliefs < 0).sum())
    print(f"[SYNTHETIC][C1] n={len(beliefs)} mean_net_belief={mean_belief:.4f} "
          f"long={n_long} short={n_short}")

    assert abs(mean_belief) < 0.05, (
        f"net_directional_belief is structurally biased on symmetric driftless data "
        f"(mean={mean_belief:.4f}, long={n_long}, short={n_short}) -- some non-directional "
        f"source is very likely casting a directional vote again"
    )
    # And the sign split must be roughly balanced, not overwhelmingly one-sided.
    directional = n_long + n_short
    assert directional > 0
    assert 0.35 < n_long / directional < 0.65, (
        f"LONG/SHORT split is lopsided ({n_long}/{n_short}) on symmetric data"
    )


def test_non_directional_sources_never_cast_a_directional_vote():
    """The mechanism behind the test above, pinned directly: none of the five
    non-directional registered sources may ever appear in
    `load_bearing_sources` (the directional-vote record), while the
    directional ones do."""
    registry = build_default_registry()
    non_directional = {
        name for name, spec in registry.specs().items() if not spec.is_directional
    }
    assert non_directional == {
        "multiscale_vol_ratio", "vol_regime_transition", "garch_conditional_variance",
        "rolling_skew", "rolling_excess_kurtosis",
    }

    rng = np.random.default_rng(4242)
    closes = 1900.0 + np.cumsum(rng.normal(0.05, 0.35, size=300))
    reasoner = FastTierReasoner(registry, refit_interval=50)
    trust = ToolTrust()
    seen = set()
    for i in range(200, 300, 10):
        hyp = reasoner.hypothesis(closes[:i], None, trust)
        seen.update(name for name, _bucket, _c in hyp.load_bearing_sources)
    assert seen, "expected at least some load-bearing sources over this run"
    assert not (seen & non_directional), (
        f"non-directional sources cast a directional vote: {sorted(seen & non_directional)}"
    )


# ---------------------------------------------------------------------------
# Task 3: bounded look-back window
# ---------------------------------------------------------------------------

def test_max_history_window_bounds_what_sources_see(monkeypatch):
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry, max_history_window=100)
    seen_lengths = []
    original_specs = registry.specs()
    for name, spec in original_specs.items():
        orig_compute = spec.compute
        def wrapped(closes_so_far, orig_compute=orig_compute):
            seen_lengths.append(len(closes_so_far))
            return orig_compute(closes_so_far)
        spec.compute = wrapped
    closes = np.cumsum(np.random.default_rng(4).normal(0, 1, 5000)) + 2000.0
    reasoner._compute_evidence(closes)
    assert max(seen_lengths) <= 100


def test_windowed_garch_output_close_to_full_history_output():
    from research.phase4_garch_volatility_mechanism import fit_garch11
    closes = np.cumsum(np.random.default_rng(5).normal(0, 1, 3000)) + 2000.0
    returns_full = np.diff(closes, prepend=closes[0])
    returns_windowed = np.diff(closes[-2000:], prepend=closes[-2000])
    _params_full, sigma2_full = fit_garch11(returns_full)
    _params_win, sigma2_win = fit_garch11(returns_windowed)
    # Compare the LAST (most recent, decision-relevant) conditional variance --
    # not the whole series, which legitimately differs by construction.
    rel_diff = abs(sigma2_full[-1] - sigma2_win[-1]) / max(abs(sigma2_full[-1]), 1e-12)
    print(f"[TASK3][GARCH windowed-vs-full] rel_diff={rel_diff:.6f}")
    assert rel_diff < 0.5  # same order of magnitude, not identical
