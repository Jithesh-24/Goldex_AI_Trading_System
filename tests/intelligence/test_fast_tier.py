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
    ))
    registry.register(EvidenceSourceSpec(
        name="source_b",
        mathematical_formulation="constant test stub",
        required_inputs=["closes"],
        assumptions="test stub",
        known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(value_b, conf, "source_b"),
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
    ))
    registry.register(EvidenceSourceSpec(
        name="kalman_filtered_velocity", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=kalman_v_compute,
    ))
    registry.register(EvidenceSourceSpec(
        name="momentum_scalar", mathematical_formulation="stub",
        required_inputs=["closes"], assumptions="stub", known_failure_conditions="none",
        compute=cheap_compute,
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
    # No exception, and it must fall inside the valid bucket range.
    bucket = context_bucket({})
    assert 0 <= bucket < N_CONTEXT_BUCKETS

    bucket_none_value = context_bucket({
        "garch_conditional_variance": EvidenceValue(None, 0.0, "garch_conditional_variance"),
        "kalman_filtered_velocity": EvidenceValue(None, 0.0, "kalman_filtered_velocity"),
    })
    assert bucket_none_value == bucket
