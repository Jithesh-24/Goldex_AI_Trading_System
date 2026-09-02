"""Adversarial test for context-conditional trust in ToolTrust.

This test verifies that ToolTrust correctly conditions trust on context bucket,
not a single global score. We construct two synthetic evidence sources with
opposite patterns: source A is predictive in context bucket 2 but unreliable in
bucket 0, and source B is the reverse. The test confirms that the same source
is judged differently depending on the context bucket.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
from intelligence.fast_tier import FastTierReasoner, ToolTrust


def test_trust_is_conditioned_on_context_not_global():
    trust = ToolTrust()
    for _ in range(30):
        trust.update("source_a", context_bucket=2, agreed=True)
        trust.update("source_a", context_bucket=0, agreed=False)
        trust.update("source_b", context_bucket=0, agreed=True)
        trust.update("source_b", context_bucket=2, agreed=False)

    assert trust.posterior_mean("source_a", 2) > 0.9
    assert trust.posterior_mean("source_a", 0) < 0.1
    assert trust.posterior_mean("source_b", 0) > 0.9
    assert trust.posterior_mean("source_b", 2) < 0.1
    # The SAME source must be judged differently depending on context --
    # this is the entire point of conditioning on context_bucket instead
    # of a single global trust score.
    assert trust.posterior_mean("source_a", 2) - trust.posterior_mean("source_a", 0) > 0.7


def test_hypothesis_weighting_upweights_trustworthy_source():
    """Verifies the trust asymmetry proven above actually flows through the
    REAL `FastTierReasoner.hypothesis()` weighting formula (trust_mean *
    ev.confidence), not just through `posterior_mean` in isolation.

    Two directional stub sources vote in fixed, opposite directions every
    call: `trusty_long` always votes LONG (+1.0, confidence 1.0), and
    `untrusty_short` always votes SHORT (-1.0, confidence 1.0). Neither
    source is `garch_conditional_variance` or `kalman_filtered_velocity`,
    so `context_bucket()` finds no usable context evidence for either of
    those two names and returns the GATED_OUT_CONTEXT_BUCKET sentinel (-1)
    -- a real, reachable, single bucket for this stub registry, which is
    exactly the bucket we pre-train trust in.
    """
    registry = EvidenceRegistry()

    def _long_compute(closes_so_far: np.ndarray) -> EvidenceValue:
        return EvidenceValue(1.0, 1.0, "trusty_long")

    def _short_compute(closes_so_far: np.ndarray) -> EvidenceValue:
        return EvidenceValue(-1.0, 1.0, "untrusty_short")

    registry.register(EvidenceSourceSpec(
        name="trusty_long",
        mathematical_formulation="constant +1.0",
        required_inputs=["closes_so_far"],
        assumptions="stub for adversarial trust-weighting test",
        known_failure_conditions="none -- always votes LONG",
        compute=_long_compute,
        is_directional=True,
    ))
    registry.register(EvidenceSourceSpec(
        name="untrusty_short",
        mathematical_formulation="constant -1.0",
        required_inputs=["closes_so_far"],
        assumptions="stub for adversarial trust-weighting test",
        known_failure_conditions="none -- always votes SHORT",
        compute=_short_compute,
        is_directional=True,
    ))

    trust = ToolTrust()
    # Same training pattern as test_trust_is_conditioned_on_context_not_global:
    # many agreed=True updates for the source we want trusted, many
    # agreed=False for the other, all in the single bucket (-1) this stub
    # registry's evidence actually falls into.
    for _ in range(30):
        trust.update("trusty_long", context_bucket=-1, agreed=True)
        trust.update("untrusty_short", context_bucket=-1, agreed=False)

    assert trust.posterior_mean("trusty_long", -1) > 0.9
    assert trust.posterior_mean("untrusty_short", -1) < 0.1

    reasoner = FastTierReasoner(registry)
    closes = np.linspace(100.0, 110.0, 50)
    hyp = reasoner.hypothesis(closes, None, trust)

    # The REAL weighting formula (trust.posterior_mean * ev.confidence) must
    # upweight the trustworthy LONG source over the untrustworthy SHORT
    # source, so the net directional belief leans LONG (positive) despite
    # both sources casting equal-magnitude, opposite-direction votes.
    assert hyp.net_directional_belief > 0.5
