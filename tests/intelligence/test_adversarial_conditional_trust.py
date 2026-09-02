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

from intelligence.fast_tier import ToolTrust


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
