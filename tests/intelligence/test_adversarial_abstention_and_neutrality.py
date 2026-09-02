"""tests/intelligence/test_adversarial_abstention_and_neutrality.py -- Task 8
(part 1 of 2): adversarial coverage for two of the mandate's required
properties -- abstention under genuine uncertainty, and direction
neutrality that correctly TRACKS a real trend (rather than merely being
unbiased on driftless data). See
tests/intelligence/test_adversarial_credit_and_reassessment.py for the
remaining three properties (thesis invalidation, continuous reassessment)
plus the two cross-reference notes (credit assignment, causality)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


def test_low_confidence_evidence_produces_no_trade():
    """A single registered source with a near-zero-confidence EvidenceValue
    must not produce a confident directional call -- the reasoner has to
    genuinely abstain (near-zero belief) or flag high uncertainty, not
    manufacture conviction out of a source that admits it barely knows
    anything."""
    from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
    from intelligence.fast_tier import FastTierReasoner, ToolTrust

    registry = EvidenceRegistry()
    registry.register(EvidenceSourceSpec(
        name="weak_source", mathematical_formulation="stub", required_inputs=[],
        assumptions="stub", known_failure_conditions="none",
        compute=lambda closes: EvidenceValue(1.0, 0.02, "weak_source"),  # confidence near zero
        is_directional=True, computational_cost_hint="stub",
    ))
    reasoner = FastTierReasoner(registry)
    hyp = reasoner.hypothesis(np.linspace(2000, 2001, 50), market_state=None, trust=ToolTrust())
    assert abs(hyp.net_directional_belief) < 0.05 or hyp.aggregate_uncertainty > 0.5


def test_direction_neutrality_tracks_real_trend_not_a_permanent_bias():
    """Complements test_directional_belief_unbiased_on_symmetric_data (Phase 2
    fix wave, tests/intelligence/test_fast_tier.py) which proves no bias on
    DRIFTLESS data. This proves the belief correctly FLIPS sign when the
    underlying trend flips -- a permanently-biased system could coincidentally
    pass the driftless test while still being unable to track a real reversal."""
    from intelligence.evidence_sources import build_default_registry
    from intelligence.fast_tier import FastTierReasoner, ToolTrust

    registry = build_default_registry()
    reasoner = FastTierReasoner(registry)
    trust = ToolTrust()
    up = np.linspace(2000.0, 2100.0, 400)
    down = np.linspace(2100.0, 2000.0, 400)
    hyp_up = reasoner.hypothesis(up, market_state=None, trust=trust)
    hyp_down = reasoner.hypothesis(down, market_state=None, trust=trust)
    assert hyp_up.net_directional_belief > 0
    assert hyp_down.net_directional_belief < 0
