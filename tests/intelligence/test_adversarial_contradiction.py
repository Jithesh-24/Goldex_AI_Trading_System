"""tests/intelligence/test_adversarial_contradiction.py -- Task 7:
adversarial characterization coverage proving that when evidence sources
genuinely CONTRADICT each other (half strongly LONG, half strongly SHORT,
equal confidence and equal trust), the Fast Tier does not average that
disagreement into a false-confidence trade. It produces high
`aggregate_uncertainty` (the `disagreement` term in
`FastTierReasoner.hypothesis`) and `FastTierDecisionEngine.decide` abstains
with NO_TRADE -- despite `net_directional_belief` potentially being
non-trivial in magnitude, uncertainty (not just belief magnitude) must gate
the decision.

This is a characterization test of ALREADY-CORRECT behavior (verified in
the Phase 2 whole-branch review's own contradiction test) -- it adds
permanent regression coverage, not a bugfix. If either assertion below
genuinely fails against current code, that is a real defect to report, not
a reason to weaken the assertion."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from contracts.market_state import DataQuality, FeedHealthState, MarketState
from intelligence.bootstrap import analytical_sizing_bootstrap, analytical_sltp_bootstrap
from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
from intelligence.fast_tier import FastTierReasoner, ToolTrust
from simulator.contracts import AccountState, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r


def _build_contradictory_registry() -> EvidenceRegistry:
    registry = EvidenceRegistry()
    for i in range(3):
        registry.register(EvidenceSourceSpec(
            name=f"long_source_{i}", mathematical_formulation="stub", required_inputs=[],
            assumptions="stub", known_failure_conditions="none",
            compute=lambda closes: EvidenceValue(1.0, 1.0, "long_source"),
            is_directional=True, computational_cost_hint="stub",
        ))
        registry.register(EvidenceSourceSpec(
            name=f"short_source_{i}", mathematical_formulation="stub", required_inputs=[],
            assumptions="stub", known_failure_conditions="none",
            compute=lambda closes: EvidenceValue(-1.0, 1.0, "short_source"),
            is_directional=True, computational_cost_hint="stub",
        ))
    return registry


def test_contradictory_evidence_produces_high_uncertainty_not_averaged_confidence():
    registry = _build_contradictory_registry()
    reasoner = FastTierReasoner(registry)
    trust = ToolTrust()  # uninformative prior for all -- equal trust
    closes = np.linspace(2000.0, 2010.0, 50)
    hyp = reasoner.hypothesis(closes, market_state=None, trust=trust)

    assert hyp.aggregate_uncertainty > 0.9
    assert abs(hyp.net_directional_belief) < 0.1  # equal-strength opposition nets near zero


def test_decision_engine_abstains_under_contradiction():
    registry = _build_contradictory_registry()
    trust = ToolTrust()
    reasoner = FastTierReasoner(registry)
    config = SimulatedExecutionConfig()

    def _cheap_cost_gate(market_state, candidate_sl_distance_r):
        return round_trip_cost_r(market_state, candidate_sl_distance_r, max_staleness_seconds=float("inf"))

    engine = FastTierDecisionEngine(
        registry=registry,
        trust=trust,
        reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate,
        sizing_bootstrap=analytical_sizing_bootstrap(config),
        sltp_bootstrap=analytical_sltp_bootstrap,
    )

    now = datetime.now(timezone.utc)
    market_state = MarketState(
        symbol="XAUUSD", source="synthetic_replay", sequence=1,
        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
        bid=1999.9, ask=2000.1, mid=2000.0, spread=0.2,
        data_quality=DataQuality.VALID, tick_count_60s=10, tick_count_300s=50,
        tick_rate_per_sec=1.0, market_closed=False, feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5, realized_vol_60s=0.0005,
    )
    account = AccountState.initial(config, now)

    action, sl, tp, size = engine.decide(market_state, account)

    assert action == "NO_TRADE"
    assert sl is None and tp is None and size is None
