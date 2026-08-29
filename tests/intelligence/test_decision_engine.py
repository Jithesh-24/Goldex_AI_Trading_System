"""tests/intelligence/test_decision_engine.py -- integration test running
FastTierDecisionEngine through simulator.replay.run_replay on synthetic
data, per Task 6's brief. Uses trivial fixed-value test-double bootstraps
(sltp_bootstrap/sizing_bootstrap) -- the real analytical bootstrap is
Task 11's job; this test's target is the DECIDE/MANAGE contract and the
NO_TRADE/LONG/SHORT gating logic in intelligence/decision_engine.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import FastTierReasoner, Hypothesis, ToolTrust
from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r
from simulator.replay import run_replay


def _make_df(n=20, start_price=1500.0, step=0.1):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [start_price + i * step for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.2 for p in prices], "low": [p - 0.2 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def _fixed_sltp(mid_offset=1.0):
    # NOTE: offset must be large enough to (a) clear entry_fill_price's
    # half-spread adjustment (simulator/engine.py's open_position uses
    # entry_price = mid +/- spread/2, not raw mid -- an offset smaller than
    # half the spread produces an SL/TP on the wrong side of the actual
    # entry price and gets rejected as INVALID_SL/TP_WRONG_SIDE) and (b)
    # translate to a non-negligible R-multiple once divided by
    # realized_vol_60s * mid (a tiny offset makes candidate_sl_distance_r
    # ~0, which makes round_trip_cost_r's cost blow up and always fails the
    # EV gate). 1.0 price-unit comfortably clears both for this test's
    # synthetic spread (~0.2) and vol (~0.0005 * mid=2000 => ~1.0 = 1R).
    def sltp_bootstrap(hyp, market_state):
        mid = market_state.mid
        if hyp.net_directional_belief >= 0:
            return mid - mid_offset, mid + mid_offset
        return mid + mid_offset, mid - mid_offset
    return sltp_bootstrap


def _fixed_sizing(size=0.01):
    return lambda hyp, account: size


def _cheap_cost_gate(market_state, candidate_sl_distance_r):
    """Test-double ev_cost_gate: reuses the real round_trip_cost_r but with
    infinite staleness tolerance (matches SimulatedExecutionConfig's replay
    convention -- historical replay timestamps are years old)."""
    return round_trip_cost_r(market_state, candidate_sl_distance_r, max_staleness_seconds=float("inf"))


def _make_engine(reasoner=None):
    registry = build_default_registry()
    trust = ToolTrust()
    reasoner = reasoner or FastTierReasoner(registry)
    return FastTierDecisionEngine(
        registry=registry,
        trust=trust,
        reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate,
        sizing_bootstrap=_fixed_sizing(),
        sltp_bootstrap=_fixed_sltp(),
    )


class _StubReasoner:
    """A reasoner test double that returns a fixed Hypothesis regardless of
    input -- used to deterministically exercise decide()'s gating branches
    without depending on the real registry's early-history NO_TRADE
    behavior (build_default_registry's sources need dozens of bars before
    they're applicable at all)."""

    def __init__(self, hyp: Hypothesis):
        self.hyp = hyp

    def hypothesis(self, closes_so_far, market_state, trust):
        return self.hyp


def test_decide_and_manage_contract_shapes():
    """DECIDE must return a 4-tuple (action, sl, tp, size) and MANAGE a
    string, honoring the existing DecideFn/ManageFn contract exactly."""
    df = _make_df(n=5)
    engine = _make_engine(reasoner=_StubReasoner(Hypothesis(0.0, 1.0, [])))
    config = SimulatedExecutionConfig()
    recorder = run_replay(df, engine.decide, engine.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    assert len(records) > 0
    for r in records:
        if r.event_type == "DECIDE":
            assert r.action in ("NO_TRADE", "LONG", "SHORT")


def test_high_uncertainty_near_zero_belief_is_no_trade():
    engine = _make_engine(reasoner=_StubReasoner(Hypothesis(0.01, 0.9, [])))
    from contracts.market_state import DataQuality, FeedHealthState, MarketState
    from datetime import datetime, timezone
    from simulator.contracts import AccountState

    now = datetime.now(timezone.utc)
    ms = MarketState(
        symbol="XAUUSD", source="synthetic_replay", sequence=1,
        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
        bid=1999.9, ask=2000.1, mid=2000.0, spread=0.2,
        data_quality=DataQuality.VALID, tick_count_60s=10, tick_count_300s=50,
        tick_rate_per_sec=1.0, market_closed=False, feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5, realized_vol_60s=0.0005,
    )
    config = SimulatedExecutionConfig()
    account = AccountState.initial(config, now)
    action, sl, tp, size = engine.decide(ms, account)
    assert action == "NO_TRADE"
    assert sl is None and tp is None and size is None


def test_strong_directional_low_uncertainty_produces_long_or_short_with_sensible_sltp_size():
    engine = _make_engine(reasoner=_StubReasoner(Hypothesis(0.8, 0.05, [("momentum_scalar", 0, 0.8)])))
    from contracts.market_state import DataQuality, FeedHealthState, MarketState
    from datetime import datetime, timezone
    from simulator.contracts import AccountState

    now = datetime.now(timezone.utc)
    ms = MarketState(
        symbol="XAUUSD", source="synthetic_replay", sequence=1,
        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
        bid=1999.9, ask=2000.1, mid=2000.0, spread=0.02,
        data_quality=DataQuality.VALID, tick_count_60s=10, tick_count_300s=50,
        tick_rate_per_sec=1.0, market_closed=False, feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5, realized_vol_60s=0.0005,
    )
    config = SimulatedExecutionConfig()
    account = AccountState.initial(config, now)
    action, sl, tp, size = engine.decide(ms, account)
    assert action == "LONG"
    assert sl < ms.mid < tp
    assert size == 0.01


def test_strong_negative_belief_produces_short():
    engine = _make_engine(reasoner=_StubReasoner(Hypothesis(-0.8, 0.05, [])))
    from contracts.market_state import DataQuality, FeedHealthState, MarketState
    from datetime import datetime, timezone
    from simulator.contracts import AccountState

    now = datetime.now(timezone.utc)
    ms = MarketState(
        symbol="XAUUSD", source="synthetic_replay", sequence=1,
        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
        bid=1999.9, ask=2000.1, mid=2000.0, spread=0.02,
        data_quality=DataQuality.VALID, tick_count_60s=10, tick_count_300s=50,
        tick_rate_per_sec=1.0, market_closed=False, feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5, realized_vol_60s=0.0005,
    )
    config = SimulatedExecutionConfig()
    account = AccountState.initial(config, now)
    action, sl, tp, size = engine.decide(ms, account)
    assert action == "SHORT"
    assert tp < ms.mid < sl


def test_manage_stub_always_returns_hold():
    engine = _make_engine()
    assert engine.manage(None, None, None) == "HOLD"
    assert engine.manage(object(), object(), object()) == "HOLD"


def test_rejected_entry_does_not_break_decide_fn_contract_through_replay():
    """Force a Phase 1 rejection (INSUFFICIENT_MARGIN via a tiny account
    balance) and confirm the DECIDE/MANAGE contract still holds -- no
    exception, the DECIDE record carries the rejection_reason, and no
    POSITION_CLOSED record ever references it. Task 9 handles proper
    credit-assignment exclusion; this task only needs the seam not to
    break."""
    df = _make_df(n=5)
    engine = _make_engine(reasoner=_StubReasoner(Hypothesis(0.8, 0.05, [])))
    # sizing_bootstrap returns a size guaranteed to blow through margin_free
    # on a tiny starting balance.
    engine.sizing_bootstrap = lambda hyp, account: 1_000_000.0
    config = SimulatedExecutionConfig(starting_balance=10.0)
    recorder = run_replay(df, engine.decide, engine.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    rejected = [r for r in records if r.action == "LONG" and r.rejection_reason is not None]
    assert len(rejected) >= 1
    assert rejected[0].rejection_reason == "INSUFFICIENT_MARGIN"
    closed_for_rejected = [
        r for r in records
        if r.event_type == "POSITION_CLOSED" and r.decision_id == rejected[0].decision_id
    ]
    assert closed_for_rejected == []


def test_engine_runs_through_replay_with_real_reasoner_no_crash():
    """Full end-to-end sanity check with the REAL FastTierReasoner (not a
    stub) over enough bars for the registry's sources to become applicable
    -- confirms the whole seam (registry -> reasoner -> engine -> replay)
    runs without raising, and every DECIDE action is a valid contract
    value."""
    df = _make_df(n=60, step=0.3)
    engine = _make_engine()
    config = SimulatedExecutionConfig()
    recorder = run_replay(df, engine.decide, engine.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    decide_records = [r for r in records if r.event_type == "DECIDE"]
    assert len(decide_records) > 0
    for r in decide_records:
        assert r.action in ("NO_TRADE", "LONG", "SHORT")


if __name__ == "__main__":
    test_decide_and_manage_contract_shapes()
    test_high_uncertainty_near_zero_belief_is_no_trade()
    test_strong_directional_low_uncertainty_produces_long_or_short_with_sensible_sltp_size()
    test_strong_negative_belief_produces_short()
    test_manage_stub_always_returns_hold()
    test_rejected_entry_does_not_break_decide_fn_contract_through_replay()
    test_engine_runs_through_replay_with_real_reasoner_no_crash()
    print("tests/intelligence/test_decision_engine.py: OK")
