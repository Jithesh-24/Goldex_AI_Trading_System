"""Regression tests for the Opportunity fail-open -> fail-closed fix
(targeted correction pass, 2026-08-24)."""
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.ev_engine import evaluate
from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from contracts.market_state import MarketState, FeedHealthState


def _market_state():
    now = datetime.now(timezone.utc)
    return MarketState(symbol="XAUUSD", source="synthetic_replay", sequence=1,
                        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
                        bid=100.0, ask=100.02, mid=100.01, spread=0.02,
                        tick_count_60s=10, tick_count_300s=50, tick_rate_per_sec=0.17,
                        realized_vol_60s=0.001, feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.1)


def _strong_direction():
    return DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                            probability_long=0.9, probability_short=0.1, calibrated=True)


def _strong_barrier():
    return BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                          p_tp=0.9, calibrated=True)


def _mae_mfe():
    return (MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                       q50=0.3, q75=0.5, q90=0.8),
            MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                      q50=0.6, q75=1.5, q90=2.5))


@pytest.mark.parametrize("bad_status", ["UNAVAILABLE", "DATA_LIMITED", "INVALID", "STALE"])
def test_opportunity_untrusted_status_forces_no_trade(bad_status):
    mae, mfe = _mae_mfe()
    opp = OpportunityOutput(model_id="opportunity_v3_candidate_h15", horizon=15, model_status=bad_status,
                             probability_take=0.9, calibrated=True)
    d = evaluate(_market_state(), _strong_direction(), opp, _strong_barrier(), 0.3, mae, mfe,
                 timeout_r=0.2, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


def test_opportunity_ok_status_missing_probability_forces_no_trade():
    mae, mfe = _mae_mfe()
    opp = OpportunityOutput(model_id="opportunity_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                             probability_take=None, calibrated=True)
    d = evaluate(_market_state(), _strong_direction(), opp, _strong_barrier(), 0.3, mae, mfe,
                 timeout_r=0.2, timeout_r_provisional_proxy=False)
    assert d.decision == "NO_TRADE"


@pytest.mark.parametrize("ok_status", ["VALIDATED", "CANDIDATE"])
def test_opportunity_trusted_high_probability_unchanged_behavior(ok_status):
    mae, mfe = _mae_mfe()
    opp = OpportunityOutput(model_id="opportunity_v3_candidate_h15", horizon=15, model_status=ok_status,
                             probability_take=0.9, calibrated=True)
    d = evaluate(_market_state(), _strong_direction(), opp, _strong_barrier(), 0.3, mae, mfe,
                 timeout_r=0.2, timeout_r_provisional_proxy=False)
    # Must NOT be forced to NO_TRADE by the Opportunity gate itself -- a trusted
    # status with a high take-probability must behave exactly as before this fix.
    assert d.decision != "NO_TRADE"


if __name__ == "__main__":
    for status in ["UNAVAILABLE", "DATA_LIMITED", "INVALID", "STALE"]:
        test_opportunity_untrusted_status_forces_no_trade(status)
    test_opportunity_ok_status_missing_probability_forces_no_trade()
    for status in ["VALIDATED", "CANDIDATE"]:
        test_opportunity_trusted_high_probability_unchanged_behavior(status)
    print("tests/test_ev_engine_opportunity_gate.py: OK")
