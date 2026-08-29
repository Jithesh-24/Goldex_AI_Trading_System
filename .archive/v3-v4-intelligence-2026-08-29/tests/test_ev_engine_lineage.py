"""Regression test: feature_schema_ids must be a genuinely separate dict
from calibration_ids, containing real feature-schema IDs (targeted
correction pass, 2026-08-24, defect #3)."""
import os
import sys
from datetime import datetime, timezone

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


def test_feature_schema_ids_distinct_from_calibration_ids_and_real():
    direction = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                                 probability_long=0.9, probability_short=0.1, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                                     probability_take=0.9, calibrated=True)
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                             p_tp=0.9, calibrated=True)
    mae = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.3, q75=0.5, q90=0.8)
    mfe = MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.6, q75=1.0, q90=1.5)

    d = evaluate(_market_state(), direction, opportunity, barrier, 0.3, mae, mfe,
                 timeout_r=0.2, timeout_r_provisional_proxy=False)

    assert d.feature_schema_ids != d.calibration_ids
    assert d.feature_schema_ids["direction"] == "direction_v3_h15__2026-08-22"
    assert d.feature_schema_ids["opportunity"] == "opportunity_v3_h15__2026-08-22"
    assert d.feature_schema_ids["barrier"] == "barrier_v3_h15__2026-08-22"
    assert d.feature_schema_ids["mae"] == "mae_quantile_v3_h15__2026-08-22"
    assert d.feature_schema_ids["mfe"] == "mfe_quantile_v3_h15__2026-08-22"

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "features", "registry", "schemas")
    for sid in d.feature_schema_ids.values():
        assert os.path.exists(os.path.join(base, f"{sid}.json")), f"missing schema file for {sid}"


if __name__ == "__main__":
    test_feature_schema_ids_distinct_from_calibration_ids_and_real()
    print("tests/test_ev_engine_lineage.py: OK")
