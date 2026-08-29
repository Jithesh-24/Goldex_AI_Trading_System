"""python3 tests/test_feature_schema.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from contracts.feature_schema import (
    FeatureDescriptor, FeatureSetSchema, FeatureStatus, HistoricalCoverage,
    ComputationalCost, UpdateTrigger,
)


def test_feature_descriptor_valid():
    d = FeatureDescriptor(
        feature_id="ret_5", family="baseline_v1",
        mathematical_definition="log(c_t) - log(c_{t-5})",
        source_module="features.features.build_tier1_features",
        required_state=["close"], update_trigger=UpdateTrigger.M1_CLOSE,
        window=5, causal=True, live_compatible=True,
        computational_cost=ComputationalCost.LOW,
        missing_value_policy="NaN during warmup, never zero-filled",
        warmup_bars=5, historical_coverage=HistoricalCoverage.FULL_HISTORY,
        status=FeatureStatus.REQUIRED,
        status_reason="core production feature, deployed since 2026-08-18",
        version="v1",
    )
    assert d.feature_id == "ret_5"
    assert d.causal is True


def test_feature_set_schema_round_trip():
    s = FeatureSetSchema(schema_id="baseline_v1", schema_version="v1",
                          feature_ids=["ret_5", "ret_15"],
                          created_at=datetime.now(timezone.utc))
    data = s.model_dump_json()
    s2 = FeatureSetSchema.model_validate_json(data)
    assert s2.feature_ids == ["ret_5", "ret_15"]


if __name__ == "__main__":
    test_feature_descriptor_valid()
    test_feature_set_schema_round_trip()
    print("tests/test_feature_schema.py: OK")
