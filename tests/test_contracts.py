"""Contract validation smoke tests. Plain-assert, run directly:
python3 tests/test_contracts.py -- matches core/test_smoke.py convention,
no pytest dependency."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.model_registry import ModelRegistryEntry
from contracts.market_state import MarketState
from contracts.feature_schema import FeatureDescriptor, FeatureSetSchema
from contracts.virtual_trade import VirtualTrade
from contracts.journal import SignalEvent, LearningEvent


def test_model_registry_entry_valid():
    entry = ModelRegistryEntry(
        model_id="direction_catboost_20260818",
        family="direction",
        algorithm="catboost",
        artifact_path="active/primary.cbm",
        created_at="2026-08-18T09:16:20",
        status="active",
        is_champion=True,
    )
    assert entry.model_id == "direction_catboost_20260818"
    assert entry.status == "active"


def test_model_registry_entry_rejects_bad_family():
    try:
        ModelRegistryEntry(
            model_id="x", family="not_a_real_family", algorithm="catboost",
            artifact_path="x.cbm", created_at="2026-08-18T09:16:20", status="active",
        )
        assert False, "expected validation error for bad family"
    except Exception as e:
        assert "family" in str(e).lower() or "literal" in str(e).lower()


def test_market_state_valid():
    ms = MarketState(
        symbol="GOLD.i#", source="synthetic_replay", sequence=1,
        market_timestamp="2026-08-18T12:00:00", ingestion_timestamp="2026-08-18T12:00:00.010",
        processing_timestamp="2026-08-18T12:00:00.011", bid=2500.10, ask=2500.35,
        mid=2500.225, spread=0.25, tick_count_60s=1, tick_count_300s=1,
        tick_rate_per_sec=0.2, feed_health="CONNECTED", last_tick_age_sec=0.01,
    )
    assert ms.spread == 0.25
    assert ms.ask > ms.bid
    assert ms.last_quality.value == "UNAVAILABLE"


def test_market_state_rejects_nonpositive_bid():
    try:
        MarketState(
            symbol="GOLD.i#", source="synthetic_replay", sequence=1,
            market_timestamp="2026-08-18T12:00:00", ingestion_timestamp="2026-08-18T12:00:00.010",
            processing_timestamp="2026-08-18T12:00:00.011", bid=0, ask=2500.35,
            mid=1250.175, spread=2500.35, tick_count_60s=1, tick_count_300s=1,
            tick_rate_per_sec=0.2, feed_health="CONNECTED", last_tick_age_sec=0.01,
        )
        assert False, "expected validation error for bid <= 0"
    except Exception:
        pass


def test_feature_set_schema_valid():
    schema = FeatureSetSchema(
        schema_id="root-28col-2026-08-18", schema_version="v1",
        feature_ids=["ewma_vol", "volatility"],
        created_at="2026-08-18T09:16:20"
    )
    assert len(schema.feature_ids) == 2
    assert schema.schema_id == "root-28col-2026-08-18"


def test_feature_descriptor_rejects_missing_required_field():
    try:
        FeatureDescriptor(
            feature_id="x", family="y", mathematical_definition="z",
            source_module="m", causal=True, live_compatible=True,
            computational_cost="LOW", missing_value_policy="NaN",
            warmup_bars=0, historical_coverage="FULL_HISTORY",
            status="REQUIRED",
            # Missing: status_reason (required field with no default)
            version="v1",
            update_trigger="M1_CLOSE"
        )
        assert False, "expected validation error for missing status_reason field"
    except Exception:
        pass


def test_virtual_trade_valid():
    vt = VirtualTrade(
        trade_id="t-1", signal_timestamp="2026-08-18T12:00:00",
        direction=1, entry=2500.0, sl=2495.0, tp=2510.0,
        model_versions={"direction": "direction_catboost_20260818"},
    )
    assert vt.direction == 1
    assert vt.expected_value is None


def test_virtual_trade_rejects_bad_direction_type():
    try:
        VirtualTrade(
            trade_id="t-1", signal_timestamp="2026-08-18T12:00:00",
            direction="up", entry=2500.0, sl=2495.0, tp=2510.0,
        )
        assert False, "expected validation error for non-int direction"
    except Exception:
        pass


def test_journal_events_valid():
    sig = SignalEvent(trade_id="t-1", timestamp="2026-08-18T12:00:00", payload={"side": 1})
    learn = LearningEvent(timestamp="2026-08-18T23:59:00", payload={"note": "eod"})
    assert sig.schema_version == "v1"
    assert learn.payload["note"] == "eod"


if __name__ == "__main__":
    test_model_registry_entry_valid()
    test_model_registry_entry_rejects_bad_family()
    test_market_state_valid()
    test_market_state_rejects_nonpositive_bid()
    test_feature_set_schema_valid()
    test_feature_descriptor_rejects_missing_required_field()
    test_virtual_trade_valid()
    test_virtual_trade_rejects_bad_direction_type()
    test_journal_events_valid()
    print("contracts/: ALL OK")
