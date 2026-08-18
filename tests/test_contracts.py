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
    ms = MarketState(timestamp="2026-08-18T12:00:00", bid=2500.10, ask=2500.35)
    assert ms.spread is None
    assert ms.ask > ms.bid


def test_market_state_rejects_nonpositive_bid():
    try:
        MarketState(timestamp="2026-08-18T12:00:00", bid=0, ask=2500.35)
        assert False, "expected validation error for bid <= 0"
    except Exception as e:
        assert "bid" in str(e).lower() or "greater than" in str(e).lower()


def test_feature_set_schema_valid():
    fd = FeatureDescriptor(
        name="ewma_vol", family="volatility", source="m1_bars", frequency="per_bar",
        causal=True, required_data=["close"], update_mechanism="incremental",
        version="1", dtype="float64", missing_value_policy="drop_row",
    )
    schema = FeatureSetSchema(schema_version="root-28col-2026-08-18", features=[fd])
    assert schema.features[0].causal is True


def test_feature_descriptor_rejects_missing_required_field():
    try:
        FeatureDescriptor(name="x", family="y", source="z")
        assert False, "expected validation error for missing required fields"
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


if __name__ == "__main__":
    test_model_registry_entry_valid()
    test_model_registry_entry_rejects_bad_family()
    test_market_state_valid()
    test_market_state_rejects_nonpositive_bid()
    test_feature_set_schema_valid()
    test_feature_descriptor_rejects_missing_required_field()
    test_virtual_trade_valid()
    test_virtual_trade_rejects_bad_direction_type()
    print("contracts/: OK")
