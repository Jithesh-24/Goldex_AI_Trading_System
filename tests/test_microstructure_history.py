"""python3 tests/test_microstructure_history.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.microstructure_history import compute_microstructure_history
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_microstructure_history_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_microstructure_history(shared)
    assert set(out.keys()) == {
        "tick_volume_zscore_60", "tick_volume_accel_20", "spread_change_1",
        "spread_percentile_252", "spread_volatility_60", "tick_volume_spread_ratio",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_microstructure_history(shared).keys())
    registered_ids = {d.feature_id for d in load_family("microstructure_history")}
    assert computed_keys == registered_ids


def test_all_six_are_non_survivors_and_registered_redundant():
    # No family-H feature survived v3 feature selection (research/output/
    # v3_feature_survivors.json's decisions dict: all keep=False) --
    # verify every registry entry reflects that with status=REDUNDANT.
    descriptors = {d.feature_id: d for d in load_family("microstructure_history")}
    assert len(descriptors) == 6
    for feature_id, d in descriptors.items():
        assert d.status.value == "REDUNDANT", feature_id


def test_live_compatible_split():
    # tick_volume-only features CAN run live (M1 bar tick-count is tracked
    # per-bar); spread-dependent features (including the mixed ratio) CANNOT
    # -- MarketState's bounded live buffer has no per-bar spread history.
    descriptors = {d.feature_id: d for d in load_family("microstructure_history")}
    live_true = {"tick_volume_zscore_60", "tick_volume_accel_20"}
    live_false = {"spread_change_1", "spread_percentile_252",
                  "spread_volatility_60", "tick_volume_spread_ratio"}
    for feature_id in live_true:
        assert descriptors[feature_id].live_compatible is True, feature_id
    for feature_id in live_false:
        assert descriptors[feature_id].live_compatible is False, feature_id


if __name__ == "__main__":
    test_compute_microstructure_history_shape_and_keys()
    test_registry_matches_computed_keys()
    test_all_six_are_non_survivors_and_registered_redundant()
    test_live_compatible_split()
    print("tests/test_microstructure_history.py: OK")
