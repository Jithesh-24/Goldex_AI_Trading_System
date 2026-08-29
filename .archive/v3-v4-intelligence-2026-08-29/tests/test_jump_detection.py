"""python3 tests/test_jump_detection.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.jump_detection import compute_jump_detection
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_jump_detection_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_jump_detection(shared, cusum_k=2.5)
    assert set(out.keys()) == {
        "cusum_distance_to_threshold", "jump_intensity_60",
        "jump_magnitude_mean_60", "jump_direction_bias_60",
        "bars_since_last_changepoint", "changepoint_intensity_240",
        "vol_shock_zscore", "_bars_since_last_changepoint_internal",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_internal_key_mirrors_public_bars_since():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_jump_detection(shared, cusum_k=2.5)
    np.testing.assert_array_equal(
        out["_bars_since_last_changepoint_internal"],
        out["bars_since_last_changepoint"],
    )


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_jump_detection(shared, cusum_k=2.5).keys())
    public_keys = {k for k in computed_keys if not k.startswith("_")}
    registered_ids = {d.feature_id for d in load_family("jump_detection")}
    assert public_keys == registered_ids
    assert len(registered_ids) == 7


if __name__ == "__main__":
    test_compute_jump_detection_shape_and_keys()
    test_internal_key_mirrors_public_bars_since()
    test_registry_matches_computed_keys()
    print("tests/test_jump_detection.py: OK")
