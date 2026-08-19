"""python3 tests/test_market_geometry.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.market_geometry import compute_market_geometry
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_market_geometry_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_market_geometry(shared)
    assert set(out.keys()) == {
        "dist_from_high_20", "dist_from_low_20", "range_position_20",
        "range_position_60", "range_width_20", "range_width_ratio_20_60",
        "displacement_from_equilibrium_60", "breakout_magnitude_20",
        "breakout_failure_magnitude_20", "reversal_frequency_60",
        "avg_run_length_60", "excursion_from_recent_distribution_20",
        "high_low_density_60",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_market_geometry(shared).keys())
    registered_ids = {d.feature_id for d in load_family("market_geometry")}
    assert computed_keys == registered_ids


if __name__ == "__main__":
    test_compute_market_geometry_shape_and_keys()
    test_registry_matches_computed_keys()
    print("tests/test_market_geometry.py: OK")
