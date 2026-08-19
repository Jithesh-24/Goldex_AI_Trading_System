"""python3 tests/test_distribution_info.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.distribution_info import compute_distribution_info
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_distribution_info_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_distribution_info(shared)
    assert set(out.keys()) == {
        "tail_probability_60", "shannon_entropy_returns_60",
        "permutation_entropy_60", "sample_entropy_20",
        "return_concentration_60", "mi_proxy_sign_lag5_240",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_distribution_info(shared).keys())
    registered_ids = {d.feature_id for d in load_family("distribution_info")}
    assert computed_keys == registered_ids


if __name__ == "__main__":
    test_compute_distribution_info_shape_and_keys()
    test_registry_matches_computed_keys()
    print("tests/test_distribution_info.py: OK")
