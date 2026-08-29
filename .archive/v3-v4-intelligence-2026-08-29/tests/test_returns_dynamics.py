"""python3 tests/test_returns_dynamics.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.returns_dynamics import compute_returns_dynamics
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_returns_dynamics_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_returns_dynamics(shared)
    assert set(out.keys()) == {
        "ret_240", "sign_ret_240", "ret_accel_5_15", "ret_decel_15_60",
        "run_length_signed", "return_autocorr_20", "return_autocorr_60",
        "return_pacf1_60", "sign_flip_rate_20", "rolling_mean_ret_20",
        "rolling_median_ret_20", "return_dispersion_20",
        "upside_downside_asymmetry_60", "return_skew_60", "return_kurt_60",
        "return_skew_240", "return_percentile_rank_60",
        "return_quantile_pos_240", "directional_entropy_60",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_returns_dynamics(shared).keys())
    registered_ids = {d.feature_id for d in load_family("returns_dynamics")}
    assert computed_keys == registered_ids


if __name__ == "__main__":
    test_compute_returns_dynamics_shape_and_keys()
    test_registry_matches_computed_keys()
    print("tests/test_returns_dynamics.py: OK")
