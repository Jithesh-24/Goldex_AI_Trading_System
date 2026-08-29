"""python3 tests/test_volatility_dynamics.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.volatility_dynamics import compute_volatility_dynamics
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_volatility_dynamics_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_volatility_dynamics(shared)
    assert set(out.keys()) == {
        "realized_variance_20", "realized_semivar_upside_20",
        "realized_semivar_downside_20", "parkinson_vol_60",
        "vol_acceleration_30", "vol_of_vol_60", "vol_percentile_252",
        "vol_zscore_60", "vol_compression_ratio",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_volatility_dynamics(shared).keys())
    registered_ids = {d.feature_id for d in load_family("volatility_dynamics")}
    assert computed_keys == registered_ids


if __name__ == "__main__":
    test_compute_volatility_dynamics_shape_and_keys()
    test_registry_matches_computed_keys()
    print("tests/test_volatility_dynamics.py: OK")
