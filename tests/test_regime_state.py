"""python3 tests/test_regime_state.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.jump_detection import compute_jump_detection
from features.distribution_info import compute_distribution_info
from features.microstructure_history import compute_microstructure_history
from features.regime_state import compute_regime_state
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_regime_state():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    jump = compute_jump_detection(shared, cusum_k=2.5)
    dist = compute_distribution_info(shared)
    micro = compute_microstructure_history(shared)
    upstream = {**jump, **dist, **micro}
    out = compute_regime_state(shared, upstream)
    assert set(out.keys()) == {
        "vol_state_tercile", "jump_state", "persistence_state",
        "entropy_state", "activity_state", "changepoint_state", "composite_state_id",
    }


def test_registry_matches_computed_keys():
    registered_ids = {d.feature_id for d in load_family("regime_state")}
    assert registered_ids == {
        "vol_state_tercile", "jump_state", "persistence_state",
        "entropy_state", "activity_state", "changepoint_state", "composite_state_id",
    }


if __name__ == "__main__":
    test_compute_regime_state()
    test_registry_matches_computed_keys()
    print("tests/test_regime_state.py: OK")
