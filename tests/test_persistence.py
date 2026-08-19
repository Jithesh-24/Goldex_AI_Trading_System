"""python3 tests/test_persistence.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.persistence import compute_persistence
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_persistence_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_persistence(shared)
    assert set(out.keys()) == {
        "hurst_240", "mean_reversion_speed_60", "half_life_60",
        "autocorr_decay_rate_60", "persistence_score",
        "residual_mean_reversion_60", "fracdiff_slope_60",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_persistence(shared).keys())
    registered_ids = {d.feature_id for d in load_family("persistence")}
    assert computed_keys == registered_ids


def test_no_family_f_survivors():
    """All 7 family-F feature_ids are REDUNDANT in the real evidence file
    (research/output/v3_feature_survivors.json) -- family ablation delta
    <0.1pp per SUMMARY.md finding #7."""
    for d in load_family("persistence"):
        assert d.status == "REDUNDANT", d.feature_id


if __name__ == "__main__":
    test_compute_persistence_shape_and_keys()
    test_registry_matches_computed_keys()
    test_no_family_f_survivors()
    print("tests/test_persistence.py: OK")
