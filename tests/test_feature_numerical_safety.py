"""python3 tests/test_feature_numerical_safety.py -- degenerate inputs
(zero-variance windows, constant price, insufficient samples) must
produce NaN or an explicit quality flag, never inf or silent corruption."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features.replay_engine import build_candidate_features


def test_constant_price_no_inf():
    n = 400
    close = np.full(n, 2000.0)
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({"time": time, "open": close, "high": close, "low": close,
                        "close": close, "tick_volume": np.zeros(n), "spread": np.full(n, 20.0)})
    base = build_features(df)
    feat = build_candidate_features(df, base)
    for col in feat.columns:
        if col == "time":
            continue
        vals = feat[col].to_numpy(dtype=np.float64)
        assert not np.isinf(vals).any(), f"{col} produced inf on constant-price input"


def test_zero_tick_volume_no_inf():
    n = 400
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({"time": time, "open": close, "high": close + 0.5, "low": close - 0.5,
                        "close": close, "tick_volume": np.zeros(n), "spread": np.full(n, 20.0)})
    base = build_features(df)
    feat = build_candidate_features(df, base)
    for col in feat.columns:
        if col == "time":
            continue
        vals = feat[col].to_numpy(dtype=np.float64)
        assert not np.isinf(vals).any(), f"{col} produced inf on zero-tick_volume input"


if __name__ == "__main__":
    test_constant_price_no_inf()
    test_zero_tick_volume_no_inf()
    print("tests/test_feature_numerical_safety.py: OK")
