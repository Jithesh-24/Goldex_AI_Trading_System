"""python3 tests/test_causality.py -- for every non-NaN feature value at
row i, changing rows AFTER i must never change that value. This is the
executable version of the causality claim research/features_v3.py's old
docstring made but never actually tested (Phase 3 spec section 2)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features.replay_engine import build_candidate_features


def _synthetic_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_truncation_does_not_change_past_values():
    df_full = _synthetic_df(n=500)
    df_truncated = df_full.iloc[:300].copy()

    base_full = build_features(df_full)
    base_trunc = build_features(df_truncated)
    feat_full = build_candidate_features(df_full, base_full)
    feat_trunc = build_candidate_features(df_truncated, base_trunc)

    compare_cols = [c for c in feat_full.columns if c != "time"]
    check_rows = 250  # inside df_truncated's range, deep enough past every warmup window (max 252)
    for col in compare_cols:
        a = feat_full[col].to_numpy(dtype=np.float64)[:check_rows]
        b = feat_trunc[col].to_numpy(dtype=np.float64)[:check_rows]
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"{col} changed when future rows were truncated -- causality violation")


def test_perturbing_future_rows_does_not_change_past_values():
    rng = np.random.default_rng(1)
    df = _synthetic_df(n=500)
    df_perturbed = df.copy()
    df_perturbed.loc[300:, "close"] = df_perturbed.loc[300:, "close"] + rng.normal(0, 50, len(df_perturbed) - 300)
    df_perturbed.loc[300:, "high"] = df_perturbed.loc[300:, "close"] + 0.5
    df_perturbed.loc[300:, "low"] = df_perturbed.loc[300:, "close"] - 0.5

    base = build_features(df)
    base_perturbed = build_features(df_perturbed)
    feat = build_candidate_features(df, base)
    feat_perturbed = build_candidate_features(df_perturbed, base_perturbed)

    compare_cols = [c for c in feat.columns if c != "time"]
    check_rows = 250
    for col in compare_cols:
        a = feat[col].to_numpy(dtype=np.float64)[:check_rows]
        b = feat_perturbed[col].to_numpy(dtype=np.float64)[:check_rows]
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"{col} changed when future rows were perturbed -- causality violation")


if __name__ == "__main__":
    test_truncation_does_not_change_past_values()
    test_perturbing_future_rows_does_not_change_past_values()
    print("tests/test_causality.py: OK")
