"""python3 tests/test_causality.py -- for every non-NaN feature value at
row i, changing rows AFTER i must never change that value. This is the
executable version of the causality claim research/features_v3.py's old
docstring made but never actually tested (Phase 3 spec section 2).

Coverage note: the primary n=500 series (check_rows=260) exercises all 28
baseline columns (features.features.build_features) and all 92 candidate
columns (features.replay_engine.build_candidate_features) for real (i.e.
compares actual non-NaN values, not NaN-vs-NaN). A handful of features have
warmup windows longer than 260 bars and would be compared vacuously (both
sides all-NaN) at that series length -- those get dedicated longer/differently
-shaped series below so every column is genuinely exercised somewhere in this
file:
  - fracdiff_slope_60 (candidate) / fracdiff_log_price (baseline): fracdiff
    warmup is 1457 bars (features/registry/baseline_v1/fracdiff_log_price.json),
    +60 more for fracdiff_slope_60's own rolling window -> first real value
    around bar 1516. Covered by *_long_warmup_features below (n=2500).
  - hurst_480 (baseline): 480-bar rolling window, first real value at bar
    480 -- past the primary test's 260-row check window. Covered by
    *_long_warmup_features below (n=2500).
  - vol_percentile_252 / vol_state_tercile / spread_percentile_252
    (candidate): computed on a `.resample("1D")` of the input series with
    `min_periods=60`, so they need ~60 distinct calendar days of history,
    not 60 rows. On a 1-minute-bar series that's ~86,400 rows; on a daily-bar
    series it's just 60 rows. Covered by *_daily_resample_features below
    (daily-frequency synthetic series, n=90).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features.replay_engine import build_candidate_features


def _synthetic_df(n=500, seed=0, freq="1min", start="2026-01-01"):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def _assert_prefix_matches(a_df, b_df, check_rows, label, only_cols=None,
                            require_non_vacuous_cols=()):
    """Compare the first `check_rows` rows of every column (except 'time')
    between a_df and b_df, NaN-aware. If `only_cols` is given, restrict the
    comparison to those columns. `require_non_vacuous_cols` additionally
    asserts that at least one row in [0, check_rows) is non-NaN on both
    sides for each named column -- guards against a test that "passes" only
    because both sides are entirely NaN in the checked range."""
    cols = only_cols if only_cols is not None else [c for c in a_df.columns if c != "time"]
    for col in cols:
        a = a_df[col].to_numpy(dtype=np.float64)[:check_rows]
        b = b_df[col].to_numpy(dtype=np.float64)[:check_rows]
        both_nan = np.isnan(a) & np.isnan(b)
        if col in require_non_vacuous_cols:
            assert not both_nan.all(), (
                f"{col} is all-NaN in both runs over the first {check_rows} rows -- "
                f"{label} comparison would be vacuous for this column")
        assert np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"{col} changed when {label} -- causality violation")


def test_truncation_does_not_change_past_values():
    df_full = _synthetic_df(n=500)
    df_truncated = df_full.iloc[:300].copy()

    base_full = build_features(df_full)
    base_trunc = build_features(df_truncated)
    feat_full = build_candidate_features(df_full, base_full)
    feat_trunc = build_candidate_features(df_truncated, base_trunc)

    # check_rows=260: inside df_truncated's range (300), deep enough past every
    # warmup window <=260 bars (covers all baseline + candidate columns except
    # the long-warmup / daily-resample ones handled by the dedicated tests below).
    check_rows = 260
    _assert_prefix_matches(base_full, base_trunc, check_rows, "future rows were truncated")
    _assert_prefix_matches(feat_full, feat_trunc, check_rows, "future rows were truncated")


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

    check_rows = 260
    _assert_prefix_matches(base, base_perturbed, check_rows, "future rows were perturbed")
    _assert_prefix_matches(feat, feat_perturbed, check_rows, "future rows were perturbed")


# --- Long-warmup features (fracdiff, hurst_480) ----------------------------
# fracdiff_log_price needs 1457 bars of warmup (truncated FFD weight window,
# features/registry/baseline_v1/fracdiff_log_price.json) and fracdiff_slope_60
# needs 60 more on top of that (~1516 total); hurst_480 needs 480. All are
# all-NaN within the primary test's n=500/check_rows=260 window, which would
# make that comparison vacuous. Use a longer minute-frequency series instead.

_LONG_WARMUP_COLS_BASE = ["fracdiff_log_price", "hurst_480"]
_LONG_WARMUP_COLS_FEAT = ["fracdiff_slope_60"]


def test_truncation_long_warmup_features():
    df_full = _synthetic_df(n=2500)
    df_truncated = df_full.iloc[:2000].copy()

    base_full = build_features(df_full)
    base_trunc = build_features(df_truncated)
    feat_full = build_candidate_features(df_full, base_full)
    feat_trunc = build_candidate_features(df_truncated, base_trunc)

    check_rows = 1900  # inside df_truncated's range (2000), past fracdiff_slope_60's ~1516-bar warmup
    _assert_prefix_matches(base_full, base_trunc, check_rows, "future rows were truncated",
                            only_cols=_LONG_WARMUP_COLS_BASE,
                            require_non_vacuous_cols=_LONG_WARMUP_COLS_BASE)
    _assert_prefix_matches(feat_full, feat_trunc, check_rows, "future rows were truncated",
                            only_cols=_LONG_WARMUP_COLS_FEAT,
                            require_non_vacuous_cols=_LONG_WARMUP_COLS_FEAT)


def test_perturbing_future_rows_long_warmup_features():
    rng = np.random.default_rng(1)
    df = _synthetic_df(n=2500)
    df_perturbed = df.copy()
    df_perturbed.loc[2000:, "close"] = df_perturbed.loc[2000:, "close"] + rng.normal(0, 50, len(df_perturbed) - 2000)
    df_perturbed.loc[2000:, "high"] = df_perturbed.loc[2000:, "close"] + 0.5
    df_perturbed.loc[2000:, "low"] = df_perturbed.loc[2000:, "close"] - 0.5

    base = build_features(df)
    base_perturbed = build_features(df_perturbed)
    feat = build_candidate_features(df, base)
    feat_perturbed = build_candidate_features(df_perturbed, base_perturbed)

    check_rows = 1900
    _assert_prefix_matches(base, base_perturbed, check_rows, "future rows were perturbed",
                            only_cols=_LONG_WARMUP_COLS_BASE,
                            require_non_vacuous_cols=_LONG_WARMUP_COLS_BASE)
    _assert_prefix_matches(feat, feat_perturbed, check_rows, "future rows were perturbed",
                            only_cols=_LONG_WARMUP_COLS_FEAT,
                            require_non_vacuous_cols=_LONG_WARMUP_COLS_FEAT)


# --- Daily-resample features (vol_percentile_252, vol_state_tercile,
# spread_percentile_252) ----------------------------------------------------
# These resample the input to "1D" and require min_periods=60 daily
# observations. On a 1-minute-bar series that's ~86,400 rows; a daily-bar
# synthetic series needs only ~90 rows to exercise the same code path in a
# fraction of a second.

_DAILY_RESAMPLE_COLS = ["vol_percentile_252", "vol_state_tercile", "spread_percentile_252"]


def test_truncation_daily_resample_features():
    df_full = _synthetic_df(n=90, freq="1D", start="2020-01-01")
    df_truncated = df_full.iloc[:80].copy()

    base_full = build_features(df_full)
    base_trunc = build_features(df_truncated)
    feat_full = build_candidate_features(df_full, base_full)
    feat_trunc = build_candidate_features(df_truncated, base_trunc)

    check_rows = 65  # inside df_truncated's range (80), past the ~60-day warmup
    _assert_prefix_matches(feat_full, feat_trunc, check_rows, "future rows were truncated",
                            only_cols=_DAILY_RESAMPLE_COLS,
                            require_non_vacuous_cols=_DAILY_RESAMPLE_COLS)


def test_perturbing_future_rows_daily_resample_features():
    rng = np.random.default_rng(1)
    df = _synthetic_df(n=90, freq="1D", start="2020-01-01")
    df_perturbed = df.copy()
    df_perturbed.loc[80:, "close"] = df_perturbed.loc[80:, "close"] + rng.normal(0, 50, len(df_perturbed) - 80)
    df_perturbed.loc[80:, "high"] = df_perturbed.loc[80:, "close"] + 0.5
    df_perturbed.loc[80:, "low"] = df_perturbed.loc[80:, "close"] - 0.5

    base = build_features(df)
    base_perturbed = build_features(df_perturbed)
    feat = build_candidate_features(df, base)
    feat_perturbed = build_candidate_features(df_perturbed, base_perturbed)

    check_rows = 65
    _assert_prefix_matches(feat, feat_perturbed, check_rows, "future rows were perturbed",
                            only_cols=_DAILY_RESAMPLE_COLS,
                            require_non_vacuous_cols=_DAILY_RESAMPLE_COLS)


if __name__ == "__main__":
    test_truncation_does_not_change_past_values()
    test_perturbing_future_rows_does_not_change_past_values()
    test_truncation_long_warmup_features()
    test_perturbing_future_rows_long_warmup_features()
    test_truncation_daily_resample_features()
    test_perturbing_future_rows_daily_resample_features()
    print("tests/test_causality.py: OK")
