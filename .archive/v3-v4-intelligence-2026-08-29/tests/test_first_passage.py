"""python3 tests/test_first_passage.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.first_passage import compute_first_passage
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_first_passage_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_first_passage(shared)
    assert set(out.keys()) == {
        "hist_p_reach_10bps_10b_60", "hist_time_to_10bps_60",
        "hist_barrier_hit_freq_60", "hist_path_asymmetry_60",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_first_passage(shared).keys())
    registered_ids = {d.feature_id for d in load_family("first_passage")}
    assert computed_keys == registered_ids


def test_warmup_bars_all_nan_and_first_valid_row_finite():
    # window=60, sub_horizon=10 -> first valid row is index 70 (0-indexed);
    # rows before that must be NaN, and hist_p_reach/hit_freq (which have
    # n_checked always == window once past warmup) must be finite there.
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_first_passage(shared)
    for k in ("hist_p_reach_10bps_10b_60", "hist_time_to_10bps_60",
              "hist_barrier_hit_freq_60"):
        assert np.all(np.isnan(out[k][:70])), k
        assert np.isfinite(out[k][70]), k


def test_causal_no_lookahead():
    # Truncating the tail of the series must not change any already-computed
    # value at or before the truncation point -- proof the kernel never
    # reads beyond the current row (spec section 2 causality requirement).
    df = _synthetic_df(200)
    base_full = build_features(df)
    shared_full = build_shared_inputs(df, base_full)
    out_full = compute_first_passage(shared_full)

    cutoff = 150
    df_trunc = df.iloc[:cutoff].reset_index(drop=True)
    base_trunc = build_features(df_trunc)
    shared_trunc = build_shared_inputs(df_trunc, base_trunc)
    out_trunc = compute_first_passage(shared_trunc)

    for k in out_full:
        full_slice = out_full[k][:cutoff]
        trunc_slice = out_trunc[k]
        both_nan = np.isnan(full_slice) & np.isnan(trunc_slice)
        close = np.isclose(full_slice, trunc_slice, equal_nan=False)
        assert np.all(both_nan | close), k


if __name__ == "__main__":
    test_compute_first_passage_shape_and_keys()
    test_registry_matches_computed_keys()
    test_warmup_bars_all_nan_and_first_valid_row_finite()
    test_causal_no_lookahead()
    print("tests/test_first_passage.py: OK")
