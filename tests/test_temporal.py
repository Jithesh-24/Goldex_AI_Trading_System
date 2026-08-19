"""python3 tests/test_temporal.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.temporal import compute_temporal
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_temporal_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_temporal(shared)
    assert set(out.keys()) == {
        "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
        "session_asian", "session_london", "session_ny", "session_london_ny_overlap",
        "session_transition_flag", "vol_conditional_on_session",
        "ret_conditional_on_session", "activity_conditional_on_session",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_temporal(shared).keys())
    registered_ids = {d.feature_id for d in load_family("temporal")}
    assert computed_keys == registered_ids


def test_session_bands_mutually_consistent():
    # session_london_ny_overlap must always be a subset of both
    # session_london and session_ny (hour in [13,16) is inside [8,16) and [13,21)).
    df = _synthetic_df(n=2000)
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_temporal(shared)
    overlap = out["session_london_ny_overlap"] > 0
    assert np.all(out["session_london"][overlap] > 0)
    assert np.all(out["session_ny"][overlap] > 0)


def test_conditional_on_session_features_marked_research_only():
    conditional_ids = {
        "vol_conditional_on_session",
        "ret_conditional_on_session",
        "activity_conditional_on_session",
    }
    by_id = {d.feature_id: d for d in load_family("temporal")}
    for feature_id in conditional_ids:
        d = by_id[feature_id]
        assert d.live_compatible is False, feature_id
        assert d.historical_coverage == "RESEARCH_ONLY", feature_id
        assert d.numerical_stability_notes, feature_id


def test_survivors_marked_useful():
    survivor_ids = {"hour_sin", "hour_cos", "ret_conditional_on_session"}
    by_id = {d.feature_id: d for d in load_family("temporal")}
    for feature_id in survivor_ids:
        assert by_id[feature_id].status == "USEFUL", feature_id
    non_survivors = {d.feature_id for d in load_family("temporal")} - survivor_ids
    for feature_id in non_survivors:
        assert by_id[feature_id].status != "USEFUL", feature_id


if __name__ == "__main__":
    test_compute_temporal_shape_and_keys()
    test_registry_matches_computed_keys()
    test_session_bands_mutually_consistent()
    test_conditional_on_session_features_marked_research_only()
    test_survivors_marked_useful()
    print("tests/test_temporal.py: OK")
