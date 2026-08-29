"""python3 tests/test_phase4_dataset.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset, HORIZONS


def test_horizons_constant_matches_spec():
    assert HORIZONS == (15, 45, 90)


def test_select_top_features_ranks_by_mean_importance():
    from research.phase4_dataset import select_top_features
    importances = [
        {"a": 10.0, "b": 1.0, "c": 5.0},
        {"a": 8.0, "b": 2.0, "c": 6.0},
    ]
    top = select_top_features(importances, top_n=2)
    assert top == ["a", "c"], f"expected mean-importance ranking [a, c], got {top}"


def test_assemble_v3_dataset_shapes_and_no_lookahead(tmp_path=None):
    # Real 6.7yr bar history -- capped to a fast dry run via `rows`.
    out = assemble_v3_dataset(max_holding=45, rows=5000)
    feat_v3 = out["feat_v3"]
    t0_idx = out["t0_idx"]
    assert len(t0_idx) > 0, "no CUSUM events found in a 5000-row dry run -- dataset assembly is broken"
    assert set(out["baseline_cols"]) <= set(feat_v3.columns)
    assert set(out["useful_cols"]) <= set(feat_v3.columns)
    # causality smoke check: every event's feature row must have no NaN in
    # the columns this plan will actually train on (mirrors learning/train.py's
    # warmup_ok gate, applied here to the V3-augmented column set).
    cols = out["baseline_cols"] + out["useful_cols"]
    assert feat_v3.loc[t0_idx, cols].notna().all().all(), "warmup NaNs leaked into selected events"
    # every event must resolve strictly in the future and within max_holding+1 bars
    close, high, low, vol_tb = out["close"], out["high"], out["low"], out["vol_tb"]
    from features.labeling import TripleBarrierConfig, triple_barrier_labels
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    assert (labels["t1"].to_numpy() > labels.index.to_numpy()).all()
    assert (labels["holding_bars"].to_numpy() <= 45).all()


if __name__ == "__main__":
    test_horizons_constant_matches_spec()
    test_select_top_features_ranks_by_mean_importance()
    test_assemble_v3_dataset_shapes_and_no_lookahead()
    print("tests/test_phase4_dataset.py: OK")
