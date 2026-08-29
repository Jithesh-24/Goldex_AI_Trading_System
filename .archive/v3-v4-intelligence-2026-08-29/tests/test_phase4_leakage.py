"""python3 tests/test_phase4_leakage.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from learning.cv import PurgedWalkForwardCV, purge_and_embargo_mask


def test_no_train_test_t0_t1_overlap_across_all_folds():
    """Independently re-derives one role's events (direction, h=45) and
    proves NO training event's [t0,t1] window overlaps ANY test fold's
    [test_start,test_end] -- the exact property purge+embargo exists to
    guarantee, checked here from outside learning/cv.py's own unit tests."""
    ds = assemble_v3_dataset(max_holding=45, rows=20000)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    nz = labels["label"].to_numpy() != 0
    t0 = labels.index.to_numpy()[nz]
    t1 = labels["t1"].to_numpy()[nz]

    cv = PurgedWalkForwardCV(n_splits=6, embargo_bars=90, min_train_bars=500)
    checked_folds = 0
    for train_pos, test_pos in cv.split(t0, t1):
        test_start, test_end = int(t0[test_pos].min()), int(t0[test_pos].max())
        train_t0, train_t1 = t0[train_pos], t1[train_pos]
        overlaps = ~((train_t1 < test_start) | (train_t0 > test_end))
        assert not overlaps.any(), f"found {overlaps.sum()} training events overlapping test fold [{test_start},{test_end}]"
        checked_folds += 1
    assert checked_folds >= 2, "not enough folds produced in this dry run to trust the check"


def test_no_future_bars_beyond_t1_influence_features():
    """Feature warmup gate in assemble_v3_dataset only looks backward
    (rolling windows over PAST bars) -- confirm no selected event's
    feature columns depend on the horizon_ok cutoff being violated, i.e.
    every t0 has strictly more than max_holding bars remaining in the
    raw series (already asserted structurally in Task 3's test; re-checked
    here against a second, independent horizon to catch a hardcoded-45
    regression in any role script that copy-pasted Task 4's max_holding)."""
    ds = assemble_v3_dataset(max_holding=90, rows=20000)
    n_bars = len(ds["close"])
    assert (ds["t0_idx"] < n_bars - 90 - 1).all()


def test_calibration_fit_only_uses_passed_rows_not_global_state():
    """PlattCalibrator.fit is a pure function of its arguments -- construct
    it twice with disjoint synthetic data and confirm the two fits differ,
    proving no hidden global/cached state could let a later fit see
    earlier (potentially test-fold) rows."""
    from decision.calibration import PlattCalibrator
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0.1, 0.9, 200)
    y1 = (rng.uniform(0, 1, 200) < p1).astype(float)
    p2 = rng.uniform(0.1, 0.9, 200)
    y2 = (rng.uniform(0, 1, 200) < (1 - p2)).astype(float)  # deliberately inverted relationship
    cal1 = PlattCalibrator.fit(p1, y1)
    cal2 = PlattCalibrator.fit(p2, y2)
    assert abs(cal1.b - cal2.b) > 0.1 or abs(cal1.a - cal2.a) > 0.1, \
        "two calibrators fit on data with opposite relationships produced near-identical params -- suspect shared state"


def test_registry_entries_never_set_champion_or_active():
    """Every models/registry/*_v3_*.json and *_stub.json produced by this
    plan's tasks must never claim is_champion or status=active -- that
    would silently promote an unvalidated research artifact into the
    production champion set this plan's Global Constraints forbid
    touching."""
    import glob
    import json
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REGISTRY_DIR = os.path.join(BASE, "models", "registry")
    phase4_files = glob.glob(os.path.join(REGISTRY_DIR, "*_v3_*.json")) + \
        glob.glob(os.path.join(REGISTRY_DIR, "*_stub.json"))
    assert len(phase4_files) > 0, "expected Phase 4 registry entries to exist by the time this test runs"
    for path in phase4_files:
        with open(path) as f:
            entry = json.load(f)
        assert entry.get("is_champion", False) is False, f"{path} illegally sets is_champion"
        assert entry.get("status") != "active", f"{path} illegally sets status=active"


if __name__ == "__main__":
    test_no_train_test_t0_t1_overlap_across_all_folds()
    test_no_future_bars_beyond_t1_influence_features()
    test_calibration_fit_only_uses_passed_rows_not_global_state()
    test_registry_entries_never_set_champion_or_active()
    print("tests/test_phase4_leakage.py: OK")
