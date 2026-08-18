"""Quick correctness + performance smoke test for learning/cv.py (and an
integration check spanning features/labeling.py + features/volatility.py).
Run directly: python3 tests/test_cv.py
Split from core/test_smoke.py during the Phase 1 V3 learning/ relocation."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from features.labeling import TripleBarrierConfig, cusum_filter, triple_barrier_labels
from features.volatility import ewma_vol
from learning.cv import PurgedWalkForwardCV, purge_and_embargo_mask


def test_purge_removes_overlapping_and_embargoes():
    t0 = np.array([0, 10, 19, 20, 25, 40])
    t1 = np.array([5, 25, 22, 24, 30, 45])  # events 1 and 2 overlap test window [20,30]
    mask = purge_and_embargo_mask(t0, t1, test_start=20, test_end=30, embargo_bars=5)
    # event0 [0,5] no overlap, no embargo -> keep
    assert mask[0]
    # event1 [10,25] overlaps [20,30] -> purge
    assert not mask[1]
    # event2 [19,22] overlaps -> purge
    assert not mask[2]
    # event3 t0=20 inside test window itself -> overlaps -> purge
    assert not mask[3]
    # event4 t0=25 inside test window -> purge
    assert not mask[4]
    # event5 t0=40, test_end=30, embargo_bars=5 -> embargo zone is (30,35], 40 is outside -> keep
    assert mask[5]
    print("OK  purge_and_embargo_mask drops overlapping + embargoed events, keeps the rest")


def test_perf_at_scale():
    rng = np.random.default_rng(2)
    n = 2_000_000
    close = 2000 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + np.abs(rng.normal(0, 0.05, n))
    low = close - np.abs(rng.normal(0, 0.05, n))
    returns = np.diff(np.log(close), prepend=np.log(close[0]))

    t0 = time.time()
    vol = ewma_vol(returns, span=100)
    t_vol = time.time() - t0

    t0 = time.time()
    threshold = np.clip(vol * 2, 1e-5, None)
    events = cusum_filter(close, threshold)
    t_cusum = time.time() - t0
    event_idx = np.where(events)[0]
    event_idx = event_idx[event_idx < n - 200]

    t0 = time.time()
    cfg = TripleBarrierConfig(pt_mult=2.0, sl_mult=1.5, max_holding=100)
    res = triple_barrier_labels(close, high, low, event_idx, vol, cfg)
    t_tb = time.time() - t0

    t0 = time.time()
    cv = PurgedWalkForwardCV(n_splits=5, embargo_bars=100)
    n_folds = 0
    for train_idx, test_idx in cv.split(res.index.to_numpy(), res["t1"].to_numpy()):
        n_folds += 1
        assert len(train_idx) > 0 and len(test_idx) > 0
    t_cv = time.time() - t0

    print(f"OK  perf @ {n:,} bars, {len(event_idx):,} CUSUM events: "
          f"ewma_vol={t_vol:.2f}s cusum={t_cusum:.2f}s triple_barrier={t_tb:.2f}s "
          f"cv({n_folds} folds)={t_cv:.2f}s")


if __name__ == "__main__":
    test_purge_removes_overlapping_and_embargoes()
    test_perf_at_scale()
    print("tests/test_cv.py: OK")
