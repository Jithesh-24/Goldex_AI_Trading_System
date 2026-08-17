"""Quick correctness + performance smoke test for core/labeling.py and
core/cv.py. Run directly: python3 core/test_smoke.py
Not a full test suite — just enough to catch a broken barrier scan or a
purge/embargo mask that leaks, before anything is trained on top of it."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.labeling import TripleBarrierConfig, cusum_filter, triple_barrier_labels
from core.volatility import ewma_vol, garman_klass, rogers_satchell, yang_zhang
from core.cv import PurgedWalkForwardCV, purge_and_embargo_mask


def test_triple_barrier_hits_expected_side():
    close = np.full(20, 100.0)
    high = close.copy()
    low = close.copy()
    # event at t0=5: force an upclose at bar 8 that should trip the upper barrier
    high[8] = 110.0
    low[8] = 100.0
    vol = np.full(20, 0.02)  # 2% vol -> pt/sl width = 2% of price = 2.0
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=10)
    res = triple_barrier_labels(close, high, low, np.array([5]), vol, cfg)
    assert res.loc[5, "touch"] == 1, f"expected upper touch, got {res.loc[5, 'touch']}"
    assert res.loc[5, "t1"] == 8, f"expected t1=8, got {res.loc[5, 't1']}"
    print("OK  triple_barrier upper-touch detection")


def test_triple_barrier_vertical_timeout():
    close = np.full(20, 100.0)
    high = close.copy()
    low = close.copy()
    vol = np.full(20, 0.02)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=5)
    res = triple_barrier_labels(close, high, low, np.array([3]), vol, cfg)
    assert res.loc[3, "touch"] == 0
    assert res.loc[3, "t1"] == 8  # 3 + max_holding
    print("OK  triple_barrier vertical-timeout when nothing touched")


def test_meta_label_side_aware():
    close = np.full(20, 100.0)
    high = close.copy()
    low = close.copy()
    low[6] = 90.0  # a down move
    vol = np.full(20, 0.02)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=10)
    # side = SHORT (-1): a down move is FAVORABLE -> should be a win (label=1)
    res_short = triple_barrier_labels(close, high, low, np.array([4]), vol, cfg,
                                       side=np.array([-1.0]))
    assert res_short.loc[4, "label"] == 1, "short into a down move should win"
    # side = LONG (+1): a down move is a stop-out -> label=0
    res_long = triple_barrier_labels(close, high, low, np.array([4]), vol, cfg,
                                      side=np.array([1.0]))
    assert res_long.loc[4, "label"] == 0, "long into a down move should lose"
    print("OK  meta-label is side-aware (same event, opposite verdicts)")


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


def test_cusum_filter_sane_event_count():
    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.normal(0, 0.1, 5000))
    threshold = np.full(5000, 0.5)
    events = cusum_filter(price, threshold)
    n_events = events.sum()
    assert 0 < n_events < 5000, f"expected a strict subset of bars as events, got {n_events}"
    print(f"OK  cusum_filter selects {n_events}/5000 bars as events (noise reduction working)")


def test_vol_estimators_run_and_are_finite_in_steady_state():
    rng = np.random.default_rng(1)
    n = 2000
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.1, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.1, n))
    returns = np.diff(np.log(close), prepend=np.log(close[0]))

    ev = ewma_vol(returns, span=50)
    gk = garman_klass(open_, high, low, close, window=20)
    rs = rogers_satchell(open_, high, low, close, window=20)
    yz = yang_zhang(open_, high, low, close, window=20)

    for name, arr in [("ewma_vol", ev), ("garman_klass", gk), ("rogers_satchell", rs),
                       ("yang_zhang", yz)]:
        tail = arr[200:]
        assert np.isfinite(tail).mean() > 0.95, f"{name} produced too many NaNs in steady state"
    print("OK  volatility estimators run and are finite in steady state")


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
    test_triple_barrier_hits_expected_side()
    test_triple_barrier_vertical_timeout()
    test_meta_label_side_aware()
    test_purge_removes_overlapping_and_embargoes()
    test_cusum_filter_sane_event_count()
    test_vol_estimators_run_and_are_finite_in_steady_state()
    test_perf_at_scale()
    print("\nALL SMOKE TESTS PASSED")
