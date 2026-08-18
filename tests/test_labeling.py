"""Quick correctness smoke test for features/labeling.py. Run directly:
python3 tests/test_labeling.py
Split from core/test_smoke.py during the Phase 1 V3 features/ relocation."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from features.labeling import TripleBarrierConfig, cusum_filter, triple_barrier_labels


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


def test_cusum_filter_sane_event_count():
    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.normal(0, 0.1, 5000))
    threshold = np.full(5000, 0.5)
    events = cusum_filter(price, threshold)
    n_events = events.sum()
    assert 0 < n_events < 5000, f"expected a strict subset of bars as events, got {n_events}"
    print(f"OK  cusum_filter selects {n_events}/5000 bars as events (noise reduction working)")


if __name__ == "__main__":
    test_triple_barrier_hits_expected_side()
    test_triple_barrier_vertical_timeout()
    test_meta_label_side_aware()
    test_cusum_filter_sane_event_count()
    print("tests/test_labeling.py: OK")
