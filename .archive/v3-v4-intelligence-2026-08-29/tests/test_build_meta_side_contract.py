"""tests/test_build_meta_side_contract.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.audit_edge import build_meta


def _synthetic_bars(n=2000):
    rng = np.random.default_rng(0)
    close = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    vol = np.full(n, 0.01)
    return close, high, low, vol


def test_build_meta_uses_caller_supplied_side_directly():
    close, high, low, vol = _synthetic_bars()
    t0_nz = np.arange(10, 1900, 50)
    n = len(t0_nz)
    side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # caller-supplied, NOT derived internally
    has_oof = np.ones(n, dtype=bool)
    side_sub, meta_labels = build_meta(close, high, low, vol, t0_nz, side, has_oof)
    assert np.array_equal(side_sub, side), "build_meta must pass the caller's side through unchanged, not recompute it"
    assert len(meta_labels) == n
    assert set(meta_labels["label"].unique()) <= {0, 1}


def test_build_meta_respects_has_oof_mask():
    close, high, low, vol = _synthetic_bars()
    t0_nz = np.arange(10, 1900, 50)
    n = len(t0_nz)
    side = np.ones(n, dtype=np.float64)
    has_oof = np.arange(n) % 3 == 0
    side_sub, meta_labels = build_meta(close, high, low, vol, t0_nz, side, has_oof)
    assert len(side_sub) == int(has_oof.sum())
    assert len(meta_labels) == int(has_oof.sum())


if __name__ == "__main__":
    test_build_meta_uses_caller_supplied_side_directly()
    test_build_meta_respects_has_oof_mask()
    print("tests/test_build_meta_side_contract.py: OK")
