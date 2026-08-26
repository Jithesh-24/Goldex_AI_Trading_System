"""tests/candidates/test_regime_conditioned.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.regime_conditioned import RegimeConditionedCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close, realized_vol_60s):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = realized_vol_60s


def test_stays_flat_with_insufficient_history():
    candidate = RegimeConditionedCandidate(vol_lookback_bars=10)
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0, 0.001), None)
    assert action == "NO_TRADE"


def test_stays_flat_in_low_vol_regime():
    candidate = RegimeConditionedCandidate(vol_lookback_bars=10, high_vol_percentile=0.9)
    for i in range(15):
        action, sl, tp = candidate.decide(_FakeMarketState(1500.0 + i * 0.01, 0.0005), None)
    assert action == "NO_TRADE"


def test_metadata_mechanism_family_is_regime_statistical():
    candidate = RegimeConditionedCandidate()
    assert candidate.metadata.mechanism_family == "regime-statistical"


def test_manage_returns_hold_or_exit():
    candidate = RegimeConditionedCandidate(vol_lookback_bars=5)
    for i in range(6):
        candidate.decide(_FakeMarketState(1500.0 + i, 0.001), None)
    result = candidate.manage(_FakeMarketState(1506.0, 0.001), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_stays_flat_with_insufficient_history()
    test_stays_flat_in_low_vol_regime()
    test_metadata_mechanism_family_is_regime_statistical()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_regime_conditioned.py: OK")
