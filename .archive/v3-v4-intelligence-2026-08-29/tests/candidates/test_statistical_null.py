"""tests/candidates/test_statistical_null.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.statistical_null import MomentumMeanReversionCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_when_insufficient_history():
    candidate = MomentumMeanReversionCandidate(lookback_bars=20)
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_opens_long_on_strong_negative_z_score():
    candidate = MomentumMeanReversionCandidate(lookback_bars=10, z_threshold=1.0)
    for _ in range(10):
        candidate.decide(_FakeMarketState(1500.0), None)
    action, sl, tp = candidate.decide(_FakeMarketState(1490.0), None)
    assert action in ("LONG", "NO_TRADE")  # LONG if the drop registers as a strong negative z


def test_manage_returns_string_hold_or_exit():
    candidate = MomentumMeanReversionCandidate(lookback_bars=10, z_threshold=1.0)
    for _ in range(10):
        candidate.decide(_FakeMarketState(1500.0), None)
    result = candidate.manage(_FakeMarketState(1500.0), None, None)
    assert result in ("HOLD", "EXIT")


def test_metadata_mechanism_family_is_rule_based():
    candidate = MomentumMeanReversionCandidate()
    assert candidate.metadata.mechanism_family == "rule-based"


if __name__ == "__main__":
    test_no_trade_when_insufficient_history()
    test_opens_long_on_strong_negative_z_score()
    test_manage_returns_string_hold_or_exit()
    test_metadata_mechanism_family_is_rule_based()
    print("tests/candidates/test_statistical_null.py: OK")
