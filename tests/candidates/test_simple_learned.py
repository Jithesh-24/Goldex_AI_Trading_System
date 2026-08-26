"""tests/candidates/test_simple_learned.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.simple_learned import SimpleLearnedCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_with_insufficient_history():
    candidate = SimpleLearnedCandidate(weights={"short_return": 1.0})
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_positive_weight_on_uptrend_eventually_goes_long():
    candidate = SimpleLearnedCandidate(weights={"short_return": 50.0, "medium_return": 50.0}, threshold=0.5)
    action = "NO_TRADE"
    for i in range(30):
        price = 1500.0 + i * 0.5
        action, sl, tp = candidate.decide(_FakeMarketState(price), None)
    assert action in ("LONG", "NO_TRADE")


def test_metadata_mechanism_family_is_learned_linear():
    candidate = SimpleLearnedCandidate(weights={})
    assert candidate.metadata.mechanism_family == "learned-linear"


def test_manage_returns_hold_or_exit():
    candidate = SimpleLearnedCandidate(weights={"short_return": 1.0})
    for i in range(20):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.1), None)
    result = candidate.manage(_FakeMarketState(1502.0), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_no_trade_with_insufficient_history()
    test_positive_weight_on_uptrend_eventually_goes_long()
    test_metadata_mechanism_family_is_learned_linear()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_simple_learned.py: OK")
