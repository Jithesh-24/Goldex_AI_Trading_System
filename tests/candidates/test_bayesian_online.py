"""tests/candidates/test_bayesian_online.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.bayesian_online import BayesianOnlineCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_with_insufficient_history():
    candidate = BayesianOnlineCandidate()
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_no_trade_with_uninformative_prior_even_on_momentum():
    candidate = BayesianOnlineCandidate(confidence_threshold=0.65)
    action = "NO_TRADE"
    for i in range(20):
        action, sl, tp = candidate.decide(_FakeMarketState(1500.0 + i * 0.5), None)
    assert action == "NO_TRADE"  # prior (0.5) never clears 0.65 without learn()


def test_learn_updates_posterior_from_wins():
    candidate = BayesianOnlineCandidate(confidence_threshold=0.4)
    for i in range(10):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.5), None)
    assert len(candidate._open_sides) >= 1, "test setup must actually open a position"
    before = (candidate.long_alpha, candidate.long_beta)
    candidate.learn([{"event_type": "POSITION_CLOSED", "realized_pnl": 5.0}])
    after = (candidate.long_alpha, candidate.long_beta)
    assert after != before


def test_metadata_mechanism_family_is_bayesian():
    candidate = BayesianOnlineCandidate()
    assert candidate.metadata.mechanism_family == "bayesian-online"


if __name__ == "__main__":
    test_no_trade_with_insufficient_history()
    test_no_trade_with_uninformative_prior_even_on_momentum()
    test_learn_updates_posterior_from_wins()
    test_metadata_mechanism_family_is_bayesian()
    print("tests/candidates/test_bayesian_online.py: OK")
