"""tests/candidates/test_hmm_regime.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from candidates.hmm_regime import HMMRegimeCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_untrained_candidate_always_no_trade():
    candidate = HMMRegimeCandidate()
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"
    assert candidate.manage(_FakeMarketState(1500.0), None, None) == "HOLD"


def test_learn_fits_parameters_and_marks_trained():
    candidate = HMMRegimeCandidate(max_em_iterations=5)
    rng = np.random.default_rng(0)
    prices = 1500.0 + np.cumsum(rng.normal(0, 0.05, 200))
    records = []
    for i, p in enumerate(prices):
        records.append({"event_type": "DECIDE", "market_state_snapshot": {"mid": float(p)}})
    candidate.learn(records)
    assert candidate.is_trained is True
    assert candidate.means is not None and len(candidate.means) == candidate.n_states


def test_metadata_mechanism_family_is_regime_generative():
    candidate = HMMRegimeCandidate()
    assert candidate.metadata.mechanism_family == "regime-generative"


def test_decide_after_learn_returns_valid_action():
    candidate = HMMRegimeCandidate(max_em_iterations=5)
    rng = np.random.default_rng(1)
    prices = 1500.0 + np.cumsum(rng.normal(0, 0.05, 200))
    records = [{"event_type": "DECIDE", "market_state_snapshot": {"mid": float(p)}} for p in prices]
    candidate.learn(records)
    action, sl, tp = candidate.decide(_FakeMarketState(float(prices[-1]) + 0.1), None)
    assert action in ("NO_TRADE", "LONG", "SHORT")


if __name__ == "__main__":
    test_untrained_candidate_always_no_trade()
    test_learn_fits_parameters_and_marks_trained()
    test_metadata_mechanism_family_is_regime_generative()
    test_decide_after_learn_returns_valid_action()
    print("tests/candidates/test_hmm_regime.py: OK")
