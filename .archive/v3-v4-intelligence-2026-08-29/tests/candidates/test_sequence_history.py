"""tests/candidates/test_sequence_history.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.sequence_history import SequenceHistoryCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_with_insufficient_history():
    candidate = SequenceHistoryCandidate()
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_learn_updates_weights():
    candidate = SequenceHistoryCandidate()
    for i in range(15):
        candidate.decide(_FakeMarketState(1500.0 + i * 150.0), None)
    assert len(candidate._open_features) >= 2, "test setup must actually open positions"
    weights_before = dict(candidate.weights)
    records = [
        {"event_type": "POSITION_CLOSED", "realized_pnl": 5.0,
         "market_state_snapshot": {"mid": 1503.0}},
        {"event_type": "POSITION_CLOSED", "realized_pnl": -3.0,
         "market_state_snapshot": {"mid": 1504.0}},
    ]
    candidate.learn(records)
    assert candidate.weights != weights_before


def test_metadata_mechanism_family_is_sequence_history():
    candidate = SequenceHistoryCandidate()
    assert candidate.metadata.mechanism_family == "sequence-history"


def test_manage_returns_hold_or_exit():
    candidate = SequenceHistoryCandidate()
    for i in range(15):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.2), None)
    result = candidate.manage(_FakeMarketState(1503.0), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_no_trade_with_insufficient_history()
    test_learn_updates_weights()
    test_metadata_mechanism_family_is_sequence_history()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_sequence_history.py: OK")
