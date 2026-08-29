"""tests/candidates/test_tabular_qlearning.py"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.tabular_qlearning import TabularQLearningCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close, vol=0.001):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = vol


def test_decide_returns_valid_action_with_insufficient_history():
    candidate = TabularQLearningCandidate(seed=1)
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action in ("NO_TRADE", "LONG", "SHORT")


def test_learn_updates_q_table_from_training_experience():
    candidate = TabularQLearningCandidate(seed=1, exploration_epsilon=1.0)
    for i in range(20):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.1), None)
    assert len(candidate._open_state_actions) >= 2, "test setup must actually open positions"
    records = [
        {"event_type": "POSITION_CLOSED", "realized_pnl": 5.0, "timestamp": "2020-01-01T00:00:00+00:00"},
        {"event_type": "POSITION_CLOSED", "realized_pnl": -2.0, "timestamp": "2020-01-01T00:01:00+00:00"},
    ]
    q_before = copy.deepcopy(candidate.q_table)
    candidate.learn(records)
    assert candidate.q_table != q_before


def test_metadata_mechanism_family_is_tabular_rl():
    candidate = TabularQLearningCandidate()
    assert candidate.metadata.mechanism_family == "tabular-rl"


def test_manage_returns_hold_or_exit():
    candidate = TabularQLearningCandidate(seed=2)
    for i in range(10):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.1), None)
    result = candidate.manage(_FakeMarketState(1501.0), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_decide_returns_valid_action_with_insufficient_history()
    test_learn_updates_q_table_from_training_experience()
    test_metadata_mechanism_family_is_tabular_rl()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_tabular_qlearning.py: OK")
