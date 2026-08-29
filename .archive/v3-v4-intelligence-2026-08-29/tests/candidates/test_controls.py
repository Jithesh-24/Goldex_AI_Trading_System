"""tests/candidates/test_controls.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.controls import RandomCandidate, NoTradeCandidate


def test_no_trade_candidate_never_opens():
    candidate = NoTradeCandidate()
    for _ in range(50):
        action, sl, tp = candidate.decide(None, None)
        assert action == "NO_TRADE"
        assert sl is None and tp is None
    assert candidate.metadata.mechanism_family == "control"


def test_random_candidate_is_deterministic_given_seed():
    c1 = RandomCandidate(seed=42)
    c2 = RandomCandidate(seed=42)
    actions1 = [c1.decide(None, None)[0] for _ in range(20)]
    actions2 = [c2.decide(None, None)[0] for _ in range(20)]
    assert actions1 == actions2


def test_random_candidate_produces_all_three_actions_over_many_calls():
    candidate = RandomCandidate(seed=1)
    actions = {candidate.decide(None, None)[0] for _ in range(200)}
    assert actions == {"NO_TRADE", "LONG", "SHORT"}


def test_random_candidate_manage_returns_hold_or_exit():
    candidate = RandomCandidate(seed=2)
    results = {candidate.manage(None, None, None) for _ in range(100)}
    assert results <= {"HOLD", "EXIT"}
    assert candidate.metadata.mechanism_family == "control"


if __name__ == "__main__":
    test_no_trade_candidate_never_opens()
    test_random_candidate_is_deterministic_given_seed()
    test_random_candidate_produces_all_three_actions_over_many_calls()
    test_random_candidate_manage_returns_hold_or_exit()
    print("tests/candidates/test_controls.py: OK")
