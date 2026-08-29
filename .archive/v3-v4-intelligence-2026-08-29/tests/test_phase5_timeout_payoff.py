"""tests/test_phase5_timeout_payoff.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_timeout_payoff import estimate_timeout_payoff


def test_estimate_timeout_payoff_returns_expected_keys():
    result = estimate_timeout_payoff(max_holding=15, rows=20000)
    for key in ("n_timeout_events", "timeout_R_mean", "provisional_proxy"):
        assert key in result
    assert isinstance(result["provisional_proxy"], bool)


if __name__ == "__main__":
    test_estimate_timeout_payoff_returns_expected_keys()
    print("tests/test_phase5_timeout_payoff.py: OK")
