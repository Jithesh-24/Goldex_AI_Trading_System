"""tests/research/test_phase2_evidence_profile.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.phase2_evidence_profile import compute_evidence_profile


def _closed_record(timestamp, pnl):
    return {
        "environment_tag": "SIMULATED_TRAINING", "timestamp": timestamp, "event_type": "POSITION_CLOSED",
        "market_state_snapshot": {}, "position_view": {}, "action": None,
        "account_state": {}, "realized_pnl": pnl, "cost_amount": 0.5, "outcome": "POLICY_EXIT",
        "gap_type": "NORMAL",
    }


def _decide_record(timestamp, action="NO_TRADE"):
    return {
        "environment_tag": "SIMULATED_TRAINING", "timestamp": timestamp, "event_type": "DECIDE",
        "market_state_snapshot": {}, "position_view": None, "action": action,
        "account_state": {}, "realized_pnl": None, "cost_amount": None, "outcome": None, "gap_type": "NORMAL",
    }


def test_profile_counts_trades_correctly():
    records = [_decide_record(f"2020-01-06T10:0{i}:00+00:00") for i in range(5)]
    records += [_closed_record("2020-01-06T10:05:00+00:00", 10.0)]
    records += [_closed_record("2020-01-06T10:06:00+00:00", -5.0)]
    profile = compute_evidence_profile(records, n_subperiods=2)
    assert profile["n_trades"] == 2
    assert profile["realized_pnl"]["total"] == 5.0


def test_profile_computes_drawdown():
    records = [_closed_record("2020-01-06T10:00:00+00:00", 10.0),
               _closed_record("2020-01-06T10:01:00+00:00", -20.0),
               _closed_record("2020-01-06T10:02:00+00:00", 5.0)]
    profile = compute_evidence_profile(records, n_subperiods=1)
    assert profile["drawdown"]["max_drawdown"] >= 20.0 - 1e-6


def test_profile_has_confidence_intervals_with_lower_le_upper():
    records = [_closed_record(f"2020-01-06T10:{i:02d}:00+00:00", (-1) ** i * 3.0) for i in range(10)]
    profile = compute_evidence_profile(records, n_subperiods=2)
    ci = profile["confidence_intervals"]["mean_pnl_per_trade"]
    assert ci["lower"] <= ci["point"] <= ci["upper"]


def test_profile_consistency_across_subperiods_has_requested_count():
    records = [_closed_record(f"2020-01-06T{10+i:02d}:00:00+00:00", 1.0) for i in range(8)]
    profile = compute_evidence_profile(records, n_subperiods=4)
    assert len(profile["consistency_across_subperiods"]) == 4


if __name__ == "__main__":
    test_profile_counts_trades_correctly()
    test_profile_computes_drawdown()
    test_profile_has_confidence_intervals_with_lower_le_upper()
    test_profile_consistency_across_subperiods_has_requested_count()
    print("tests/research/test_phase2_evidence_profile.py: OK")
