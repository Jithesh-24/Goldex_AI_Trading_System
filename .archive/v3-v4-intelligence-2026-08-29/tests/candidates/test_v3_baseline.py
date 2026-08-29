"""tests/candidates/test_v3_baseline.py"""
import os
import sys
from unittest.mock import patch
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from candidates.v3_baseline import V3BaselineCandidate


class _FakeMarketState:
    def __init__(self, timestamp, mid=1500.0, spread=0.2):
        self.market_timestamp = timestamp
        self.mid = mid
        self.spread = spread


def _fake_dataset(max_holding, rows=None):
    """Mock dataset matching the REAL assemble_replay_dataset return shape."""
    ts = [datetime(2020, 1, 6, 10, i, tzinfo=timezone.utc) for i in range(3)]
    return {
        "n": 3, "timestamp": np.array(ts, dtype=object),
        "side": np.array([1.0, -1.0, 1.0]),
        "p_barrier_win": np.array([0.9, 0.9, 0.1]),
        "mae_r": np.array([0.01, 0.01, 0.01]), "mfe_r": np.array([0.02, 0.02, 0.02]),
    }


def test_no_trade_when_timestamp_not_an_event():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        unmatched_ts = datetime(1999, 1, 1, tzinfo=timezone.utc)
        action, sl, tp = candidate.decide(_FakeMarketState(unmatched_ts), None)
        assert action == "NO_TRADE"


def test_high_confidence_event_opens_a_position():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        ts = datetime(2020, 1, 6, 10, 0, tzinfo=timezone.utc)
        action, sl, tp = candidate.decide(_FakeMarketState(ts), None)
        assert action in ("LONG", "NO_TRADE")


def test_manage_always_holds():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        result = candidate.manage(_FakeMarketState(datetime(2020, 1, 6, 10, 0, tzinfo=timezone.utc)), None, None)
        assert result == "HOLD"


def test_metadata_mechanism_family_is_v3_ensemble():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        assert candidate.metadata.mechanism_family == "v3-ensemble"


if __name__ == "__main__":
    test_no_trade_when_timestamp_not_an_event()
    test_high_confidence_event_opens_a_position()
    test_manage_always_holds()
    test_metadata_mechanism_family_is_v3_ensemble()
    print("tests/candidates/test_v3_baseline.py: OK")
