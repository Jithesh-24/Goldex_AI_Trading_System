"""Regression tests for the replay cost model fix (targeted correction
pass, 2026-08-24, defect #4): _ReplayMarketState must use real per-event
historical mid/vol, not the two previously-hardcoded constants
(mid=2350.0, realized_vol_60s=0.0006)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_ev_engine import _ReplayMarketState
from research.phase5_ev_dataset import assemble_replay_dataset


def test_replay_market_state_takes_real_per_event_values():
    ms1 = _ReplayMarketState(spread=0.015, mid=1234.5, vol_60s=0.0009)
    ms2 = _ReplayMarketState(spread=0.015, mid=5678.9, vol_60s=0.0021)
    assert ms1.mid == 1234.5
    assert ms1.realized_vol_60s == 0.0009
    assert ms2.mid == 5678.9
    assert ms2.realized_vol_60s == 0.0021
    assert ms1.mid != ms2.mid
    assert ms1.realized_vol_60s != ms2.realized_vol_60s


def test_dataset_provides_varying_real_mid_and_vol():
    # Try multiple row counts to find one with OOF coverage
    for rows in [100000, 200000, 500000, None]:
        d = assemble_replay_dataset(max_holding=15, rows=rows)
        if d["n"] > 0:
            # Found data with OOF coverage
            assert len(set(d["mid"].tolist())) > 1
            assert len(set(d["vol_60s_proxy"].tolist())) > 1
            assert all(v > 0 for v in d["vol_60s_proxy"])
            # must not equal the old hardcoded constants for every event
            assert not all(m == 2350.0 for m in d["mid"].tolist())
            assert not all(v == 0.0006 for v in d["vol_60s_proxy"].tolist())
            return
    # If we get here, no row count produced events, which indicates data issue
    raise AssertionError("Could not find OOF-covered events in any row count tested")


if __name__ == "__main__":
    test_replay_market_state_takes_real_per_event_values()
    test_dataset_provides_varying_real_mid_and_vol()
    print("tests/test_phase5_ev_engine_replay_cost.py: OK")
