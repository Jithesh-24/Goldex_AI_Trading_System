"""python3 tests/test_tick_contract.py"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.tick import Tick


def test_tick_valid():
    t = Tick(symbol="GOLD.i#", market_timestamp="2026-08-18T12:00:00",
              ingestion_timestamp="2026-08-18T12:00:00.010", bid=2500.10,
              ask=2500.35, mid=2500.225, spread=0.25, source="synthetic_replay",
              internal_seq=1)
    assert t.last is None
    assert t.tick_volume is None


def test_tick_rejects_nonpositive_bid():
    try:
        Tick(symbol="GOLD.i#", market_timestamp="2026-08-18T12:00:00",
             ingestion_timestamp="2026-08-18T12:00:00.010", bid=0, ask=2500.35,
             mid=1250.175, spread=2500.35, source="synthetic_replay", internal_seq=1)
        assert False, "expected validation error for bid <= 0"
    except Exception:
        pass


def test_tick_rejects_bad_source_literal():
    try:
        Tick(symbol="GOLD.i#", market_timestamp="2026-08-18T12:00:00",
             ingestion_timestamp="2026-08-18T12:00:00.010", bid=2500.10, ask=2500.35,
             mid=2500.225, spread=0.25, source="made_up_source", internal_seq=1)
        assert False, "expected validation error for bad source literal"
    except Exception:
        pass


if __name__ == "__main__":
    test_tick_valid()
    test_tick_rejects_nonpositive_bid()
    test_tick_rejects_bad_source_literal()
    print("contracts/tick.py: OK")
