"""python3 tests/test_daily_buffer.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from features.daily_buffer import DailyBuffer


def test_bootstrap_from_real_csv():
    buf = DailyBuffer(size=252)
    buf.bootstrap_from_csv("data/gold_seed.csv", value_cols=["close", "spread"])
    s = buf.series("close")
    assert len(s) <= 252
    assert s.index.is_monotonic_increasing


def test_record_and_ring_eviction():
    buf = DailyBuffer(size=3)
    buf.record(date(2026, 1, 1), {"x": 1.0})
    buf.record(date(2026, 1, 2), {"x": 2.0})
    buf.record(date(2026, 1, 3), {"x": 3.0})
    buf.record(date(2026, 1, 4), {"x": 4.0})  # evicts 2026-01-01
    s = buf.series("x")
    assert len(s) == 3
    assert list(s.values) == [2.0, 3.0, 4.0]


if __name__ == "__main__":
    test_bootstrap_from_real_csv()
    test_record_and_ring_eviction()
    print("tests/test_daily_buffer.py: OK")
