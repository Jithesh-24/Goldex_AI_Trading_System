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


def test_series_index_reflects_key_specific_days_when_keys_absent_on_some_days():
    # A key that's missing from record() on some days (e.g. NaN-filtered in
    # bootstrap_from_csv) must NOT have its series index positionally
    # length-matched against the shared _days deque -- that misattributes
    # values to the wrong days.
    buf = DailyBuffer(size=10)
    day1, day2, day3 = date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)
    buf.record(day1, {"x": 1.0, "y": 10.0})
    buf.record(day2, {"x": 2.0})  # y missing this day
    buf.record(day3, {"x": 3.0, "y": 30.0})

    sx = buf.series("x")
    assert list(sx.index) == [day1, day2, day3]
    assert list(sx.values) == [1.0, 2.0, 3.0]

    sy = buf.series("y")
    assert list(sy.index) == [day1, day3]
    assert list(sy.values) == [10.0, 30.0]


def test_same_day_update_in_place_per_key():
    buf = DailyBuffer(size=10)
    day1, day2 = date(2026, 1, 1), date(2026, 1, 2)
    buf.record(day1, {"x": 1.0, "y": 10.0})
    buf.record(day1, {"x": 1.5})  # same-day update for x only, y untouched
    buf.record(day2, {"x": 2.0, "y": 20.0})

    sx = buf.series("x")
    assert list(sx.index) == [day1, day2]
    assert list(sx.values) == [1.5, 2.0]

    sy = buf.series("y")
    assert list(sy.index) == [day1, day2]
    assert list(sy.values) == [10.0, 20.0]


if __name__ == "__main__":
    test_bootstrap_from_real_csv()
    test_record_and_ring_eviction()
    test_series_index_reflects_key_specific_days_when_keys_absent_on_some_days()
    test_same_day_update_in_place_per_key()
    print("tests/test_daily_buffer.py: OK")
