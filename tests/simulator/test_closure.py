"""tests/simulator/test_closure.py"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.closure import classify_gap, is_weekend_close_start


def test_is_weekend_close_start_friday_evening():
    friday_2100 = datetime(2020, 1, 3, 21, 0, tzinfo=timezone.utc)  # a Friday
    assert is_weekend_close_start(friday_2100) is True


def test_is_weekend_close_start_friday_afternoon_not_close():
    friday_1400 = datetime(2020, 1, 3, 14, 0, tzinfo=timezone.utc)
    assert is_weekend_close_start(friday_1400) is False


def test_classify_gap_normal_one_minute():
    prev = datetime(2020, 1, 6, 10, 0, tzinfo=timezone.utc)
    curr = prev + timedelta(minutes=1)
    assert classify_gap(prev, curr) == "NORMAL"


def test_classify_gap_weekend_closure():
    prev = datetime(2020, 1, 3, 21, 0, tzinfo=timezone.utc)  # Friday 21:00
    curr = datetime(2020, 1, 5, 22, 0, tzinfo=timezone.utc)  # Sunday 22:00
    assert classify_gap(prev, curr) == "WEEKEND_CLOSURE"


def test_classify_gap_data_gap_midweek():
    prev = datetime(2020, 1, 7, 10, 0, tzinfo=timezone.utc)  # Tuesday
    curr = prev + timedelta(hours=3)
    assert classify_gap(prev, curr) == "DATA_GAP"


if __name__ == "__main__":
    test_is_weekend_close_start_friday_evening()
    test_is_weekend_close_start_friday_afternoon_not_close()
    test_classify_gap_normal_one_minute()
    test_classify_gap_weekend_closure()
    test_classify_gap_data_gap_midweek()
    print("tests/simulator/test_closure.py: OK")
