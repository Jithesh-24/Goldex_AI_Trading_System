"""python3 tests/test_tick_capture.py"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market.tick_capture import TickCapture


def test_disabled_by_default_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ticks.csv")
        cap = TickCapture(out_path=path)  # enabled not passed -- must default False
        cap.on_tick({"time": "2026-08-22T00:00:00Z", "bid": 2400.1, "ask": 2400.3})
        cap.close()
        assert not os.path.exists(path), "TickCapture must be opt-in, never write when disabled"


def test_enabled_appends_real_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ticks.csv")
        cap = TickCapture(out_path=path, enabled=True)
        cap.on_tick({"time": "2026-08-22T00:00:00Z", "bid": 2400.1, "ask": 2400.3})
        cap.on_tick({"time": "2026-08-22T00:00:01Z", "bid": 2400.2, "ask": 2400.4})
        cap.close()
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["bid"] == "2400.1"


if __name__ == "__main__":
    test_disabled_by_default_writes_nothing()
    test_enabled_appends_real_rows()
    print("tests/test_tick_capture.py: OK")
