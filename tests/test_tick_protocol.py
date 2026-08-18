"""python3 tests/test_tick_protocol.py"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market.tick_protocol import encode_tick_frame, encode_backfill_frame, decode_frame, FRAME_TICK, FRAME_BACKFILL


def test_tick_frame_roundtrip():
    line = encode_tick_frame("GOLD.i#", "2026-08-18T12:00:00",
                              2500.10, 2500.35, 3, "mt5_live", 1)
    frame = decode_frame(line)
    assert frame["type"] == FRAME_TICK
    assert frame["bid"] == 2500.10
    assert frame["internal_seq"] == 1


def test_backfill_frame_roundtrip():
    bars = [{"time_iso": "2026-08-18T11:59:00", "open": 2500.0, "high": 2500.5,
             "low": 2499.8, "close": 2500.2, "tick_volume": 42, "spread": 25}]
    line = encode_backfill_frame("GOLD.i#", bars)
    frame = decode_frame(line)
    assert frame["type"] == FRAME_BACKFILL
    assert frame["bars"][0]["close"] == 2500.2


def test_decode_rejects_malformed():
    try:
        decode_frame("not json at all")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        decode_frame(json.dumps({"type": "not_a_real_type"}))
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_tick_frame_roundtrip()
    test_backfill_frame_roundtrip()
    test_decode_rejects_malformed()
    print("market/tick_protocol.py: OK")
