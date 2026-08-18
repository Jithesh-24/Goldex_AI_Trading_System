"""Shared wire format between market/mt5_feed.py (Wine Python 3.11,
stdlib-only) and market/feed_listener.py (native). Newline-delimited
JSON, one frame per line: {"type": "tick"|"backfill", ...}. Deliberately
plain-dict, not pydantic -- pydantic validation happens once, on the
native side, in feed_listener.py; the wire format itself must import
cleanly under a bare Wine Python with no third-party packages besides
MetaTrader5."""
import json

FRAME_TICK = "tick"
FRAME_BACKFILL = "backfill"


def encode_tick_frame(symbol, market_timestamp_iso, bid, ask, tick_volume, source, internal_seq):
    """No ingestion_timestamp on the wire, deliberately: ingestion_timestamp
    means "when feed_listener.py received it" -- the sender (mt5_feed.py)
    cannot know that in advance, and stamping it here would make
    feed_latency_sec (ingestion - market) meaningless. feed_listener.py
    stamps it itself, at actual receipt."""
    return json.dumps({
        "type": FRAME_TICK,
        "symbol": symbol,
        "market_timestamp": market_timestamp_iso,
        "bid": bid,
        "ask": ask,
        "tick_volume": tick_volume,
        "source": source,
        "internal_seq": internal_seq,
    }) + "\n"


def encode_backfill_frame(symbol, bars):
    """bars: list of dicts with time_iso, open, high, low, close,
    tick_volume, spread -- same shape xm_ticker.py's backfill CSV rows
    already use, just JSON instead of CSV."""
    return json.dumps({"type": FRAME_BACKFILL, "symbol": symbol, "bars": bars}) + "\n"


def decode_frame(line):
    """Returns a dict with at least "type", or raises ValueError on
    malformed input -- callers decide reject-vs-crash, this function
    never silently returns a partial/guessed frame."""
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    frame = json.loads(line)
    if "type" not in frame or frame["type"] not in (FRAME_TICK, FRAME_BACKFILL):
        raise ValueError(f"unknown frame type: {frame.get('type')!r}")
    return frame
