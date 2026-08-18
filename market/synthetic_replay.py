"""Synthetic tick stream generator -- built from the real field schema
xm_ticker.py's own code proves (bid/ask, ~25-40ms inter-arrival jitter,
realistic spread magnitude around 0.20-0.30 for XAUUSD). Explicitly
labeled synthetic (source="synthetic_replay") everywhere it's used --
never presented as real broker data. There is no persisted real XM
tick-level dataset to replay instead (Section 2 of the design spec)."""
import random
from datetime import datetime, timedelta, timezone


def generate_ticks(n, start_time=None, seed=42, base_price=2500.0):
    rng = random.Random(seed)
    start_time = start_time or datetime.now(timezone.utc)
    ticks = []
    t = start_time
    price = base_price
    for i in range(n):
        t = t + timedelta(milliseconds=rng.randint(20, 45))
        price += rng.gauss(0, 0.03)
        spread = max(0.15, rng.gauss(0.22, 0.04))
        bid = round(price, 2)
        ask = round(price + spread, 2)
        ticks.append({
            "symbol": "GOLD.i#",
            "market_timestamp": t.isoformat(),
            "ingestion_timestamp": (t + timedelta(milliseconds=rng.randint(1, 8))).isoformat(),
            "bid": bid, "ask": ask,
            "tick_volume": rng.randint(1, 5),
            "source": "synthetic_replay",
            "internal_seq": i + 1,
        })
    return ticks
