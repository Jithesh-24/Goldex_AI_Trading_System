"""python3 tests/test_state_engine.py"""
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.tick import Tick
from market.state_engine import StateEngine, is_market_closed


def _tick(t, bid, ask, seq):
    return Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t + timedelta(milliseconds=10),
                bid=bid, ask=ask, mid=(bid + ask) / 2, spread=ask - bid,
                source="synthetic_replay", internal_seq=seq)


def test_m1_construction_within_one_minute():
    eng = StateEngine("GOLD.i#")
    t0 = datetime(2026, 8, 18, 12, 0, 5, tzinfo=timezone.utc)
    ms = eng.on_tick(_tick(t0, 2500.0, 2500.2, 1))
    assert ms.current_m1.open == 2500.0 and ms.current_m1.complete is False
    ms = eng.on_tick(_tick(t0 + timedelta(seconds=10), 2500.5, 2500.7, 2))
    assert ms.current_m1.high == 2500.5 and ms.current_m1.tick_count == 2
    ms = eng.on_tick(_tick(t0 + timedelta(seconds=20), 2499.8, 2500.0, 3))
    assert ms.current_m1.low == 2499.8 and ms.current_m1.close == 2499.8
    print("OK  M1 bar accumulates open/high/low/close/tick_count within a minute")


def test_m1_boundary_rollover():
    eng = StateEngine("GOLD.i#")
    t0 = datetime(2026, 8, 18, 12, 0, 55, tzinfo=timezone.utc)
    eng.on_tick(_tick(t0, 2500.0, 2500.2, 1))
    t1 = datetime(2026, 8, 18, 12, 1, 5, tzinfo=timezone.utc)  # crosses into the next minute
    ms = eng.on_tick(_tick(t1, 2501.0, 2501.2, 2))
    assert ms.completed_m1 is not None and ms.completed_m1.complete is True
    assert ms.completed_m1.close == 2500.0
    assert ms.current_m1.open == 2501.0 and ms.current_m1.complete is False
    print("OK  minute rollover produces correct completed_m1/current_m1 split")


def test_duplicate_tick_rejected():
    eng = StateEngine("GOLD.i#")
    t0 = datetime(2026, 8, 18, 12, 0, 5, tzinfo=timezone.utc)
    ms1 = eng.on_tick(_tick(t0, 2500.0, 2500.2, 1))
    ms2 = eng.on_tick(_tick(t0, 2500.0, 2500.2, 2))  # identical ts+bid+ask
    assert ms1 is not None and ms2 is None
    print("OK  duplicate tick (identical market_timestamp+bid+ask) rejected")


def test_out_of_order_tick_rejected():
    eng = StateEngine("GOLD.i#")
    t0 = datetime(2026, 8, 18, 12, 0, 10, tzinfo=timezone.utc)
    eng.on_tick(_tick(t0, 2500.0, 2500.2, 1))
    earlier = t0 - timedelta(seconds=5)
    ms = eng.on_tick(_tick(earlier, 2499.0, 2499.2, 2))
    assert ms is None
    print("OK  out-of-order tick (timestamp reversal) rejected")


def test_incremental_matches_reference_spread_stats():
    """Incremental spread_mean_60s/spread_std_60s must match a from-scratch
    recomputation on the same window -- this is the incremental-correctness
    proof required by the spec."""
    import statistics
    eng = StateEngine("GOLD.i#")
    t0 = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)
    spreads = []
    ms = None
    for i in range(20):
        bid = 2500.0 + i * 0.01
        spread = 0.2 + (i % 5) * 0.01
        ask = bid + spread
        ms = eng.on_tick(_tick(t0 + timedelta(seconds=i), bid, ask, i + 1))
        spreads.append(spread)
    ref_mean = sum(spreads) / len(spreads)
    ref_std = statistics.pstdev(spreads)
    assert abs(ms.spread_mean_60s - ref_mean) < 1e-9
    assert abs(ms.spread_std_60s - ref_std) < 1e-9
    print("OK  incremental spread_mean_60s/spread_std_60s match from-scratch reference")


def test_bootstrap_seeds_completed_bars_without_live_ticks():
    eng = StateEngine("GOLD.i#")
    eng.bootstrap([
        {"time_iso": "2026-08-18T11:58:00+00:00", "open": 2499.0, "high": 2499.5,
         "low": 2498.8, "close": 2499.2, "tick_volume": 30, "spread": 25},
        {"time_iso": "2026-08-18T11:59:00+00:00", "open": 2499.2, "high": 2500.0,
         "low": 2499.0, "close": 2500.0, "tick_volume": 45, "spread": 24},
    ])
    assert len(eng.completed_m1) == 2
    assert eng.completed_m1[-1].close == 2500.0
    assert eng.current_m1 is None  # bootstrap seeds history only, not an in-progress bar
    print("OK  bootstrap() seeds completed_m1 from backfill without touching current_m1")


def test_is_market_closed_matches_known_hours():
    # Saturday, always closed
    assert is_market_closed(datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)) is True
    # Wednesday midday, open
    assert is_market_closed(datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)) is False
    # Wednesday daily break 21:00-22:00 UTC
    assert is_market_closed(datetime(2026, 8, 19, 21, 30, tzinfo=timezone.utc)) is True
    print("OK  is_market_closed matches the empirically-derived XM session hours")


if __name__ == "__main__":
    test_m1_construction_within_one_minute()
    test_m1_boundary_rollover()
    test_duplicate_tick_rejected()
    test_out_of_order_tick_rejected()
    test_incremental_matches_reference_spread_stats()
    test_bootstrap_seeds_completed_bars_without_live_ticks()
    test_is_market_closed_matches_known_hours()
    print("market/state_engine.py: OK")
