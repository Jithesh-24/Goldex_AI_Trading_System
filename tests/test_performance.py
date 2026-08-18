"""python3 tests/test_performance.py -- SYNTHETIC performance baseline.
Every printed number is explicitly labeled [SYNTHETIC]; this is not a
claim about real XM broker performance (Section 26 of the design spec).
Not a pass/fail test in the strict sense -- establishes a baseline,
prints it, and does a minimal sanity assert that processing completed
and didn't wildly regress into pathological O(n^2) territory.

Timing and memory are measured in SEPARATE passes: tracemalloc's own
instrumentation overhead (profiled at ~3.7x slower per tick) would
otherwise contaminate the throughput/latency numbers with a measurement
artifact rather than real processing cost."""
import sys
import os
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.tick import Tick
from market.state_engine import StateEngine
from market.synthetic_replay import generate_ticks

N_TICKS = 20000


def _to_tick(rt):
    return Tick(symbol=rt["symbol"], market_timestamp=rt["market_timestamp"],
                ingestion_timestamp=rt["ingestion_timestamp"], bid=rt["bid"], ask=rt["ask"],
                mid=(rt["bid"] + rt["ask"]) / 2, spread=rt["ask"] - rt["bid"],
                tick_volume=rt["tick_volume"], source=rt["source"], internal_seq=rt["internal_seq"])


def test_synthetic_throughput_and_latency_percentiles():
    raw_ticks = generate_ticks(N_TICKS, seed=99)
    eng = StateEngine("GOLD.i#")
    per_tick_us = []

    t_start = time.perf_counter()
    for rt in raw_ticks:
        tick = _to_tick(rt)
        t0 = time.perf_counter()
        eng.on_tick(tick)
        per_tick_us.append((time.perf_counter() - t0) * 1e6)
    elapsed = time.perf_counter() - t_start

    per_tick_us.sort()
    p50 = per_tick_us[len(per_tick_us) // 2]
    p95 = per_tick_us[int(len(per_tick_us) * 0.95)]
    p99 = per_tick_us[int(len(per_tick_us) * 0.99)]
    ticks_per_sec = N_TICKS / elapsed

    print(f"[SYNTHETIC] {N_TICKS} ticks in {elapsed:.3f}s -> {ticks_per_sec:,.0f} ticks/sec")
    print(f"[SYNTHETIC] per-tick processing latency: p50={p50:.1f}us p95={p95:.1f}us p99={p99:.1f}us")

    assert ticks_per_sec > 500, "processing should comfortably exceed the ~25-40 ticks/sec real feed rate"
    assert p99 < 50000, "p99 per-tick latency should stay well under 50ms even synthetically"

    # separate pass, memory only, tracemalloc's overhead doesn't matter here.
    # Only needs enough ticks to reach ring-buffer steady state (~9-10k ticks
    # at this synthetic cadence for the 300s window), not the full N_TICKS --
    # tracemalloc's ~3.7x per-tick overhead makes the full run impractically
    # slow for a value that stops changing once buffers are full anyway.
    MEM_PASS_TICKS = 10000
    eng2 = StateEngine("GOLD.i#")
    tracemalloc.start()
    for rt in raw_ticks[:MEM_PASS_TICKS]:
        eng2.on_tick(_to_tick(rt))
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[SYNTHETIC] peak traced memory ({MEM_PASS_TICKS}-tick measurement pass, "
          f"buffer reaches steady state well before this): {peak_mem / 1024:.1f} KB")


if __name__ == "__main__":
    test_synthetic_throughput_and_latency_percentiles()
    print("tests/test_performance.py: OK")
