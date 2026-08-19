"""python3 tests/test_feature_performance.py -- [SYNTHETIC] benchmark of
LiveFeatureEngine.on_m1_close latency/throughput, same two-pass
timing/memory-separation pattern as Phase 2's test_performance.py (see that
file's module docstring for why timing and tracemalloc are measured in
SEPARATE passes: tracemalloc's own instrumentation overhead would otherwise
contaminate the latency numbers with a measurement artifact rather than
real processing cost).

Known bug fixed here (4th recurrence, also hit in Tasks 22/23/24):
MarketState.completed_m1 is level-held (market/state_engine.py:155) -- it
stays non-None on every tick after the first M1 bar completes, not just the
boundary tick. Naively gating on `state.completed_m1 is not None` calls
on_m1_close() on EVERY tick after the first bar (~20000 calls instead of
~666), which caused a real OOM (exit 137) in Task 22 at this exact tick
count. Fixed by edge-triggering on state.completed_m1.start_time change,
mirroring the already-reviewed-clean pattern in tests/test_live_engine.py."""
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contracts.tick import Tick
from market.state_engine import StateEngine
from market.synthetic_replay import generate_ticks
from features.live_engine import LiveFeatureEngine

N_TICKS = 20000


def _to_tick(rt):
    # generate_ticks() returns raw dicts (matching the real feed_listener
    # wire format), not Tick objects -- same conversion test_performance.py
    # uses.
    return Tick(symbol=rt["symbol"], market_timestamp=rt["market_timestamp"],
                ingestion_timestamp=rt["ingestion_timestamp"], bid=rt["bid"], ask=rt["ask"],
                mid=(rt["bid"] + rt["ask"]) / 2, spread=rt["ask"] - rt["bid"],
                tick_volume=rt["tick_volume"], source=rt["source"], internal_seq=rt["internal_seq"])


def test_m1_close_latency_synthetic():
    raw_ticks = generate_ticks(n=N_TICKS, seed=7)
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    m1_close_latencies_us = []
    last_bar_start = None
    for rt in raw_ticks:
        state = engine.on_tick(_to_tick(rt))
        if state is None:
            continue
        live.on_tick(state)
        # Edge-trigger on start_time change -- completed_m1 is level-held
        # (rides on every subsequent MarketState once the first bar closes),
        # so a naive "is not None" gate calls on_m1_close() once per tick
        # instead of once per bar. See module docstring.
        if state.completed_m1 is not None and state.completed_m1.start_time != last_bar_start:
            last_bar_start = state.completed_m1.start_time
            t0 = time.perf_counter()
            live.on_m1_close(engine.completed_m1_window(480))
            m1_close_latencies_us.append((time.perf_counter() - t0) * 1e6)

    assert len(m1_close_latencies_us) > 5, "not enough M1 closes in this synthetic run to measure"
    arr = np.array(m1_close_latencies_us)
    # Caveat: at n≈10 samples, p99 is essentially the observed max, not a statistically robust tail estimate.
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"[SYNTHETIC] on_m1_close latency over {len(arr)} bar closes: "
          f"p50={p50:.0f}us p95={p95:.0f}us p99={p99:.0f}us")
    # Investigated (Phase 2 precedent: treat a wide p50-p99 spread as a
    # signal, not noise): the VERY FIRST real on_m1_close call (2 bars) is
    # a ~450-500ms outlier that dominates p95/p99 out of only ~11 samples
    # at this synthetic cadence; calls 3-11 are consistently ~25-30ms. This
    # is one-time numba JIT compilation (@numba.njit(cache=True) kernels in
    # first_passage.py, distribution_info.py, persistence.py, hurst.py,
    # fracdiff.py, kalman.py, market_geometry.py, returns_dynamics.py -- the family modules on_m1_close
    # calls), not a bug in live_engine.py or the families themselves: a
    # real live process pays this cost exactly once at process warmup, not
    # per bar close, and even including it p99 is still <1s, well under
    # the 2s budget. No fix needed.
    assert p99 < 2_000_000, f"on_m1_close p99={p99:.0f}us exceeds 2s budget"


def test_m1_close_memory_synthetic():
    # Separate, shorter pass under tracemalloc -- matches test_performance.py's
    # established two-pass pattern. Doesn't need the full N_TICKS: DailyBuffer
    # and the tick-level ring buffers reach steady state well before that,
    # and tracemalloc's per-call overhead makes the full run impractically
    # slow for a value that stops changing once buffers are full anyway.
    MEM_PASS_TICKS = 6000
    raw_ticks = generate_ticks(n=MEM_PASS_TICKS, seed=7)
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    last_bar_start = None
    n_closes = 0
    tracemalloc.start()
    for rt in raw_ticks:
        state = engine.on_tick(_to_tick(rt))
        if state is None:
            continue
        live.on_tick(state)
        if state.completed_m1 is not None and state.completed_m1.start_time != last_bar_start:
            last_bar_start = state.completed_m1.start_time
            live.on_m1_close(engine.completed_m1_window(480))
            n_closes += 1
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[SYNTHETIC] peak traced memory ({MEM_PASS_TICKS}-tick / {n_closes}-bar-close "
          f"measurement pass): {peak_mem / 1024:.1f} KB")


if __name__ == "__main__":
    test_m1_close_latency_synthetic()
    test_m1_close_memory_synthetic()
    print("tests/test_feature_performance.py: OK")
