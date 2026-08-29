"""tests/simulator/test_replay_performance.py -- Task 9: REAL performance
measurement (not sanity-only). Section 12 of the mandate: "do not claim
millisecond-capable without measurement." Every number below is printed,
not just asserted below some threshold -- the printed number IS the
deliverable, following the same [SYNTHETIC]-labeled convention already
used by tests/test_performance.py (time.perf_counter for latency,
tracemalloc kept out of the timed path so its own overhead doesn't
contaminate the numbers).

Four things measured in isolation:
  1. End-to-end historical replay throughput (bars/sec via run_replay).
  2. build_snapshot construction latency (per-call mean/p50/p99).
  3. execution.py fill computation latency (entry_fill_price / exit_fill_price
     / round_trip_cost_r -- the actual per-decision hot path: entry/exit
     fills happen on every open/close, cost_r on every DECIDE record).
  4. ExperienceRecorder.record latency -- this is where Task 8's widened
     market_state_snapshot (.model_dump() of the full MarketState) and
     widened account dict actually land, so this number reflects the
     CURRENT (heavier) record shape, not a pre-Task-8 assumption.

Dataset size: 5,000 synthetic 1-minute bars for run_replay (amortizes
Python/pandas per-call overhead while finishing in well under a second),
1,000 iterations for the isolated build_snapshot/execution/record
microbenchmarks (enough to get stable p50/p99 percentiles without slowing
the suite down)."""
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from contracts.market_state import MarketState, M1BarState, DataQuality, FeedHealthState
from simulator.contracts import AccountState, EnvironmentTag, PositionOutcome, Side, SimulatedExecutionConfig
from simulator.execution import entry_fill_price, exit_fill_price, compute_cost_r
from simulator.experience import ExperienceRecord, ExperienceRecorder, write_tag_guard
from simulator.market_state_builder import build_snapshot
from simulator.replay import run_replay

N_REPLAY_BARS = 5000
N_MICROBENCH_ITERS = 1000


def _make_df(n=N_REPLAY_BARS, start_price=1900.0):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [start_price + (i % 200) * 0.05 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [12 + (i % 5) for i in range(n)],
        "spread": [22.0 + (i % 7) for i in range(n)],
    })


def _percentiles(values_us):
    values_us = sorted(values_us)
    n = len(values_us)
    return {
        "mean": sum(values_us) / n,
        "p50": values_us[n // 2],
        "p99": values_us[int(n * 0.99)],
    }


def _print_stats(label, unit, stats, n):
    print(f"[SYNTHETIC][Task 9] {label}: n={n} mean={stats['mean']:.2f}{unit} "
          f"p50={stats['p50']:.2f}{unit} p99={stats['p99']:.2f}{unit}")


# A trivial, deterministic policy: open a LONG on the first decide() call,
# then hold until forced closure. This exercises the full DECIDE -> MANAGE*
# -> POSITION_CLOSED record path (all three ExperienceRecord event types),
# not just the flat NO_TRADE path, so the throughput number reflects
# realistic per-bar work rather than the cheapest possible branch.
def _make_policy():
    state = {"opened": False}

    def decide(market_state, account):
        if not state["opened"]:
            state["opened"] = True
            return ("LONG", market_state.mid - 5.0, market_state.mid + 5.0)
        return ("NO_TRADE", None, None)

    def manage(market_state, position_view, account):
        return "HOLD"

    return decide, manage


def test_replay_throughput_bars_per_sec():
    """1. End-to-end historical replay throughput (bars/sec via run_replay)."""
    df = _make_df()
    config = SimulatedExecutionConfig()
    decide, manage = _make_policy()

    t0 = time.perf_counter()
    recorder = run_replay(df, decide, manage, config, EnvironmentTag.SIMULATED_TRAINING)
    elapsed = time.perf_counter() - t0

    bars_per_sec = N_REPLAY_BARS / elapsed
    records = recorder.all_records()
    print(f"[SYNTHETIC][Task 9] run_replay: {N_REPLAY_BARS} bars in {elapsed:.3f}s "
          f"-> {bars_per_sec:,.0f} bars/sec ({len(records)} experience records produced)")

    assert len(records) > 0
    assert bars_per_sec > 50, "replay throughput should comfortably exceed the ~1 bar/min real feed rate"


def test_build_snapshot_construction_latency():
    """2. build_snapshot construction latency in isolation (per-call)."""
    df = _make_df(n=N_MICROBENCH_ITERS + 1)
    per_call_us = []
    for i in range(1, N_MICROBENCH_ITERS + 1):
        t0 = time.perf_counter()
        build_snapshot(df, i)
        per_call_us.append((time.perf_counter() - t0) * 1e6)

    stats = _percentiles(per_call_us)
    _print_stats("build_snapshot construction latency", "us", stats, N_MICROBENCH_ITERS)
    assert stats["p99"] < 50000, "build_snapshot p99 should stay well under 50ms even synthetically"


def _sample_market_state():
    ts = datetime(2020, 1, 6, 10, 0, 0, tzinfo=timezone.utc)
    m1 = M1BarState(open=1900.0, high=1900.5, low=1899.5, close=1900.2,
                     tick_count=12, start_time=ts, end_time=ts, complete=True)
    return MarketState(
        symbol="XAUUSD", source="synthetic_replay", state_version="v1", sequence=0,
        market_timestamp=ts, ingestion_timestamp=ts, processing_timestamp=ts,
        bid=1899.9, ask=1900.1, mid=1900.0, spread=0.22, last=1900.0,
        last_quality=DataQuality.VALID,
        tick_count_60s=60, tick_count_300s=300, tick_rate_per_sec=1.0,
        current_m1=m1, completed_m1=m1,
        realized_vol_60s=0.0008, spread_mean_60s=0.22, spread_std_60s=0.01,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.0,
        feed_latency_sec=0.0, state_update_latency_sec=0.0,
    )


def test_execution_fill_computation_latency():
    """3. execution.py fill computation latency -- entry_fill_price,
    exit_fill_price, and compute_cost_r (round_trip_cost_r) are the actual
    per-decision hot path: entry/exit fills happen on every open/close,
    cost_r is computed (via decision.ev_cost callers) on the same cadence
    as market_state is built."""
    market_state = _sample_market_state()
    config = SimulatedExecutionConfig(max_staleness_seconds=float("inf"))
    per_call_us = []
    for _ in range(N_MICROBENCH_ITERS):
        t0 = time.perf_counter()
        entry_fill_price(Side.LONG, market_state.mid, market_state.spread, config)
        exit_fill_price(Side.LONG, market_state.mid, market_state.spread, config)
        compute_cost_r(market_state, 1.5, config)
        per_call_us.append((time.perf_counter() - t0) * 1e6)

    stats = _percentiles(per_call_us)
    _print_stats("execution fill computation (entry+exit+cost_r combined per call)", "us", stats,
                 N_MICROBENCH_ITERS)
    assert stats["p99"] < 10000, "fill computation p99 should stay well under 10ms even synthetically"


def test_experience_recorder_record_latency():
    """4. ExperienceRecorder.record latency -- this is where Task 8's
    widened market_state_snapshot (.model_dump() of the full MarketState)
    and widened account dict land, so this measures the CURRENT (heavier)
    record shape."""
    market_state = _sample_market_state()
    market_state_snapshot = market_state.model_dump()  # Task 8 widening: full dump, not mid/spread only
    account = AccountState.initial(SimulatedExecutionConfig(), market_state.market_timestamp)
    account_dict = {
        "balance": account.balance, "equity": account.equity, "margin_used": account.margin_used,
        "margin_free": account.margin_free, "exposure": account.exposure,
        "open_position_id": account.open_position_id,
        "realized_pnl_total": account.realized_pnl_total, "drawdown": account.drawdown,
        "currency": account.currency,
    }

    recorder = ExperienceRecorder()
    per_call_us = []
    for _ in range(N_MICROBENCH_ITERS):
        record = ExperienceRecord(
            environment_tag=EnvironmentTag.SIMULATED_TRAINING, timestamp=market_state.market_timestamp,
            event_type="DECIDE", market_state_snapshot=market_state_snapshot, position_view=None,
            action="NO_TRADE", account_state=account_dict, realized_pnl=None, cost_amount=None,
            outcome=None, gap_type="NORMAL",
        )
        t0 = time.perf_counter()
        write_tag_guard(EnvironmentTag.SIMULATED_TRAINING, record)
        recorder.record(record)
        per_call_us.append((time.perf_counter() - t0) * 1e6)

    stats = _percentiles(per_call_us)
    _print_stats("ExperienceRecorder.record latency (post-Task-8 widened snapshot)", "us", stats,
                 N_MICROBENCH_ITERS)
    assert len(recorder.all_records()) == N_MICROBENCH_ITERS
    assert stats["p99"] < 10000, "record() p99 should stay well under 10ms even synthetically"

    # Memory footprint of the widened record shape, measured in a separate
    # pass so tracemalloc's own overhead doesn't contaminate the timing
    # numbers above (same convention as tests/test_performance.py).
    tracemalloc.start()
    mem_recorder = ExperienceRecorder()
    for _ in range(N_MICROBENCH_ITERS):
        mem_recorder.record(ExperienceRecord(
            environment_tag=EnvironmentTag.SIMULATED_TRAINING, timestamp=market_state.market_timestamp,
            event_type="DECIDE", market_state_snapshot=market_state.model_dump(), position_view=None,
            action="NO_TRADE", account_state=dict(account_dict), realized_pnl=None, cost_amount=None,
            outcome=None, gap_type="NORMAL",
        ))
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[SYNTHETIC][Task 9] peak traced memory for {N_MICROBENCH_ITERS} widened experience records: "
          f"{peak_mem / 1024:.1f} KB ({peak_mem / 1024 / N_MICROBENCH_ITERS:.2f} KB/record)")


if __name__ == "__main__":
    test_replay_throughput_bars_per_sec()
    test_build_snapshot_construction_latency()
    test_execution_fill_computation_latency()
    test_experience_recorder_record_latency()
    print("tests/simulator/test_replay_performance.py: OK")
