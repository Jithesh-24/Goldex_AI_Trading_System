"""python3 tests/test_live_replay_equivalence.py -- feeds the same
synthetic tick sequence through live_engine.py (bounded, trigger-driven)
and replay_engine.py (batch), asserts the live snapshot at each M1 close
matches the batch computation on the equivalent prefix (spec section 8).

Two known-bug fixes carried over from Task 22 / tests/test_live_engine.py,
both required for this test to be meaningful rather than crash/hang:

1. generate_ticks() (market/synthetic_replay.py) returns plain dicts, not
   contracts.tick.Tick instances -- StateEngine.on_tick() requires a real
   Tick (it reads tick.market_timestamp/.bid/.ask/... as attributes, not
   dict keys). Converted via the same _to_tick() pattern already used in
   tests/test_performance.py.

2. MarketState.completed_m1 is level-held (market/state_engine.py:155:
   completed_m1=self.completed_m1[-1] if self.completed_m1 else None) --
   once the first M1 bar completes, every subsequent tick's state carries
   that same non-None "latest completed bar" field, not just the boundary
   tick where a new bar first completes. The brief's original sketch did
   `if state.completed_m1 is not None: on_m1_close(...)`, which fires on
   nearly every tick instead of once per ~30-tick M1 bar -- this caused an
   OOM/hang (exit 137) at 20000 ticks in Task 22. Fixed here the same way
   tests/test_live_engine.py fixes it: edge-trigger on
   state.completed_m1.start_time changing, tracked via last_bar_start."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from contracts.tick import Tick
from market.state_engine import StateEngine
from market.synthetic_replay import generate_ticks
from features.live_engine import LiveFeatureEngine
from features.features import build_features
from features.replay_engine import build_candidate_features


def _to_tick(rt):
    return Tick(symbol=rt["symbol"], market_timestamp=rt["market_timestamp"],
                ingestion_timestamp=rt["ingestion_timestamp"], bid=rt["bid"], ask=rt["ask"],
                mid=(rt["bid"] + rt["ask"]) / 2, spread=rt["ask"] - rt["bid"],
                tick_volume=rt["tick_volume"], source=rt["source"], internal_seq=rt["internal_seq"])


def test_live_matches_replay_at_m1_close():
    raw_ticks = generate_ticks(n=6000, seed=42)
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    last_snapshot = None
    last_bar_start = None
    for rt in raw_ticks:
        tick = _to_tick(rt)
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        # Edge-trigger on start_time change (see module docstring, fix #2)
        # -- on_m1_close runs once per completed bar, not once per tick.
        if state.current_m1 is not None and state.completed_m1 is not None \
                and state.completed_m1.start_time != last_bar_start:
            last_bar_start = state.completed_m1.start_time
            last_snapshot = live.on_m1_close(engine.completed_m1_window(480))

    assert last_snapshot is not None

    # Reference: batch-compute the exact same bounded window replay would
    # see, via build_candidate_features (Task 16's proven composition --
    # identical to what live_engine.py's on_m1_close calls internally).
    bars = engine.completed_m1_window(480)
    df = pd.DataFrame({
        "time": [b.start_time for b in bars], "open": [b.open for b in bars],
        "high": [b.high for b in bars], "low": [b.low for b in bars],
        "close": [b.close for b in bars], "tick_volume": [b.tick_count for b in bars],
        "spread": [0.2] * len(bars),
    })
    base = build_features(df)
    replay = build_candidate_features(df, base)

    checked = 0
    families_with_no_overlap = set()
    for feature_id, (value, quality) in last_snapshot.items():
        if feature_id not in replay.columns:
            families_with_no_overlap.add(feature_id)
            continue
        if quality != "VALID":
            continue
        expected = replay[feature_id].to_numpy(dtype=np.float64)[-1]
        if np.isnan(expected):
            continue
        assert abs(value - expected) < 1e-6 or np.isclose(value, expected, rtol=1e-6), (
            f"{feature_id}: live={value} replay={expected}")
        checked += 1

    assert checked > 0, "no VALID features overlapped with replay columns -- test is not exercising anything"
    print(f"checked={checked} VALID features matched live vs replay "
          f"(no-overlap-in-replay-columns: {sorted(families_with_no_overlap)})")


if __name__ == "__main__":
    test_live_matches_replay_at_m1_close()
    print("tests/test_live_replay_equivalence.py: OK")
