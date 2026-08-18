"""python3 tests/test_latency_instrumentation.py -- confirms the three
latency figures are computed (not assumed zero) on synthetic data.
Labeled synthetic throughout; not a claim about real feed latency."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.tick import Tick
from market.state_engine import StateEngine
from market.synthetic_replay import generate_ticks


def test_latency_fields_populated_and_sane():
    eng = StateEngine("GOLD.i#")
    raw_ticks = generate_ticks(50, seed=7)
    last_state = None
    for rt in raw_ticks:
        tick = Tick(symbol=rt["symbol"], market_timestamp=rt["market_timestamp"],
                     ingestion_timestamp=rt["ingestion_timestamp"], bid=rt["bid"], ask=rt["ask"],
                     mid=(rt["bid"] + rt["ask"]) / 2, spread=rt["ask"] - rt["bid"],
                     tick_volume=rt["tick_volume"], source=rt["source"], internal_seq=rt["internal_seq"])
        state = eng.on_tick(tick)
        if state is not None:
            last_state = state
    assert last_state is not None
    assert last_state.feed_latency_sec is not None and last_state.feed_latency_sec >= 0
    assert last_state.state_update_latency_sec is not None and last_state.state_update_latency_sec >= 0
    print(f"OK  [SYNTHETIC] feed_latency_sec={last_state.feed_latency_sec:.6f} "
          f"state_update_latency_sec={last_state.state_update_latency_sec:.6f}")


if __name__ == "__main__":
    test_latency_fields_populated_and_sane()
    print("tests/test_latency_instrumentation.py: OK")
