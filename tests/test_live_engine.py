"""python3 tests/test_live_engine.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta

from contracts.tick import Tick
from market.state_engine import StateEngine
from features.live_engine import LiveFeatureEngine


def test_live_engine_produces_snapshot_after_enough_bars():
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)  # no bootstrap in this unit test

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = None
    last_bar_start = None
    for i in range(20000):  # enough ticks to cross several M1 boundaries
        t = base + timedelta(seconds=i * 2)
        tick = Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t,
                    bid=2000.0 + (i % 100) * 0.01, ask=2000.2 + (i % 100) * 0.01,
                    mid=2000.1, spread=0.2, source="synthetic_replay", internal_seq=i)
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        # completed_m1 is level-held (the latest completed bar rides on
        # every subsequent MarketState, not just the boundary tick) --
        # edge-trigger on start_time change so on_m1_close runs once per
        # bar, not once per tick (once-per-tick caused an OOM/hang: ~20000
        # full family recomputes instead of ~666).
        if state.current_m1 is not None and state.completed_m1 is not None \
                and state.completed_m1.start_time != last_bar_start:
            last_bar_start = state.completed_m1.start_time
            snapshot = live.on_m1_close(engine.completed_m1_window(480))

    assert snapshot is not None
    assert len(snapshot) > 0
    for feature_id, (value, quality) in snapshot.items():
        assert quality in ("VALID", "WARMING_UP", "UNAVAILABLE")


if __name__ == "__main__":
    test_live_engine_produces_snapshot_after_enough_bars()
    print("tests/test_live_engine.py: OK")
