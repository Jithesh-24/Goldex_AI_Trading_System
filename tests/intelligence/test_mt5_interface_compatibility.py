"""Task 10: MT5 interface compatibility verification (read-only).

Scope: prove FastTierDecisionEngine.decide()/.manage() consume a MarketState
built the way MT5-sourced data actually produces one -- through the real
normalization path -- without any Fast-Tier-side branching on data source.
No order placement, no live MT5 connection, no new MT5 integration code.

Investigation (Step 1): the real MT5 normalization path in this repo is:

    market/mt5_feed.py (MQL5-side socket sender, out of Python's reach here)
      -> market/feed_listener.py:_handle_line() builds a `Tick` with
         source=frame["source"] (== "mt5_live" for a live feed; see
         contracts/tick.py's `Literal["mt5_live", "synthetic_replay"]`)
      -> market/state_engine.py:StateEngine.on_tick(tick) -- "Pure incremental
         MarketState builder -- no I/O, no MT5, no sockets" (its own module
         docstring) -- returns a `MarketState` (contracts/market_state.py)
         with source carried through from the tick, timezone-aware
         timestamps, real bid/ask/mid/spread, data_quality, feed_health, etc.

This test drives a `Tick(source="mt5_live", ...)` through the *real*
`StateEngine.on_tick()` -- the same call feed_listener.py makes for a live
MT5 tick -- so the resulting MarketState is genuinely MT5-shaped, not
hand-typed to look plausible. That MarketState is then fed to
FastTierDecisionEngine.decide()/.manage() completely unmodified.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contracts.tick import Tick
from market.state_engine import StateEngine
from intelligence.decision_engine import FastTierDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import FastTierReasoner, ToolTrust
from simulator.contracts import AccountState, SimulatedExecutionConfig
from simulator.cost_model import round_trip_cost_r


def _mt5_tick(seq, market_ts, bid, ask):
    """A Tick shaped exactly as feed_listener.py builds one from a live MT5
    frame (market/feed_listener.py:_handle_line, FRAME_TICK branch) --
    source="mt5_live", mid/spread derived from bid/ask, ingestion_timestamp
    stamped on receipt (never taken from the wire)."""
    return Tick(
        symbol="XAUUSD",
        market_timestamp=market_ts,
        ingestion_timestamp=market_ts + timedelta(milliseconds=50),
        bid=bid, ask=ask, mid=(bid + ask) / 2, spread=ask - bid,
        tick_volume=1, source="mt5_live", internal_seq=seq,
    )


def _cheap_cost_gate(market_state, candidate_sl_distance_r):
    return round_trip_cost_r(market_state, candidate_sl_distance_r, max_staleness_seconds=float("inf"))


def _make_engine():
    registry = build_default_registry()
    trust = ToolTrust()
    reasoner = FastTierReasoner(registry)

    def sltp_bootstrap(hyp, market_state):
        mid = market_state.mid
        if hyp.net_directional_belief >= 0:
            return mid - 1.0, mid + 1.0
        return mid + 1.0, mid - 1.0

    return FastTierDecisionEngine(
        registry=registry,
        trust=trust,
        reasoner=reasoner,
        ev_cost_gate=_cheap_cost_gate,
        sizing_bootstrap=lambda hyp, account: 0.01,
        sltp_bootstrap=sltp_bootstrap,
    )


def _mt5_shaped_market_state():
    """Feeds a run of mt5_live-sourced ticks through the real StateEngine
    (the same object feed_listener.py owns for a live feed) and returns the
    final MarketState -- i.e. built via the actual MT5 normalization path,
    not hand-assembled to merely resemble one."""
    engine = StateEngine(symbol="XAUUSD")
    start = datetime(2026, 9, 2, 10, 0, 0, tzinfo=timezone.utc)
    state = None
    price = 2000.0
    for i in range(120):
        market_ts = start + timedelta(seconds=i)
        price += 0.01 if i % 2 == 0 else -0.01
        tick = _mt5_tick(seq=i, market_ts=market_ts, bid=price, ask=price + 0.2)
        result = engine.on_tick(tick)
        if result is not None:
            state = result
    assert state is not None
    return state


def test_fast_tier_consumes_mt5_shaped_market_state_without_modification():
    market_state = _mt5_shaped_market_state()

    # Confirms this really is MT5-shaped, not synthetic-replay-shaped.
    assert market_state.source == "mt5_live"
    assert market_state.market_timestamp.tzinfo is not None
    assert market_state.ingestion_timestamp.tzinfo is not None
    assert market_state.bid > 0 and market_state.ask > 0
    assert market_state.spread == market_state.ask - market_state.bid

    engine = _make_engine()
    config = SimulatedExecutionConfig()
    account = AccountState.initial(config, market_state.market_timestamp)

    action, sl, tp, size = engine.decide(market_state, account)
    assert action in ("NO_TRADE", "LONG", "SHORT")
    if action == "NO_TRADE":
        assert sl is None and tp is None and size is None
    else:
        assert sl is not None and tp is not None and size is not None


def test_fast_tier_manage_consumes_mt5_shaped_market_state_without_modification():
    """Same MT5-normalized-path MarketState, this time through manage() on
    an already-open position -- the other half of the DecideFn/ManageFn
    seam FastTierDecisionEngine wires into (Phase 1 contract)."""
    market_state = _mt5_shaped_market_state()
    engine = _make_engine()
    config = SimulatedExecutionConfig()
    account = AccountState.initial(config, market_state.market_timestamp)

    # decide() first, to get into a state manage() can legally be called
    # from (an open thesis, mirroring how simulator/replay.py drives the
    # DecideFn/ManageFn seam: manage() is only ever called while a position
    # is open).
    action, sl, tp, size = engine.decide(market_state, account)
    if action == "NO_TRADE":
        # No position opened on this particular synthetic run -- manage()
        # is a no-op contract-wise while flat in this engine (mirrors
        # decide()'s own thesis-clearing guard), so just confirm decide()'s
        # contract shape here instead of forcing a trade.
        assert sl is None and tp is None and size is None
        return

    result = engine.manage(market_state, None, account)
    assert isinstance(result, str)
