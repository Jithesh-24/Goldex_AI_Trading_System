# Phase 2 Hardening -- Task 10: MT5 interface compatibility verification

Scope: read-only verification. No order placement, no live MT5 connection,
no new MT5 integration code. Goal: confirm `FastTierDecisionEngine` consumes
an MT5-shaped `MarketState` through the same interface it already uses for
synthetic-replay-shaped data, with zero Fast-Tier-side branching on source.

## Step 1 finding: a real MT5 normalization path already exists

Searched `contracts/` and `market/` per the brief. This repo already has a
live, non-archived MT5 tick-to-`MarketState` pipeline from Phase 1:

- `market/mt5_feed.py` -- MQL5-side socket sender that reads live MT5 ticks
  and pushes them over a socket as wire frames (`market/tick_protocol.py`).
- `market/feed_listener.py` (`FeedListener._handle_line`, lines ~92-124) --
  receives a `FRAME_TICK` wire frame and constructs a `contracts.tick.Tick`
  with `source=frame["source"]` (`"mt5_live"` for a genuine live feed --
  see `contracts/tick.py`'s `Literal["mt5_live", "synthetic_replay"]`), then
  calls `self.engine.on_tick(tick)`.
- `market/state_engine.py` (`StateEngine.on_tick`) -- "Pure incremental
  MarketState builder -- no I/O, no MT5, no sockets" (its own module
  docstring). Converts a `Tick` into a `contracts.market_state.MarketState`:
  timezone-aware `market_timestamp`/`ingestion_timestamp`, real
  `bid`/`ask`/`mid`/`spread`, `data_quality`, `feed_health`,
  `tick_count_60s`/`tick_count_300s`, `realized_vol_60s`, M1 bar state, etc.
  `source` is carried straight through from the tick, so a live-fed
  `MarketState` legitimately has `source="mt5_live"`.

This is Phase 1 work, still wired and actively used (not archived --
contrast with `.archive/v3-v4-intelligence-2026-08-29/`, which holds a
different, retired V3/V4 pipeline). `tests/simulator/test_historical_live_interface_consistency.py`
and several other active test files already import `StateEngine` and rely
on it.

**Conclusion for Step 1: no fabrication was needed.** A real MT5
normalization path exists and was used as-is.

## Step 2/3: test built and run

`tests/intelligence/test_mt5_interface_compatibility.py` drives a run of
`Tick(source="mt5_live", ...)` objects -- shaped exactly as
`feed_listener.py` builds them from a live MT5 frame -- through the *real*
`StateEngine.on_tick()` (the same call `feed_listener.py` makes for a live
feed), producing a genuinely MT5-normalized `MarketState` (not a
hand-typed lookalike). That `MarketState` is then passed unmodified into
`FastTierDecisionEngine.decide()` and, when a position was opened, into
`.manage()`.

Two tests:

1. `test_fast_tier_consumes_mt5_shaped_market_state_without_modification`
   -- asserts the state really is MT5-shaped (`source == "mt5_live"`,
   timezone-aware timestamps, positive bid/ask, spread consistency), then
   asserts `decide()` returns a valid 4-tuple with `action` in
   `("NO_TRADE", "LONG", "SHORT")` and consistent sl/tp/size nullness.
2. `test_fast_tier_manage_consumes_mt5_shaped_market_state_without_modification`
   -- same MT5-normalized-path state fed to `manage()` on the
   decide()-then-manage() seam (position_view=None per the existing
   `test_position_management.py` convention), asserting the string-typed
   return contract.

```
$ .venv/bin/pytest tests/intelligence/test_mt5_interface_compatibility.py -v
tests/intelligence/test_mt5_interface_compatibility.py::test_fast_tier_consumes_mt5_shaped_market_state_without_modification PASSED [ 50%]
tests/intelligence/test_mt5_interface_compatibility.py::test_fast_tier_manage_consumes_mt5_shaped_market_state_without_modification PASSED [100%]

2 passed in 1.11s
```

## Step 4: compatibility finding

**Yes.** `FastTierDecisionEngine.decide()`/`.manage()` handle an
MT5-normalized `MarketState` with zero Fast-Tier-side branching. Neither
method, nor anything they call (`FastTierReasoner`, `ToolTrust`,
`EvidenceRegistry`, the injected `ev_cost_gate`/`sizing_bootstrap`/
`sltp_bootstrap`), inspects `MarketState.source` anywhere -- confirmed by
`grep -n "\.source" intelligence/decision_engine.py intelligence/fast_tier.py`
returning no matches. The engine only ever consumes the `MarketState`
contract's price/quality/timing fields, never a data-source-specific type
or a source-conditional code path. This matches the expected outcome the
brief anticipated ("since `decide`/`manage` only ever consumed the
`MarketState` contract, never a data-source-specific type").

This is exactly what the `MarketState` contract's docstring promises
(`contracts/market_state.py`: "the single source of truth every future V3
component reads") -- the source-tagging (`mt5_live` vs. `synthetic_replay`)
exists for provenance/audit purposes, not for downstream consumers to
branch on.

## What this does not cover

- No live MT5 connection or `market/mt5_feed.py`'s MQL5-side socket sender
  was exercised -- out of scope per the brief, and `mt5_feed.py` isn't
  reachable from Python tests anyway (it runs inside the MT5 terminal).
- No order placement or execution path was touched.
- Task 11 (real SL/TP/sizing bootstrap) is still a stub in this test, as it
  is in the existing `tests/intelligence/test_decision_engine.py` -- not
  this task's concern.
