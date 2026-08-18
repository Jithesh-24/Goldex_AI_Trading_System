# market/

Managed real-time MT5 market-data pipeline (Phase 2).

- `mt5_feed.py` — Wine-side process (`wine python.exe market/mt5_feed.py`),
  the only MT5 IPC owner. Behavior-preserving superset of the old
  `xm_ticker.py`: same STATE/BARS_LIVE/BARS_BACKFILL/OFFSET_FILE file
  outputs (app/engine.py's real signal generation, market-closed gating,
  and verdict-tracking still depend on these, unchanged), plus one
  additive capability — pushes normalized ticks + one backfill frame per
  connection over `tick_protocol.py`'s wire format to `feed_listener.py`.
  Live-verified 2026-08-18: 504 real ticks processed end-to-end over a
  90s window against a real XM connection.
- `tick_protocol.py` — shared, stdlib-only wire format between the Wine
  and native sides. No ingestion_timestamp on the wire, deliberately —
  that's stamped by `feed_listener.py` at actual receipt.
- `feed_listener.py` — native TCP server + feed-health state machine,
  runs as a background thread inside `app/engine.py`'s process, feeds...
- `state_engine.py` — the pure incremental `MarketState` builder. O(1)
  amortized tick-count tracking, O(window)-not-O(history) spread stats
  using plain float arithmetic (not `statistics.pstdev`, which profiled
  at ~94% of runtime -- see git history).
- `synthetic_replay.py` — labeled synthetic tick generator for tests/
  performance benchmarking (no real XM tick-level dataset exists to
  replay instead).

The old `xm_ticker.py` is archived at
`.archive/legacy-xm-ticker-2026-08-18/xm_ticker.py` — `mt5_feed.py` is a
strict superset of its behavior, so nothing is lost. `trading/watchdog.py`
now launches/monitors `mt5_feed.py` instead.

`app/engine.py`'s `get_market_state()` accessor (Task 11) proves the new
pipeline reaches the V3 application boundary but is **not** wired into
the live signal-generation loop — that loop still reads the same STATE
file `mt5_feed.py` writes, exactly as before Phase 2. Cutting the
decision loop over to the new `MarketState` is Phase 3+ work, once a
real feature/decision layer exists to consume it.
