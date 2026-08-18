# Golex V3 — Phase 2: Real-Time XM/MT5 Market-State Pipeline

Status: approved for implementation planning
Date: 2026-08-18
Scope: Phase 2 only — a managed MT5 feed process, a canonical Tick contract,
an incremental MarketState engine, and retirement of the legacy
file-polling path as the V3 runtime dependency. No feature library, no
specialist models, no dynamic SL/TP, no EV, no virtual trade management, no
EOD learning, no champion/challenger.

## 1. Purpose

Build the "eyes and nervous system" of V3: a reliable, low-latency,
causally-correct real-time market-state pipeline that becomes the
canonical source of live market information for every future V3
component, replacing the external file-polling contract Phase 1
explicitly marked as temporary legacy.

## 2. Research findings (hard constraints, not assumptions)

**`MetaTrader5` is a Windows-only Python package.** It is not installed in
this repo's native venv (`ModuleNotFoundError`, confirmed) and cannot be —
it wraps IPC to the Windows MT5 terminal. `market/xm_ticker.py` runs under
`wine python.exe`, a completely separate Python 3.11 interpreter with no
third-party packages beyond `MetaTrader5` itself. **This means a managed
MT5 feed process is unavoidably a separate OS process across a Wine↔native
boundary** — no design can eliminate this boundary, only make it fast and
clean. Wine, the MT5 terminal, and Xvfb are genuinely installed and were
live-used previously (a real persisted server-offset file exists:
`{"offset_h": 3.0, "ts": 1786938629.003645}`).

**What the existing `xm_ticker.py` proves empirically about XM/`GOLD.i#`**
(mined from its code and dated bug-fix comments, not generic MT5 docs):

- Symbol is `GOLD.i#`, not literally `XAUUSD`.
- Only `bid`/`ask` are ever read from `symbol_info_tick()`. `last` is never
  read; `real_volume` is hardcoded to `0` in the backfill writer. Both are
  `UNSUPPORTED_BY_DATA` for this instrument until proven otherwise.
- `tick_volume` (integer tick count per bar, from `copy_rates_from_pos`) is
  real and used.
- `t.time` (integer seconds) is used throughout. MT5's Python API also
  exposes `time_msc` (milliseconds) — never read by the existing code.
  Available per the documented API surface; sub-second precision for
  XM's actual feed is **not verified** in this codebase's history.
- Server time offset is not fixed — it drifts with DST and is measured
  live from either a fresh tick or, when the tick path is stale
  (market-closed weekends), from `copy_rates_from_pos` M1 bar timestamps,
  with a persisted fallback. XM's summer offset is empirically +3.0h.
- Session hours are **empirically reverse-engineered from live bars**
  (comment: "verified from live bars 2026-08-10"): Fri ≥21:00 UTC close,
  Sat all day, Sun <22:00 UTC still closed, daily break 21:00–22:00 UTC
  Mon–Thu. Not derived from any MT5 session API — no such API is used or
  assumed available.
- Real, tested reconnect logic already exists: explicit
  `symbol_select(True)` re-subscription (a documented XM-specific gotcha —
  weekend zombie churn desyncs Market Watch), a wait-for-first-tick loop
  after reconnect, and a deliberate avoidance of `mt5.shutdown()` churn on
  transient `None` ticks (documented outage: aggressive shutdown/init
  wedged the module's IPC).
- **No raw tick history is persisted anywhere** — only M1 OHLC bars
  (`xm_bars_backfill.csv`, `xm_live_bars.jsonl`). There is no real XM
  tick-level dataset to replay; any tick-level replay must be synthetic,
  clearly labeled as such (matches the user's own Section 26 rule).
- `xm_ticker.py` currently does **two jobs**: market data (bid/ask/M1 bar
  construction) and first-touch TP/SL verdict tracking for the currently
  open trade (reads `.active_signal_ai.json`, writes SL/TP-touch verdicts
  into the same state file). The second job is trade-management, not
  market-state, and is explicitly out of Phase 2 scope.

**Scope decision (user-confirmed):** the verdict-tracking half is left
exactly as-is, on its existing file-based channel, unmigrated. Only the
market-data half is replaced by the new pipeline. Redesigning
verdict-tracking is virtual-trade-management work for a later phase.

**Live verification (user-authorized):** Xvfb + Wine + MT5 terminal will
be launched during implementation to verify the new feed process against
a real XM connection — read-only, no order placement, stopped again
afterward (Phase 2 does not restart the trading services). Numbers from
this window are labeled "live-measured"; everything else is labeled
"synthetic replay." Never blended or presented as equivalent.

## 3. Data honesty rule

No microstructure variable is computed unless the underlying feed
genuinely supports it. Order-book imbalance, true trade-flow imbalance,
VPIN, Kyle's lambda, institutional order flow, and hidden liquidity are
all `UNSUPPORTED_BY_DATA` for a retail CFD feed with bid/ask + tick-count
only — `contracts/tick.py` and `contracts/market_state.py` do not define
fields for them. What XM/MT5 genuinely supports and Phase 2 preserves the
raw state for (without computing the Phase 3 feature itself): tick-level
returns, inter-arrival time, tick intensity, bid/ask dynamics, spread
dynamics, a legitimate micro-price (bid/ask weighted, if quote sizes were
available — they are not for this feed, so micro-price reduces to mid,
documented as such), short-horizon realized volatility from ticks,
directional tick imbalance (buy-tick vs sell-tick counts — already
computed today as `imb_60s`/`imb_300s`, a legitimate quantity from
bid-direction ticks, not a fabricated order-flow claim).

## 4. Architecture

```
market/
├── mt5_feed.py         Wine-side process (rewritten xm_ticker.py's
│                        market-data half): connect, offset detection,
│                        backfill, 25ms tick poll, pushes normalized
│                        ticks as JSON lines over a TCP socket to
│                        feed_listener.py. Verdict-tracking logic stays
│                        here unchanged, on its existing file channel,
│                        clearly separated and documented as
│                        out-of-Phase-2-scope.
├── tick_protocol.py     Shared wire format: a plain-dict JSON schema +
│                        newline-delimited framing, importable by both
│                        Wine's bare Python 3.11 and the native venv
│                        (stdlib only, no pydantic dependency here).
├── feed_listener.py     Native-side TCP server (native process is the
│                        server; mt5_feed.py is the client and owns
│                        reconnect). Accepts the connection, parses each
│                        line, validates into contracts.tick.Tick,
│                        feeds state_engine.py. Owns the feed-health
│                        state machine (CONNECTED/STALE/RECONNECTING/
│                        DISCONNECTED/INVALID).
├── state_engine.py      Pure incremental MarketState builder, no I/O.
│                        new tick -> update current M1 bar + bounded
│                        rolling buffers in place -> publish MarketState.
│                        Never reloads/recomputes from history.
├── xm_ticker.py          Retired from the V3 runtime path once
│                        mt5_feed.py is live-verified; archived with the
│                        rest of Phase 1's legacy set, not deleted.
└── README.md
```

`app/engine.py` gains a thin in-process accessor to `feed_listener.py`'s
latest `MarketState` — Phase 2 keeps `feed_listener.py` and `app/engine.py`
in the same OS process for simplicity; there is no reason to add a second
network hop between two native components that already share a venv.
Concretely: `feed_listener.py`'s TCP server (accepting `mt5_feed.py`'s
connection) runs on a background thread; each accepted tick updates one
lock-protected "latest `MarketState`" reference (a single `threading.Lock`
around a plain variable swap — the update is O(1) and infrequent enough
at tick cadence that a lock is simpler and sufficient, no need for a queue
or async runtime). `get_latest_state()` takes the same lock to read.
This is sufficient to prove `MT5 -> managed feed -> MarketState -> V3
application boundary` end-to-end. No Phase 3 feature calculation is added
merely to demonstrate this — `app/engine.py`'s existing CatBoost signal
path is untouched (Section 22/23: no model or trading changes).

## 5. Canonical Tick contract (`contracts/tick.py`)

```python
class Tick(BaseModel):
    symbol: str
    market_timestamp: datetime       # broker-clock-derived, UTC-normalized
    ingestion_timestamp: datetime    # when feed_listener.py received it
    bid: float
    ask: float
    mid: float                       # derived, not broker-supplied
    spread: float                    # derived, not broker-supplied
    last: Optional[float] = None     # UNSUPPORTED_BY_DATA for GOLD.i# today
    tick_volume: Optional[int] = None
    source: Literal["mt5_live", "synthetic_replay"]
    internal_seq: int                # feed_listener.py's own monotonic
                                      # counter -- MT5 provides no reliable
                                      # tick sequence ID, this is NOT a
                                      # broker sequence, clearly named as
                                      # internal to avoid that confusion
```

`last` and any volume/order-flow field beyond `tick_volume` stay
`Optional`/absent rather than defaulting to `0.0` — a silent zero would be
indistinguishable from a real zero-value observation.

## 6. Timestamp discipline

Two timestamps travel with every tick and every `MarketState` update:
`market_timestamp` (MT5 server time, offset-corrected to true UTC using
the existing measured-offset logic ported into `mt5_feed.py`) and
`ingestion_timestamp` (native-side wall clock at receipt). A third,
`processing_timestamp`, is stamped by `state_engine.py` when it finishes
updating state from that tick. These three points are never mixed into a
single field — each is named and typed separately, matching Section 6's
requirement, and are exactly what Section 7's latency instrumentation
subtracts.

## 7. Latency measurement

`feed_listener.py` and `state_engine.py` compute and expose (not just
log) three latency figures per tick, attached to the resulting
`MarketState.feed_health` block:

- **feed latency** = `ingestion_timestamp - market_timestamp`
- **state-update latency** = `processing_timestamp - ingestion_timestamp`
- **decision-ready latency** = time from `processing_timestamp` to the
  moment `app/engine.py`'s in-process accessor observes the new state
  (measured, not assumed zero, even though same-process)

No latency number is claimed without a real measured value behind it —
synthetic-replay latency figures are labeled synthetic; live-window
figures are labeled live.

## 8. MarketState contract (`contracts/market_state.py`, expanded)

Phase 1's `MarketState` had mostly-`Optional` placeholder fields. Phase 2
populates it for real and adds the identity/versioning/health structure:

```python
class FeedHealthState(str, Enum):
    UNKNOWN = "UNKNOWN"
    CONNECTED = "CONNECTED"
    STALE = "STALE"
    RECONNECTING = "RECONNECTING"
    DISCONNECTED = "DISCONNECTED"
    INVALID = "INVALID"

class DataQuality(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"   # feed doesn't currently have it
    UNKNOWN = "UNKNOWN"           # not yet determined (e.g. at startup)

class M1BarState(BaseModel):
    open: float; high: float; low: float; close: float
    tick_count: int
    start_time: datetime; end_time: Optional[datetime] = None
    complete: bool          # explicit -- current vs completed bar,
                             # never inferred from a missing field

class MarketState(BaseModel):
    # IDENTITY
    symbol: str
    source: Literal["mt5_live", "synthetic_replay"]
    state_version: str = "v1"
    sequence: int
    # TIME
    market_timestamp: datetime
    ingestion_timestamp: datetime
    processing_timestamp: datetime
    # PRICE
    bid: float; ask: float; mid: float; spread: float
    last: Optional[float] = None
    last_quality: DataQuality = DataQuality.UNAVAILABLE
    # ACTIVITY
    tick_count_60s: int; tick_count_300s: int
    tick_rate_per_sec: float
    # BAR STATE
    current_m1: Optional[M1BarState] = None
    completed_m1: Optional[M1BarState] = None
    # VOLATILITY STATE (raw inputs only -- not the Phase 3 feature library)
    realized_vol_60s: Optional[float] = None
    spread_mean_60s: Optional[float] = None
    spread_std_60s: Optional[float] = None
    # FEED HEALTH
    feed_health: FeedHealthState
    last_tick_age_sec: float
    feed_latency_sec: Optional[float] = None
    state_update_latency_sec: Optional[float] = None
```

Missing data is represented by `DataQuality.UNAVAILABLE`/`UNKNOWN` on the
relevant field's companion, never silently coerced to `0.0` or `None`
treated as zero downstream.

## 9. Incremental state engine

`state_engine.py.StateEngine.on_tick(tick: Tick) -> MarketState` is the
only entry point. Internally: extend/roll the current M1 bar in O(1),
update fixed-size ring buffers (60s/300s tick timestamps, spread samples)
by appending and evicting from the front (same pattern `xm_ticker.py`
already uses for its microstructure buffers — proven, kept), recompute
`realized_vol_60s`/`spread_mean_60s`/`spread_std_60s` from the current
ring buffer contents only (O(window size), not O(history)). No historical
reload ever occurs inside `on_tick`. A separate, explicit `StateEngine.bootstrap(recent_m1_bars)` method seeds
the ring buffers from a bounded recent window — this is initialization,
not per-tick recomputation, and is the only place historical bars enter
the live engine (Section 13). Since `mt5_feed.py` (Wine) and the
`StateEngine` (native) are different processes, the backfill bars have to
cross the same socket as ticks: `tick_protocol.py` defines a `backfill`
frame type alongside `tick`, sent once by `mt5_feed.py` immediately after
connecting (from its existing `copy_rates_from_pos` call — same call
`xm_ticker.py` already makes on every connect/reconnect, cheap and fast).
`feed_listener.py` routes `backfill` frames to `StateEngine.bootstrap()`
and `tick` frames to `StateEngine.on_tick()`; no `tick` frame is processed
until the one `backfill` frame for that connection has been applied.

## 10. M1 bar construction

Ported behavior-preserving from `xm_ticker.py`'s proven bar-build logic
(MT5 bar convention: OHLC from bid, minute-bucketed by
`int(t_utc // 60) * 60`). `current_m1` (`complete=False`) is updated in
place tick-by-tick; on minute rollover the just-finished bar is copied
into `completed_m1` (`complete=True`) and a fresh `current_m1` starts.
Downstream code reads `complete` explicitly rather than inferring
completeness from field presence — this is the concrete mechanism that
prevents the lookahead Section 10 warns about.

## 11. Multi-horizon state

Phase 2 keeps three horizons available in the engine's internal buffers —
tick-level (raw ring buffer), M1 (current + completed bar), and a bounded
rolling window (60s/300s, matching what's already computed today) — without
building the hundreds of Phase 3 features those horizons will eventually
feed. `MarketState`'s volatility-state fields are the minimum slice
already required at this layer (Section 8), not a preview of the full
feature library.

## 12. State buffer

In-memory only, bounded: last 300s of raw ticks (ring buffer, matches the
existing 5-minute microstructure window), last N completed M1 bars where
N is set by `state_engine.py`'s own needs (default: 480 bars / 8 hours —
enough for session-relative context without approaching the six-year
research dataset). No CSV/dataset load happens in the live process.
Historical data (`data/gold_seed_merged_full6yr.csv`) stays exclusively a
`learning/`/`research/` concern.

## 13. Historical/live separation

The live `MarketState` is generated exclusively from the current tick
stream plus the engine's own previous in-memory state.
`StateEngine.bootstrap()` (Section 9) is the one explicit,
clearly-named exception — a startup-only initialization from a bounded
recent backfill (the same `copy_rates_from_pos` call `xm_ticker.py`
already makes), never invoked mid-stream, never able to leak
future-relative information since it only ever runs before the first live
tick is processed.

## 14. Feed gap handling

`feed_listener.py` classifies every anomaly explicitly rather than
silently continuing:

- **no tick for >5s** (tick cadence is ~25-40ms; 5s is a generous multiple)
  → `FeedHealthState.STALE`
- **socket disconnect** → `FeedHealthState.DISCONNECTED`
- **malformed tick line** (bad JSON, missing required field, non-finite
  float) → rejected, logged, does not update state, does not crash the
  listener
- **timestamp reversal** (`market_timestamp` older than the last accepted
  tick's) → rejected as out-of-order (Section 25's out-of-order test)
- **duplicate tick** (identical `market_timestamp` + `bid` + `ask` as the
  last accepted tick) → deduplicated, not double-counted into tick-rate
  stats
- **spread anomaly** (negative spread, or spread > a generous sanity
  multiple of the trailing mean) → tick accepted for bid/ask but flagged;
  `spread` field's data quality marked, not silently trusted
- **symbol unavailable at connect** → `FeedHealthState.INVALID`, does not
  fabricate a state

The engine never emits an apparently-healthy `MarketState` while the feed
is actually stale — `feed_health` is authoritative and checked by any
future consumer before trusting price fields.

## 15. Reconnection

State machine, `mt5_feed.py` (Wine-client) side, extending the existing
proven MT5-reconnect logic one level out to also cover the TCP link to
`feed_listener.py`:

```
CONNECTED -> (no tick / socket error) -> STALE -> RECONNECTING -> CONNECTED
```

Bounded exponential backoff (1s, 2s, 4s, 8s, capped at 8s — short enough
to recover fast from a transient blip, bounded enough to never busy-loop),
with each attempt logged. `feed_listener.py`'s own health state mirrors
this from the server side (socket accept/drop), so a consumer sees
`DISCONNECTED` promptly regardless of which side of the link actually
failed.

## 16. Market session / clock

UTC internally, everywhere, no exceptions — `market_timestamp` is always
the offset-corrected true-UTC value, never raw MT5 server time. The
empirically-derived XM session-hours logic (Section 2) is ported
unchanged into `state_engine.py` as a documented, named function
(`is_market_closed(utc_time) -> bool`) rather than re-derived from
scratch or assumed to be generic Forex hours. Presentation-layer
conversion (e.g., IST for a human-readable log line) happens only at the
log-formatting boundary, never stored.

## 17. Feed health monitor

`FeedHealthState` (Section 8/14) is exposed on every `MarketState`
instance — any future consumer (the eventual decision engine) can check
`market_state.feed_health` and refuse to act on `STALE`/`DISCONNECTED`/
`INVALID` data. Phase 2 exposes this signal; it does not implement the
refusal logic itself (that's decision-engine work, a later phase).

## 18. Persistence / recovery

- **Ephemeral, rebuild freely:** the tick ring buffer, computed
  volatility-state fields — lost on restart, reconstructed by
  `bootstrap()` + a few seconds of live ticks.
- **Recoverable, needs persistence:** only `xm_server_offset.json` (the
  measured server-clock offset — genuinely expensive to redetect when the
  tick path is stale over a weekend, exactly why `xm_ticker.py` already
  persists it, reused unchanged). The M1 backfill window itself does
  *not* need disk persistence: `copy_rates_from_pos` is a cheap, fast MT5
  call `mt5_feed.py` already makes fresh on every connect/reconnect
  (proven existing behavior) — a restart just re-fetches it via the
  `backfill` frame (Section 9) rather than reading a stale on-disk copy.
  `internal_seq`'s last value is *not* persisted either — a restart is
  expected to reset it, and a consumer can already detect a restart from
  the accompanying `feed_health` transition, so a resurrected counter
  would add complexity without a real recovery need it solves.
- **Historical data:** belongs to `data/`/`research/`, never touched by
  the live process, per Section 12/13.

## 19. Legacy file-polling retirement — precise scope

**Retired from the V3 runtime path:** `app/engine.py` no longer reads
`xm_tick_state.json` for bid/ask or market-closed detection — those come
from `feed_listener.py`'s in-process `MarketState` instead.
`xm_bars_backfill.csv`/`xm_live_bars.jsonl` as the seed-building
intermediary are retired — `mt5_feed.py`'s backfill feeds
`StateEngine.bootstrap()` directly.

**Not retired, deliberately:** the verdict-tracking half of
`xm_tick_state.json` (trade_id/verdict/sl_first_px/tp_first_px fields) —
per the confirmed scope boundary (Section 2), this stays exactly as it is
today, on its existing file channel, since it's real trade-management
logic Phase 2 does not touch.

**Archived, not deleted:** the old `market/xm_ticker.py` (once
`mt5_feed.py` is live-verified), plus any now-superseded backfill CSV
artifacts — moved to `.archive/`, preserved for research/replay value,
same policy as Phase 1.

## 20. App/engine integration

`app/engine.py` gains a minimal accessor
(`market.feed_listener.get_latest_state() -> Optional[MarketState]`) used
only to prove the pipeline reaches the V3 application boundary. No change
to `LiveEngine`'s CatBoost signal path, feature computation, or Telegram
logic — those keep using the existing `gold_seed.csv`-buffer +
`build_features()` path untouched (Section 22/23). This accessor is
additive and inert until a later phase wires it into the actual decision
loop.

## 21. Model routing foundation — preserved, not extended

No change to `decision/router.py`, `models/registry/`, or
`config/models.yaml` — Phase 1's router already supports arbitrary future
model families via the `ModelFamily` literal and per-role config; Phase 2
adds no new roles and does not hardcode anything to CatBoost that wasn't
already CatBoost-specific in Phase 1 (the two active model registrations).
`MarketState`'s shape (Section 8) is deliberately family-agnostic — it's
raw state, not a feature vector tied to any one model's input schema.

## 22-24. No model/trading changes, read-only feed

No CatBoost/model file, feature schema, threshold, calibration, SL/TP, or
Telegram logic changes anywhere in Phase 2. `mt5_feed.py` only calls
read-only MT5 API functions (`initialize`, `symbol_select`,
`symbol_info`, `symbol_info_tick`, `copy_rates_from_pos`) — the same set
`xm_ticker.py` already calls today. No order-placement, position-close,
or account-modification call is added; this is enforced by code review
during implementation (no such MT5 function appears anywhere in the new
module), not by a runtime guard, since the read-only call set is small
and enumerable.

## 25. Testing

All in `tests/`, plain-assert style, matching the existing convention:

- `tests/test_tick_contract.py` — valid tick accepted; malformed tick
  (missing field, non-finite float, bid>ask... note: bid>ask can be a
  real crossed-market artifact briefly, so this is flagged not rejected —
  see Section 14) rejected
- `tests/test_market_state_contract.py` — `MarketState` validation,
  `DataQuality`/`FeedHealthState` enum behavior
- `tests/test_state_engine.py` — incremental M1 construction from a known
  synthetic tick sequence produces the expected OHLC; minute-boundary
  rollover produces the correct `completed_m1`/`current_m1` split;
  duplicate tick handling; out-of-order tick handling per the documented
  policy (Section 14); incremental volatility/spread stats match a
  reference (non-incremental, recomputed-from-scratch) calculation on the
  same synthetic window — this is the incremental-correctness proof
- `tests/test_feed_listener.py` — stale-feed transition, disconnect
  transition, reconnect recovery, using a synthetic socket client (no
  Wine/MT5 needed)
- `tests/test_boundary.py` (extended) — `market/` (the new live modules)
  still never imports `learning/`/`research/`
- `tests/test_latency_instrumentation.py` — the three latency figures
  (Section 7) are computed and non-negative/sane on synthetic data
- A live-verification pass (not a `tests/` script — a one-time manual run
  during implementation, Xvfb+Wine+MT5, real XM connection) confirms
  `mt5_feed.py` connects, streams real ticks, and `feed_listener.py`
  produces a real `MarketState` — its output (raw log, latency numbers)
  is captured and reported as live evidence, then both processes are
  stopped.

## 26. Performance test

`tests/test_performance.py` (or a `scripts/` benchmark script, decided at
plan time) replays a **synthetic** tick stream built from the real schema
proven by `xm_ticker.py`'s own field usage (realistic spread/price
dynamics, realistic ~25-40ms inter-arrival jitter) through
`feed_listener.py` + `state_engine.py`, measuring ticks/sec, p50/p95/p99
processing latency, memory, and CPU. Explicitly labeled synthetic in
every report line. If the live-verification window (Section 25) produces
enough real ticks to compute a meaningful comparison figure, it's reported
separately and never averaged together with the synthetic numbers.

## 27. Research opportunity — noted, not built

`MarketState`'s raw-state fields (Section 8/11) already preserve what
tick-level returns, inter-arrival time, tick intensity, bid/ask dynamics,
spread dynamics, and directional tick imbalance would need as their
inputs. No Phase 3 feature computation is added — this section documents
that the plumbing doesn't block those features later, nothing more.

## 28. Documentation

`docs/ARCHITECTURE.md` gets a new "Phase 2" section: live data flow
(Mermaid), `MarketState` lifecycle, tick normalization, timestamp
handling, latency measurement, feed health, reconnect behavior, M1
construction, state buffering, legacy-path retirement (precise scope per
Section 19), model-routing compatibility statement, plus a documented list
of actual MT5/XM limitations discovered (Section 2's findings,
consolidated).

## 29. Completion criteria

Managed feed process exists and is live-verified against real XM (once,
bounded window) · canonical `Tick`/`MarketState` contracts work · M1
state correctly distinguishes current vs completed · incremental updates
proven correct against reference calculations · feed health observable ·
stale/disconnect/reconnect implemented and tested · timestamps rigorous
(three distinct, named, never mixed) · latency measured (not assumed) ·
legacy file-polling no longer the V3 runtime dependency for market data
(verdict-tracking explicitly excluded, Section 19) · persistence/recovery
defined and tested · Phase 3 has a clean, family-agnostic state interface
· all `tests/` pass · a labeled synthetic performance baseline exists ·
`docs/ARCHITECTURE.md` updated with Phase 2 section + Mermaid diagram.

## 30. Explicitly out of scope

Feature library, specialist models beyond what Phase 1 already
registered, dynamic SL/TP, EV, virtual trade management, EOD learning,
champion/challenger execution, any change to CatBoost models/thresholds/
calibration/Telegram logic, migrating verdict-tracking off its existing
channel, restarting `ai-engine.service`/`gold-shadow.service` as
persistent live services (the live-verification window is bounded and
manual, not a service restart).
