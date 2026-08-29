# GOLDEX Phase 1: Market & Execution Nervous System — Design + Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans

Governs: "GOLDEX — AUTHORIZE PHASE 1: MARKET & EXECUTION NERVOUS SYSTEM" mandate
(2026-08-29, same session as the foundation cleanup, commits `af01bdc`/`70012d8`/`8f92275`).

Goal: make historical replay and the MT5 live path expose the same MarketState
shape, close the identified leakage-test gaps, make account/position state
complete, make execution realism configurable rather than partially-dead, let a
future decision seam specify size (not just side/SL/TP), widen experience
recording to raw facts, and add real (not sanity-only) latency/throughput
measurement. **No trading intelligence, no live orders, no strategy.**

Spec basis: this plan is built directly from a read-only gap analysis of the
current foundation (`simulator/`, `market/`, `contracts/`, `journal/`, `config/`)
against the mandate's Section 18 acceptance criteria — findings cited inline per
task, file:line, not re-derived from scratch.

Global constraints: no new trading logic; every task ships with a test; no task
claims a validated trading signal or profitability; MT5 stays live-order-free
(Section 16); reuse `simulator/replay.py`'s existing `DecideFn`/`ManageFn` seam
rather than inventing a new decision interface (the gap analysis confirmed it
already has the right shape); `.venv` (numpy/pandas/pydantic/pytest/numba/
scikit-learn) is the reproducible test environment going forward — every task's
tests run via `.venv/bin/python -m pytest`.

## Gap summary (from read-only analysis, condensed)

| Area | State |
|---|---|
| Unified MarketState (historical vs. live) | **Diverges**: `simulator/market_state_builder.py` hardcodes tick_count/spread_mean/std to 0/None and collapses ingestion=processing=market timestamp; `market/state_engine.py` computes those for real but never sets `realized_vol_60s` (always None) |
| MT5 bridge isolation | Already clean — zero V3/V4 imports in `market/`, confirmed by grep |
| Historical replay chronology/no-look-ahead | Solid mechanically; only 2 of 7 mandate leakage categories have dedicated tests |
| Account state | Missing realized_pnl accumulation, drawdown tracking, currency |
| Position state | Missing exit_reason on the position object itself, cumulative execution cost, stored current_price |
| Execution realism | `latency_ms` field declared, never used anywhere — dead; no rejected/failed fill states |
| Trade lifecycle seam | Already exists (`DecideFn`/`ManageFn` in `simulator/replay.py`) — just needs `size` added to the tuple |
| Clock abstraction | market/ingestion/processing timestamps real on live path, collapsed on historical path; no decision/order/ack timestamps anywhere (no order object exists yet) |
| Experience recording | Reward-shaping is clean (no R-multiple baked in) but `market_state_snapshot` is truncated to `{mid, spread}`, not raw MarketState |
| Performance measurement | Only live tick throughput/latency measured; historical replay throughput, construction/execution/recording latency unmeasured |
| Data quality | `is_market_closed()` exists but is dead code, never called from `on_tick`; no invalid-price/spread-anomaly/corrupted-record handling either side |
| Architecture neutrality | Confirmed clean — no fixed horizon, no strategy branch anywhere in the foundation |
| Account currency/leverage/margin config | Missing — `RiskConfig` is empty, leverage is a hardcoded default in `SimulatedExecutionConfig` |

## Tasks

Each task: failing test first, minimal implementation, verify pass, commit.
Ordered by dependency, not by mandate section number.

### Task 1 — Unify MarketState construction (historical ↔ live)

Fix `simulator/market_state_builder.py` so it either computes real values for
`tick_count_60s/300s`, `spread_mean_60s/std_60s` from the bar-derived data it
has (M1 OHLC + spread series), or leaves them `None` with an explicit
`"structurally unavailable at bar granularity"` docstring note instead of a
silent `0`/`0.0`. Add `realized_vol_60s` computation to
`market/state_engine.py:StateEngine.on_tick` (currently the one field the live
path never sets). Do not collapse `ingestion_timestamp`/`processing_timestamp`
to the exact market timestamp in the historical path if a configurable
simulated-latency offset would be more honest — keep it simple: document
zero-offset as a deliberate simplification if kept, rather than an oversight.

New test: `tests/simulator/test_historical_live_interface_consistency.py` —
feed equivalent synthetic tick/bar data through both `build_snapshot` and
`StateEngine.on_tick`, diff the resulting `MarketState` field-by-field, assert
only genuinely bar-granularity-unavailable fields differ (and that those are
`None`, not a fabricated zero).

### Task 2 — Wire market-closure detection into the live tick path

`market/state_engine.py`'s `is_market_closed()` exists but is never called
from `on_tick`. Wire it in: closed-market ticks should be flagged (e.g. via
`MarketState.data_quality`/`feed_health`, not silently processed as normal).

Test: extend `tests/market` (new file if none exists) or add to
`test_state_engine.py`'s successor — feed a tick during a closure window,
assert the resulting state reflects it.

### Task 3 — Fill the leakage-test gap (Section 11)

Of the mandate's 7 leakage categories, only future-price leakage and
observation-feature look-ahead are currently tested
(`tests/simulator/test_no_leakage.py`, `test_observation_features_no_lookahead.py`).
Add explicit tests for the remaining categories against `simulator/replay.py`:
future timestamp leakage (distinct from price), future account-state leakage,
future position-outcome leakage. The historical/live interface-consistency
test from Task 1 covers the 7th category (historical/live interface
differences).

### Task 4 — Extend AccountState: realized PnL, drawdown, currency

`simulator/contracts.py:AccountState` currently has balance, equity,
margin_used, margin_free, exposure, open_position_id, simulation_timestamp.
Add `realized_pnl_total` (accumulated on every `close_position` call, not just
returned per-trade), `peak_equity`/`drawdown` (updated on every equity change),
`currency` (plain string field, config-driven default).

Test: extend `tests/simulator/test_engine.py`/`test_experience.py` — open and
close a sequence of positions with known PnL, assert `realized_pnl_total` and
`drawdown` track correctly across the sequence.

### Task 5 — Account/margin/leverage configuration

`config/schema.py:RiskConfig` is currently empty (`pass`); `leverage` is a
hardcoded default (100.0) inside `SimulatedExecutionConfig`. Add `currency`,
`leverage`, `margin_call_level` (or equivalent) fields to `RiskConfig`, read
from a non-empty `config/risk.yaml`, and have `SimulatedExecutionConfig`'s
leverage default come from config rather than a hardcoded literal.

Test: `tests/test_config.py`-equivalent (new, minimal — the old one was
archived with the V3 config fields) asserting `load_config().risk.leverage`
matches `config/risk.yaml`.

### Task 6 — Position state completeness

Add `exit_reason` directly to the `Position`/`PositionView` object (currently
only appears on `ExperienceRecord.outcome` after close — the live position
object itself has nowhere to carry it while being monitored), a stored
`current_price` field (currently fed in transiently to PnL methods, not
retained), and cumulative `execution_cost_total` (currently only
`entry_cost_amount`; exit cost is computed separately and never summed onto
the position).

Test: extend `tests/simulator/test_engine.py` — open, hold across several
bars, close; assert `execution_cost_total` reflects both entry and exit cost,
`current_price` updates on each monitor step, `exit_reason` is set at close.

### Task 7 — Size in the decision seam

`simulator/replay.py`'s `DecideFn`/`ManageFn` already implement the
FLAT→ENTRY→OPEN→MONITOR→EXIT→FLAT lifecycle with `action`/`sl_price`/
`tp_price` — this is the right seam, confirmed by the gap analysis, and no new
interface is needed. Extend the `DecideFn` return tuple with an optional
`size` (defaulting to `None`, meaning "engine decides" — preserves backward
compatibility with `simulator/engine.py:23`'s current hardcoded
risk-fraction sizing), and have `engine.py` use the caller-supplied size when
present.

Test: extend `tests/simulator/test_replay.py` — a `DecideFn` that returns an
explicit size, assert the resulting position uses it instead of the
risk-fraction default.

### Task 8 — Widen experience recording to raw MarketState

`ExperienceRecord.market_state_snapshot` (`simulator/replay.py:94,139,157,186`)
is currently truncated to `{"mid": ..., "spread": ...}`. Widen it to the full
`MarketState` (as a dict), consistent with Section 10's "record raw facts, do
not compress into a strategy-specific label." Reward-shaping stays untouched —
this task does not add or change any reward/R-multiple logic, only what raw
observation data is captured per record.

Test: extend `tests/simulator/test_experience.py` — assert a recorded
`ExperienceRecord`'s `market_state_snapshot` contains the full field set
`MarketState` exposes, not just mid/spread.

### Task 9 — Real performance measurement (not sanity-only)

`tests/test_latency_instrumentation.py` only asserts latency fields are
non-negative; `tests/test_performance.py` measures live synthetic-tick
throughput only. Add real measured benchmarks (reported, not just
pass/fail-asserted, following the existing `tracemalloc`/`time.perf_counter`
convention already used in `test_performance.py`) for: end-to-end historical
replay throughput (bars/sec via `run_replay` on a synthetic dataset),
`build_snapshot` construction latency in isolation, `execution.py` fill
computation latency, `ExperienceRecorder.record` latency.

Test: new `tests/simulator/test_replay_performance.py`, `Task 9` numbers
printed/logged (not just asserted below some threshold) so real figures are
visible in the deliverable report — per mandate Section 12, "do not claim
millisecond-capable without measurement."

### Task 10 — Execution latency: implement or remove

`SimulatedExecutionConfig.latency_ms` is declared but has zero usages anywhere
in `simulator/` (dead field). Decide by evidence, not convenience: if a
configurable execution-latency model is cheap to add (delay the fill by N ms
relative to the decision bar/tick, using data already available), add it,
tested. If tick-level timing data isn't available in the historical replay
path to make this meaningful (replay is currently bar-level per the gap
analysis, Task 11 below), remove the dead field rather than leaving an unused
knob that implies a capability that doesn't exist.

Test: if implemented — `tests/simulator/test_execution.py` extension proving
a fill is delayed by the configured amount. If removed — no new test, just
confirm no remaining reference (grep) and existing execution tests still pass.

### Task 11 — Rejected/failed execution states

Every `open_position`/`close_position` call currently always fills — no
insufficient-margin or invalid-SL/TP rejection path exists. Add explicit
rejection: `open_position` returns a rejection reason (not a fabricated fill)
when requested size exceeds available margin, or SL/TP violate basic sanity
(e.g. SL on the wrong side of entry for the given direction).

Test: `tests/simulator/test_engine.py` extension — request a position that
exceeds available margin, assert rejection (no position opened, account state
unchanged) rather than a silent fill.

### Task 12 — Data quality: invalid price / spread anomaly / corrupted record handling

Neither the historical (`simulator/closure.py` handles gaps/weekend closure
only) nor live (`market/state_engine.py` handles out-of-order/duplicate only)
path rejects invalid prices (zero/negative/NaN) or flags spread anomalies
(e.g. spread suddenly 100x normal). Add explicit checks to both paths,
surfaced via `MarketState.data_quality`/equivalent rather than silently
passed through.

Test: extend `tests/simulator/test_market_state_builder.py` and a live-side
equivalent — feed a NaN/negative price or an anomalous spread, assert it's
flagged rather than silently accepted.

### Task 13 — Tick-level historical replay: scope decision, not build

Mandate Section 4 asks for "tick-level progression where source data
permits." Current `data/` only has bar CSVs (`gold_seed.csv`,
`xm_bars_backfill.csv`) — no tick-level historical data exists in this repo.
This task is a documentation task, not a build task: record explicitly in the
final Phase 1 report that tick-level historical replay is not implemented
because no tick-level historical dataset currently exists, and that
`run_replay` remains bar-level. No code change.

### Task 14 — Decision/order/ack timestamp chain: scope decision

Mandate Section 3 wants the full market-timestamp → receipt → decision →
order-submission → execution-ack chain measurable. Phase 1 has no live-order
path (Section 16 forbids it) and no order-submission object exists yet
(intelligence isn't built). Building placeholder timestamp fields for a
submission step that can't happen yet would be speculative. This task
documents the chain that DOES exist today (market_timestamp →
ingestion_timestamp → processing_timestamp, real on the live path per Task 1)
and explicitly notes decision/order/ack timestamps are deferred to whichever
phase adds the actual decision-and-order pipeline — not built here.

### Task 15 — Whole-branch review

After Tasks 1-12 (13/14 are documentation, not code), run the full retained
+ new test suite via `.venv/bin/python -m pytest`, re-run the
decision/candidates/learning import-cleanliness grep from the cleanup phase
against the now-larger foundation, and produce the mandate's Section 20 final
deliverable report.

## Execution order

Tasks 1→9 are independent enough to parallelize (each touches a different
file/concern); Task 10 depends on Task 1's confirmation of whether replay
stays bar-level; Task 11 is independent; Task 12 is independent; Tasks 13/14
are pure documentation, do any time; Task 15 is last, always.

## What this plan does not do

No `intelligence/` code, no decision engine, no strategy, no live MT5 order
submission, no reward function, no R-multiple, no regime classifier — all
explicitly out of scope per mandate Sections 2, 16, 17.
