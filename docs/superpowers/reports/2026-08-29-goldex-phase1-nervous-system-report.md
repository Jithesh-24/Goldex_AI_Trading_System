# GOLDEX Phase 1: Market & Execution Nervous System — Final Report

Governs: "GOLDEX — AUTHORIZE PHASE 1: MARKET & EXECUTION NERVOUS SYSTEM" mandate.
Plan: `docs/superpowers/plans/2026-08-29-goldex-phase1-nervous-system-plan.md`.
Range: `90aa1cf..b557bf4` (16 commits, all reviewed via subagent-driven development —
12 task reviews, 1 whole-branch review, 1 fix wave, 1 scoped re-review).

No trading intelligence, no strategy, no live orders were built or connected. `intelligence/`
remains empty.

## FOUNDATION BUILT

- Unified historical/live `MarketState` construction — the two paths had silently diverged
  (fabricated `0`/`None` activity fields on the historical side, missing `realized_vol_60s`
  on the live side, and — caught only in final review — a `market_closed` field populated on
  one side and not the other). All three closed.
- Market-closure detection wired into the live tick path (was dead code).
- Account state completed: `realized_pnl_total` (accumulated), `drawdown`/`peak_equity`,
  `currency`, all config-driven via a new `config/risk.yaml` (leverage, currency,
  `margin_call_level`).
- Position state completed: `current_price`, `execution_cost_total` (entry+exit combined),
  `exit_reason` (on `Position`, deliberately not `PositionView` — adding it there would fail
  an existing leakage-guard test and has no legitimate use before exit occurs).
- Rejected/failed execution states: `open_position` now rejects on insufficient margin or
  invalid SL/TP direction, returning the *same* unmutated account object rather than a
  fabricated fill. The experience log correctly marks a rejected DECIDE record rather than
  leaving it looking like a real trade (caught in final review, fixed).
- Invalid-price and spread-anomaly detection on both paths, via a shared
  `contracts/data_quality.py` module (extracted mid-task after a reviewer caught duplication).
- Optional caller-supplied position size threaded through the `DecideFn` seam, backward
  compatible with existing 3-tuple callers.
- `SimulatedExecutionConfig`'s dead `latency_ms` field removed (evidence-based: replay is
  bar-level, no sub-bar timing exists to model against — implementing it would have been
  fake precision).
- `config.loader.load_config` cached (`@lru_cache`) — was doing uncached 6-file YAML I/O on
  every `SimulatedExecutionConfig()` construction (3.36ms/call, caught in final review).

## MARKET INTERFACE

**Historical:** `simulator/market_state_builder.py:build_snapshot` — chronological, no-look-
ahead (row `i`'s H/L/C withheld), now genuinely matches the live path's field set including
`market_closed`. Bar-level only (see Limitations).

**MT5/live:** `market/mt5_feed.py` → `market/tick_protocol.py` → `market/feed_listener.py` →
`market/state_engine.py:StateEngine.on_tick`. Confirmed zero decision/candidates/learning
imports anywhere in `market/` — already isolated before this phase, and this phase's changes
didn't reintroduce coupling. `on_tick` now has a documented 3-way return contract: out-of-
order/duplicate ticks → `None`; invalid-price/anomalous-spread ticks → flagged `MarketState`
(`data_quality=INVALID`); normal ticks → `MarketState`.

**Not built:** no live order submission, no XM account connection (forbidden by mandate
Section 16).

## EXECUTION

Spread-crossing entry/exit fills, configurable slippage fraction, round-trip transaction cost
(`simulator/cost_model.py`, extracted from the retired V3 `decision/` module in the prior
cleanup phase), now with explicit rejection instead of always-fills. No latency model —
removed as dead/unimplementable at current bar-level granularity, documented rather than
faked.

## ACCOUNT

Balance, equity, margin_used, margin_free, exposure, `realized_pnl_total`, `drawdown`,
`peak_equity`, `currency`, open_position_id, simulation_timestamp. Leverage/currency/margin
constraint are config-driven (`config/risk.yaml`), cached at load time.

## EXPERIENCE

`ExperienceRecord.market_state_snapshot` widened from `{mid, spread}` to the full
`MarketState` via `.model_dump()`. Account dict widened to include the new
`realized_pnl_total`/`drawdown`/`currency` fields. Rejection reason now recorded on the
DECIDE record when an entry is rejected. Reward-shaping untouched throughout — no R-multiple,
no reward function was added or modified at any point in this phase (mandate Section 10).

## LEAKAGE

Of the mandate's 7 leakage categories: future-price leakage and observation-feature look-
ahead were already tested before this phase. This phase added 3 more (future timestamp,
future account-state, future position-outcome — `tests/simulator/test_leakage_extended.py`,
8 tests, each independently verified by a task reviewer to catch a concrete, plausible
leakage bug, not a tautology) and the 7th (historical/live interface differences —
`tests/simulator/test_historical_live_interface_consistency.py`, strengthened during the
final fix wave from a `None`-only check to a real field-by-field-or-declared-divergence
check, which is what caught the `market_closed` gap in the first place). All 7 categories
now have dedicated tests.

## PERFORMANCE (measured, `tests/simulator/test_replay_performance.py`)

- End-to-end historical replay: **606 bars/sec**
- `build_snapshot` construction: p50 **1520µs**, p99 **1967µs** — dominant cost is the
  pandas trailing-window slicing (plausible attribution, not independently profiled)
- Execution fill computation (entry+exit+cost_r): p50 **1.23µs**, p99 **2.03µs**
- `ExperienceRecorder.record`: p50 **0.16µs**, p99 **0.65µs** (~1.80 KB/record)

No claim of "millisecond-capable" beyond what these numbers show. `build_snapshot` is the
one component worth optimizing first if higher replay throughput is needed later.

## TESTS

**127 passed, 0 failed**, `.venv/bin/python -m pytest tests/simulator tests/research
tests/test_boundary.py tests/test_kalman_incremental.py tests/test_latency_instrumentation.py
tests/test_performance.py tests/test_tick_contract.py tests/test_tick_protocol.py
tests/test_market_closure_detection.py tests/test_config.py`, 93s. Only pre-existing numpy
deprecation warnings (unrelated to this phase). Verified independently twice: once by the
final whole-branch reviewer, once by me after the fix-wave re-review.

Environment: `.venv` at repo root, dependencies pinned in `requirements.txt`
(numpy/pandas/pydantic/pytest/PyYAML/numba/scikit-learn) — this environment didn't exist
before this phase; a missing-`pyyaml` gap was caught and fixed mid-phase (Task 5), which is
exactly why the pinning exists now.

Import-cleanliness re-confirmed at final review: `grep` across `simulator/market/contracts/
config/` for `decision|candidates|learning|intelligence` returns only two source-code
*comments*, zero real imports.

## REMAINING LIMITATIONS

- **Tick-level historical replay is not implemented.** `data/` contains only bar CSVs
  (`gold_seed.csv`, `xm_bars_backfill.csv`) — no tick-level historical dataset exists in this
  repo. `run_replay` remains bar-level. Building tick-level replay against absent data would
  have been speculative; documented rather than attempted (mandate Section 4).
- **Decision/order/execution-ack timestamp chain is not built.** The chain that exists today
  is real: market_timestamp → ingestion_timestamp → processing_timestamp (genuine on the live
  path). Decision/order-submission/ack timestamps don't exist because no order-submission
  object exists yet — that's the intelligence layer's job, explicitly out of scope for Phase 1
  (mandate Section 16 forbids live orders; Section 3's fuller chain is deferred to whichever
  phase adds the actual decision-and-order pipeline).
- **`data_quality` and `market_closed` are produced but consumed by nothing yet.** Correct for
  infrastructure-only Phase 1 — a future `DecisionEngine` is what should read and act on them.
  `run_replay` today will still open a position on a bar flagged `INVALID`; no filtering logic
  was added because filtering is a decision, and Phase 1 does not make decisions.
- **The historical 60-second activity/spread window is structurally one bar** at M1
  granularity (60 seconds = exactly 1 prior 1-minute bar). `spread_std_60s` is therefore
  always `0.0` on the historical side, and the 5-sigma spread-anomaly path never fires there
  — only the 10x-mean-ratio fallback provides real anomaly detection at bar granularity. This
  is now documented in code rather than silently misleading; it was not "fixed" by widening
  the window, since that would fabricate precision the bar data doesn't support.
- **Live vs. historical spread-anomaly baselines use different window sizes** (live: 300s tick
  ring buffer; historical: the single-bar window above) — same detection function, genuinely
  different inputs, so the two paths can disagree on identical underlying data. Not resolved;
  flagged for whoever next touches this.
- A handful of deferred minors from task reviews remain (duplicated `VOL_LOOKBACK_BARS`
  constant across two files with no shared source/drift test; two untested degenerate-input
  sentinel paths; a second, unrelated "market_closed" concept inside `market/mt5_feed.py`'s
  supervisor state dict, unconnected to `MarketState.market_closed`; `AccountState.currency`'s
  dataclass default is a hardcoded literal alongside the config-driven factory version).
  None are blocking; none affect correctness of what Phase 1 actually built.

## PHASE 2 READINESS

What's now available to build GOLDEX intelligence on top of:

- A `MarketState` that means the same thing whether it came from historical replay or MT5 live
  — the mandate's Section 1 "one world, two environments" principle, verified by a real test.
- The `DecideFn`/`ManageFn` seam in `simulator/replay.py`, unchanged in shape from before this
  phase except for the additive, backward-compatible `size` element — this is still the single
  place a future decision mechanism plugs in, per the architecture-decision spec's
  `DecisionEngine` contract.
- Complete account/position state, with rejection semantics an intelligence can learn from
  (a rejected entry is now distinguishable in the experience log from a real one).
- Full raw-fact experience recording (`ExperienceRecord.market_state_snapshot` +
  account dict), not a curated subset — ready for whatever credit-assignment/learning
  architecture Track E eventually specifies.
- Real measured performance numbers to reason about decision-loop frequency against.
- A leakage-test suite covering all 7 mandate categories, as the safety net for correctness
  claims Phase 2's work will be judged against.

Not decided by this phase, deliberately: which decision architecture fills the `DecideFn`
seam, whether entry/exit/SL/TP/sizing end up jointly or separately learned, what Tracks B–F
(cross-instrument, tick data, conditional quant-mechanism evidence, credit assignment,
trade-management research) find. None of that was Phase 1's job.

No claim of a validated trading signal or profitability is made anywhere in this report.
