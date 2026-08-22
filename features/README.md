# features/

Quantitative feature fabric (Phase 3): the causal, versioned,
live-and-replay-compatible feature universe built on top of Phase 2's
`MarketState`. Full design: `docs/superpowers/specs/2026-08-19-golex-v3-phase3-feature-fabric-design.md`.

## Family modules

- `_shared.py` — `SharedInputs`/`build_shared_inputs`: the common
  precomputed arrays (OHLC, `ret1`/`sign1`, `ewma_vol`, Kalman residual)
  every family's `compute_<family>()` needs, built once per call instead
  of independently by each family.
- `returns_dynamics.py` (Family A) — return-dynamics candidates: signed
  run-length, lag-1 autocorrelation, rolling return-percentile rank.
- `volatility_dynamics.py` (Family B) — realized variance/semivariance,
  Parkinson volatility, vol acceleration/of-vol, 252-day vol percentile.
- `jump_detection.py` (Family C) — CUSUM-based jump/changepoint
  detection: distance-to-threshold, bars-since-last-changepoint, jump
  intensity.
- `distribution_info.py` (Family D) — information-theoretic shape of the
  return distribution: Shannon/permutation/sample entropy, a
  mutual-information-proxy sign feature, tail probability.
- `market_geometry.py` (Family E) — price-range positioning: distance
  from rolling high/low, range position/width ratio, breakout-failure
  magnitude, displacement from equilibrium.
- `persistence.py` (Family F) — mean-reversion speed/half-life,
  autocorrelation-based persistence (reuses `returns_dynamics`'s
  `rolling_autocorr_lag1` — the one kernel genuinely shared across two
  families).
- `temporal.py` (Family G) — cyclical hour/minute/day-of-week encodings
  and UTC session-band flags (a different, independent concept from
  `market/state_engine.py`'s XM-specific `is_market_closed()`).
- `microstructure_history.py` (Family H) — historical tick_volume/spread
  derived features, honestly scoped to what the 6.7yr CSV actually has
  (no order book, no real tick stream).
- `regime_state.py` (Family I) — discretized regime/state variables
  (causal tercile buckets), built from other families' already-computed
  continuous outputs, not an HMM.
- `first_passage.py` (Family J) — retrospective, empirical first-passage
  probability/time/frequency stats; causal by construction (every inner
  loop index strictly `< i`, verified by direct code read).
- `microstructure_live.py` — new **live-only** TICK-triggered family
  (`TickActivityTracker`): spread change, spread-shock z-score, tick
  inter-arrival mean/std, arrival burstiness. No historical analogue
  exists or can exist (no tick-level history in the 6.7yr CSV).
- `hurst.py`, `fracdiff.py`, `volatility.py`, `labeling.py` — pre-existing
  supporting math (rolling Hurst, fractional differencing, GK/RS/YZ
  volatility estimators, triple-barrier labeling + CUSUM event sampling),
  untouched by Phase 3.
- `kalman.py` — `kalman_local_level` (the original batch numba filter,
  unchanged) plus `StatefulKalman`, a new O(1)-per-update incremental
  class with identical math, persisting `(x0, x1, p00, p01, p10, p11)`
  across calls instead of looping over a full array.
- `daily_buffer.py` — `DailyBuffer`, a new bounded daily-resampled ring
  buffer for the handful of ~252-observation features
  (`vol_percentile_252`, `spread_percentile_252`) that need more history
  than the live process's bounded M1 window holds.
- `features.py` — the CURRENT PRODUCTION 28-column feature set
  (`build_tier1_features`), imported live by `app/engine.py`/
  `app/shadow.py`. Unmodified by Phase 3; registered in the registry as
  family `baseline_v1`.
- `replay_engine.py` — batch entry point. `build_candidate_features()`
  composes every family's `compute_<family>()` into one DataFrame from a
  historical CSV; also the reference implementation `live_engine.py`'s
  per-M1-close recompute is checked against.
- `live_engine.py` — trigger-driven live entry point. `LiveFeatureEngine`
  exposes `on_tick()`/`on_m1_close()`, additive only — never called from
  `app/engine.py`'s decision loop.

## Registry layout

`features/registry/<family>/<feature_id>.json` — one JSON file per
registered feature, one subdirectory per family, each validated through
`contracts/feature_schema.py`'s `FeatureDescriptor` pydantic model
(mathematical definition, source module, causality/live-compatibility
flags, historical coverage, warmup bars, status + status_reason, evidence
reference, version, ...). `features/registry/__init__.py` provides
`load_descriptor()`/`load_family()`/`load_all()` (read and validate) and
`build_schema(schema_id, schema_version, feature_ids) -> FeatureSetSchema`
— the hook a future Phase 4 specialist model uses to construct its own
named feature-set slice from this universe, without the registry ever
assuming one universal vector. `features/registry/diagnostics.py` holds
reusable `correlation_redundancy()`/`distribution_stability()` functions,
generalized from the original 92-feature research selection methodology
so they can be run against any family (including new ones with no prior
evidence, e.g. `microstructure_live`).

The registry currently holds 125 descriptors: 28 `baseline_v1` (the
production feature set) + 92 candidates moved from
`research/features_v3.py` across families A-J + 5 new live-only
`microstructure_live` features.

## `replay_engine.py` vs `live_engine.py`

Two entry points, same math. Both call the exact same
`features/<family>.py` `compute_<family>()` functions — `replay_engine.py`
in batch mode over a historical CSV-derived DataFrame, `live_engine.py`
per M1-close over a bounded window (`StateEngine.completed_m1_window(n)`,
O(window) not O(history)) pulled from live `MarketState`. Sharing function
bodies instead of hand-deriving a second incremental implementation per
feature is the core design choice: there is nothing to drift between live
and replay except real bootstrap/window-edge effects, and those are
checked directly by `tests/test_live_replay_equivalence.py` rather than
trusted by construction alone.

## Live-only additions

Three pieces exist only because a live process has state a one-shot batch
replay doesn't:

- **`DailyBuffer`** (`daily_buffer.py`) — bootstrapped once at
  `live_engine` startup from the small rolling `data/gold_seed.csv`
  (~2.5 months), not the full 6.7yr merge, capped at a bounded ring size
  (252 entries). Backs `vol_percentile_252` live without ever loading full
  history into the live process.
- **`StatefulKalman`** (`kalman.py`) — O(1) incremental Kalman filter,
  equation-for-equation identical to `kalman_local_level`, verified to
  rtol=1e-9/atol=1e-12 over 200 samples. Not yet wired into
  `live_engine.py` (the existing 28-col baseline Kalman features already
  run fine at window≤480 without it) — a tested, standalone utility for a
  future genuinely-per-tick consumer.
- **`TickActivityTracker`** (`microstructure_live.py`) — small per-tick
  ring buffers (same pattern as `market/state_engine.py`'s
  `_tick_times_60s`/`_spreads`), producing the 5 TICK-triggered
  `microstructure_live` features with no batch/replay analogue at all.

## Full design

`docs/superpowers/specs/2026-08-19-golex-v3-phase3-feature-fabric-design.md`
has the complete Phase 3 design: architecture diagram, registry contracts
(`FeatureStatus`/`HistoricalCoverage`/`ComputationalCost`/`UpdateTrigger`
enums), the live-engine trigger model, the live-only feature rationale,
testing strategy, and the one-sentence rule the whole design turns on
(§9): historical survivor evidence describes what helped one past
model/config — it is not a universal filter, and the registry preserves
the entire quantitative universe, not a pre-selected subset.
