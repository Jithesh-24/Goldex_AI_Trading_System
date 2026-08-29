# GOLDEX Foundation Cleanup Plan

STATUS: proposed, not executed. No deletions/moves/commits made. Awaiting approval per mandate Section 15.

Governs: "GOLDEX — ARCHITECTURE RESET & PHASE 1 FOUNDATION CLEANUP" mandate +
"APPROVE AUDIT / AUTHORIZE DEPENDENCY-AWARE CLEANUP PLAN" directive (same session,
2026-08-29). Built from two read-only audit passes (dir-level, then file-level).

## A. File-level deletion candidates

None. Per Section 5/8 of the mandate ("do not erase historical evidence"), nothing
is hard-deleted in this plan. Everything currently classified RETIRE moves to an
archive location (Section B) or is removed from the active import graph while the
file itself survives in git history + archive. The only exception ever considered
(`rotator_keys.json`) is explicitly untouched (Section 8 of the user's directive).

## B. File-level archive candidates

Move under `.archive/v3-v4-intelligence-2026-08-29/` (mirroring the existing
`.archive/pre-rebuild-2026-08-17/` precedent), preserving relative paths:

- `decision/` (all files) — except `round_trip_cost_r`, extracted first (Section D.1)
- `candidates/` (all files)
- `learning/` (all files)
- `models/registry/` (all files)
- `models/active/*.cbm` (archived as historical artifacts, not deleted, not loaded by anything)
- `app/engine.py`, `app/shadow.py`
- `contracts/specialist_output.py`, `contracts/ev_decision.py`, `contracts/model_registry.py`,
  `contracts/feature_schema.py`, `contracts/virtual_trade.py`
- `features/labeling.py`, `features/regime_state.py`, `features/live_engine.py`,
  `features/replay_engine.py`, `features/features.py`, `features/registry/diagnostics.py`,
  `features/registry/schemas.py`
- `features/daily_buffer.py`, `features/microstructure_live.py` (ARCHIVE, not RETIRE —
  historical-research-only, not decision-coupled, but not a going-forward dependency either)
- `research/` ARCHIVE list (full 51-file list from the file-level audit — everything
  not on the ACTIVE list in Section C below)
- `research/phase5b_diagnostics/` (all 8 files, subpackage)
- `config/decision.yaml`, `config/models.yaml`, `config/learning.yaml`
- `tests/candidates/` (all 9 files)
- `tests/` (flat) RETIRE list from the file-level audit (~60 files: `test_ev_*`,
  `test_calibration_registry.py`, `test_router.py`, `test_model_registry.py`,
  `test_specialist_*`, `test_phase4_*`, `test_phase5*`, `test_phase5b_*`,
  `test_baseline_registry.py`, `test_build_meta_side_contract.py`,
  `test_direction_side.py`, `test_labeling.py`, `test_live_engine.py`,
  `test_live_replay_equivalence.py`, `test_replay_engine.py`, `test_feature_*`,
  `test_regime_state.py`, `test_state_engine.py`, `test_feed_listener.py`,
  `test_tick_capture.py`, `test_market_geometry.py`, `test_microstructure_*`,
  `test_persistence.py`, `test_temporal.py`, `test_returns_dynamics.py`,
  `test_volatility*`, `test_jump_detection.py`, `test_first_passage.py`,
  `test_distribution_info.py`, `test_daily_buffer.py`, `test_historical_coverage.py`,
  `test_causality.py`, `test_config.py`, `test_contracts.py`, `test_cv.py`)
- `tests/research/` RETIRE list (`test_phase2_*`, `test_phase3_*`,
  `test_phase4_trajectory_assembly.py`)

Archive is a `git mv`, not a copy — history stays attached to each file.

## C. Files to preserve (active foundation)

- `simulator/` in full (contracts.py, engine.py, execution.py, replay.py, experience.py,
  market_state_builder.py, closure.py) — after the Section D.1 cost-dependency fix
- `contracts/market_state.py`, `contracts/tick.py`, `contracts/journal.py`
- `market/` in full (mt5_feed.py, feed_listener.py, tick_protocol.py, state_engine.py,
  synthetic_replay.py, tick_capture.py) — confirmed zero decision/candidates coupling
- `journal/` (already declared architecture-neutral by its own docstring)
- `trading/` (disk_monitor.py, space_guard.py, watchdog.py — ops safety, non-coupled)
- `scripts/` (data acquisition utilities, non-coupled)
- `services/` (shell ops scripts, non-Python-coupled)
- `memory-compressor.py`
- `features/`: `_shared.py`, `returns_dynamics.py`, `volatility.py`,
  `volatility_dynamics.py`, `hurst.py`, `fracdiff.py`, `kalman.py`, `first_passage.py`,
  `market_geometry.py`, `distribution_info.py`, `jump_detection.py`, `temporal.py`,
  `microstructure_history.py`, `persistence.py` — generic causal primitives, candidate
  future evidence-source inputs, currently used by nothing in the active foundation
  (zero simulator/genesis-research imports today) but kept per Section 4/7 of the
  original mandate as reusable infra, not V3 debt
- `research/`: `genesis_event_time_test.py`, `genesis_event_time_test1b_falsification.py`,
  `genesis_horizon_sweep.py`, `phase3a_representation_experiments.py`,
  `phase3a_nonlinear_smoke_test.py`, `phase3a_raw_path_geometry_probe.py`,
  `phase3a_volatility_conditioned_direction.py`, `phase4_garch_volatility_mechanism.py`,
  `phase4_kalman_trend_mechanism.py`, `phase4_distributional_mechanism.py`,
  `phase4_trajectory_vs_snapshot_test.py`, `phase4_mechanism_oos_check.py`
- `config/market.yaml`, `config/runtime.yaml`, `config/risk.yaml`
- `tests/simulator/*` (10 files), `tests/research/test_genesis_event_time_test.py`,
  `tests/research/test_genesis_event_time_test1b.py`
- `tests/test_boundary.py`, `tests/test_performance.py`,
  `tests/test_latency_instrumentation.py`, `tests/test_tick_protocol.py`,
  `tests/test_tick_contract.py` (resolved from UNKNOWN, see Section E)
- `docs/` in full, all history — genesis-track docs tagged active, V3 docs tagged
  history (Section G)
- `.archive/` existing contents, unchanged
- `data/` — see Section F
- `rotator_keys.json` — untouched, flagged only (Section 8 of directive)
- `logs/` — untouched, runtime output

**9 previously-UNKNOWN tests, resolved by reading contents (not filename):**

| File | Imports | Class | Reason |
|---|---|---|---|
| `test_causality.py` | `features.features`, `features.replay_engine` | ARCHIVE | both imports are V3-retired feature-assembly shells |
| `test_boundary.py` | none (stdlib `ast` only) | KEEP | AST-based enforcement that `app/`, `market/`, `features/` never import `learning`/`research` — a real architecture-boundary guarantee; `os.walk` over the archived `app/` path yields nothing once moved, so it degrades gracefully rather than breaking |
| `test_config.py` | `config.loader.load_config` | ARCHIVE | asserts `cfg.decision.meta_prob_threshold`, `cfg.models.direction`, `cfg.learning.acc_regression_tolerance` — all fields removed from `Config` in D.5 below |
| `test_contracts.py` (flat) | `contracts.model_registry`, `contracts.feature_schema`, `contracts.virtual_trade`, `contracts.market_state`, `contracts.journal` | ARCHIVE | majority of asserted contracts (model_registry, feature_schema, virtual_trade) are V3-specific and archive in this plan; the file tests them as one unit |
| `test_cv.py` | `features.labeling`, `features.volatility`, `learning.cv` | ARCHIVE | `features.labeling` and `learning.cv` are both archived |
| `test_performance.py` | `contracts.tick`, `market.state_engine`, `market.synthetic_replay` | KEEP | pure retained-infra imports |
| `test_latency_instrumentation.py` | `contracts.tick`, `market.state_engine`, `market.synthetic_replay` | KEEP | pure retained-infra imports |
| `test_tick_protocol.py` | `market.tick_protocol` | KEEP | pure retained-infra import |
| `test_tick_contract.py` | `contracts.tick` | KEEP | pure retained-infra import |

**Two additional contracts discovered while reading `test_contracts.py`, not classified
in the prior audit passes:** `contracts/feature_schema.py` (docstring: "each future
specialist model construct its own slice" — V3 model-routing concept) and
`contracts/virtual_trade.py` (`model_versions`, `expected_value`, `confidence` fields
tied to the V3 EV/specialist pipeline). Both classified **ARCHIVE**, added to Section B.

## D. Dependency changes

**D.1 — simulator/execution.py → decision.ev_cost (the blocking entanglement)**

`round_trip_cost_r` is generic: reads only `market_state` fields (spread,
realized_vol_60s, mid, market_timestamp) and a float `candidate_sl_distance`; it does
not touch `MAEOutput`/`MFEOutput`. Action: copy it verbatim into a new
`simulator/cost_model.py`, drop the unused `contracts.specialist_output` import,
repoint `simulator/execution.py:11` at `simulator.cost_model`. `candidate_sl_tp`
(the actually V3-coupled sibling function, keyed on `MAEOutput`/`MFEOutput` and
`model_status`) archives with `decision/` and is not carried forward.

Correction from the dir-level audit: `simulator/execution.py` line 6 mentioning
`features.labeling` is a docstring credit only, not a real import — no second
dependency to sever there.

**D.2 — app/engine.py, app/shadow.py → decision.signal / decision.router /
features.features / features.labeling**

These two files are the only place `market/` reaches the retired stack, and it's
indirect: `app/engine.py` imports `market.feed_listener.FeedListener` (clean) plus
`decision.signal.SignalEngine`, `decision.router.ModelRouter`,
`features.features.build_features`, `features.labeling.cusum_filter` (retired).
Action: archive both files whole with `decision/`/`features/registry`. `market/`
itself needs no code change — confirmed zero direct coupling to `decision/` or
`candidates/`. A future GOLDEX live engine reads `market/feed_listener.py` directly;
no new engine is built in this cleanup (mandate Section 14).

**D.3 — features/live_engine.py, features/replay_engine.py, features/registry/*
→ decision/candidates/config**

Archive with the rest of `decision/`/`candidates/` — these are V3 feature-matrix
assembly shells around the KEEP-classified math primitives, not the primitives
themselves. No active foundation file currently imports them.

**D.4 — models/active/*.cbm**

No code path loads these automatically today outside the retired `learning/` and
`app/engine.py`. Archiving removes them from any future accidental load surface.

**D.5 — config/loader.py + config/schema.py → decision.yaml/models.yaml/learning.yaml**

`config.loader.load_config()` unconditionally reads all nine yaml files into one
`Config` object; archiving `decision.yaml`/`models.yaml`/`learning.yaml` without
touching the loader would make `load_config()` raise `FileNotFoundError` for every
caller, including ones that only need `market`/`risk`/`runtime`. This is the config
equivalent of the D.1 simulator entanglement, so it gets the same treatment: remove
`DecisionConfig`, `ModelRoleConfig`, `LearningConfig` and their fields from
`config/schema.py`'s `Config`, and remove the corresponding three `_load(...)` calls
from `config/loader.py`. `MarketConfig`, `FeaturesConfig`, `RiskConfig`,
`TelegramConfig`, `JournalConfig`, `RuntimeConfig` are untouched. `tests/test_config.py`
asserted the removed fields and archives with `decision.yaml` (Section C).

## E. Feature classification (features/, full second pass — see also C and B)

| File | Class |
|---|---|
| `_shared.py`, `returns_dynamics.py`, `volatility.py`, `volatility_dynamics.py`, `hurst.py`, `fracdiff.py`, `kalman.py`, `first_passage.py`, `market_geometry.py`, `distribution_info.py`, `jump_detection.py`, `temporal.py`, `microstructure_history.py`, `persistence.py` | KEEP |
| `daily_buffer.py`, `microstructure_live.py` | ARCHIVE |
| `labeling.py`, `regime_state.py`, `live_engine.py`, `replay_engine.py`, `features.py`, `registry/diagnostics.py`, `registry/schemas.py` | ARCHIVE (with decision/candidates) |

No UNKNOWN remained in `features/` after the file-level pass.

## F. Data classification (untracked `data/`)

| File | Size | Disposition |
|---|---|---|
| `gold_seed.csv` | 6.4MB | source data — commit to git (small enough, reproducibility-relevant) |
| `xm_bars_backfill.csv` | 138KB | source/live-capture data — commit to git |
| `gold_seed.csv.bak_pre_20260825refresh` | 4.5MB | runtime backup artifact — add to `.gitignore`, not committed |
| `gold_seed_merged_full6yr.csv` | 166MB | generated/merged, reproducible from source + `scripts/` — add to `.gitignore`, not committed (too large for git regardless) |

Repository `.gitignore` already has a blanket `*.csv` rule (pre-existing, not added
by this cleanup) — that is why all of `data/` shows untracked rather than committed.
Per the directive, the narrowest fix is two explicit negations immediately after the
`*.csv` line: `!data/gold_seed.csv` and `!data/xm_bars_backfill.csv`, then
`git add` those two files explicitly. The backup (`.bak_pre_20260825refresh`) and the
159MB merged/reproducible file stay excluded under the existing blanket rule — no new
ignore pattern needed for them, and the blanket rule itself is left as-is since it
correctly covers stray CSVs elsewhere in the repo (candidates/research output, etc.),
which is outside this cleanup's scope to re-audit. Nothing in `data/` is deleted.

## G. Test migration

- **Active** (run in CI/local going forward): `tests/simulator/*` (10 files),
  `tests/research/test_genesis_event_time_test.py`,
  `tests/research/test_genesis_event_time_test1b.py`.
- **Archived with source** (moved alongside the code they test, not deleted):
  `tests/candidates/*`, the ~60-file flat-`tests/` RETIRE list, and the
  `tests/research/` V3/V4-tournament test files — see Section B for exact list.
- **Pending 9 files** (Section C) get classified before archiving, not defaulted
  to either bucket.
- No `tests/intelligence/` exists — Track A scaffold was never executed, confirmed
  consistent with "do not implement the new AI" (mandate Section 14).

## H. Final dependency graph (post-cleanup target)

```
ACTIVE FOUNDATION                          ARCHIVED (historical, not imported)
------------------                         -----------------------------------
simulator/  ----------------\              .archive/v3-v4-intelligence-2026-08-29/
  cost_model.py (new, from    \                decision/
   decision.ev_cost.round_     \               candidates/
   trip_cost_r)                 \              learning/
contracts/                       \             models/registry/, models/active/
  market_state.py, tick.py,       \            app/engine.py, app/shadow.py
  journal.py                       \           contracts/specialist_output.py,
market/ (mt5_feed, feed_listener,   \            ev_decision.py, model_registry.py
  state_engine, tick_protocol,      \          features/labeling.py, regime_state.py,
  synthetic_replay, tick_capture)    \           live_engine.py, replay_engine.py,
journal/                             \          features.py, registry/*
trading/, scripts/, services/         \        research/ (51 V3/V4 files, incl.
features/ (13 primitive files)         \        phase5b_diagnostics/)
research/ (12 genesis-track files)      \      config/decision.yaml, models.yaml,
config/market.yaml, runtime.yaml,        \      learning.yaml
  risk.yaml                                 \  tests/candidates/*, ~60 flat tests/,
tests/simulator/*, 2 genesis                \   tests/research/ tournament tests
  research tests
```

Verification after execution: `grep -rn "^from decision\|^import decision\|^from candidates\|^import candidates"`
across every file under `simulator/`, `market/`, `contracts/`, `journal/`, kept
`features/*`, kept `research/*`, `trading/`, `scripts/` must return zero matches.
This is the concrete pass/fail test for "FOUNDATION does not depend on retired
V3/V4 intelligence."

---

## Not done in this plan (per mandate Section 14 / directive Section 15)

No new AI, market perception, decision engine, learning loop, or live-engine
replacement is designed or built here. `intelligence/` stays empty. The 9 UNKNOWN
tests, the `rotator_keys.json` usage question, and the exact `.gitignore` pattern
wording are the only open items before execution.

**STOP — awaiting approval before any archive move, extraction, gitignore edit, or commit.**
