# Golex V3 — Phase 1: Foundation & Repository Reconstruction

Status: approved for implementation planning
Date: 2026-08-18
Scope: Phase 1 only (foundation/structure). No new models, no EV gate, no
dynamic SL/TP, no EOD learning, no champion/challenger execution logic.

## 1. Purpose

Rebuild the repo from an accumulated V1/V2 research-production hybrid into a
clean V3 skeleton: explicit production/research boundary, versioned model
registry, config as single source of truth, formal contracts (market state,
feature schema, virtual trade, journal, model registry), and a documented
target architecture — without breaking anything that currently works and
without implementing later-phase intelligence.

## 2. Current-state inventory (as found 2026-08-18)

**Live processes (both to be stopped before Phase 1 work begins, not
restarted at the end of Phase 1 — cutover to the new layout happens
deliberately in a later phase, not as a side effect of stopping/renaming):**
- `ai-engine.service` → root `ai_signal_engine.py`
- `gold-shadow.service` → root `shadow_engine.py`

Both already import from `core/` (`core.signal.SignalEngine`,
`core.features.build_features`, `core.labeling.cusum_filter`,
`core.calibration`) — production already runs the new two-stage
CatBoost pipeline, not the old v7 LightGBM stack. `ARCHITECTURE.md` at repo
root (dated 2026-08-02, describes v7: 95-feature LightGBM dual-ensemble) is
stale and superseded; it gets archived, not updated.

**`xm_ticker.py`** (direct `MetaTrader5` import, the only MT5 connection) is
not currently a running process. The live engine talks to it only through
file state (`xm_tick_state.json`, `.active_signal_ai.json`,
`xm_live_bars.jsonl`) written to an **external** directory,
`/home/jith/.hermes/profiles/trading/cron/output/` — outside this repo
entirely. This is a real architectural gap: today's "market feed" is a
polling-file contract with a process this repo doesn't manage or version.
**Phase 1 relocates and cleans `xm_ticker.py`'s code but explicitly does NOT
treat it as integrated live infrastructure.** It is marked in the new
`market/` module and in the architecture doc as **temporary legacy
architecture** — the external `cron/output/` file-polling contract is a
known-bad interim state. Phase 2 is where the real real-time MT5 market-state
pipeline (Step 7's `MarketState` contract, actually wired to a feed) gets
built properly.

**`research/`** (untracked, dated today) contains an in-progress, valuable
empirical audit (`research/audit_edge.py` + `research/output/`) that found a
live/OOF calibration-drift bug: the deployed 0.60 meta threshold hasn't fired
in honest out-of-fold validation for ~14 months even though the live system
(fit in-sample) reports ~14.5 signals/day. This is preserved byte-for-byte —
content, not just existence.

**Two near-duplicate CatBoost training runs exist:** `models/primary.cbm` +
`models/meta.cbm` (root — the ones `core/signal.SignalEngine` actually
loads, `model_dir=models/` is hardcoded) vs `models/v2/primary.cbm` +
`models/v2/meta.cbm` (same feature set, same config, slightly different OOF
accuracy — an earlier/parallel run, never loaded by any code). Root pair is
the only "active" model; `v2/` and the four snapshots under
`models/archive/` are superseded runs.

**Data file audit** — grepped `core/` and `research/` for every root-level
CSV/NPY: only `gold_seed.csv` is referenced by current code
(`core/data.py`, `core/seed_refresh.py`, `research/audit_edge.py`,
`research/ev_surface.py`). Every other root data file (`dukascopy_m1_features.csv`,
`xauusd_rally.csv`, `gap_m1_*.csv`, `gold_m1_2021.csv`, `gold_m1_history.csv`,
`gold_recent.csv`, `gold_seed_full6yr.csv`, `gold_seed_merged_full6yr.csv`,
`gold_seed_multi.csv`, `gold_seed.csv.bak_1238`, `backtest_*.csv`,
`_feat_signals.npy`, `quant_features_116.npy`, `prices_tail.npy`,
`train_data_t.npy`, `train_data_y.npy`) is dead to current code but is
**not deleted** — archived with full paths preserved, since it may back
claims in old research docs (`RESEARCH_PROOF.md`, `AI_SYSTEM_SUMMARY.md`).
`catboost_info/` is CatBoost's own transient training-log scratch dir
(regenerated every run) — this one genuinely is disposable, deleted not
archived, and added to `.gitignore`.

**No config framework exists.** Decision thresholds and training-locked
model parameters are currently mixed together in `models/feature_cols.json`
(read by `core/signal.SignalEngine.__init__`): `meta_prob_threshold` is a
tunable decision-policy knob, while `primary_cols`/`meta_cols`/`tb_cfg_trade`/
`horizon_vol_scale`/`max_holding` are training-locked (the code's own
docstring: "so training and inference can never silently drift apart").
Phase 1 splits these by nature, not by convenience:
- **Training-locked parameters → model registry entry** (immutable once a
  model is trained; changing them means training a new model).
- **Decision-policy thresholds → `config/decision.yaml`** (editable without
  retraining).

Runtime venv (`/home/jith/.hermes/hermes-agent/venv`) has **pydantic
2.13.4** and **catboost 1.2.10**, no pytest. Existing test convention is a
plain-assert script run directly (`python3 core/test_smoke.py`) — Phase 1
keeps this convention rather than introducing a pytest dependency.

## 3. V3 directory structure

```
app/            live orchestrator entrypoint — thin process that wires
                market → decision → trading → journal. Replaces the
                monolithic root ai_signal_engine.py/shadow_engine.py loops.
market/         xm_ticker.py (feed code, relocated — NOT wired as live
                infra this phase) + MarketState contract usage
features/       core/features.py, fracdiff.py, hurst.py, kalman.py,
                volatility.py — feature computation, causal only
decision/       core/signal.py → decision/signal.py (the scorer — the ONLY
                model-inference code app/ is allowed to import) +
                decision/router.py (model router, see §6)
trading/        VirtualTrade contract + trade-state process supervision
                (watchdog.py, disk-monitor.py, space_guard.py)
journal/        Journal contract (schema only — actual .jsonl event files
                keep living in the external cron/output/ dir this phase;
                only the schema/contract is versioned here)
learning/       core/train.py, retrain_daily.py, seed_refresh.py,
                calibration.py, evaluate.py, cv.py, labeling.py, data.py —
                research/batch-callable, never imported by app/
research/       unchanged, untouched content
contracts/      canonical pydantic contract layer (see §5) — single
                authoritative location for every cross-domain schema;
                domain folders may define domain-internal helper models but
                MUST import shared contracts from here, never redefine them
config/         config/*.yaml + pydantic Settings loader — single source of
                truth (see §7)
models/         registry/, active/, candidates/, archive/ (see §6)
data/           gold_seed.csv (the one live-referenced dataset)
tests/          consolidated, mirrors the module tree, plain-assert style
scripts/        one-off utilities: download_*.py, fetch_dukascopy_m1.py,
                merge/backfill tools
services/       shell wrappers + systemd unit definitions, self-heal/health
                scripts (self-heal.sh, system-health.sh, camofox-watchdog.sh,
                model_staleness_watch.sh)
docs/           new ARCHITECTURE.md (V3, with Mermaid diagram) + this spec
.archive/       existing archive dirs + newly archived old v7 model
                artifacts, superseded planning docs, dead root data files
```

`core/` retires as a name — its contents move into `features/`,
`learning/`, and `decision/` by responsibility, which is the concrete
enforcement of the production/research boundary (Step 4): `app/` may import
`decision/`, `features/`, `market/`, `trading/`, `journal/`, `contracts/`,
`config/`. `app/` must never import anything from `learning/` or `research/`.

## 4. Production vs research boundary

**Live-importable** (reachable from `app/`): `market/`, `features/`,
`decision/`, `trading/`, `journal/`, `contracts/`, `config/`.
**Research-only** (never imported by `app/`): `learning/`, `research/`,
`scripts/`. These may import live-importable modules (e.g. `learning/train.py`
imports `features/` to build training data) — the boundary is one-directional.
This is enforced by convention + the test foundation (§9 includes an import-
graph check), not by a runtime firewall — Phase 1 doesn't build a plugin
sandbox, just makes the violation visible and cheap to catch.

## 5. Contracts (`contracts/`)

Every contract is a **pydantic v2** `BaseModel` (validation, dtype
enforcement, range constraints on `Field(...)` where meaningful). One file
per contract family, all re-exported from `contracts/__init__.py`:

- `contracts/market_state.py` — `MarketState`: timestamp, bid, ask, spread,
  mid, tick info placeholder, M1 state placeholder, multi-horizon state
  placeholder, volatility placeholder, activity placeholder, session,
  current regime, feature state reference. Phase 1 defines the shape;
  most fields are `Optional` until Phase 2 populates them from a real feed.
- `contracts/feature_schema.py` — `FeatureDescriptor` (name, family, source,
  frequency, causal: bool, required_data, update_mechanism, version, dtype,
  valid_range: Optional[tuple], missing_value_policy) + `FeatureSetSchema`
  (ordered list of descriptors + a schema version string). The current
  28-column primary/meta feature lists get expressed as one `FeatureSetSchema`
  instance, generated from `models/feature_cols.json` — this becomes the
  reproducible source `features/build_features` is checked against.
- `contracts/virtual_trade.py` — `VirtualTrade`: trade_id, signal_timestamp,
  direction, entry, sl, tp, expected_value (Optional — not computed until a
  later phase), confidence, model_versions (dict[role, model_id]),
  feature_schema_version, probability_state, mae_forecast/mfe_forecast
  (Optional), regime, execution_metadata, management_state, resolution,
  outcome, journal_ref.
- `contracts/journal.py` — one `BaseModel` per lifecycle event type from the
  original spec (`SignalEvent`, `MarketStateEvent`, `ManagementEvent`,
  `ExecutionEvent`, `ResolutionEvent`, `LearningEvent`), each carrying a
  `schema_version` field and a `trade_id` linking key.
- `contracts/model_registry.py` — `ModelRegistryEntry`: model_id, family
  (`Literal["direction","opportunity_meta","regime","mae_quantile",
  "mfe_quantile","barrier_probability"]`), algorithm, artifact_path,
  feature_schema_version, feature_cols (locked), target_definition,
  training_config (locked — tb_cfg, horizon_vol_scale, max_holding, cusum_k,
  embargo_bars, model hyperparameters), training_period, validation_period,
  created_at, status (`Literal["candidate","active","archived","rejected"]`),
  is_champion: bool, metrics (free-form dict), lineage (data snapshot
  reference, code commit sha, config snapshot reference — Step 10).

Duplicate-definition rule: if two domains need the same shape (e.g. both
`decision/router.py` and `learning/train.py` need to know a model's
`feature_cols`), they both import `contracts.model_registry.ModelRegistryEntry`
— nobody re-declares a parallel model.

## 6. Model registry (`models/`)

```
models/
├── registry/     one JSON file per model_id, validated against
│                 ModelRegistryEntry at load time. This is the only place
│                 that says what a model IS.
├── active/       artifact files (.cbm etc) for entries with status="active"
│                 — today: primary.cbm + meta.cbm (direction +
│                 opportunity_meta roles), promoted from the current root
│                 files, registered with is_champion=True.
├── candidates/   artifact files for status="candidate" entries — empty in
│                 Phase 1 (no challenger workflow yet), directory exists so
│                 later phases have a place to land.
└── archive/      everything superseded: models/v2/*, models/archive/*
                  (existing timestamped snapshots), old v7 LightGBM .txt
                  files, calibration_by_drr*.json — registered with
                  status="archived", is_champion=False, so they stay
                  queryable via registry even though not live-loadable.
```

The live engine never globs a directory or guesses a filename: it asks
the registry (via `decision/router.py`) for "the active model for role X"
and gets back an `artifact_path` it loads by exact path. This is what makes
champion/challenger swaps later a metadata change, not a file shuffle.

## 7. Model router (`decision/router.py`)

Target shape (from the approved diagram):

```
Market State
     ↓
Model Router
     ├── Direction        → best validated model
     ├── Opportunity/Meta → best validated model
     ├── Regime           → best validated model
     ├── MAE (quantile)   → best validated model
     ├── MFE (quantile)   → best validated model
     └── Barrier          → best validated model
```

The router is **static and config-driven, never dynamic**: it reads
`config/models.yaml`, a hand-maintained (later: champion/challenger-process-
maintained) mapping of `role → model_id`, resolves each `model_id` against
`models/registry/`, and loads the artifact. It does not compare live
performance or pick a model based on today's data — model *selection* is
exclusively a research/champion-challenger decision (future phase); the
router's only job at inference time is "load what research approved."

Phase 1 scope: the `ModelRouter` class and the `config/models.yaml` schema
exist and work for the two roles that currently have real models
(`direction`, `opportunity_meta`) — `decision/signal.py`'s
`SignalEngine` is refactored to obtain its two CatBoost models through the
router instead of hardcoding `models/primary.cbm` / `models/meta.cbm`. The
other four roles (`regime`, `mae_quantile`, `mfe_quantile`,
`barrier_probability`) are defined in the `family` Literal and in
`config/models.yaml` as explicitly absent (e.g. `model_id: null`) — the
router returns `None` for an unconfigured role rather than fabricating a
model. This is the concrete foundation the target architecture's "Specialist
Model Layer" plugs into later without a rebuild — new families arrive by
adding a registry entry + a `config/models.yaml` line, not by changing
`decision/router.py`'s shape.

## 8. Configuration (`config/`)

Single source of truth, pydantic v2 `BaseSettings`-style models, one file
per category, loaded once at process start:

- `config/market.yaml` — symbol, feed connection parameters (Phase 1: the
  external cron/output paths, explicitly commented as temporary legacy)
- `config/features.yaml` — feature schema version pointer
- `config/models.yaml` — the router's role→model_id map (§7)
- `config/decision.yaml` — `meta_prob_threshold` and other decision-policy
  knobs (moved out of `models/feature_cols.json`)
- `config/risk.yaml` — placeholder for future risk parameters (empty/defaults
  in Phase 1, not populated with invented numbers)
- `config/telegram.yaml` — bot/env references (no secrets committed — path
  to the existing `.env`, same as today)
- `config/journal.yaml` — journal event schema version, output paths
- `config/learning.yaml` — retrain tolerances (e.g. today's
  `ACC_REGRESSION_TOLERANCE = 0.01` from `retrain_daily.py`)
- `config/runtime.yaml` — BASE/OUTDIR-equivalent paths (today hardcoded as
  `BASE = "/home/jith/..."` string literals in `ai_signal_engine.py`)

No Python file gets a hardcoded threshold, path, model ID, or feature list
after Phase 1 for the modules touched (`decision/`, `market/` relocated
code, `learning/retrain_daily.py`). A `config/loader.py` provides one
`load_config()` entry point returning a composed settings object.

## 9. Tests (`tests/`)

Mirrors the module tree, plain-assert scripts (matching existing
`core/test_smoke.py` convention — no new pytest dependency):
- `tests/test_config.py` — every `config/*.yaml` loads and validates
- `tests/test_contracts.py` — each contract accepts a valid example and
  rejects an invalid one (bad dtype, out-of-range value)
- `tests/test_model_registry.py` — every `models/registry/*.json` parses as
  `ModelRegistryEntry`; `active/` artifact paths referenced by
  status="active" entries exist on disk
- `tests/test_router.py` — router resolves `direction` and `opportunity_meta`
  to loadable artifacts; unconfigured roles return `None` without raising
  `tests/test_boundary.py` — import-graph check: nothing under `app/`
  transitively imports `learning/` or `research/`
- `tests/test_labeling.py`, `test_cv.py`, etc. — relocated from
  `core/test_smoke.py`, split by module, same assertions

## 10. What gets archived vs deleted vs kept live

Note: `.gitignore` already excludes `*.csv`, `*.npy`, `*.log`, `*.jsonl`,
`models/*.txt` — most of what's listed below as "archived" is filesystem-
only (plain `mv`, no git history to preserve), not a git operation. Only the
doc files (`.md`) and the macro-removal diff are tracked and need commits.

**Archived (moved to `.archive/`, fully recoverable, nothing deleted):**
old v7 model artifacts (`models/*.txt`, `calibration_by_drr*.json`,
`models/regime_*`, `models/spec_*`, `models/oof_spec_*`, `models/drr_spec_*`,
`models/dirmask_spec_*`, etc.), `models/v2/`, existing `models/archive/`
snapshots (re-homed under the new `models/archive/`), stale root docs
(`ARCHITECTURE.md`, `COMPLETE_PLAN.md`, `JANE_STREET_PLAN.md`,
`MILLIONAIRE_PLAN.md`, `RESEARCH_PROOF.md`, `AI_SYSTEM_SUMMARY.md`,
`FEATURE_ANALYSIS.md`, `audit_*.md`), dead root data files (the CSV/NPY list
in §2), stale logs (`*.log`), `.matrix_schema*.json`, `features.json`,
`features.py.fixed`, `train_data_meta.json`, `quant_features_meta.json`.

**Left untouched, not in scope:** `rotator_keys.json` — gitignored
credential file, no reference from any trading script, unrelated to this
repo's pipeline; not moved, not archived.

**Deleted (genuinely transient/regenerable, not evidence):**
`catboost_info/`, `__pycache__/` everywhere (added to `.gitignore`).

**Committed as part of Phase 1 cleanup (already-validated dead code):**
the in-progress uncommitted macro removal (`event_calendar.py`,
`fetch_macro_context.py`, `macro/` deleted; `core/features.py`,
`core/train.py` modified) — grep-verified no remaining references.

**Kept live / relocated, not archived:** `gold_seed.csv` → `data/`,
`research/` untouched, everything under §3's new tree.

Nothing is archived or deleted without first grep-checking for references
from `core/`/`research/` (already done for the data-file list in §2) — the
per-category sweep in the implementation plan repeats this check for the
docs/models/logs categories before moving them.

## 11. Service handling

Both `ai-engine.service` and `gold-shadow.service` are **stopped** before any
file moves begin (explicit user instruction: implementation phase, no live
risk needed, architect properly rather than avoid-restart). They are **not
restarted** at the end of Phase 1 — Phase 1 produces a correct, importable
new layout and updates the systemd unit files under `services/` to point at
the new paths, but actually cutting live traffic over to the new `app/`
entrypoint is a deliberate later-phase decision, not an automatic side
effect of this phase.

## 12. Architecture document

`docs/ARCHITECTURE.md` (new, replaces the stale root one — old one archived):
purpose, system boundaries, data flow, component responsibilities, model
responsibilities, research/live separation, versioning, journal lineage,
future learning architecture, and a Mermaid diagram covering the full target
architecture from the original spec (Market State → Model Router →
Decision Engine → Trade Construction → Telegram → Virtual Trade → Journal →
Learning → Champion/Challenger), with Phase 1's actual scope (what's real vs
placeholder) marked explicitly so the doc doesn't overclaim.

## 13. Explicitly out of scope for Phase 1

New feature library, new ML models, dynamic SL/TP, EV gate, barrier engine,
EOD learning, champion/challenger *execution* (the registry/router
*foundation* is in scope, running an actual challenger process is not),
advanced trade management, real-time MT5 pipeline wiring (Phase 2),
restarting the live services.

## 14. Completion criteria

Repository inventory documented (§2, this file) · legacy/current/V3
boundaries clear (§3, §10) · obsolete files archived not deleted (§10) · V3
directory structure exists (§3) · production/research separation exists and
is testable (§4, §9) · feature schema, market state, virtual trade, journal,
model registry contracts exist as pydantic models (§5) · model registry
foundation exists with active/candidates/archive split (§6) · router
foundation exists for the two real roles (§7) · configuration foundation
exists, no hardcoded thresholds/paths/model IDs in touched modules (§8) ·
service boundaries documented (§3, §11) · test foundation exists and passes
(§9) · `docs/ARCHITECTURE.md` with Mermaid exists (§12) · project starts
cleanly (`python3 -c "import app"` and each new package imports without
error) · existing functionality not unintentionally broken (verified by
re-running relocated `tests/` against the new paths — services stay stopped
per §11, so "not broken" means "importable and logically equivalent," not
"still running").
