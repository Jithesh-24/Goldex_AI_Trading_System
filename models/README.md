# models/

Phase 4 specialist quantitative model layer: seven independent roles, each answering a distinct risk/opportunity question on the XAUUSD market. Full design: `docs/superpowers/specs/2026-08-22-golex-v3-phase4-specialist-models-design.md`.

## The seven specialist roles

- **Direction** — Will price move up or down over the next N bars (15/45/90)? Classifier (binary), outputs probability of up-move.
- **Opportunity/Meta** — Is the current bar a genuine trade entry condition, or just noise? Trade-filter classifier, validates whether Direction's signal is actionable.
- **Regime** — Which market regime are we in (mean-reversion vs momentum vs low-vol, etc.)? Unsupervised state classifier (Gaussian HMM), discretizes continuous market behavior into causal regime buckets.
- **MAE Quantile** — How much is price likely to move *against* the intended entry direction before the stop-loss is hit? Quantile regressor (targets 0.5/0.75/0.9 quantiles), estimates maximum adverse excursion distribution.
- **MFE Quantile** — How much profit potential is available if price moves *with* the trade? Quantile regressor (targets 0.5/0.75/0.9 quantiles), estimates maximum favorable excursion distribution.
- **Barrier Probability** — Given market state, what is the probability price hits a given barrier level within N bars? Calibrated probability estimator, distinct from Direction's binary classifier — this role specializes in probability calibration over the full [0,1] range.
- **Execution/Decay** — How does entry-signal quality degrade if execution is delayed? Time-decay model, estimates post-signal price drift under latency (30s/60s/120s).

Each role trains on the same **shared Phase 3 V3-feature foundation** (28 baseline columns + 17 useful-status candidates selected by importance, Task 3) but narrows to its own feature schema via out-of-sample importance ranking — no role reuses another's feature set unnarrowed.

## Model registry: contract and lifecycle

`contracts/model_registry.py` defines `ModelRegistryEntry`, persisted as JSON in `models/registry/<model_id>.json`. Two independent status fields control the deployment and validation lifecycle:

- **`status`** — Five-state validation workflow: `candidate` (research prototype), `validated` (OOS metrics meet spec), `active` (approved for production), `archived` (superseded), `rejected` (spec-compliant finding that the role is not viable). A validated model may be archived or rejected; neither is a failure — both are legitimate research outcomes.
- **`is_champion`** — Separate boolean flag, independent of `status`. Only `true` for the single approved production model per role (set by the EOD learning promotion process, Phase 5). Allows candidate/challenger pairs to coexist without ambiguity.

Each entry locks down:
- `model_id` — unique identifier (e.g. `direction_v3_h15_catboost_2026-08-22`)
- `family` — the role (from `ModelFamily` literal: one of the seven above plus `execution_decay`)
- `algorithm` — implementation (CatBoost, scikit-learn, custom)
- `artifact_path` — relative path to serialized model file
- `feature_schema_version` — reference to a `FeatureSetSchema` JSON in `features/registry/schemas/`
- `feature_cols` — exact column order, locked at registry creation time
- `target_definition` — what the model predicts (e.g. "direction binary label, CUSUM events, h=15")
- `training_config` — hyperparameters and config dict, locked
- `training_period` / `validation_period` — date ranges for cross-validation folds
- `created_at` — timestamp
- `metrics` — OOS evaluation metrics (accuracy, log_loss, win_rate, calibration gap, etc., role-dependent)
- `lineage` — optional reference to training data snapshot, code commit, config

## Router: static, config-driven, not a champion/challenger engine

`decision/router.py`'s `ModelRouter` is deliberately **static and simple**. It does not compare live performance, does not pick models based on today's data, and does not implement any champion/challenger logic. Its single responsibility: given a `role` (string), look up the corresponding `model_id` in `config/models.yaml`, load the registry entry from `models/registry/`, and return it for inference.

Why static? Model *selection* (choosing which `model_id` to use) is exclusively a research-phase decision — the EOD learning loop (Phase 5) compares OOS metrics, detects drift, and promotes/demotes candidates to `active` status. The router trusts research has already approved; at inference time, the router's only job is "load what research said to use." This separation ensures:

1. **Auditability** — every model ID in production is a reviewed research entry with locked metrics.
2. **Safety** — no live decision-making about model selection; that's offline.
3. **Simplicity** — the router is a pure lookup, testable in isolation from performance comparison logic.

The `role_map` in `config/models.yaml` is the single place to edit which model goes live per role — e.g. `direction: direction_v3_h15_catboost_2026-08-22`. Swapping a model means editing YAML and restarting the app; no code changes required.

## Feature schema persistence

`features/registry/schemas.py` provides `save_schema()`/`load_schema()` to persist `FeatureSetSchema` objects as JSON files in `features/registry/schemas/`. Each file is named `{schema_id}__{schema_version}.json` — e.g. `direction_v3_h15__2026-08-22.json` for the Direction model's h=15 horizon feature set.

Why persist schemas separately from the registry entries? Because the feature list for each specialist role can change without the registry entry being rewritten — a schema captures "which exact features, in which order, does this model expect at inference time," independent of the model's other metadata. A `ModelRegistryEntry` references its schema via `feature_schema_version` and loads it only when needed. This decoupling lets:

- Audit what features a model actually used (the schema is an immutable artifact).
- Rebuild training datasets deterministically (load the schema, fetch those exact features in that order).
- Catch feature-order mismatches at inference time (load schema, validate inference vector matches).

Schemas are created by `features/registry/__init__.py`'s `build_schema()` function, which takes a list of feature IDs and returns a timestamped `FeatureSetSchema`; persisting the result is a one-liner.

## Registry layout

```
models/
├── registry/
│   ├── direction_v3_h15_catboost_2026-08-22.json
│   ├── direction_v3_h45_catboost_2026-08-22.json
│   ├── direction_v3_h90_catboost_2026-08-22.json
│   ├── opportunity_v3_h15_catboost_2026-08-22.json
│   ├── ...
│   └── regime_v3_hmm_2026-08-22.json
├── active/
│   ├── direction_v3_h15.pkl  (may be symlinked to a candidates/ model during evaluation)
│   ├── opportunity_v3_h15.pkl
│   ├── regime_v3.pkl
│   └── ...
├── candidates/
│   ├── direction_v3_h15_cb.pkl  (challenger pairs under evaluation)
│   ├── direction_v3_h15_lgb.pkl  (old archive, may be sparse)
│   └── ...
└── archive/
    └── ...pre-Phase-4 models...
```

Each registry entry is `contracts.model_registry.ModelRegistryEntry` (pydantic-validated).

## Full design

`docs/superpowers/specs/2026-08-22-golex-v3-phase4-specialist-models-design.md` contains:
- Architecture diagram: MARKET → MARKETSTATE → FEATURES → SPECIALISTS → CALIBRATED PROBABILITIES → PHASE 5.
- Each role's target definition and why it's separate from the others.
- Feature-schema construction and per-specialist importance ranking.
- OOS evaluation contracts for each role (what metrics matter and why).
- Integration points: how the router loads models, how `decision/signal.py` orchestrates the specialist outputs, how `app/engine.py` wires everything.
- Leakage audit protocol and explicit confirmation that train/test [t0,t1] never overlap and PlattCalibrator remains stateless.
