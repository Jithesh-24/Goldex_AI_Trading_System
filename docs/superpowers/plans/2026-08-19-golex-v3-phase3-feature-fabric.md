# GOLEX V3 Phase 3: Quantitative Feature Fabric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the registered, versioned, live-compatible quantitative
feature fabric on top of Phase 2's `MarketState`, reusing (not
re-deriving) the existing production 28-feature set and the existing
92-candidate research library, adding a formal registry, a trigger-driven
live incremental engine, a matching batch replay engine, real causality
tests, and one genuinely new live-only microstructure family.

**Architecture:** `features/<family>.py` modules hold pure, causal
compute functions (moved from `research/features_v3.py`, math
unchanged). `features/replay_engine.py` (batch, historical CSV) and
`features/live_engine.py` (bounded, trigger-driven, MarketState-fed) both
call the *same* family functions — replay on a full DataFrame, live on a
bounded window pulled from `market/state_engine.py`'s ring buffers — so
live/replay equivalence is close to true by construction, verified by
explicit tests. `features/registry/` holds one `FeatureDescriptor` JSON
per feature with real evidence citations from `research/output/`.

**Tech Stack:** Python 3.13 (native venv
`/home/jith/.hermes/hermes-agent/venv/bin/python3`, has pydantic 2.13.4,
numpy, pandas, numba, catboost). No pytest — plain-assert scripts run
directly (`python3 tests/test_x.py`), matching every existing test file
in this repo.

**Spec:** `docs/superpowers/specs/2026-08-19-golex-v3-phase3-feature-fabric-design.md`

## Global Constraints

- Production signal path (`app/engine.py`'s CatBoost call via
  `features.features.build_features`, `app/shadow.py`) is NEVER modified.
- `features/features.py` (the 28-col production baseline) is NEVER
  edited.
- No new models, no dynamic SL/TP, no EV gate, no virtual trade
  management, no Telegram changes, no EOD learning, no
  champion/challenger. `decision/`, `learning/`, `models/` files
  untouched except where explicitly noted.
- Every moved/new feature function must be causal (value at row i depends
  only on data at or before i) — this is checked by a real test (Task
  17), not assumed.
- `research/output/*.json,csv` (existing OOF evidence) is READ, never
  recomputed or duplicated.
- No hardcoded thresholds/paths outside `config/` — new paths go through
  `config/features.yaml` + `config/schema.py`'s `FeaturesConfig`.
- Every registered feature gets a mandatory `status_reason` string citing
  its real evidence (or explicitly stating "no evidence yet, Phase 4
  evaluates").
- Live engine loads bounded state only — never the 6.7-year historical
  CSV into the live process (matches Phase 2's established principle).
- Run `python3 tests/test_boundary.py` after every task that touches
  `features/` or `market/` — must stay green throughout.

---

## File Structure

```
features/
  features.py                 EXISTING, untouched (production 28-col)
  volatility.py, hurst.py,     EXISTING, untouched (baseline support)
  fracdiff.py, labeling.py
  kalman.py                    EXISTING + NEW StatefulKalman class appended (Task 19)
  _shared.py                   NEW -- SharedInputs dataclass + builder (Task 5)
  returns_dynamics.py          NEW -- family A (Task 5)
  volatility_dynamics.py       NEW -- family B (Task 6)
  jump_detection.py            NEW -- family C (Task 7)
  distribution_info.py         NEW -- family D (Task 8)
  market_geometry.py           NEW -- family E (Task 9)
  persistence.py                NEW -- family F (Task 10)
  temporal.py                   NEW -- family G (Task 11)
  microstructure_history.py      NEW -- family H (Task 12)
  regime_state.py                 NEW -- family I (Task 13)
  first_passage.py                 NEW -- family J (Task 14)
  microstructure_live.py            NEW -- new live-only family (Task 21)
  daily_buffer.py                    NEW -- DailyBuffer ring buffer (Task 20)
  replay_engine.py                    NEW -- batch orchestration (Task 16)
  live_engine.py                       NEW -- trigger-driven live orchestration (Task 22)
  registry/
    __init__.py                        NEW -- loader + build_schema() (Task 3)
    diagnostics.py                      NEW -- redundancy/stability tooling (Task 26)
    baseline_v1/*.json                  NEW -- 28 descriptor JSONs (Task 15)
    returns_dynamics/*.json             NEW (Task 5)
    volatility_dynamics/*.json          NEW (Task 6)
    jump_detection/*.json               NEW (Task 7)
    distribution_info/*.json            NEW (Task 8)
    market_geometry/*.json              NEW (Task 9)
    persistence/*.json                  NEW (Task 10)
    temporal/*.json                     NEW (Task 11)
    microstructure_history/*.json       NEW (Task 12)
    regime_state/*.json                 NEW (Task 13)
    first_passage/*.json                NEW (Task 14)
    microstructure_live/*.json          NEW (Task 21)
  README.md                            NEW -- Task 29

contracts/
  feature_schema.py            MODIFIED -- extended (Task 2)

config/
  schema.py                    MODIFIED -- FeaturesConfig extended (Task 4)
  features.yaml                MODIFIED -- new fields (Task 4)

market/
  state_engine.py               MODIFIED -- completed_m1_window() accessor (Task 18)

research/
  features_v3.py                 MODIFIED -- becomes thin deprecated re-export shim (Task 16)
  build_v3_dataset.py, v3_pipeline_checks.py, v3_family_ablation.py,
  v3_feature_selection.py         MODIFIED -- import from features.* instead of research.features_v3 (Task 16)
  historical_coverage.py          NEW -- Task 1

tests/
  test_historical_coverage.py     NEW (Task 1)
  test_feature_schema.py          NEW (Task 2)
  test_feature_registry.py        NEW (Task 3)
  test_config.py                  MODIFIED (Task 4)
  test_returns_dynamics.py ... test_first_passage.py   NEW, one per family (Tasks 5-14)
  test_baseline_registry.py       NEW (Task 15)
  test_replay_engine.py           NEW (Task 16)
  test_causality.py               NEW (Task 17)
  test_state_engine.py            MODIFIED -- completed_m1_window test added (Task 18)
  test_kalman_incremental.py      NEW (Task 19)
  test_daily_buffer.py            NEW (Task 20)
  test_microstructure_live.py     NEW (Task 21)
  test_live_engine.py             NEW (Task 22)
  test_live_replay_equivalence.py NEW (Task 23)
  test_feature_warmup_missing.py  NEW (Task 24)
  test_feature_numerical_safety.py NEW (Task 25)
  test_feature_diagnostics.py     NEW (Task 26)
  test_boundary.py                MODIFIED (Task 27)
  test_feature_performance.py     NEW (Task 28)

docs/ARCHITECTURE.md              MODIFIED -- Phase 3 section (Task 29)
```

---

## Task 1: Historical coverage measurement

**Files:**
- Create: `research/historical_coverage.py`
- Test: `tests/test_historical_coverage.py`

**Interfaces:**
- Produces: `research/historical_coverage.py::measure_coverage(csv_path: str) -> dict` returning
  `{"real_volume_nonzero_frac": float, "tick_volume_nonzero_frac": float,
  "tick_volume_degrades_after": str (ISO date, first date where the trailing
  30-day nonzero fraction drops below 0.05), "spread_constant_frac": float,
  "spread_unique_values": list[float]}`. Later tasks (registry entries)
  cite this output by re-running it, not by re-deriving the numbers by
  hand.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_historical_coverage.py
"""python3 tests/test_historical_coverage.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.historical_coverage import measure_coverage

def test_measure_coverage_real_data():
    result = measure_coverage("data/gold_seed_merged_full6yr.csv")
    assert 0.0 <= result["real_volume_nonzero_frac"] < 0.5, result["real_volume_nonzero_frac"]
    assert result["tick_volume_nonzero_frac"] < 1.0
    assert result["tick_volume_degrades_after"] is not None
    assert result["spread_constant_frac"] > 0.9
    assert len(result["spread_unique_values"]) < 20

if __name__ == "__main__":
    test_measure_coverage_real_data()
    print("tests/test_historical_coverage.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_historical_coverage.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.historical_coverage'`

- [ ] **Step 3: Write the implementation**

```python
# research/historical_coverage.py
"""Real measurement of per-column historical data quality -- feeds the
registry's historical_coverage/status metadata (Phase 3 spec section 2).
Never hand-wave these numbers; always re-run this against the real CSV."""
import pandas as pd


def measure_coverage(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, usecols=["time", "tick_volume", "spread", "real_volume"],
                      parse_dates=["time"])
    n = len(df)
    real_volume_nonzero_frac = float((df["real_volume"] != 0).mean())
    tick_volume_nonzero_frac = float((df["tick_volume"] != 0).mean())

    daily = df.set_index("time")["tick_volume"].resample("1D").apply(lambda s: (s != 0).mean())
    trailing30 = daily.rolling(30, min_periods=1).mean()
    degraded = trailing30[trailing30 < 0.05]
    tick_volume_degrades_after = str(degraded.index[0].date()) if len(degraded) else None

    spread_counts = df["spread"].value_counts(normalize=True)
    spread_constant_frac = float(spread_counts.iloc[0])
    spread_unique_values = sorted(df["spread"].unique().tolist())

    return {
        "n_rows": n,
        "real_volume_nonzero_frac": real_volume_nonzero_frac,
        "tick_volume_nonzero_frac": tick_volume_nonzero_frac,
        "tick_volume_degrades_after": tick_volume_degrades_after,
        "spread_constant_frac": spread_constant_frac,
        "spread_unique_values": spread_unique_values,
    }


if __name__ == "__main__":
    import json
    result = measure_coverage("data/gold_seed_merged_full6yr.csv")
    print(json.dumps(result, indent=2, default=str))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_historical_coverage.py`
Expected: PASS. Note the printed `tick_volume_degrades_after` date and
`spread_unique_values` — used verbatim in Task 12's registry entries'
`status_reason` fields.

- [ ] **Step 5: Commit**

```bash
git add research/historical_coverage.py tests/test_historical_coverage.py
git commit -m "Add real historical-coverage measurement for real_volume/tick_volume/spread"
```

---

## Task 2: Extend `contracts/feature_schema.py`

**Files:**
- Modify: `contracts/feature_schema.py` (currently 25 lines, Phase 1 stub)
- Test: `tests/test_feature_schema.py`

**Interfaces:**
- Produces: `FeatureStatus`, `HistoricalCoverage`, `ComputationalCost`,
  `UpdateTrigger` enums; `FeatureDescriptor`, `FeatureSetSchema` pydantic
  models — imported by every later task that writes a registry entry or
  builds the registry loader (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_schema.py
"""python3 tests/test_feature_schema.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from contracts.feature_schema import (
    FeatureDescriptor, FeatureSetSchema, FeatureStatus, HistoricalCoverage,
    ComputationalCost, UpdateTrigger,
)


def test_feature_descriptor_valid():
    d = FeatureDescriptor(
        feature_id="ret_5", family="baseline_v1",
        mathematical_definition="log(c_t) - log(c_{t-5})",
        source_module="features.features.build_tier1_features",
        required_state=["close"], update_trigger=UpdateTrigger.M1_CLOSE,
        window=5, causal=True, live_compatible=True,
        computational_cost=ComputationalCost.LOW,
        missing_value_policy="NaN during warmup, never zero-filled",
        warmup_bars=5, historical_coverage=HistoricalCoverage.FULL_HISTORY,
        status=FeatureStatus.REQUIRED,
        status_reason="core production feature, deployed since 2026-08-18",
        version="v1",
    )
    assert d.feature_id == "ret_5"
    assert d.causal is True


def test_feature_set_schema_round_trip():
    s = FeatureSetSchema(schema_id="baseline_v1", schema_version="v1",
                          feature_ids=["ret_5", "ret_15"],
                          created_at=datetime.now(timezone.utc))
    data = s.model_dump_json()
    s2 = FeatureSetSchema.model_validate_json(data)
    assert s2.feature_ids == ["ret_5", "ret_15"]


if __name__ == "__main__":
    test_feature_descriptor_valid()
    test_feature_set_schema_round_trip()
    print("tests/test_feature_schema.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_schema.py`
Expected: FAIL with `ImportError: cannot import name 'FeatureStatus'`

- [ ] **Step 3: Write the implementation**

```python
# contracts/feature_schema.py
"""Canonical feature schema contract -- prevents the feature-mismatch
problems between training and inference that motivated this rebuild.
Extended in Phase 3 with full per-feature metadata (spec section 5) and
model-routing schema slicing (spec section 9): the registry preserves the
entire quantitative universe, and FeatureSetSchema.feature_ids lets each
future specialist model construct its own slice -- no feature is
pre-selected here."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FeatureStatus(str, Enum):
    REQUIRED = "REQUIRED"
    USEFUL = "USEFUL"
    OPTIONAL = "OPTIONAL"
    UNSUPPORTED_BY_DATA = "UNSUPPORTED_BY_DATA"
    REDUNDANT = "REDUNDANT"
    REJECTED = "REJECTED"


class HistoricalCoverage(str, Enum):
    FULL_HISTORY = "FULL_HISTORY"
    PARTIAL_HISTORY = "PARTIAL_HISTORY"
    LIVE_ONLY = "LIVE_ONLY"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    UNSUPPORTED = "UNSUPPORTED"


class ComputationalCost(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    RESEARCH_ONLY = "RESEARCH_ONLY"


class UpdateTrigger(str, Enum):
    TICK = "TICK"
    M1_CLOSE = "M1_CLOSE"
    DAILY = "DAILY"
    EVENT = "EVENT"


class FeatureDescriptor(BaseModel):
    feature_id: str
    family: str
    mathematical_definition: str
    source_module: str
    required_state: list[str] = Field(default_factory=list)
    update_trigger: UpdateTrigger
    window: Optional[int] = None
    causal: bool
    live_compatible: bool
    computational_cost: ComputationalCost
    numerical_stability_notes: Optional[str] = None
    missing_value_policy: str
    warmup_bars: int
    dependencies: list[str] = Field(default_factory=list)
    units: Optional[str] = None
    normalization: Optional[str] = None
    expected_range: Optional[tuple[float, float]] = None
    historical_coverage: HistoricalCoverage
    status: FeatureStatus
    status_reason: str
    evidence_ref: Optional[str] = None
    version: str


class FeatureSetSchema(BaseModel):
    schema_id: str
    schema_version: str
    feature_ids: list[str]
    created_at: datetime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_schema.py`
Expected: PASS.

- [ ] **Step 5: Run existing contracts test to check no regression**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_contracts.py`
Expected: PASS (this file doesn't touch `FeatureDescriptor`, only confirms the module still imports cleanly).

- [ ] **Step 6: Commit**

```bash
git add contracts/feature_schema.py tests/test_feature_schema.py
git commit -m "Extend contracts/feature_schema.py with full Phase 3 metadata + status enums"
```

---

## Task 3: `features/registry/` loader package

**Files:**
- Create: `features/registry/__init__.py`
- Test: `tests/test_feature_registry.py`

**Interfaces:**
- Consumes: `contracts.feature_schema.FeatureDescriptor`, `FeatureSetSchema` (Task 2).
- Produces: `features.registry.load_descriptor(path: str) -> FeatureDescriptor`,
  `features.registry.load_family(family: str) -> list[FeatureDescriptor]`,
  `features.registry.load_all() -> list[FeatureDescriptor]`,
  `features.registry.build_schema(schema_id: str, schema_version: str, feature_ids: list[str]) -> FeatureSetSchema`.
  Every family task (5-14, 21) writes JSON files this loader reads;
  `replay_engine.py`/`live_engine.py` (Tasks 16, 22) call `load_all()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_registry.py
"""python3 tests/test_feature_registry.py"""
import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.registry import load_descriptor, load_family, load_all, build_schema, REGISTRY_DIR


def test_load_descriptor_fixture():
    fixture_dir = tempfile.mkdtemp()
    try:
        family_dir = os.path.join(fixture_dir, "test_family")
        os.makedirs(family_dir)
        payload = {
            "feature_id": "fixture_feat", "family": "test_family",
            "mathematical_definition": "x", "source_module": "features.fixture",
            "required_state": ["close"], "update_trigger": "M1_CLOSE",
            "window": 5, "causal": True, "live_compatible": True,
            "computational_cost": "LOW", "missing_value_policy": "NaN",
            "warmup_bars": 5, "historical_coverage": "FULL_HISTORY",
            "status": "REQUIRED", "status_reason": "fixture", "version": "v1",
        }
        path = os.path.join(family_dir, "fixture_feat.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        d = load_descriptor(path)
        assert d.feature_id == "fixture_feat"
        fam = load_family("test_family", registry_dir=fixture_dir)
        assert len(fam) == 1 and fam[0].feature_id == "fixture_feat"
    finally:
        shutil.rmtree(fixture_dir)


def test_load_all_real_registry_empty_before_family_tasks():
    # REGISTRY_DIR exists (created this task) but has no family JSON yet --
    # later tasks populate it. Must not crash on an empty/partial registry.
    all_descriptors = load_all()
    assert isinstance(all_descriptors, list)


def test_build_schema():
    schema = build_schema("test_schema", "v1", ["a", "b", "c"])
    assert schema.feature_ids == ["a", "b", "c"]
    assert schema.schema_id == "test_schema"


if __name__ == "__main__":
    test_load_descriptor_fixture()
    test_load_all_real_registry_empty_before_family_tasks()
    test_build_schema()
    print("tests/test_feature_registry.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_registry.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.registry'`

- [ ] **Step 3: Write the implementation**

```python
# features/registry/__init__.py
"""Feature registry loader -- reads FeatureDescriptor JSON files organized
one-subdirectory-per-family under features/registry/, and builds named
FeatureSetSchema slices for future model-specific schemas (spec section
9: the registry preserves the ENTIRE quantitative universe; nothing here
pre-selects a "final" feature set)."""
import json
import os
from datetime import datetime, timezone

from contracts.feature_schema import FeatureDescriptor, FeatureSetSchema

REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))


def load_descriptor(path: str) -> FeatureDescriptor:
    with open(path) as f:
        return FeatureDescriptor(**json.load(f))


def load_family(family: str, registry_dir: str = REGISTRY_DIR) -> list[FeatureDescriptor]:
    family_dir = os.path.join(registry_dir, family)
    if not os.path.isdir(family_dir):
        return []
    out = []
    for fname in sorted(os.listdir(family_dir)):
        if fname.endswith(".json"):
            out.append(load_descriptor(os.path.join(family_dir, fname)))
    return out


def load_all(registry_dir: str = REGISTRY_DIR) -> list[FeatureDescriptor]:
    out = []
    for entry in sorted(os.listdir(registry_dir)):
        full = os.path.join(registry_dir, entry)
        if os.path.isdir(full) and not entry.startswith("__"):
            out.extend(load_family(entry, registry_dir=registry_dir))
    return out


def build_schema(schema_id: str, schema_version: str, feature_ids: list[str]) -> FeatureSetSchema:
    return FeatureSetSchema(schema_id=schema_id, schema_version=schema_version,
                             feature_ids=feature_ids, created_at=datetime.now(timezone.utc))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_registry.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/registry/__init__.py tests/test_feature_registry.py
git commit -m "Add features/registry/ loader: load_descriptor, load_family, load_all, build_schema"
```

---

## Task 4: `config/features.yaml` + `FeaturesConfig` extension

**Files:**
- Modify: `config/schema.py:28-29` (`FeaturesConfig`)
- Modify: `config/features.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `cfg.features.registry_dir: str`,
  `cfg.features.daily_buffer_bootstrap_csv: str`,
  `cfg.features.daily_buffer_size: int` — consumed by Task 20
  (`DailyBuffer`) and Task 22 (`live_engine.py`).

- [ ] **Step 1: Modify `config/features.yaml`**

```yaml
schema_version: root-28col-2026-08-18
registry_dir: features/registry
daily_buffer_bootstrap_csv: data/gold_seed.csv
daily_buffer_size: 252
```

- [ ] **Step 2: Modify `config/schema.py`**

```python
class FeaturesConfig(BaseModel):
    schema_version: str
    registry_dir: str
    daily_buffer_bootstrap_csv: str
    daily_buffer_size: int
```

- [ ] **Step 3: Add assertions to `tests/test_config.py`**

```python
    assert cfg.features.registry_dir == "features/registry"
    assert cfg.features.daily_buffer_size == 252
```

(Insert inside the existing `test_load_config_valid` function, after the
existing `assert cfg.market.feed_mode ...` line.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_config.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/schema.py config/features.yaml tests/test_config.py
git commit -m "Extend FeaturesConfig with registry_dir/daily_buffer settings for Phase 3"
```

---

## Task 5: Family A (`returns_dynamics.py`) + `features/_shared.py`

**Files:**
- Create: `features/_shared.py`
- Create: `features/returns_dynamics.py`
- Create: `features/registry/returns_dynamics/*.json` (19 files)
- Test: `tests/test_returns_dynamics.py`

**Interfaces:**
- Produces: `features._shared.SharedInputs` (dataclass), `build_shared_inputs(df, base_feat) -> SharedInputs`.
  Consumed by every subsequent family module (Tasks 6-14) and by
  `replay_engine.py`/`live_engine.py` (Tasks 16, 22).
- Produces: `features.returns_dynamics.compute_returns_dynamics(shared: SharedInputs) -> dict[str, np.ndarray]`,
  and the 5 numba kernels: `run_length_signed`, `rolling_autocorr_lag1`,
  `rolling_pacf1_ar2`, `sign_flip_rate`, `directional_entropy` (all
  module-level, importable — `rolling_autocorr_lag1` is reused by Task 10,
  family F).

Source: `research/features_v3.py` lines 1-471 (top-of-file docstring +
kernels + shared-variable setup) and lines 475-499 (family A assembly
block). Move, do not rewrite — every numba kernel's body is copied
verbatim; only names lose their leading underscore (module-private `_x`
becomes public `x` since these are now the public interface of
`returns_dynamics.py`) and call sites in the assembly function update to
match.

- [ ] **Step 1: Write `features/_shared.py`**

```python
# features/_shared.py
"""Common precomputed arrays every family-Batch compute_<family>()
function needs -- built once per call (replay: once per DataFrame; live:
once per bounded window) instead of every family recomputing ret1/sign1/
etc independently. Mirrors research/features_v3.py's former
build_candidate_features() setup block (lines 455-471), unchanged math."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SharedInputs:
    df: pd.DataFrame
    base_feat: pd.DataFrame
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    log_c: np.ndarray
    ret1: np.ndarray
    sign1: np.ndarray
    times: pd.DatetimeIndex
    ewma_vol: np.ndarray
    kalman_resid: np.ndarray
    hurst_120: np.ndarray
    tick_vol: np.ndarray
    spread: np.ndarray


def build_shared_inputs(df: pd.DataFrame, base_feat: pd.DataFrame) -> SharedInputs:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    log_c = np.log(c)
    ret1 = np.diff(log_c, prepend=log_c[0])
    ret1[0] = np.nan
    sign1 = np.nan_to_num(np.sign(ret1), nan=0.0)
    times = pd.to_datetime(df["time"].to_numpy())
    ewma_vol = base_feat["ewma_vol"].to_numpy(dtype=np.float64)
    kalman_resid = base_feat["kalman_residual_z"].to_numpy(dtype=np.float64)
    hurst_120 = base_feat["hurst_120"].to_numpy(dtype=np.float64)
    tick_vol = df["tick_volume"].to_numpy(dtype=np.float64) if "tick_volume" in df.columns else np.full(len(df), np.nan)
    spread = df["spread"].to_numpy(dtype=np.float64) if "spread" in df.columns else np.full(len(df), np.nan)
    return SharedInputs(df=df, base_feat=base_feat, o=o, h=h, l=l, c=c, log_c=log_c,
                         ret1=ret1, sign1=sign1, times=times, ewma_vol=ewma_vol,
                         kalman_resid=kalman_resid, hurst_120=hurst_120,
                         tick_vol=tick_vol, spread=spread)
```

- [ ] **Step 2: Write `features/returns_dynamics.py`**

Copy `research/features_v3.py` lines 35-51 (`_run_length_signed`), 54-65
(`_rolling_autocorr_lag1`), 68-87 (`_rolling_pacf1_ar2`), 90-101
(`_sign_flip_rate`), 104-129 (`_directional_entropy`) verbatim, renaming
each to drop its leading underscore. Then add:

```python
"""Family A -- return dynamics. Moved from research/features_v3.py,
math unchanged (spec section 4). Causal by construction: only
np.roll/pd.Series.rolling/.shift and backward-scanning numba loops."""
import numba
import numpy as np
import pandas as pd

from features._shared import SharedInputs

# ... (the 5 numba kernel functions go here, copied per Step 2 above) ...


def compute_returns_dynamics(shared: SharedInputs) -> dict:
    ret1, sign1 = shared.ret1, shared.sign1
    ret1_s = pd.Series(ret1)
    log_c = shared.log_c
    base_feat = shared.base_feat
    f = {}
    f["ret_240"] = log_c - np.roll(log_c, 240)
    f["ret_240"][:240] = np.nan
    f["sign_ret_240"] = np.sign(f["ret_240"])
    f["ret_accel_5_15"] = base_feat["ret_5"].to_numpy() - base_feat["ret_15"].to_numpy()
    f["ret_decel_15_60"] = base_feat["ret_15"].to_numpy() - base_feat["ret_60"].to_numpy()
    f["run_length_signed"] = run_length_signed(sign1)
    f["return_autocorr_20"] = rolling_autocorr_lag1(ret1, 20)
    f["return_autocorr_60"] = rolling_autocorr_lag1(ret1, 60)
    f["return_pacf1_60"] = rolling_pacf1_ar2(ret1, 60)
    f["sign_flip_rate_20"] = sign_flip_rate(sign1, 20)
    f["rolling_mean_ret_20"] = ret1_s.rolling(20).mean().to_numpy()
    f["rolling_median_ret_20"] = ret1_s.rolling(20).median().to_numpy()
    f["return_dispersion_20"] = ret1_s.rolling(20).std().to_numpy()
    up = np.where(ret1 > 0, ret1, np.nan)
    down = np.where(ret1 < 0, -ret1, np.nan)
    up_mean_60 = pd.Series(up).rolling(60, min_periods=5).mean()
    down_mean_60 = pd.Series(down).rolling(60, min_periods=5).mean()
    f["upside_downside_asymmetry_60"] = (up_mean_60 / down_mean_60).to_numpy()
    f["return_skew_60"] = ret1_s.rolling(60).skew().to_numpy()
    f["return_kurt_60"] = ret1_s.rolling(60).kurt().to_numpy()
    f["return_skew_240"] = ret1_s.rolling(240).skew().to_numpy()
    f["return_percentile_rank_60"] = ret1_s.rolling(60).rank(pct=True).to_numpy()
    ret15 = base_feat["ret_15"].to_numpy()
    f["return_quantile_pos_240"] = pd.Series(ret15).rolling(240).rank(pct=True).to_numpy()
    f["directional_entropy_60"] = directional_entropy(sign1, 60)
    return f
```

- [ ] **Step 3: Write 19 registry JSON files** (one per key in `f`)

`features/registry/returns_dynamics/ret_240.json`:

```json
{
  "feature_id": "ret_240", "family": "returns_dynamics",
  "mathematical_definition": "log(c_t) - log(c_{t-240})",
  "source_module": "features.returns_dynamics.compute_returns_dynamics",
  "required_state": ["close"], "update_trigger": "M1_CLOSE", "window": 240,
  "causal": true, "live_compatible": true, "computational_cost": "LOW",
  "missing_value_policy": "NaN for first 240 bars, never zero-filled",
  "warmup_bars": 240, "historical_coverage": "FULL_HISTORY",
  "status": "REDUNDANT",
  "status_reason": "Not in the 17 v3 survivors (research/output/v3_feature_survivors.json); redundant/low-importance in the 92-candidate ablation. Kept registered per spec section 9 -- Phase 4 may find it useful for a different target.",
  "evidence_ref": "research/output/v3_feature_survivors.json",
  "version": "v1"
}
```

Repeat for the other 18 features in `f`, reading `status`/`status_reason`
directly off `research/output/v3_feature_survivors.json`'s `decisions`
dict for each `feature_id`: `status="USEFUL"` if
`decisions[id]["keep"] is true` (cite the real `catboost_importance`/
`mi_rank`/`family` numbers from that JSON in `status_reason`), else
`status="REDUNDANT"` (cite the real reason string already present in
`decisions[id]["reasons"]`). `run_length_signed` and
`return_percentile_rank_60` are the two family-A survivors — mark
`status="USEFUL"`. Every other field (`update_trigger`, `causal`,
`live_compatible`, `warmup_bars` matching each feature's window) follows
the same pattern as `ret_240` above, window value taken from the rolling
window each feature actually uses (20/60/240 per the code).

- [ ] **Step 4: Write the test**

```python
# tests/test_returns_dynamics.py
"""python3 tests/test_returns_dynamics.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.returns_dynamics import compute_returns_dynamics
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_returns_dynamics_shape_and_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    out = compute_returns_dynamics(shared)
    assert set(out.keys()) == {
        "ret_240", "sign_ret_240", "ret_accel_5_15", "ret_decel_15_60",
        "run_length_signed", "return_autocorr_20", "return_autocorr_60",
        "return_pacf1_60", "sign_flip_rate_20", "rolling_mean_ret_20",
        "rolling_median_ret_20", "return_dispersion_20",
        "upside_downside_asymmetry_60", "return_skew_60", "return_kurt_60",
        "return_skew_240", "return_percentile_rank_60",
        "return_quantile_pos_240", "directional_entropy_60",
    }
    for k, v in out.items():
        assert len(v) == len(df), k


def test_registry_matches_computed_keys():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    computed_keys = set(compute_returns_dynamics(shared).keys())
    registered_ids = {d.feature_id for d in load_family("returns_dynamics")}
    assert computed_keys == registered_ids


if __name__ == "__main__":
    test_compute_returns_dynamics_shape_and_keys()
    test_registry_matches_computed_keys()
    print("tests/test_returns_dynamics.py: OK")
```

- [ ] **Step 5: Run test to verify it fails, then implement, then pass**

Run `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_returns_dynamics.py`
before Steps 1-3 exist (expect `ModuleNotFoundError`), then after writing
`_shared.py` + `returns_dynamics.py` + the 19 JSON files, run again.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/_shared.py features/returns_dynamics.py features/registry/returns_dynamics/ tests/test_returns_dynamics.py
git commit -m "Add family A (returns_dynamics): move from research/features_v3.py, register 19 features"
```

---

## Task 6: Family B (`volatility_dynamics.py`)

**Files:**
- Create: `features/volatility_dynamics.py`
- Create: `features/registry/volatility_dynamics/*.json` (9 files)
- Test: `tests/test_volatility_dynamics.py`

**Interfaces:**
- Consumes: `features._shared.SharedInputs` (Task 5).
- Produces: `features.volatility_dynamics.compute_volatility_dynamics(shared: SharedInputs) -> dict[str, np.ndarray]`.

Source: `research/features_v3.py` lines 501-519 (family B assembly
block, no dedicated numba kernels — all pandas rolling ops). Note line
511-513's `resample("1D")` + `reindex(..., method="ffill")` pattern for
`vol_percentile_252` — this is the feature that motivates Task 20's
`DailyBuffer`; keep this exact implementation for the batch/replay path
unchanged, `live_engine.py` (Task 22) will call this same function
against a live-populated daily series instead.

- [ ] **Step 1: Write `features/volatility_dynamics.py`**

```python
"""Family B -- volatility dynamics candidates (beyond the baseline
EWMA/GK/RS/YZ/bipower/jump in features/volatility.py, which stays
untouched). Moved from research/features_v3.py lines 501-519."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def compute_volatility_dynamics(shared: SharedInputs) -> dict:
    ret1_s = pd.Series(shared.ret1)
    ewma_vol = shared.ewma_vol
    times = shared.times
    base_feat = shared.base_feat
    f = {}
    f["realized_variance_20"] = (ret1_s ** 2).rolling(20).sum().to_numpy()
    ret1_up = pd.Series(np.where(shared.ret1 > 0, shared.ret1, 0.0))
    ret1_down = pd.Series(np.where(shared.ret1 < 0, shared.ret1, 0.0))
    f["realized_semivar_upside_20"] = (ret1_up ** 2).rolling(20).sum().to_numpy()
    f["realized_semivar_downside_20"] = (ret1_down ** 2).rolling(20).sum().to_numpy()
    log_hl = np.log(shared.h / shared.l)
    f["parkinson_vol_60"] = np.sqrt(pd.Series(log_hl ** 2).rolling(60).mean().to_numpy() / (4 * np.log(2)))
    f["vol_acceleration_30"] = ewma_vol - pd.Series(ewma_vol).shift(30).to_numpy()
    f["vol_of_vol_60"] = pd.Series(ewma_vol).rolling(60).std().to_numpy()
    ev_daily = pd.Series(ewma_vol, index=times).resample("1D").last()
    vol_pctile_252 = ev_daily.rolling(252, min_periods=60).rank(pct=True).shift(1)
    f["vol_percentile_252"] = vol_pctile_252.reindex(times, method="ffill").to_numpy()
    ev_roll_mean_60 = pd.Series(ewma_vol).rolling(60).mean()
    ev_roll_std_60 = pd.Series(ewma_vol).rolling(60).std()
    f["vol_zscore_60"] = ((pd.Series(ewma_vol) - ev_roll_mean_60) / ev_roll_std_60).to_numpy()
    gk20 = base_feat["gk_vol_20"].to_numpy()
    gk240 = base_feat["gk_vol_240"].to_numpy()
    f["vol_compression_ratio"] = np.where(gk240 > 1e-12, gk20 / gk240, np.nan)
    return f
```

- [ ] **Step 2: Write 9 registry JSON files** using the same pattern as
  Task 5 Step 3 — pull `status`/`status_reason` from
  `research/output/v3_feature_survivors.json`'s `decisions` dict for each
  of these 9 `feature_id`s. `vol_acceleration_30`, `vol_of_vol_60`,
  `vol_compression_ratio` are the family-B survivors →
  `status="USEFUL"`. `vol_percentile_252` gets
  `update_trigger="DAILY"`, `required_state=["ewma_vol_daily_history"]`,
  `dependencies=["daily_buffer"]` (Task 20 dependency, noted now for
  Task 22 to pick up). The rest → `status="REDUNDANT"` citing their real
  reasons from the JSON.

- [ ] **Step 3: Write `tests/test_volatility_dynamics.py`** — same shape
  as Task 5 Step 4 (`_synthetic_df` helper duplicated locally, shape/key
  assertion, registry-matches-computed-keys assertion). Verify FAIL then
  PASS as in Task 5 Step 5.

- [ ] **Step 4: Commit**

```bash
git add features/volatility_dynamics.py features/registry/volatility_dynamics/ tests/test_volatility_dynamics.py
git commit -m "Add family B (volatility_dynamics): move from research/features_v3.py, register 9 features"
```

---

## Task 7: Family C (`jump_detection.py`)

**Files:**
- Create: `features/jump_detection.py`
- Create: `features/registry/jump_detection/*.json` (9 files)
- Test: `tests/test_jump_detection.py`

**Interfaces:**
- Consumes: `SharedInputs` (Task 5).
- Produces: `compute_jump_detection(shared, cusum_k: float) -> dict`.

Source: `research/features_v3.py` lines 521-553. **Important deviation
from a verbatim move**: line 522 does `from learning.train import
CUSUM_K` — this is a `research/`→`learning/` import inside what is about
to become `features/`, and `features/` must never import `learning/`
(Global Constraints, enforced by Task 27's boundary test). Fix during the
move: `compute_jump_detection` takes `cusum_k: float` as an explicit
parameter instead of importing the constant. `replay_engine.py` (Task 16)
passes `cusum_k=2.5` (the literal value of `learning.train.CUSUM_K` today
— confirm this by reading `learning/train.py`'s `CUSUM_K` definition
before writing Task 16, do not guess).

- [ ] **Step 1: Write `features/jump_detection.py`**

```python
"""Family C -- jump/change detection. Moved from research/features_v3.py
lines 521-553. Deviation from verbatim: cusum_k is now an explicit
parameter (was `from learning.train import CUSUM_K`) -- features/ must
never import learning/ (test_boundary.py)."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def compute_jump_detection(shared: SharedInputs, cusum_k: float) -> dict:
    ret1, sign1, ewma_vol, c = shared.ret1, shared.sign1, shared.ewma_vol, shared.c
    f = {}
    threshold = np.clip(cusum_k * np.nan_to_num(ewma_vol, nan=np.nanmedian(ewma_vol)) * c, 1e-6, None)
    cusum_pos = np.zeros(len(c)); cusum_neg = np.zeros(len(c))
    sp, sn = 0.0, 0.0
    for i in range(1, len(c)):
        diff = c[i] - c[i - 1]
        sp = max(0.0, sp + diff); sn = min(0.0, sn + diff)
        if sp > threshold[i] or sn < -threshold[i]:
            sp, sn = 0.0, 0.0
        cusum_pos[i], cusum_neg[i] = sp, sn
    f["cusum_distance_to_threshold"] = np.maximum(cusum_pos, -cusum_neg) / np.clip(threshold, 1e-9, None)
    local_vol_price = ewma_vol * c
    is_jump = np.abs(ret1 * c) > 3 * np.clip(local_vol_price, 1e-9, None)
    is_jump_s = pd.Series(is_jump.astype(np.float64))
    f["jump_intensity_60"] = is_jump_s.rolling(60).sum().to_numpy()
    jump_mag = pd.Series(np.where(is_jump, np.abs(ret1), np.nan))
    f["jump_magnitude_mean_60"] = jump_mag.rolling(60, min_periods=1).mean().to_numpy()
    jump_dir = pd.Series(np.where(is_jump, sign1, np.nan))
    f["jump_direction_bias_60"] = jump_dir.rolling(60, min_periods=1).mean().to_numpy()
    changepoint = (cusum_pos == 0) & (cusum_neg == 0)
    changepoint[0] = False
    bars_since = np.full(len(c), np.nan)
    last_cp = -1
    for i in range(len(c)):
        if changepoint[i]:
            last_cp = i
        bars_since[i] = i - last_cp if last_cp >= 0 else np.nan
    f["bars_since_last_changepoint"] = bars_since
    f["changepoint_intensity_240"] = pd.Series(changepoint.astype(np.float64)).rolling(240).sum().to_numpy()
    shock = np.abs(ret1) / np.clip(ewma_vol, 1e-9, None)
    f["vol_shock_zscore"] = (pd.Series(shock) - pd.Series(shock).rolling(60).mean()).to_numpy()
    # exposed for family I (regime_state.py, Task 13), which reuses bars_since
    f["_bars_since_last_changepoint_internal"] = bars_since
    return f
```

Note: `f["_bars_since_last_changepoint_internal"]` duplicates
`bars_since_last_changepoint` under a private-prefixed key so family I
(Task 13, `jump_state`/`changepoint_state`) can reuse the array without
recomputing the CUSUM loop. `replay_engine.py`/`live_engine.py` must
strip any `_`-prefixed key before writing final output (note this in
Task 16/22's implementation).

- [ ] **Step 2: Write 7 registry JSON files** (7 real output keys,
  excluding the internal-only `_bars_since_last_changepoint_internal`) —
  same evidence-citation pattern as Task 5. `vol_shock_zscore` is the
  family-C survivor → `status="USEFUL"`.

- [ ] **Step 3: Write `tests/test_jump_detection.py`** — same shape as
  Task 5 Step 4, but `compute_jump_detection(shared, cusum_k=2.5)`; assert
  output keys match the 7 *public* registry entries (filter out any key
  starting with `_` before comparing to `load_family(...)`).

- [ ] **Step 4: Commit**

```bash
git add features/jump_detection.py features/registry/jump_detection/ tests/test_jump_detection.py
git commit -m "Add family C (jump_detection): move from research/features_v3.py, fix learning/ import, register 7 features"
```

---

## Task 8: Family D (`distribution_info.py`)

**Files:**
- Create: `features/distribution_info.py`
- Create: `features/registry/distribution_info/*.json` (6 files)
- Test: `tests/test_distribution_info.py`

**Interfaces:**
- Consumes: `SharedInputs`.
- Produces: `compute_distribution_info(shared) -> dict`, plus kernels
  `shannon_entropy_returns`, `permutation_entropy`, `sample_entropy`,
  `mi_proxy_sign_lag` (module-level, from `research/features_v3.py` lines
  132-265).

Source: kernels at lines 132-265, assembly at lines 555-563.

- [ ] **Step 1: Write `features/distribution_info.py`** — copy the 4
  numba kernels (lines 132-156 `_shannon_entropy_returns`, 159-198
  `_permutation_entropy`, 201-236 `_sample_entropy`, 239-265
  `_mi_proxy_sign_lag`) verbatim minus leading underscore, then:

```python
def compute_distribution_info(shared: SharedInputs) -> dict:
    ret1, sign1 = shared.ret1, shared.sign1
    ret1_s = pd.Series(ret1)
    f = {}
    ret_std_60 = ret1_s.rolling(60).std()
    f["tail_probability_60"] = (np.abs(ret1_s) > 2 * ret_std_60).rolling(60).mean().to_numpy()
    f["shannon_entropy_returns_60"] = shannon_entropy_returns(ret1, 60, 8)
    f["permutation_entropy_60"] = permutation_entropy(ret1, 60, 3)
    f["sample_entropy_20"] = sample_entropy(ret1, 20, 2, 0.2)
    r2 = ret1_s ** 2
    f["return_concentration_60"] = ((r2 ** 2).rolling(60).sum() / (r2.rolling(60).sum() ** 2)).to_numpy()
    f["mi_proxy_sign_lag5_240"] = mi_proxy_sign_lag(sign1, 5, 240)
    return f
```

- [ ] **Step 2: Write 6 registry JSON files.** `tail_probability_60` is
  the family-D survivor → `status="USEFUL"`. `sample_entropy_20` gets
  `computational_cost="HIGH"` (its own docstring documents O(window²) —
  copy that note verbatim into `numerical_stability_notes`).

- [ ] **Step 3: Write `tests/test_distribution_info.py`** — same shape as
  Task 5 Step 4.

- [ ] **Step 4: Commit**

```bash
git add features/distribution_info.py features/registry/distribution_info/ tests/test_distribution_info.py
git commit -m "Add family D (distribution_info): move from research/features_v3.py, register 6 features"
```

---

## Task 9: Family E (`market_geometry.py`)

**Files:**
- Create: `features/market_geometry.py`
- Create: `features/registry/market_geometry/*.json` (13 files)
- Test: `tests/test_market_geometry.py`

**Interfaces:**
- Consumes: `SharedInputs`.
- Produces: `compute_market_geometry(shared) -> dict`, kernels
  `breakout_failure_magnitude`, `avg_run_length`, `high_low_density`
  (from lines 325-394).

Source: kernels at lines 325-394 (`_breakout_failure_magnitude`,
`_avg_run_length`, `_high_low_density`), assembly at lines 565-593.

- [ ] **Step 1: Write `features/market_geometry.py`** — copy the 3
  kernels verbatim minus underscore, then:

```python
def compute_market_geometry(shared: SharedInputs) -> dict:
    c, h, l, sign1 = shared.c, shared.h, shared.l, shared.sign1
    close_s = pd.Series(c)
    f = {}
    roll_max_h_20 = pd.Series(h).rolling(20).max()
    roll_min_l_20 = pd.Series(l).rolling(20).min()
    roll_max_h_60 = pd.Series(h).rolling(60).max()
    roll_min_l_60 = pd.Series(l).rolling(60).min()
    f["dist_from_high_20"] = ((close_s - roll_max_h_20) / close_s).to_numpy()
    f["dist_from_low_20"] = ((close_s - roll_min_l_20) / close_s).to_numpy()
    rng20 = (roll_max_h_20 - roll_min_l_20)
    f["range_position_20"] = ((close_s - roll_min_l_20) / rng20).to_numpy()
    rng60 = (roll_max_h_60 - roll_min_l_60)
    f["range_position_60"] = ((close_s - roll_min_l_60) / rng60).to_numpy()
    f["range_width_20"] = (rng20 / close_s).to_numpy()
    range_width_60 = (rng60 / close_s).to_numpy()
    f["range_width_ratio_20_60"] = np.where(range_width_60 > 1e-12, f["range_width_20"] / range_width_60, np.nan)
    roll_mean_c_60 = close_s.rolling(60).mean()
    roll_std_c_60 = close_s.rolling(60).std()
    f["displacement_from_equilibrium_60"] = ((close_s - roll_mean_c_60) / roll_std_c_60).to_numpy()
    prior_high_20 = pd.Series(h).rolling(20).max().shift(1)
    f["breakout_magnitude_20"] = (np.maximum(0, close_s - prior_high_20) / close_s).to_numpy()
    f["breakout_failure_magnitude_20"] = breakout_failure_magnitude(c, h, l, 20, 5)
    roll_median_c_60 = close_s.rolling(60).median()
    above_median = (close_s > roll_median_c_60).astype(np.float64)
    crossings = above_median.diff().abs()
    f["reversal_frequency_60"] = crossings.rolling(60).sum().to_numpy()
    f["avg_run_length_60"] = avg_run_length(sign1, 60)
    excursion_std_20 = close_s.rolling(20).std()
    f["excursion_from_recent_distribution_20"] = np.where(
        excursion_std_20 > 1e-9, (close_s - close_s.shift(20)) / excursion_std_20, np.nan)
    f["high_low_density_60"] = high_low_density(h, l, 60)
    return f
```

- [ ] **Step 2: Write 13 registry JSON files.** Survivors:
  `dist_from_high_20`, `dist_from_low_20`, `range_position_20`,
  `range_position_60`, `range_width_ratio_20_60`,
  `displacement_from_equilibrium_60`,
  `excursion_from_recent_distribution_20` → `status="USEFUL"` (7 of
  family E's 13 survive — the largest survivor count of any family, cite
  this in `status_reason`).

- [ ] **Step 3: Write `tests/test_market_geometry.py`** — same shape as
  Task 5 Step 4.

- [ ] **Step 4: Commit**

```bash
git add features/market_geometry.py features/registry/market_geometry/ tests/test_market_geometry.py
git commit -m "Add family E (market_geometry): move from research/features_v3.py, register 13 features"
```

---

## Task 10: Family F (`persistence.py`)

**Files:**
- Create: `features/persistence.py`
- Create: `features/registry/persistence/*.json` (7 files)
- Test: `tests/test_persistence.py`

**Interfaces:**
- Consumes: `SharedInputs`, `features.returns_dynamics.rolling_autocorr_lag1` (Task 5).
- Produces: `compute_persistence(shared) -> dict`, kernels
  `mean_reversion_speed`, `autocorr_decay_rate` (from lines 268-322).

Source: kernels at lines 268-322 (`_mean_reversion_speed`,
`_autocorr_decay_rate`), assembly at lines 595-610. Line 596's
`from features.hurst import rolling_hurst` is already a valid
`features`→`features` import — keep as-is.

- [ ] **Step 1: Write `features/persistence.py`**

```python
"""Family F -- mean reversion / persistence. Moved from
research/features_v3.py lines 595-610. Reuses rolling_autocorr_lag1 from
features.returns_dynamics (family A) -- the one kernel genuinely shared
across two families, not duplicated."""
import numba
import numpy as np
import pandas as pd

from features._shared import SharedInputs
from features.hurst import rolling_hurst
from features.returns_dynamics import rolling_autocorr_lag1

# copy _mean_reversion_speed (lines 268-285) and _autocorr_decay_rate
# (lines 288-322) verbatim, renamed mean_reversion_speed / autocorr_decay_rate


def compute_persistence(shared: SharedInputs) -> dict:
    ret1, c = shared.ret1, shared.c
    kalman_resid, hurst_120, base_feat = shared.kalman_resid, shared.hurst_120, shared.base_feat
    f = {}
    f["hurst_240"] = rolling_hurst(ret1, window=240)
    f["mean_reversion_speed_60"] = mean_reversion_speed(c, 60)
    speed = f["mean_reversion_speed_60"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f["half_life_60"] = np.where(speed < 0, -np.log(2) / np.log(1 + speed), np.nan)
    f["autocorr_decay_rate_60"] = autocorr_decay_rate(ret1, 240, np.array([1, 2, 3, 5, 10], dtype=np.int64))
    f["persistence_score"] = hurst_120 - 0.5
    f["residual_mean_reversion_60"] = rolling_autocorr_lag1(np.nan_to_num(kalman_resid), 60)
    fracdiff = base_feat["fracdiff_log_price"].to_numpy()
    fd_s = pd.Series(fracdiff)
    x_idx = pd.Series(np.arange(len(fd_s), dtype=np.float64))
    cov = fd_s.rolling(60).cov(x_idx)
    var = x_idx.rolling(60).var()
    f["fracdiff_slope_60"] = (cov / var).to_numpy()
    return f
```

- [ ] **Step 2: Write 7 registry JSON files.** No family-F survivors in
  `v3_feature_survivors.json` — all 7 get `status="REDUNDANT"`, citing
  the real per-feature reasons from the JSON (family ablation delta <0.1pp
  per `SUMMARY.md` finding #7).

- [ ] **Step 3: Write `tests/test_persistence.py`** — same shape as Task 5
  Step 4.

- [ ] **Step 4: Commit**

```bash
git add features/persistence.py features/registry/persistence/ tests/test_persistence.py
git commit -m "Add family F (persistence): move from research/features_v3.py, register 7 features"
```

---

## Task 11: Family G (`temporal.py`)

**Files:**
- Create: `features/temporal.py`
- Create: `features/registry/temporal/*.json` (14 files)
- Test: `tests/test_temporal.py`

**Interfaces:**
- Consumes: `SharedInputs`.
- Produces: `compute_temporal(shared) -> dict`.

Source: `research/features_v3.py` lines 612-641, no dedicated numba
kernels. **Document, don't fix, a real naming distinction**: this family's
session encoding (`session_asian`/`session_london`/`session_ny`, hardcoded
generic UTC hour bands 0-8/8-16/13-21) is a *different, independent*
concept from `market/state_engine.py`'s `is_market_closed()` (the
empirically-derived XM GOLD.i# open/closed schedule from Phase 2). Do not
merge them — `is_market_closed` answers "is the market open at all,"
this family answers "which trading session, given the market is open." A
one-line comment noting this distinction goes at the top of
`temporal.py`.

- [ ] **Step 1: Write `features/temporal.py`**

```python
"""Family G -- time/session encoding (UTC; MT5 server-time offset not
modeled -- relative encodings only). Moved from research/features_v3.py
lines 612-641. NOTE: session_asian/london/ny below are generic UTC hour
bands, a DIFFERENT and independent concept from
market/state_engine.py's is_market_closed() (the empirically-derived XM
GOLD.i# open/closed schedule) -- this family encodes WHICH session,
is_market_closed answers WHETHER the market is open at all. Not merged."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def compute_temporal(shared: SharedInputs) -> dict:
    times, ewma_vol, tick_vol, base_feat = shared.times, shared.ewma_vol, shared.tick_vol, shared.base_feat
    hour = times.hour.to_numpy(dtype=np.float64)
    minute = times.minute.to_numpy(dtype=np.float64)
    dow = times.dayofweek.to_numpy(dtype=np.float64)
    f = {}
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    f["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["session_asian"] = ((hour >= 0) & (hour < 8)).astype(np.float64)
    f["session_london"] = ((hour >= 8) & (hour < 16)).astype(np.float64)
    f["session_ny"] = ((hour >= 13) & (hour < 21)).astype(np.float64)
    f["session_london_ny_overlap"] = ((hour >= 13) & (hour < 16)).astype(np.float64)
    session_id = np.select(
        [f["session_london_ny_overlap"] > 0, f["session_london"] > 0, f["session_ny"] > 0, f["session_asian"] > 0],
        [3, 1, 2, 0], default=0)
    session_change = pd.Series(session_id).diff().abs() > 0
    f["session_transition_flag"] = session_change.astype(np.float64).to_numpy()
    hour_key = pd.Series(hour, index=times)
    ev_by_hour_ref = pd.Series(ewma_vol, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["vol_conditional_on_session"] = (pd.Series(ewma_vol) - ev_by_hour_ref.to_numpy()).to_numpy()
    ret5 = base_feat["ret_5"].to_numpy()
    ret5_by_hour_ref = pd.Series(ret5, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["ret_conditional_on_session"] = (pd.Series(ret5) - ret5_by_hour_ref.to_numpy()).to_numpy()
    tv_by_hour_ref = pd.Series(tick_vol, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["activity_conditional_on_session"] = (pd.Series(tick_vol) - tv_by_hour_ref.to_numpy()).to_numpy()
    return f
```

- [ ] **Step 2: Write 14 registry JSON files.** Survivors: `hour_sin`,
  `hour_cos`, `ret_conditional_on_session` → `status="USEFUL"` (the only
  3 of family G's 14 to survive). The `_conditional_on_session` features
  get `update_trigger="M1_CLOSE"` but note `dependencies=["expanding_hour_group_state"]`
  in their descriptor — flag in `numerical_stability_notes` that the
  `groupby(...).expanding()` pattern needs a full historical hour-grouped
  state to reproduce live, making these 3 features **live_compatible:
  false / historical_coverage: RESEARCH_ONLY** for now (an expanding
  per-hour-of-day mean since inception is not something the bounded live
  buffer can reproduce) — document this explicitly rather than silently
  attempting and getting it wrong; Phase 4 or a later phase can build a
  proper incremental per-hour accumulator if these prove valuable.

- [ ] **Step 3: Write `tests/test_temporal.py`** — same shape as Task 5
  Step 4.

- [ ] **Step 4: Commit**

```bash
git add features/temporal.py features/registry/temporal/ tests/test_temporal.py
git commit -m "Add family G (temporal): move from research/features_v3.py, register 14 features, document session-vs-is_market_closed distinction"
```

---

## Task 12: Family H (`microstructure_history.py`)

**Files:**
- Create: `features/microstructure_history.py`
- Create: `features/registry/microstructure_history/*.json` (6 files)
- Test: `tests/test_microstructure_history.py`

**Interfaces:**
- Consumes: `SharedInputs`, `research.historical_coverage.measure_coverage` output (Task 1, for registry metadata only — not imported by the compute function itself).
- Produces: `compute_microstructure_history(shared) -> dict`.

Source: `research/features_v3.py` lines 643-653, no numba kernels.

- [ ] **Step 1: Write `features/microstructure_history.py`**

```python
"""Family H -- historical microstructure, honestly scoped: tick_volume +
spread only (no tick stream, no order book in the 6.7yr CSV -- see spec
section 2/Task 1's real historical_coverage.py measurement for exact
tick_volume degradation date and spread's near-constant-98.9%-of-history
fact). Moved from research/features_v3.py lines 643-653."""
import pandas as pd

from features._shared import SharedInputs


def compute_microstructure_history(shared: SharedInputs) -> dict:
    tv_s = pd.Series(shared.tick_vol)
    sp_s = pd.Series(shared.spread)
    times = shared.times
    f = {}
    f["tick_volume_zscore_60"] = ((tv_s - tv_s.rolling(60).mean()) / tv_s.rolling(60).std()).to_numpy()
    f["tick_volume_accel_20"] = (tv_s.rolling(20).mean() - tv_s.rolling(20).mean().shift(20)).to_numpy()
    f["spread_change_1"] = sp_s.diff().to_numpy()
    sp_daily = sp_s.copy(); sp_daily.index = times
    spread_pctile = sp_daily.resample("1D").last().rolling(252, min_periods=60).rank(pct=True).shift(1)
    f["spread_percentile_252"] = spread_pctile.reindex(times, method="ffill").to_numpy()
    f["spread_volatility_60"] = sp_s.rolling(60).std().to_numpy()
    f["tick_volume_spread_ratio"] = (tv_s / (sp_s + 1.0)).to_numpy()
    return f
```

- [ ] **Step 2: Write 6 registry JSON files.** No family-H survivors. All
  6 → `status="REDUNDANT"` for the batch/replay path, citing
  `SUMMARY.md` finding #9 (spread: zero CatBoost importance, constant
  98.9% of history) for the 3 spread-dependent features and finding #8
  (tick_volume: small/unstable) for the 3 tick_volume-dependent ones. Run
  `python3 research/historical_coverage.py` (Task 1) and use its real
  `tick_volume_degrades_after` date in `historical_coverage:
  "PARTIAL_HISTORY"` + `status_reason` for `tick_volume_zscore_60`,
  `tick_volume_accel_20`, `tick_volume_spread_ratio`. Mark
  `spread_change_1`, `spread_percentile_252`, `spread_volatility_60` as
  `live_compatible: false` — Task 22's `live_engine.py` has no per-bar
  spread history available from `MarketState`'s bounded M1 buffer (only
  the current live tick's spread and Task 21's dedicated tick-level ring
  buffers), so these specific 3 stay research-only; the *new* live-tick
  spread family (Task 21) is what replaces them for live use, per spec
  section 7.

- [ ] **Step 3: Write `tests/test_microstructure_history.py`** — same
  shape as Task 5 Step 4.

- [ ] **Step 4: Commit**

```bash
git add features/microstructure_history.py features/registry/microstructure_history/ tests/test_microstructure_history.py
git commit -m "Add family H (microstructure_history): move from research/features_v3.py, register 6 features, mark spread-history features live_compatible=false"
```

---

## Task 13: Family I (`regime_state.py`)

**Files:**
- Create: `features/regime_state.py`
- Create: `features/registry/regime_state/*.json` (7 files)
- Test: `tests/test_regime_state.py`

**Interfaces:**
- Consumes: `SharedInputs`, and the `bars_since_last_changepoint` array
  from `features.jump_detection.compute_jump_detection(shared, cusum_k)`
  (Task 7) — `compute_regime_state` takes `jump_features: dict` as an
  explicit second argument (the output of `compute_jump_detection`)
  instead of recomputing CUSUM state, matching the internal-key handoff
  noted in Task 7.
- Produces: `compute_regime_state(shared, jump_features) -> dict`.

Source: `research/features_v3.py` lines 655-677 (assembly, including the
local `causal_tercile` closure at lines 663-671).

- [ ] **Step 1: Write `features/regime_state.py`**

```python
"""Family I -- discretized regime/state variables from already-computed
continuous features (NOT an HMM). Moved from research/features_v3.py
lines 655-677. Consumes family C's bars_since array directly instead of
recomputing CUSUM state (see Task 7's internal-key handoff)."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def causal_tercile(x, window):
    """Rolling (trailing, shift(1)) tercile bucket -- unlike pd.cut(x, 3),
    which fixes bin edges from the WHOLE series (past+future, a leakage
    bug the original research caught during smoke-testing), this only
    ever uses thresholds computed from data strictly before the current
    row."""
    s = pd.Series(x)
    lo = s.rolling(window, min_periods=window // 4).quantile(0.333).shift(1)
    hi = s.rolling(window, min_periods=window // 4).quantile(0.667).shift(1)
    return np.where(s <= lo, 0.0, np.where(s >= hi, 2.0, 1.0))


def compute_regime_state(shared: SharedInputs, jump_features: dict) -> dict:
    ewma_vol, times = shared.ewma_vol, shared.times
    bars_since = jump_features["_bars_since_last_changepoint_internal"]
    f = {}
    ev_daily2 = pd.Series(ewma_vol, index=times).resample("1D").last()
    tercile = ev_daily2.rolling(252, min_periods=60).apply(
        lambda w: np.searchsorted(np.percentile(w, [33.3, 66.7]), w[-1]), raw=True).shift(1)
    vol_state_tercile = tercile.reindex(times, method="ffill").to_numpy()
    f["vol_state_tercile"] = vol_state_tercile
    f["jump_state"] = np.where(bars_since <= 5, 2.0, np.where(bars_since <= 20, 1.0, 0.0))
    persistence_score = shared.hurst_120 - 0.5
    f["persistence_state"] = causal_tercile(persistence_score, 500)
    # shannon_entropy_returns_60 recomputed here would duplicate family D;
    # regime_state instead requires it be passed alongside jump_features --
    # see compute_regime_state's caller in replay_engine.py/live_engine.py,
    # which merges family D's output into a combined dict before calling this.
    return f
```

Wait — `entropy_state` and `activity_state` in the original code depend
on family D's `shannon_entropy_returns_60` and family H's
`tick_volume_zscore_60` respectively (lines 674-675). Rather than a
second special-cased parameter, generalize: `compute_regime_state` takes
a third argument `upstream: dict` (the merged output of families C, D,
H), matching how `replay_engine.py` will already have all of them
computed by the time it calls family I. Revise the signature and body:

```python
def compute_regime_state(shared: SharedInputs, upstream: dict) -> dict:
    """upstream must contain: bars_since (from family C's internal key),
    shannon_entropy_returns_60 (family D), tick_volume_zscore_60 (family H)."""
    ewma_vol, times = shared.ewma_vol, shared.times
    bars_since = upstream["_bars_since_last_changepoint_internal"]
    f = {}
    ev_daily2 = pd.Series(ewma_vol, index=times).resample("1D").last()
    tercile = ev_daily2.rolling(252, min_periods=60).apply(
        lambda w: np.searchsorted(np.percentile(w, [33.3, 66.7]), w[-1]), raw=True).shift(1)
    vol_state_tercile = tercile.reindex(times, method="ffill").to_numpy()
    f["vol_state_tercile"] = vol_state_tercile
    f["jump_state"] = np.where(bars_since <= 5, 2.0, np.where(bars_since <= 20, 1.0, 0.0))
    persistence_score = shared.hurst_120 - 0.5
    f["persistence_state"] = causal_tercile(persistence_score, 500)
    f["entropy_state"] = causal_tercile(upstream["shannon_entropy_returns_60"], 500)
    f["activity_state"] = causal_tercile(upstream["tick_volume_zscore_60"], 500)
    f["changepoint_state"] = np.where(bars_since <= 10, 0.0, np.where(bars_since <= 60, 1.0, 2.0))
    f["composite_state_id"] = np.nan_to_num(vol_state_tercile, nan=0) * 3 + np.nan_to_num(f["persistence_state"], nan=0)
    return f
```

(Use this second, corrected version — delete the first draft above it
when writing the file; it's shown to make the fix visible against the
original source.)

- [ ] **Step 2: Write 7 registry JSON files.** No family-I survivors. All
  7 → `status="REDUNDANT"`, `dependencies=["jump_detection", "distribution_info", "microstructure_history"]`.

- [ ] **Step 3: Write `tests/test_regime_state.py`**

```python
"""python3 tests/test_regime_state.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features._shared import build_shared_inputs
from features.jump_detection import compute_jump_detection
from features.distribution_info import compute_distribution_info
from features.microstructure_history import compute_microstructure_history
from features.regime_state import compute_regime_state
from features.registry import load_family


def _synthetic_df(n=400):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_compute_regime_state():
    df = _synthetic_df()
    base = build_features(df)
    shared = build_shared_inputs(df, base)
    jump = compute_jump_detection(shared, cusum_k=2.5)
    dist = compute_distribution_info(shared)
    micro = compute_microstructure_history(shared)
    upstream = {**jump, **dist, **micro}
    out = compute_regime_state(shared, upstream)
    assert set(out.keys()) == {
        "vol_state_tercile", "jump_state", "persistence_state",
        "entropy_state", "activity_state", "changepoint_state", "composite_state_id",
    }


def test_registry_matches_computed_keys():
    registered_ids = {d.feature_id for d in load_family("regime_state")}
    assert registered_ids == {
        "vol_state_tercile", "jump_state", "persistence_state",
        "entropy_state", "activity_state", "changepoint_state", "composite_state_id",
    }


if __name__ == "__main__":
    test_compute_regime_state()
    test_registry_matches_computed_keys()
    print("tests/test_regime_state.py: OK")
```

- [ ] **Step 4: Commit**

```bash
git add features/regime_state.py features/registry/regime_state/ tests/test_regime_state.py
git commit -m "Add family I (regime_state): move from research/features_v3.py, wire cross-family upstream dict, register 7 features"
```

---

## Task 14: Family J (`first_passage.py`)

**Files:**
- Create: `features/first_passage.py`
- Create: `features/registry/first_passage/*.json` (4 files)
- Test: `tests/test_first_passage.py`

**Interfaces:**
- Consumes: `SharedInputs`.
- Produces: `compute_first_passage(shared) -> dict`, kernel `first_passage_stats`.

Source: kernel at lines 397-443 (`_first_passage_stats`, already verified
causal by direct code read during design — every inner-loop index is
strictly `< i`), assembly at lines 679-684.

- [ ] **Step 1: Write `features/first_passage.py`**

```python
"""Family J -- first-passage / path information, fully-resolved-past-only
(causal, verified by direct code read during Phase 3 design: every inner
loop index in first_passage_stats is strictly < i, no lookahead -- see
spec section 2). Moved from research/features_v3.py lines 397-443, 679-684."""
import numba
import numpy as np

from features._shared import SharedInputs

# copy _first_passage_stats (lines 397-443) verbatim, renamed first_passage_stats


def compute_first_passage(shared: SharedInputs) -> dict:
    ret1, c = shared.ret1, shared.c
    p_reach, time_to, hit_freq, fav_adv = first_passage_stats(ret1, c, 60, 10, 0.001)
    return {
        "hist_p_reach_10bps_10b_60": p_reach,
        "hist_time_to_10bps_60": time_to,
        "hist_barrier_hit_freq_60": hit_freq,
        "hist_path_asymmetry_60": fav_adv,
    }
```

- [ ] **Step 2: Write 4 registry JSON files.** No family-J survivors.
  All 4 → `status="REDUNDANT"`. `causal: true` with
  `numerical_stability_notes`: "verified causal by direct code read: loop
  bound i-window-sub_horizon..i-sub_horizon, all indices strictly < i
  (spec section 2)".

- [ ] **Step 3: Write `tests/test_first_passage.py`** — same shape as
  Task 5 Step 4.

- [ ] **Step 4: Commit**

```bash
git add features/first_passage.py features/registry/first_passage/ tests/test_first_passage.py
git commit -m "Add family J (first_passage): move from research/features_v3.py, register 4 features"
```

---

## Task 15: Baseline registry entries (28 production features)

**Files:**
- Create: `features/registry/baseline_v1/*.json` (28 files)
- Test: `tests/test_baseline_registry.py`

**Interfaces:**
- Consumes: `features.features.build_tier1_features` (existing, read
  only — not modified), `features.registry.load_family`/`build_schema`.
- Produces: `features.registry.build_schema("baseline_v1", "root-28col-2026-08-18", [...])`
  matching `models/registry/direction_catboost_20260818.json`'s
  `feature_cols` exactly.

`features/features.py` is never edited (Global Constraints). This task
only writes registry metadata pointing at the existing, unmodified code.

- [ ] **Step 1: Read `models/registry/direction_catboost_20260818.json`'s
  `feature_cols`** (already known from design research: `["ret_1",
  "sign_ret_1", "ret_5", "sign_ret_5", "ret_15", "sign_ret_15", "ret_60",
  "sign_ret_60", "ewma_vol", "gk_vol_20", "rs_vol_20", "yz_vol_20",
  "gk_vol_60", "rs_vol_60", "yz_vol_60", "gk_vol_240", "rs_vol_240",
  "yz_vol_240", "bipower_var_60", "jump_component_60", "kalman_level_dist",
  "kalman_velocity", "kalman_residual_z", "hurst_120", "hurst_480",
  "fracdiff_log_price", "spread", "tick_volume"]`) — confirm it still
  matches by re-reading the file at execution time, do not assume it's
  unchanged since design.

- [ ] **Step 2: Write 28 registry JSON files**, one per column above. All
  get `family="baseline_v1"`, `status="REQUIRED"`,
  `status_reason="currently deployed production feature, models/registry/direction_catboost_20260818.json"`,
  `historical_coverage="FULL_HISTORY"` except `spread`/`tick_volume`
  which get `historical_coverage="PARTIAL_HISTORY"` citing Task 1's real
  `tick_volume_degrades_after` date and `SUMMARY.md` finding #9 for
  spread, `source_module="features.features.build_tier1_features"`,
  `live_compatible=true`, `causal=true`, `update_trigger="M1_CLOSE"`,
  `evidence_ref="models/registry/direction_catboost_20260818.json"`.
  Example (`ret_1.json`):

```json
{
  "feature_id": "ret_1", "family": "baseline_v1",
  "mathematical_definition": "log(c_t) - log(c_{t-1})",
  "source_module": "features.features.build_tier1_features",
  "required_state": ["close"], "update_trigger": "M1_CLOSE", "window": 1,
  "causal": true, "live_compatible": true, "computational_cost": "LOW",
  "missing_value_policy": "NaN for first bar",
  "warmup_bars": 1, "historical_coverage": "FULL_HISTORY",
  "status": "REQUIRED",
  "status_reason": "currently deployed production feature, models/registry/direction_catboost_20260818.json",
  "evidence_ref": "models/registry/direction_catboost_20260818.json",
  "version": "v1"
}
```

- [ ] **Step 3: Write `tests/test_baseline_registry.py`**

```python
"""python3 tests/test_baseline_registry.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.registry import load_family, build_schema

_MODEL_REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "registry", "direction_catboost_20260818.json")


def test_baseline_registry_matches_deployed_model():
    with open(_MODEL_REGISTRY) as f:
        deployed = json.load(f)
    deployed_cols = set(deployed["feature_cols"])
    registered = {d.feature_id for d in load_family("baseline_v1")}
    assert registered == deployed_cols, registered.symmetric_difference(deployed_cols)


def test_build_baseline_schema():
    with open(_MODEL_REGISTRY) as f:
        deployed = json.load(f)
    schema = build_schema("baseline_v1", deployed["feature_schema_version"], deployed["feature_cols"])
    assert schema.feature_ids == deployed["feature_cols"]


if __name__ == "__main__":
    test_baseline_registry_matches_deployed_model()
    test_build_baseline_schema()
    print("tests/test_baseline_registry.py: OK")
```

- [ ] **Step 4: Run and verify pass**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_baseline_registry.py`

- [ ] **Step 5: Commit**

```bash
git add features/registry/baseline_v1/ tests/test_baseline_registry.py
git commit -m "Register the 28 deployed production features as baseline_v1, no code changes"
```

---

## Task 16: `features/replay_engine.py` + numeric equivalence regression + import migration

**Files:**
- Create: `features/replay_engine.py`
- Modify: `research/features_v3.py` (becomes a thin deprecated shim)
- Modify: `research/build_v3_dataset.py`, `research/v3_pipeline_checks.py`,
  `research/v3_family_ablation.py`, `research/v3_feature_selection.py`
  (import path updates only)
- Test: `tests/test_replay_engine.py`

**Interfaces:**
- Consumes: every `compute_<family>` function from Tasks 5-14.
- Produces: `features.replay_engine.build_candidate_features(df, base_feat) -> pd.DataFrame`
  (same signature/output shape as the original
  `research.features_v3.build_candidate_features`, used by Task 22's
  equivalence test and by Task 23's live/replay test).

- [ ] **Step 1: Read `learning/train.py`'s `CUSUM_K` definition** to
  confirm its literal value (needed for Task 7's `cusum_k` parameter).
  `grep -n "^CUSUM_K" learning/train.py`.

- [ ] **Step 2: Write `features/replay_engine.py`**

```python
"""Batch/replay feature engine -- composes every family's compute_<family>
function into one DataFrame, replacing research/features_v3.py's former
build_candidate_features (spec section 4/6). Used for historical dataset
building (Phase 4) and as the reference implementation live_engine.py's
per-M1-close recompute is checked against (Task 23)."""
import pandas as pd

from features._shared import build_shared_inputs
from features.returns_dynamics import compute_returns_dynamics
from features.volatility_dynamics import compute_volatility_dynamics
from features.jump_detection import compute_jump_detection
from features.distribution_info import compute_distribution_info
from features.market_geometry import compute_market_geometry
from features.persistence import compute_persistence
from features.temporal import compute_temporal
from features.microstructure_history import compute_microstructure_history
from features.regime_state import compute_regime_state
from features.first_passage import compute_first_passage

CUSUM_K = <literal value confirmed in Step 1>  # e.g. 2.5


def build_candidate_features(df: pd.DataFrame, base_feat: pd.DataFrame, cusum_k: float = CUSUM_K) -> pd.DataFrame:
    shared = build_shared_inputs(df, base_feat)
    a = compute_returns_dynamics(shared)
    b = compute_volatility_dynamics(shared)
    c = compute_jump_detection(shared, cusum_k)
    d = compute_distribution_info(shared)
    e = compute_market_geometry(shared)
    fam_f = compute_persistence(shared)
    g = compute_temporal(shared)
    h = compute_microstructure_history(shared)
    upstream = {**c, **d, **h}
    i = compute_regime_state(shared, upstream)
    j = compute_first_passage(shared)

    merged = {}
    for fam in (a, b, c, d, e, fam_f, g, h, i, j):
        for k, v in fam.items():
            if k.startswith("_"):
                continue
            merged[k] = v
    out = pd.DataFrame(merged, index=df.index)
    out.insert(0, "time", df["time"].to_numpy())
    return out
```

- [ ] **Step 3: Write `tests/test_replay_engine.py`** — numeric
  equivalence regression against the pre-move code, run BEFORE turning
  `research/features_v3.py` into a shim (Step 4), by temporarily
  importing the original file's function under a distinct name:

```python
"""python3 tests/test_replay_engine.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import numpy as np
import pandas as pd

from features.features import build_features
from features.replay_engine import build_candidate_features


def _synthetic_df(n=600):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def _load_original_module():
    """Loads a git-history copy of the pre-move research/features_v3.py
    (before Task 16 Step 4 shims it) via `git show`, so this equivalence
    check is real even after the shim is in place. Run this test's first
    pass BEFORE Step 4 replaces the file -- Step 4 itself is gated on this
    test passing against the then-still-original file."""
    import subprocess
    src = subprocess.run(
        ["git", "show", "HEAD:research/features_v3.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader("original_features_v3", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(src, "original_features_v3.py", "exec"), module.__dict__)
    return module


def test_replay_engine_matches_original():
    df = _synthetic_df()
    base = build_features(df)
    original = _load_original_module()
    expected = original.build_candidate_features(df, base)
    actual = build_candidate_features(df, base)
    assert set(actual.columns) - {"time"} == set(expected.columns) - {"time"}
    for col in expected.columns:
        if col == "time":
            continue
        e = expected[col].to_numpy(dtype=np.float64)
        a = actual[col].to_numpy(dtype=np.float64)
        both_nan = np.isnan(e) & np.isnan(a)
        assert np.allclose(e[~both_nan], a[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True), col


if __name__ == "__main__":
    test_replay_engine_matches_original()
    print("tests/test_replay_engine.py: OK")
```

- [ ] **Step 4: Run test, confirm PASS against the still-original
  `research/features_v3.py`** (this is the load-bearing check that the
  move preserved behavior exactly). Run:
  `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_replay_engine.py`

- [ ] **Step 5: Only after Step 4 passes, replace
  `research/features_v3.py`** with a deprecated shim:

```python
"""DEPRECATED -- moved to features/*.py in Phase 3 (spec section 4).
Kept as a thin re-export so any external reference doesn't hard-break;
new code should import features.replay_engine directly."""
from features.replay_engine import build_candidate_features  # noqa: F401
```

- [ ] **Step 6: Update `research/build_v3_dataset.py`,
  `research/v3_pipeline_checks.py`, `research/v3_family_ablation.py`,
  `research/v3_feature_selection.py`** — change any
  `from research.features_v3 import ...` or `import research.features_v3`
  to the equivalent `from features.replay_engine import build_candidate_features`
  (grep each file first to find its exact import line before editing:
  `grep -n "features_v3" research/*.py`).

- [ ] **Step 7: Re-run `tests/test_replay_engine.py`** to confirm it
  still passes reading `HEAD:research/features_v3.py` via `git show` (the
  git-history read makes this robust to the working-tree file now being a
  shim — this is intentional, it always compares against the original
  pre-move implementation regardless of the shim).

- [ ] **Step 8: Commit**

```bash
git add features/replay_engine.py research/features_v3.py research/build_v3_dataset.py research/v3_pipeline_checks.py research/v3_family_ablation.py research/v3_feature_selection.py tests/test_replay_engine.py
git commit -m "Add features/replay_engine.py composing all 10 families; verify numeric equivalence with pre-move code; migrate research/ imports"
```

---

## Task 17: Causality truncation test suite

**Files:**
- Create: `tests/test_causality.py`

**Interfaces:**
- Consumes: `features.replay_engine.build_candidate_features`,
  `features.features.build_features`.
- This closes the real gap found in design (spec section 2): the old
  docstring's claimed `research/v3_causality_check.py` never existed.

- [ ] **Step 1: Write the test**

```python
"""python3 tests/test_causality.py -- for every non-NaN feature value at
row i, changing rows AFTER i must never change that value. This is the
executable version of the causality claim research/features_v3.py's old
docstring made but never actually tested (Phase 3 spec section 2)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features.replay_engine import build_candidate_features


def _synthetic_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def test_truncation_does_not_change_past_values():
    df_full = _synthetic_df(n=500)
    df_truncated = df_full.iloc[:300].copy()

    base_full = build_features(df_full)
    base_trunc = build_features(df_truncated)
    feat_full = build_candidate_features(df_full, base_full)
    feat_trunc = build_candidate_features(df_truncated, base_trunc)

    compare_cols = [c for c in feat_full.columns if c != "time"]
    check_rows = 250  # inside df_truncated's range, deep enough past every warmup window (max 252)
    for col in compare_cols:
        a = feat_full[col].to_numpy(dtype=np.float64)[:check_rows]
        b = feat_trunc[col].to_numpy(dtype=np.float64)[:check_rows]
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"{col} changed when future rows were truncated -- causality violation")


def test_perturbing_future_rows_does_not_change_past_values():
    rng = np.random.default_rng(1)
    df = _synthetic_df(n=500)
    df_perturbed = df.copy()
    df_perturbed.loc[300:, "close"] = df_perturbed.loc[300:, "close"] + rng.normal(0, 50, len(df_perturbed) - 300)
    df_perturbed.loc[300:, "high"] = df_perturbed.loc[300:, "close"] + 0.5
    df_perturbed.loc[300:, "low"] = df_perturbed.loc[300:, "close"] - 0.5

    base = build_features(df)
    base_perturbed = build_features(df_perturbed)
    feat = build_candidate_features(df, base)
    feat_perturbed = build_candidate_features(df_perturbed, base_perturbed)

    compare_cols = [c for c in feat.columns if c != "time"]
    check_rows = 250
    for col in compare_cols:
        a = feat[col].to_numpy(dtype=np.float64)[:check_rows]
        b = feat_perturbed[col].to_numpy(dtype=np.float64)[:check_rows]
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"{col} changed when future rows were perturbed -- causality violation")


if __name__ == "__main__":
    test_truncation_does_not_change_past_values()
    test_perturbing_future_rows_does_not_change_past_values()
    print("tests/test_causality.py: OK")
```

- [ ] **Step 2: Run and debug any real failures found**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_causality.py`
If a family genuinely fails (e.g. a rolling operation using a centered
window by accident), this is a REAL bug caught for the first time — fix
the specific family module (do not weaken the test). Expected after any
fixes: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_causality.py
git commit -m "Add real causality truncation tests -- closes the gap where research/features_v3.py claimed but never built this test"
```

---

## Task 18: `market/state_engine.py` — `completed_m1_window()` accessor

**Files:**
- Modify: `market/state_engine.py` (add one method to `StateEngine`)
- Modify: `tests/test_state_engine.py`

**Interfaces:**
- Produces: `StateEngine.completed_m1_window(n: int) -> list[M1BarState]`
  — consumed by Task 22 (`live_engine.py`).

- [ ] **Step 1: Add a failing test to `tests/test_state_engine.py`**

```python
def test_completed_m1_window_returns_bounded_recent_bars():
    engine = StateEngine("GOLD.i#")
    # feed enough synthetic ticks to build several completed M1 bars
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(300):
        t = base + timedelta(seconds=i * 2)
        engine.on_tick(Tick(symbol="GOLD.i#", market_timestamp=t,
                             ingestion_timestamp=t, bid=2000.0 + i * 0.01,
                             ask=2000.2 + i * 0.01, mid=2000.1 + i * 0.01,
                             spread=0.2, source="synthetic_replay", internal_seq=i))
    window = engine.completed_m1_window(3)
    assert len(window) <= 3
    assert all(bar.complete for bar in window)
    # fewer than requested during warmup
    fresh = StateEngine("GOLD.i#")
    assert fresh.completed_m1_window(5) == []
```

(Add the necessary `from datetime import timedelta` import at the top of
the test file if not already present; check the file's existing imports
first.)

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_state_engine.py`
Expected: FAIL with `AttributeError: 'StateEngine' object has no attribute 'completed_m1_window'`

- [ ] **Step 3: Add the method to `market/state_engine.py`**, inside the
  `StateEngine` class, after `bootstrap`:

```python
    def completed_m1_window(self, n: int) -> list:
        """Last n completed M1 bars (fewer during warmup), oldest first.
        Read-only view into the same bounded ring buffer MarketState's
        completed_m1 field already draws its single latest entry from --
        no separate storage, no unbounded growth (Phase 3 spec section 6)."""
        if n <= 0:
            return []
        return list(self.completed_m1)[-n:]
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_state_engine.py`
Expected: PASS (all pre-existing tests in this file plus the new one).

- [ ] **Step 5: Re-run boundary + feed_listener tests for regression**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_boundary.py && /home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feed_listener.py`
Expected: both PASS unchanged.

- [ ] **Step 6: Commit**

```bash
git add market/state_engine.py tests/test_state_engine.py
git commit -m "Add StateEngine.completed_m1_window() for bounded live feature-fabric access"
```

---

## Task 19: `StatefulKalman` incremental class

**Files:**
- Modify: `features/kalman.py` (append a class, existing
  `kalman_local_level` function untouched)
- Test: `tests/test_kalman_incremental.py`

**Interfaces:**
- Produces: `features.kalman.StatefulKalman` with `.update(price: float) -> tuple[float, float, float]`
  (returns `(level, velocity, residual)`), consumed by Task 22.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kalman_incremental.py
"""python3 tests/test_kalman_incremental.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from features.kalman import kalman_local_level, StatefulKalman


def test_stateful_kalman_matches_batch():
    rng = np.random.default_rng(0)
    prices = 2000 + np.cumsum(rng.normal(0, 1, 200))
    q, r = 1e-5, 1.0

    batch_level, batch_velocity, batch_residual = kalman_local_level(prices, q, r)

    kf = StatefulKalman(q=q, r=r)
    live_level, live_velocity, live_residual = [], [], []
    for p in prices:
        level, velocity, residual = kf.update(p)
        live_level.append(level); live_velocity.append(velocity); live_residual.append(residual)

    assert np.allclose(batch_level, live_level, rtol=1e-9, atol=1e-12)
    assert np.allclose(batch_velocity, live_velocity, rtol=1e-9, atol=1e-12)
    assert np.allclose(batch_residual, live_residual, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    test_stateful_kalman_matches_batch()
    print("tests/test_kalman_incremental.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_kalman_incremental.py`
Expected: FAIL with `ImportError: cannot import name 'StatefulKalman'`

- [ ] **Step 3: Append `StatefulKalman` to `features/kalman.py`**

```python
class StatefulKalman:
    """O(1)-per-update incremental version of kalman_local_level -- same
    2-state (level, velocity) constant-velocity model, same math, just
    persisting state across calls instead of looping over a full array.
    First .update() call seeds state from that first price (matches
    kalman_local_level's row-0 initialization)."""

    def __init__(self, q: float, r: float):
        self.q = q
        self.r = r
        self._initialized = False
        self.x0 = 0.0
        self.x1 = 0.0
        self.p00, self.p01, self.p10, self.p11 = 1.0, 0.0, 0.0, 1.0

    def update(self, price: float) -> tuple:
        if not self._initialized:
            self.x0, self.x1 = price, 0.0
            self._initialized = True
            return self.x0, self.x1, 0.0

        q, r = self.q, self.r
        x0_pred = self.x0 + self.x1
        x1_pred = self.x1
        p00_pred = self.p00 + self.p01 + self.p10 + self.p11 + q
        p01_pred = self.p01 + self.p11
        p10_pred = self.p10 + self.p11
        p11_pred = self.p11 + q

        y = price - x0_pred
        s = p00_pred + r
        k0 = p00_pred / s
        k1 = p10_pred / s

        self.x0 = x0_pred + k0 * y
        self.x1 = x1_pred + k1 * y
        self.p00 = (1 - k0) * p00_pred
        self.p01 = (1 - k0) * p01_pred
        self.p10 = p10_pred - k1 * p00_pred
        self.p11 = p11_pred - k1 * p01_pred

        return self.x0, self.x1, y
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_kalman_incremental.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/kalman.py tests/test_kalman_incremental.py
git commit -m "Add StatefulKalman: O(1) incremental version of kalman_local_level, verified identical to batch"
```

---

## Task 20: `DailyBuffer`

**Files:**
- Create: `features/daily_buffer.py`
- Test: `tests/test_daily_buffer.py`

**Interfaces:**
- Consumes: `cfg.features.daily_buffer_bootstrap_csv`,
  `cfg.features.daily_buffer_size` (Task 4).
- Produces: `features.daily_buffer.DailyBuffer` with
  `.bootstrap_from_csv(csv_path, size)`, `.record(day: date, values: dict)`,
  `.series(key: str) -> pd.Series` (indexed by day, ascending) — consumed
  by Task 22 for `vol_percentile_252`/`spread_percentile_252`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_daily_buffer.py
"""python3 tests/test_daily_buffer.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from features.daily_buffer import DailyBuffer


def test_bootstrap_from_real_csv():
    buf = DailyBuffer(size=252)
    buf.bootstrap_from_csv("data/gold_seed.csv", value_cols=["close", "spread"])
    s = buf.series("close")
    assert len(s) <= 252
    assert s.index.is_monotonic_increasing


def test_record_and_ring_eviction():
    buf = DailyBuffer(size=3)
    buf.record(date(2026, 1, 1), {"x": 1.0})
    buf.record(date(2026, 1, 2), {"x": 2.0})
    buf.record(date(2026, 1, 3), {"x": 3.0})
    buf.record(date(2026, 1, 4), {"x": 4.0})  # evicts 2026-01-01
    s = buf.series("x")
    assert len(s) == 3
    assert list(s.values) == [2.0, 3.0, 4.0]


if __name__ == "__main__":
    test_bootstrap_from_real_csv()
    test_record_and_ring_eviction()
    print("tests/test_daily_buffer.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_daily_buffer.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `features/daily_buffer.py`**

```python
"""Bounded daily-resampled ring buffer for the handful of ~252-observation
features (vol_percentile_252, spread_percentile_252) that need more
history than the live process's bounded M1 buffer holds -- WITHOUT
loading the 6.7-year historical CSV into the live process (spec section
6). Bootstrapped once at live_engine startup from the small rolling
gold_seed.csv (~2.5mo), refreshed once/day thereafter from live values."""
from collections import deque

import pandas as pd


class DailyBuffer:
    def __init__(self, size: int):
        self.size = size
        self._days = deque(maxlen=size)
        self._values: dict = {}  # key -> deque(maxlen=size), parallel to _days

    def bootstrap_from_csv(self, csv_path: str, value_cols: list) -> None:
        df = pd.read_csv(csv_path, parse_dates=["time"])
        daily = df.set_index("time")[value_cols].resample("1D").last().dropna(how="all")
        daily = daily.tail(self.size)
        for day, row in daily.iterrows():
            self.record(day.date(), {c: float(row[c]) for c in value_cols if pd.notna(row[c])})

    def record(self, day, values: dict) -> None:
        if self._days and self._days[-1] == day:
            for k, v in values.items():
                self._values.setdefault(k, deque(maxlen=self.size))
                if self._values[k]:
                    self._values[k][-1] = v
                else:
                    self._values[k].append(v)
            return
        self._days.append(day)
        for k, v in values.items():
            self._values.setdefault(k, deque(maxlen=self.size))
            self._values[k].append(v)

    def series(self, key: str) -> pd.Series:
        vals = self._values.get(key, deque())
        return pd.Series(list(vals), index=list(self._days)[-len(vals):])
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_daily_buffer.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/daily_buffer.py tests/test_daily_buffer.py
git commit -m "Add DailyBuffer: bounded daily ring buffer for 252-window live features, no 6.7yr load"
```

---

## Task 21: New live-only family (`microstructure_live.py`)

**Files:**
- Create: `features/microstructure_live.py`
- Create: `features/registry/microstructure_live/*.json` (~6 files)
- Test: `tests/test_microstructure_live.py`

**Interfaces:**
- Consumes: `contracts.market_state.MarketState` (Phase 2, unchanged).
- Produces: `features.microstructure_live.TickActivityTracker` — a small
  stateful class (own tick-level ring buffers, same pattern as
  `market/state_engine.py`'s `_tick_times_60s`), with
  `.update(state: MarketState) -> dict[str, float]`, consumed by Task 22.

Per spec section 7: these have no historical analogue and are not forced
into the production schema. `status="OPTIONAL"` pending Phase 4
evaluation, no `evidence_ref`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_microstructure_live.py
"""python3 tests/test_microstructure_live.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta

from contracts.market_state import MarketState, FeedHealthState, DataQuality
from features.microstructure_live import TickActivityTracker


def _state(seq, ts, bid, ask, spread):
    return MarketState(
        symbol="GOLD.i#", source="synthetic_replay", sequence=seq,
        market_timestamp=ts, ingestion_timestamp=ts, processing_timestamp=ts,
        bid=bid, ask=ask, mid=(bid + ask) / 2, spread=spread,
        last=None, last_quality=DataQuality.UNAVAILABLE,
        tick_count_60s=seq, tick_count_300s=seq, tick_rate_per_sec=1.0,
        current_m1=None, completed_m1=None,
        realized_vol_60s=None, spread_mean_60s=spread, spread_std_60s=0.0,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.01,
        feed_latency_sec=0.01, state_update_latency_sec=0.0001,
    )


def test_tick_activity_tracker_basic():
    tracker = TickActivityTracker()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = None
    for i in range(20):
        ts = base + timedelta(milliseconds=i * 300)
        state = _state(i, ts, 2000.0 + i * 0.01, 2000.2 + i * 0.01, 0.2 + (i % 3) * 0.01)
        out = tracker.update(state)
    assert out is not None
    assert set(out.keys()) == {
        "spread_change_live", "spread_shock_zscore_live",
        "tick_interarrival_mean_60s", "tick_interarrival_std_60s",
        "tick_arrival_burstiness_60s",
    }
    for v in out.values():
        assert v is None or isinstance(v, float)


if __name__ == "__main__":
    test_tick_activity_tracker_basic()
    print("tests/test_microstructure_live.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_microstructure_live.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `features/microstructure_live.py`**

```python
"""New live-only microstructure family (spec section 7) -- exists only
because Phase 2 provides a real tick-level bid/ask stream; no 6.7-year
historical analogue exists or can exist. Implemented now, evaluated
against real targets in Phase 4 (registry status=OPTIONAL, no
evidence_ref). Same small-ring-buffer pattern as
market/state_engine.py's _tick_times_60s/_spreads."""
from collections import deque

TICK_WINDOW_SEC = 60.0


class TickActivityTracker:
    def __init__(self):
        self._spreads = deque()
        self._times = deque()
        self._last_spread = None

    def update(self, state) -> dict:
        ts = state.market_timestamp.timestamp()
        spread = state.spread

        spread_change = None if self._last_spread is None else spread - self._last_spread
        self._last_spread = spread

        self._times.append(ts)
        self._spreads.append(spread)
        while self._times and ts - self._times[0] > TICK_WINDOW_SEC:
            self._times.popleft()
            self._spreads.popleft()

        spread_shock = None
        if len(self._spreads) > 1:
            mean = sum(self._spreads) / len(self._spreads)
            var = sum((x - mean) ** 2 for x in self._spreads) / len(self._spreads)
            std = var ** 0.5
            spread_shock = (spread - mean) / std if std > 1e-12 else 0.0

        interarrivals = [t2 - t1 for t1, t2 in zip(self._times, list(self._times)[1:])]
        interarrival_mean = sum(interarrivals) / len(interarrivals) if interarrivals else None
        interarrival_std = None
        burstiness = None
        if len(interarrivals) > 1:
            m = interarrival_mean
            var = sum((x - m) ** 2 for x in interarrivals) / len(interarrivals)
            interarrival_std = var ** 0.5
            burstiness = (interarrival_std - m) / (interarrival_std + m) if (interarrival_std + m) > 1e-12 else 0.0

        return {
            "spread_change_live": spread_change,
            "spread_shock_zscore_live": spread_shock,
            "tick_interarrival_mean_60s": interarrival_mean,
            "tick_interarrival_std_60s": interarrival_std,
            "tick_arrival_burstiness_60s": burstiness,
        }
```

Note: the test expects 5 keys but the fixture assertion above lists 5
(`spread_change_live`, `spread_shock_zscore_live`,
`tick_interarrival_mean_60s`, `tick_interarrival_std_60s`,
`tick_arrival_burstiness_60s`) — implementation matches.

- [ ] **Step 4: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_microstructure_live.py`
Expected: PASS.

- [ ] **Step 5: Write 5 registry JSON files**, all
  `family="microstructure_live"`, `historical_coverage="LIVE_ONLY"`,
  `status="OPTIONAL"`, `evidence_ref=null`,
  `status_reason="no historical evidence exists or can exist for this feature; live data only since 2026-08-18 (Phase 2 live verification date)"`,
  `update_trigger="TICK"`, `live_compatible=true`,
  `source_module="features.microstructure_live.TickActivityTracker"`,
  `causal=true` (each only uses state up to and including the current
  tick).

- [ ] **Step 6: Commit**

```bash
git add features/microstructure_live.py features/registry/microstructure_live/ tests/test_microstructure_live.py
git commit -m "Add new live-only microstructure_live family: real tick-level spread/arrival dynamics, no historical analogue"
```

---

## Task 22: `features/live_engine.py`

**Files:**
- Create: `features/live_engine.py`
- Test: `tests/test_live_engine.py`

**Interfaces:**
- Consumes: `market.feed_listener.FeedListener` (Phase 2),
  `market.state_engine.StateEngine.completed_m1_window` (Task 18),
  `features.daily_buffer.DailyBuffer` (Task 20),
  `features.kalman.StatefulKalman` (Task 19),
  `features.microstructure_live.TickActivityTracker` (Task 21), every
  `compute_<family>` function (Tasks 5-14), `features.registry.load_all`
  (Task 3).
- Produces: `features.live_engine.LiveFeatureEngine` with
  `.on_tick(state: MarketState) -> dict` (updates tick-triggered
  features only) and `.on_m1_close(bars: list) -> dict` (recomputes
  M1_CLOSE-triggered features from the bounded window), each returning a
  `{feature_id: (value, quality)}` snapshot where `quality` is one of
  `"VALID"`, `"WARMING_UP"`, `"UNAVAILABLE"`. Additive only — not called
  from `app/engine.py`'s decision loop (spec section 6, matches Phase 2's
  `get_market_state()` precedent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_engine.py
"""python3 tests/test_live_engine.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta

from contracts.tick import Tick
from market.state_engine import StateEngine
from features.live_engine import LiveFeatureEngine


def test_live_engine_produces_snapshot_after_enough_bars():
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)  # no bootstrap in this unit test

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = None
    for i in range(20000):  # enough ticks to cross several M1 boundaries
        t = base + timedelta(seconds=i * 2)
        tick = Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t,
                    bid=2000.0 + (i % 100) * 0.01, ask=2000.2 + (i % 100) * 0.01,
                    mid=2000.1, spread=0.2, source="synthetic_replay", internal_seq=i)
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        if state.current_m1 is not None and state.completed_m1 is not None:
            snapshot = live.on_m1_close(engine.completed_m1_window(480))

    assert snapshot is not None
    assert len(snapshot) > 0
    for feature_id, (value, quality) in snapshot.items():
        assert quality in ("VALID", "WARMING_UP", "UNAVAILABLE")


if __name__ == "__main__":
    test_live_engine_produces_snapshot_after_enough_bars()
    print("tests/test_live_engine.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_live_engine.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `features/live_engine.py`**

```python
"""Trigger-driven live feature engine on top of MarketState (spec section
6). M1_CLOSE-triggered families recompute via the SAME compute_<family>
functions replay_engine.py uses, against a bounded window pulled from
StateEngine.completed_m1_window() -- O(window), not O(1), not
O(history), matching Phase 2's own established performance bar.
TICK-triggered features use small dedicated ring buffers
(TickActivityTracker) or StatefulKalman. Additive only: never called from
app/engine.py's decision loop."""
import numpy as np
import pandas as pd

from features._shared import build_shared_inputs
from features.features import build_tier1_features
from features.returns_dynamics import compute_returns_dynamics
from features.volatility_dynamics import compute_volatility_dynamics
from features.jump_detection import compute_jump_detection
from features.distribution_info import compute_distribution_info
from features.market_geometry import compute_market_geometry
from features.persistence import compute_persistence
from features.temporal import compute_temporal
from features.microstructure_history import compute_microstructure_history
from features.regime_state import compute_regime_state
from features.first_passage import compute_first_passage
from features.microstructure_live import TickActivityTracker
from features.daily_buffer import DailyBuffer
from features.registry import load_all

# StatefulKalman (Task 19) is NOT used here: kalman_residual_z etc. are
# baseline_v1 features (features/features.py, unmodified), out of
# live_engine.py's scope -- they're already computed live via
# app/engine.py's existing build_features() call, separately. Persistence
# family's residual_mean_reversion_60 gets kalman_residual_z from the
# bounded-window batch build_tier1_features() call below (O(window),
# same as every other M1_CLOSE family), which is already fast enough at
# window<=480 -- no O(1) path is needed for that. StatefulKalman remains
# a tested, standalone utility (Task 19) for a future consumer that
# genuinely needs per-tick O(1) kalman state; nothing in this plan calls
# it yet, so it is not instantiated here.

M1_LIVE_FAMILIES = (
    compute_returns_dynamics, compute_volatility_dynamics,
    compute_distribution_info, compute_market_geometry, compute_persistence,
    compute_temporal, compute_first_passage,
)
CUSUM_K_LIVE = 2.5  # matches features.replay_engine.CUSUM_K -- confirm identical at implementation time


class LiveFeatureEngine:
    def __init__(self, state_engine, daily_bootstrap_csv: str = None, daily_buffer_size: int = 252):
        self.state_engine = state_engine
        self.tick_tracker = TickActivityTracker()
        self.daily_buffer = DailyBuffer(size=daily_buffer_size)
        if daily_bootstrap_csv:
            self.daily_buffer.bootstrap_from_csv(daily_bootstrap_csv, value_cols=["close", "spread"])
        self._descriptors = {d.feature_id: d for d in load_all()}
        self._last_tick_snapshot = {}
        self._last_m1_snapshot = {}

    def on_tick(self, state) -> dict:
        live_vals = self.tick_tracker.update(state)
        out = {}
        for feature_id, value in live_vals.items():
            quality = "VALID" if value is not None else "WARMING_UP"
            out[feature_id] = (value, quality)
        self._last_tick_snapshot = out
        return {**self._last_tick_snapshot, **self._last_m1_snapshot}

    def on_m1_close(self, bars: list) -> dict:
        if len(bars) < 2:
            return {**self._last_tick_snapshot, **self._last_m1_snapshot}

        df = pd.DataFrame({
            "time": [b.start_time for b in bars],
            "open": [b.open for b in bars], "high": [b.high for b in bars],
            "low": [b.low for b in bars], "close": [b.close for b in bars],
            "tick_volume": [b.tick_count for b in bars],
            "spread": [None] * len(bars),  # not tracked per-bar live -- spread-history features stay WARMING_UP/UNAVAILABLE
        })
        base_feat = build_tier1_features(df)
        shared = build_shared_inputs(df, base_feat)

        merged = {}
        for compute_fn in M1_LIVE_FAMILIES:
            merged.update(compute_fn(shared))
        jump = compute_jump_detection(shared, CUSUM_K_LIVE)
        merged.update({k: v for k, v in jump.items() if not k.startswith("_")})
        micro_h = compute_microstructure_history(shared)
        merged.update(micro_h)
        upstream = {**jump, **{k: v for k, v in merged.items() if k in
                                ("shannon_entropy_returns_60", "tick_volume_zscore_60")}}
        regime = compute_regime_state(shared, upstream)
        merged.update(regime)

        # DailyBuffer wiring: compute_volatility_dynamics's own internal
        # daily resample (inside compute_fn above) only sees the bounded
        # ~8h window, so its vol_percentile_252 is structurally always
        # NaN live -- that's honest (WARMING_UP forever) but wastes the
        # DailyBuffer built in Task 20. Override with the real daily-buffer
        # backed value once enough days have accumulated:
        today = bars[-1].start_time.date()
        if len(shared.ewma_vol) and not np.isnan(shared.ewma_vol[-1]):
            self.daily_buffer.record(today, {"close": float(shared.ewma_vol[-1])})
        vol_hist = self.daily_buffer.series("close")
        if len(vol_hist) >= 60 and "vol_percentile_252" in merged:
            merged["vol_percentile_252"] = np.append(
                merged["vol_percentile_252"][:-1], vol_hist.rank(pct=True).iloc[-1])

        out = {}
        for feature_id, values in merged.items():
            descriptor = self._descriptors.get(feature_id)
            if descriptor is not None and not descriptor.live_compatible:
                out[feature_id] = (None, "UNAVAILABLE")
                continue
            last_val = values[-1] if hasattr(values, "__len__") and len(values) else None
            if last_val is None or (isinstance(last_val, float) and last_val != last_val):  # NaN check
                out[feature_id] = (None, "WARMING_UP")
            else:
                out[feature_id] = (float(last_val), "VALID")
        self._last_m1_snapshot = out
        return {**self._last_tick_snapshot, **self._last_m1_snapshot}
```

- [ ] **Step 4: Run to verify it passes, debug real integration issues as
  they surface** (constructing a valid bounded pseudo-DataFrame from
  `M1BarState` objects and running the real family functions against it
  end-to-end is the first true integration point in this plan — expect to
  find and fix real issues, e.g. minimum-row-count requirements for
  `.rolling(240)`-style windows when `bars` is short; these are genuine
  bugs to fix in `live_engine.py`, not to be masked).

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_live_engine.py`

- [ ] **Step 5: Commit**

```bash
git add features/live_engine.py tests/test_live_engine.py
git commit -m "Add features/live_engine.py: trigger-driven live feature snapshot, additive, not wired into decision loop"
```

---

## Task 23: Live vs replay numerical equivalence test

**Files:**
- Create: `tests/test_live_replay_equivalence.py`

**Interfaces:**
- Consumes: `features.live_engine.LiveFeatureEngine`,
  `features.replay_engine.build_candidate_features`,
  `market.synthetic_replay.generate_ticks` (Phase 2, existing).

- [ ] **Step 1: Write the test**

```python
"""python3 tests/test_live_replay_equivalence.py -- feeds the same
synthetic tick sequence through live_engine.py (bounded, trigger-driven)
and replay_engine.py (batch), asserts the live snapshot at each M1 close
matches the batch computation on the equivalent prefix (spec section 8)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from market.state_engine import StateEngine
from market.synthetic_replay import generate_ticks
from features.live_engine import LiveFeatureEngine
from features.features import build_features
from features.replay_engine import build_candidate_features


def test_live_matches_replay_at_m1_close():
    ticks = generate_ticks(n=6000, seed=42)
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    last_snapshot = None
    for tick in ticks:
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        if state.completed_m1 is not None:
            last_snapshot = live.on_m1_close(engine.completed_m1_window(480))

    assert last_snapshot is not None

    bars = engine.completed_m1_window(480)
    df = pd.DataFrame({
        "time": [b.start_time for b in bars], "open": [b.open for b in bars],
        "high": [b.high for b in bars], "low": [b.low for b in bars],
        "close": [b.close for b in bars], "tick_volume": [b.tick_count for b in bars],
        "spread": [0.2] * len(bars),
    })
    base = build_features(df)
    replay = build_candidate_features(df, base)

    checked = 0
    for feature_id, (value, quality) in last_snapshot.items():
        if quality != "VALID" or feature_id not in replay.columns:
            continue
        expected = replay[feature_id].to_numpy(dtype=np.float64)[-1]
        if np.isnan(expected):
            continue
        assert abs(value - expected) < 1e-6 or np.isclose(value, expected, rtol=1e-6), (
            f"{feature_id}: live={value} replay={expected}")
        checked += 1
    assert checked > 0, "no VALID features overlapped with replay columns -- test is not exercising anything"


if __name__ == "__main__":
    test_live_matches_replay_at_m1_close()
    print("tests/test_live_replay_equivalence.py: OK")
```

- [ ] **Step 2: Run and reconcile any real discrepancies**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_live_replay_equivalence.py`
Any real mismatch (e.g. a boundary/warmup-count difference between the
live bounded window and the replay full DataFrame) is a genuine bug —
fix in `live_engine.py`, do not loosen the tolerance to force a pass.
Expected after fixes: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_live_replay_equivalence.py
git commit -m "Add live/replay numerical equivalence test -- verifies same math, different buffer source"
```

---

## Task 24: Warmup / missing-data state tests

**Files:**
- Create: `tests/test_feature_warmup_missing.py`

- [ ] **Step 1: Write the test**

```python
"""python3 tests/test_feature_warmup_missing.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta

from contracts.tick import Tick
from market.state_engine import StateEngine
from features.live_engine import LiveFeatureEngine


def test_short_history_reports_warming_up_not_plausible_numbers():
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = None
    for i in range(200):  # only ~3 M1 bars -- nowhere near the 240/252-bar windows
        t = base + timedelta(seconds=i)
        tick = Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t,
                    bid=2000.0, ask=2000.2, mid=2000.1, spread=0.2,
                    source="synthetic_replay", internal_seq=i)
        state = engine.on_tick(tick)
        if state and state.completed_m1 is not None:
            snapshot = live.on_m1_close(engine.completed_m1_window(480))
    assert snapshot is not None
    long_window_features = ["ret_240", "return_skew_240", "hurst_240", "changepoint_intensity_240"]
    for fid in long_window_features:
        if fid in snapshot:
            value, quality = snapshot[fid]
            assert quality in ("WARMING_UP", "UNAVAILABLE"), f"{fid} claimed {quality} with only 3 bars of history"


def test_spread_history_features_marked_unavailable_live():
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = None
    for i in range(2000):
        t = base + timedelta(seconds=i)
        tick = Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t,
                    bid=2000.0 + i * 0.001, ask=2000.2 + i * 0.001, mid=2000.1,
                    spread=0.2, source="synthetic_replay", internal_seq=i)
        state = engine.on_tick(tick)
        if state and state.completed_m1 is not None:
            snapshot = live.on_m1_close(engine.completed_m1_window(480))
    assert snapshot is not None
    for fid in ("spread_change_1", "spread_percentile_252", "spread_volatility_60"):
        if fid in snapshot:
            value, quality = snapshot[fid]
            assert quality == "UNAVAILABLE", f"{fid} should be UNAVAILABLE live (no per-bar spread history), got {quality}"


if __name__ == "__main__":
    test_short_history_reports_warming_up_not_plausible_numbers()
    test_spread_history_features_marked_unavailable_live()
    print("tests/test_feature_warmup_missing.py: OK")
```

- [ ] **Step 2: Run and fix any feature that silently returns a plausible
  number instead of a quality flag during warmup** (real bug if found —
  fix in `live_engine.py`'s quality-flag logic, Task 22).

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_warmup_missing.py`

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_warmup_missing.py
git commit -m "Add warmup/missing-data state tests -- no silent plausible-looking numbers before real data exists"
```

---

## Task 25: NaN/inf numerical-safety tests

**Files:**
- Create: `tests/test_feature_numerical_safety.py`

- [ ] **Step 1: Write the test**

```python
"""python3 tests/test_feature_numerical_safety.py -- degenerate inputs
(zero-variance windows, constant price, insufficient samples) must
produce NaN or an explicit quality flag, never inf or silent corruption."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.features import build_features
from features.replay_engine import build_candidate_features


def test_constant_price_no_inf():
    n = 400
    close = np.full(n, 2000.0)
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({"time": time, "open": close, "high": close, "low": close,
                        "close": close, "tick_volume": np.zeros(n), "spread": np.full(n, 20.0)})
    base = build_features(df)
    feat = build_candidate_features(df, base)
    for col in feat.columns:
        if col == "time":
            continue
        vals = feat[col].to_numpy(dtype=np.float64)
        assert not np.isinf(vals).any(), f"{col} produced inf on constant-price input"


def test_zero_tick_volume_no_inf():
    n = 400
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({"time": time, "open": close, "high": close + 0.5, "low": close - 0.5,
                        "close": close, "tick_volume": np.zeros(n), "spread": np.full(n, 20.0)})
    base = build_features(df)
    feat = build_candidate_features(df, base)
    for col in feat.columns:
        if col == "time":
            continue
        vals = feat[col].to_numpy(dtype=np.float64)
        assert not np.isinf(vals).any(), f"{col} produced inf on zero-tick_volume input"


if __name__ == "__main__":
    test_constant_price_no_inf()
    test_zero_tick_volume_no_inf()
    print("tests/test_feature_numerical_safety.py: OK")
```

- [ ] **Step 2: Run and fix any real inf-producing division** (e.g. a
  ratio family dividing by a rolling std that's exactly 0 on constant
  price — most are already guarded with `np.clip(..., 1e-9, None)` or
  `np.where(denom > eps, ..., np.nan)` per the original code, but verify
  for real rather than assuming).

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_numerical_safety.py`

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_numerical_safety.py
git commit -m "Add numerical-safety tests: constant price and zero tick_volume must never produce inf"
```

---

## Task 26: Redundancy/stability diagnostics (`features/registry/diagnostics.py`)

**Files:**
- Create: `features/registry/diagnostics.py`
- Test: `tests/test_feature_diagnostics.py`

**Interfaces:**
- Produces: `correlation_redundancy(df: pd.DataFrame, threshold: float = 0.95) -> list[tuple[str, str, float]]`,
  `distribution_stability(series_a: pd.Series, series_b: pd.Series) -> dict`
  (mean/std/skew comparison between two periods — a simple, real drift
  check, not a re-run of the OOF importance pipeline).

Generalizes `research/v3_feature_selection.py`'s correlation-pruning
methodology (lines 56-59, 96-104 of that file) into reusable tooling, per
spec section 9 — applied fresh to the new live-only family (Task 21),
NOT re-run against the already-evidenced 92 candidates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_diagnostics.py
"""python3 tests/test_feature_diagnostics.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.registry.diagnostics import correlation_redundancy, distribution_stability


def test_correlation_redundancy_detects_duplicate_column():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    df = pd.DataFrame({"a": a, "b": a * 2 + 0.0001, "c": rng.normal(0, 1, 500)})
    pairs = correlation_redundancy(df, threshold=0.95)
    names = {(p[0], p[1]) for p in pairs}
    assert ("a", "b") in names or ("b", "a") in names


def test_distribution_stability_flags_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 500))
    b = pd.Series(rng.normal(5, 1, 500))  # large mean shift
    result = distribution_stability(a, b)
    assert result["mean_shift"] > 4.0


if __name__ == "__main__":
    test_correlation_redundancy_detects_duplicate_column()
    test_distribution_stability_flags_shift()
    print("tests/test_feature_diagnostics.py: OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_diagnostics.py`

- [ ] **Step 3: Write `features/registry/diagnostics.py`**

```python
"""Reusable redundancy/stability diagnostics -- generalizes
research/v3_feature_selection.py's correlation-pruning methodology (spec
section 9). Run fresh against NEW features (e.g. microstructure_live,
Task 21); NOT re-run against the already-OOF-evidenced 92 candidates."""
import numpy as np
import pandas as pd


def correlation_redundancy(df: pd.DataFrame, threshold: float = 0.95) -> list:
    corr = df.corr().abs()
    pairs = []
    cols = list(df.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val) and val > threshold:
                pairs.append((a, b, float(val)))
    return pairs


def distribution_stability(series_a: pd.Series, series_b: pd.Series) -> dict:
    a, b = series_a.dropna(), series_b.dropna()
    mean_a, mean_b = a.mean(), b.mean()
    std_a, std_b = a.std(), b.std()
    pooled_std = ((std_a ** 2 + std_b ** 2) / 2) ** 0.5
    mean_shift = abs(mean_a - mean_b) / pooled_std if pooled_std > 1e-12 else 0.0
    return {
        "mean_a": float(mean_a), "mean_b": float(mean_b),
        "std_a": float(std_a), "std_b": float(std_b),
        "mean_shift": float(mean_shift),
        "std_ratio": float(std_b / std_a) if std_a > 1e-12 else float("nan"),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_diagnostics.py`

- [ ] **Step 5: Apply the diagnostics to the new live-only family** —
  write a small script `research/microstructure_live_diagnostics.py`
  that runs `TickActivityTracker` over `synthetic_replay.generate_ticks`
  output, builds a DataFrame of its 5 outputs, and calls
  `correlation_redundancy` on it; print the result. This is real evidence
  generation for a family that has none yet (spec section 7/9) — run it
  and record the printed output in Task 30's final report, don't just
  write the script.

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 research/microstructure_live_diagnostics.py`

- [ ] **Step 6: Commit**

```bash
git add features/registry/diagnostics.py tests/test_feature_diagnostics.py research/microstructure_live_diagnostics.py
git commit -m "Add reusable redundancy/stability diagnostics, apply fresh to the new live-only family"
```

---

## Task 27: Boundary test extension

**Files:**
- Modify: `tests/test_boundary.py`

**Interfaces:**
- Produces: `test_features_never_imports_learning_or_research()`.

- [ ] **Step 1: Add the test function**, using the existing
  `_check_no_forbidden_imports` helper already in the file:

```python
def test_features_never_imports_learning_or_research():
    _check_no_forbidden_imports("features")
```

And add it to the `if __name__ == "__main__":` block's call list.

- [ ] **Step 2: Run to verify it passes**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_boundary.py`
Expected: PASS — if it fails, it means some family module (most likely
Task 7's jump_detection.py) still has a stray `learning`/`research`
import; fix that module, don't weaken this test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_boundary.py
git commit -m "Extend boundary test: features/ must never import learning/research"
```

---

## Task 28: Performance benchmark

**Files:**
- Create: `tests/test_feature_performance.py`

**Interfaces:**
- Consumes: `features.live_engine.LiveFeatureEngine`,
  `market.synthetic_replay.generate_ticks`.

- [ ] **Step 1: Write the test**, following the two-pass (timing,
  then separate shorter tracemalloc) pattern already established in
  `tests/test_performance.py` from Phase 2 — read that file first to
  match its structure exactly.

```python
"""python3 tests/test_feature_performance.py -- [SYNTHETIC] benchmark of
LiveFeatureEngine.on_m1_close latency/throughput, same two-pass
timing/memory-separation pattern as Phase 2's test_performance.py."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from market.state_engine import StateEngine
from market.synthetic_replay import generate_ticks
from features.live_engine import LiveFeatureEngine

N_TICKS = 20000


def test_m1_close_latency_synthetic():
    ticks = generate_ticks(n=N_TICKS, seed=7)
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    m1_close_latencies_us = []
    for tick in ticks:
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        if state.completed_m1 is not None:
            t0 = time.perf_counter()
            live.on_m1_close(engine.completed_m1_window(480))
            m1_close_latencies_us.append((time.perf_counter() - t0) * 1e6)

    assert len(m1_close_latencies_us) > 5, "not enough M1 closes in this synthetic run to measure"
    arr = np.array(m1_close_latencies_us)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"[SYNTHETIC] on_m1_close latency over {len(arr)} bar closes: "
          f"p50={p50:.0f}us p95={p95:.0f}us p99={p99:.0f}us")
    assert p99 < 2_000_000, f"on_m1_close p99={p99:.0f}us exceeds 2s budget"


if __name__ == "__main__":
    test_m1_close_latency_synthetic()
    print("tests/test_feature_performance.py: OK")
```

- [ ] **Step 2: Run, record real numbers, raise the threshold or
  optimize if it fails** (matches Phase 2's precedent: two real
  performance bugs were found and fixed by this exact kind of test, not
  assumed fine).

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 tests/test_feature_performance.py`
Record the printed p50/p95/p99 for Task 30's report.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_performance.py
git commit -m "Add live_engine.py performance benchmark, [SYNTHETIC]-labeled"
```

---

## Task 29: Documentation

**Files:**
- Modify: `docs/ARCHITECTURE.md` (append new section)
- Create: `features/README.md`

- [ ] **Step 1: Write `features/README.md`** covering: the family module
  list and what each computes (one line each), the registry layout,
  `replay_engine.py` vs `live_engine.py` and why they share function
  bodies, the `DailyBuffer`/`StatefulKalman`/`TickActivityTracker`
  live-only additions, and a pointer to
  `docs/superpowers/specs/2026-08-19-golex-v3-phase3-feature-fabric-design.md`
  for the full design — same style as the existing `market/README.md`.

- [ ] **Step 2: Append a "## Phase 3: Quantitative Feature Fabric"
  section to `docs/ARCHITECTURE.md`** covering: the architecture diagram
  (from the design spec section 3), registry design, the live/replay
  equivalence approach and Task 23's real verification result, historical
  coverage findings (Task 1's real measured numbers — read
  `docs/ARCHITECTURE.md`'s existing Phase 2 section first to match its
  style/tone), the causality-test gap that was found and closed (Task
  17), the new live-only family (Task 21) and its diagnostics result
  (Task 26 Step 5's real printed output), explicit model-routing
  compatibility (Task 15's `build_schema` demonstration), and the two
  real performance numbers from Task 28.

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md features/README.md
git commit -m "Document Phase 3: quantitative feature fabric architecture, registry, live/replay equivalence, historical coverage"
```

---

## Task 30: Final verification sweep + completion report

**Files:** none (verification only)

- [ ] **Step 1: Run every test file created/modified in this plan**, in
  order, and confirm all PASS:

```bash
for f in tests/test_historical_coverage.py tests/test_feature_schema.py \
  tests/test_feature_registry.py tests/test_config.py \
  tests/test_returns_dynamics.py tests/test_volatility_dynamics.py \
  tests/test_jump_detection.py tests/test_distribution_info.py \
  tests/test_market_geometry.py tests/test_persistence.py \
  tests/test_temporal.py tests/test_microstructure_history.py \
  tests/test_regime_state.py tests/test_first_passage.py \
  tests/test_baseline_registry.py tests/test_replay_engine.py \
  tests/test_causality.py tests/test_state_engine.py \
  tests/test_kalman_incremental.py tests/test_daily_buffer.py \
  tests/test_microstructure_live.py tests/test_live_engine.py \
  tests/test_live_replay_equivalence.py tests/test_feature_warmup_missing.py \
  tests/test_feature_numerical_safety.py tests/test_feature_diagnostics.py \
  tests/test_boundary.py tests/test_feature_performance.py \
  tests/test_contracts.py tests/test_feed_listener.py; do
  echo "=== $f ==="
  /home/jith/.hermes/hermes-agent/venv/bin/python3 "$f" || echo "FAILED: $f"
done
```

- [ ] **Step 2: Import-check every new/touched package**

```bash
/home/jith/.hermes/hermes-agent/venv/bin/python3 -c "
import contracts.feature_schema, features.registry, features._shared
import features.returns_dynamics, features.volatility_dynamics, features.jump_detection
import features.distribution_info, features.market_geometry, features.persistence
import features.temporal, features.microstructure_history, features.regime_state
import features.first_passage, features.microstructure_live, features.daily_buffer
import features.replay_engine, features.live_engine, features.kalman
import market.state_engine, config.loader
print('all imports OK')
"
```

- [ ] **Step 3: Confirm registry completeness** — count JSON files
  across all family subdirectories, cross-check against expected totals
  (28 baseline + 19+9+7+6+13+7+14+6+7+4 = 92 candidates + 5 live-only =
  125 total).

```bash
/home/jith/.hermes/hermes-agent/venv/bin/python3 -c "
from features.registry import load_all
all_d = load_all()
print(f'{len(all_d)} registered features')
from collections import Counter
print(Counter(d.family for d in all_d))
print(Counter(d.status for d in all_d))
"
```

- [ ] **Step 4: Confirm production path untouched**

```bash
git diff HEAD~30..HEAD -- features/features.py app/engine.py app/shadow.py decision/ learning/ models/ | head -50
```

Expected: `app/engine.py` shows only Phase 2's already-committed changes
(no new diff from this plan — Phase 3 never touched it); `features/features.py`,
`decision/`, `learning/`, `models/` show zero diff from this plan's
commits.

- [ ] **Step 5: Confirm no stray processes / services still inactive**

```bash
systemctl is-active ai-engine.service gold-shadow.service gold-watchdog.timer 2>&1
ps aux | grep -E "wine|mt5_feed" | grep -v grep
```

Expected: all `inactive`/`failed` (never started this phase), no stray
processes.

- [ ] **Step 6: Compose and deliver the completion report** to the user
  in the exact P-A format from the design spec's section 13 (A through
  P), using the real output captured in Steps 1-5, Task 1's real
  historical-coverage numbers, Task 16/17/23's real equivalence and
  causality test results, Task 26 Step 5's real diagnostics output, and
  Task 28's real latency numbers. End with section P: recommend Phase 4
  only, do not implement it.
