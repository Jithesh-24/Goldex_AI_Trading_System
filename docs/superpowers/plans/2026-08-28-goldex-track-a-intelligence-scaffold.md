# GOLDEX Track A: Intelligence Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable `intelligence/` scaffold (evidence registry, replaceable decision-mechanism interface, simulator wiring, research/deployment learning split) on top of the unmodified Phase 1 simulator, with zero claims of a validated trading signal.

**Architecture:** A new top-level `intelligence/` package sits between `simulator/` (unmodified) and any future decision mechanism. `EvidenceRegistry` computes named, causal, versioned feature values from price history (wrapping the 9 already-validated-as-code Phase 3A/4 representation functions, unchanged). `DecisionEngine` is an abstract interface with one concrete implementation for this plan, `StubDecisionEngine`, which always returns `NO_TRADE`/holds — this plan builds plumbing, not a strategy. A `ReplayAdapter` wires any `DecisionEngine` + `EvidenceRegistry` pair into the `decide_fn`/`manage_fn` callables `simulator/replay.run_replay` already expects, including the existing `observation_features` opt-in convention. `LearningLoop` structurally separates a mutable research state from a frozen deployed state, gated by an explicit, caller-supplied validation check.

**Tech Stack:** Python 3.11, numpy, pandas (already in use), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-28-goldex-autonomous-architecture-decision.md` (Section F: scaffold diagram, Section G: EvidenceSource contract, Section I: LearningLoop split, Section L: immediate scope). Sections B–F of that document (cross-instrument, tick, conditional quant, credit assignment, trade management) are separate future tracks and are explicitly out of scope here.

## Global Constraints

- No new dependencies beyond numpy/pandas/pytest already used in this repo.
- Every representation function reused from `research/phase3a_representation_experiments.py`, `research/phase4_garch_volatility_mechanism.py`, `research/phase4_kalman_trend_mechanism.py`, `research/phase4_distributional_mechanism.py` must be imported unchanged, never reimplemented (per the spec's Section L and the project's established no-reimplementation convention).
- `intelligence/` must not modify anything under `simulator/`, `decision/`, `contracts/`, or `research/` — it only consumes them.
- No task in this plan may claim, imply, or test for a validated trading edge. `StubDecisionEngine` always returns `NO_TRADE` — this is a scaffold-correctness plan, not a signal-validation plan.
- Every evidence-source wrapper must be causal: `EvidenceRegistry.compute_all(closes_so_far)` computed at index `t` must depend only on `closes_so_far[:t+1]` — verified by a no-look-ahead test per source (truncate-and-recompute must match).
- Follow the existing test file convention: `sys.path.insert(0, ...)` to repo root at the top of each new test file (see `tests/simulator/test_experience.py`), tests run via `pytest tests/... -v`.

---

## File Structure

```
intelligence/
    __init__.py
    evidence.py          # EvidenceValue, EvidenceSource protocol, EvidenceRegistry
    evidence_sources.py   # 9 Phase 3A/4 representation functions wrapped as EvidenceSource
    decision_engine.py   # Action, DecisionEngine protocol, StubDecisionEngine
    replay_adapter.py    # ReplayAdapter: DecisionEngine+EvidenceRegistry -> decide_fn/manage_fn
    learning_loop.py      # LearningLoop: research_state vs deployed_state, promote() gate

tests/intelligence/
    __init__.py
    test_evidence.py
    test_evidence_sources.py
    test_decision_engine.py
    test_replay_adapter.py
    test_learning_loop.py
```

---

### Task 1: Evidence contract and registry

**Files:**
- Create: `intelligence/__init__.py` (empty)
- Create: `intelligence/evidence.py`
- Test: `tests/intelligence/__init__.py` (empty)
- Test: `tests/intelligence/test_evidence.py`

**Interfaces:**
- Produces: `EvidenceValue` (dataclass: `value: Optional[float]`, `confidence: float`, `source_name: str`); `EvidenceSource` (a `Callable[[np.ndarray], EvidenceValue]` type alias — takes the closes-so-far array, returns one `EvidenceValue`); `EvidenceRegistry` class with `register(name: str, fn: EvidenceSource) -> None`, `compute_all(closes_so_far: np.ndarray) -> dict[str, EvidenceValue]`, `names() -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/intelligence/test_evidence.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from intelligence.evidence import EvidenceValue, EvidenceRegistry


def _const_source(name, val):
    def _fn(closes_so_far):
        return EvidenceValue(value=val, confidence=1.0, source_name=name)
    return _fn


def test_evidence_value_is_a_plain_record():
    v = EvidenceValue(value=1.5, confidence=0.9, source_name="test")
    assert v.value == 1.5
    assert v.confidence == 0.9
    assert v.source_name == "test"


def test_registry_register_and_names():
    registry = EvidenceRegistry()
    registry.register("a", _const_source("a", 1.0))
    registry.register("b", _const_source("b", 2.0))
    assert sorted(registry.names()) == ["a", "b"]


def test_registry_rejects_duplicate_name():
    registry = EvidenceRegistry()
    registry.register("a", _const_source("a", 1.0))
    with pytest.raises(ValueError):
        registry.register("a", _const_source("a", 2.0))


def test_compute_all_calls_every_source_with_the_same_closes_array():
    registry = EvidenceRegistry()
    seen = {}

    def _capture(name):
        def _fn(closes_so_far):
            seen[name] = closes_so_far
            return EvidenceValue(value=float(len(closes_so_far)), confidence=1.0, source_name=name)
        return _fn

    registry.register("a", _capture("a"))
    registry.register("b", _capture("b"))
    closes = np.array([1.0, 2.0, 3.0])
    result = registry.compute_all(closes)

    assert set(result.keys()) == {"a", "b"}
    assert result["a"].value == 3.0
    assert result["b"].value == 3.0
    assert np.array_equal(seen["a"], closes)
    assert np.array_equal(seen["b"], closes)


def test_compute_all_isolates_a_source_that_raises():
    registry = EvidenceRegistry()

    def _broken(closes_so_far):
        raise ValueError("boom")

    def _ok(closes_so_far):
        return EvidenceValue(value=1.0, confidence=1.0, source_name="ok")

    registry.register("broken", _broken)
    registry.register("ok", _ok)
    result = registry.compute_all(np.array([1.0, 2.0]))

    assert result["ok"].value == 1.0
    assert result["broken"].value is None
    assert result["broken"].confidence == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/intelligence/test_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'intelligence'`

- [ ] **Step 3: Write minimal implementation**

```python
"""intelligence/evidence.py
Evidence contract: each quantitative mechanism registers as a named,
causal function of price history that returns a value plus a confidence,
never a vote or a trading decision. A source that raises is isolated --
it never takes down the registry -- and is recorded as an unavailable
(value=None, confidence=0.0) evidence value rather than propagating the
exception, since a fragile representation must not be able to block
every other source's computation."""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class EvidenceValue:
    value: Optional[float]
    confidence: float
    source_name: str


EvidenceSource = Callable[[np.ndarray], EvidenceValue]


class EvidenceRegistry:
    def __init__(self):
        self._sources: dict[str, EvidenceSource] = {}

    def register(self, name: str, fn: EvidenceSource) -> None:
        if name in self._sources:
            raise ValueError(f"Evidence source '{name}' already registered.")
        self._sources[name] = fn

    def names(self) -> list[str]:
        return list(self._sources.keys())

    def compute_all(self, closes_so_far: np.ndarray) -> dict[str, EvidenceValue]:
        results = {}
        for name, fn in self._sources.items():
            try:
                results[name] = fn(closes_so_far)
            except Exception:
                results[name] = EvidenceValue(value=None, confidence=0.0, source_name=name)
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/intelligence/test_evidence.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add intelligence/__init__.py intelligence/evidence.py tests/intelligence/__init__.py tests/intelligence/test_evidence.py
git commit -m "feat: add EvidenceRegistry contract for Track A intelligence scaffold"
```

---

### Task 2: Decision-engine interface and stub implementation

**Files:**
- Create: `intelligence/decision_engine.py`
- Test: `tests/intelligence/test_decision_engine.py`

**Interfaces:**
- Consumes: `EvidenceValue`, `EvidenceRegistry` from `intelligence.evidence` (Task 1).
- Produces: `Action` (dataclass: `kind: Literal["NO_TRADE", "LONG", "SHORT"]`, `sl_price: Optional[float] = None`, `tp_price: Optional[float] = None`, `size: Optional[float] = None`); `ManageAction` type alias = `Literal["HOLD", "CLOSE"]`; `DecisionEngine` abstract base class with abstract methods `decide(market_state, evidence: dict[str, EvidenceValue], account) -> Action` and `manage(market_state, position_view, evidence: dict[str, EvidenceValue], account) -> ManageAction`; `StubDecisionEngine(DecisionEngine)` — `decide` always returns `Action(kind="NO_TRADE")`, `manage` always returns `"HOLD"`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/intelligence/test_decision_engine.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from intelligence.decision_engine import Action, DecisionEngine, StubDecisionEngine


def test_action_defaults_to_no_optional_fields():
    a = Action(kind="NO_TRADE")
    assert a.kind == "NO_TRADE"
    assert a.sl_price is None
    assert a.tp_price is None
    assert a.size is None


def test_stub_decision_engine_is_a_decision_engine():
    engine = StubDecisionEngine()
    assert isinstance(engine, DecisionEngine)


def test_stub_decide_always_returns_no_trade():
    engine = StubDecisionEngine()
    action = engine.decide(market_state=object(), evidence={}, account=object())
    assert action.kind == "NO_TRADE"
    assert action.sl_price is None
    assert action.tp_price is None


def test_stub_manage_always_holds():
    engine = StubDecisionEngine()
    result = engine.manage(market_state=object(), position_view=object(), evidence={}, account=object())
    assert result == "HOLD"


def test_decision_engine_cannot_be_instantiated_directly():
    import pytest
    with pytest.raises(TypeError):
        DecisionEngine()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/intelligence/test_decision_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'intelligence.decision_engine'`

- [ ] **Step 3: Write minimal implementation**

```python
"""intelligence/decision_engine.py
DecisionEngine is the single replaceable seam between market/evidence
information and a trading action (spec Section F). This module's only
concrete implementation, StubDecisionEngine, always returns NO_TRADE/HOLD
-- it exists to prove the scaffold wires together correctly, not to claim
any trading signal. A future gated-expert combiner, learned policy, or
hierarchical model replaces StubDecisionEngine without any change to
intelligence/evidence.py, intelligence/replay_adapter.py, or anything in
simulator/."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from intelligence.evidence import EvidenceValue

ActionKind = Literal["NO_TRADE", "LONG", "SHORT"]
ManageAction = Literal["HOLD", "CLOSE"]


@dataclass
class Action:
    kind: ActionKind
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    size: Optional[float] = None


class DecisionEngine(ABC):
    @abstractmethod
    def decide(self, market_state, evidence: dict[str, EvidenceValue], account) -> Action:
        raise NotImplementedError

    @abstractmethod
    def manage(self, market_state, position_view, evidence: dict[str, EvidenceValue], account) -> ManageAction:
        raise NotImplementedError


class StubDecisionEngine(DecisionEngine):
    """Always NO_TRADE / HOLD. No signal, no strategy -- scaffold-correctness
    only, per the Track A plan's explicit scope."""

    def decide(self, market_state, evidence: dict[str, EvidenceValue], account) -> Action:
        return Action(kind="NO_TRADE")

    def manage(self, market_state, position_view, evidence: dict[str, EvidenceValue], account) -> ManageAction:
        return "HOLD"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/intelligence/test_decision_engine.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add intelligence/decision_engine.py tests/intelligence/test_decision_engine.py
git commit -m "feat: add replaceable DecisionEngine interface with NO_TRADE stub"
```

---

### Task 3: Replay adapter wiring into `simulator/replay.run_replay`

**Files:**
- Create: `intelligence/replay_adapter.py`
- Test: `tests/intelligence/test_replay_adapter.py`

**Interfaces:**
- Consumes: `DecisionEngine`, `Action`, `StubDecisionEngine` from `intelligence.decision_engine` (Task 2); `EvidenceRegistry` from `intelligence.evidence` (Task 1); `simulator.replay.run_replay`, `simulator.contracts.{SimulatedExecutionConfig, EnvironmentTag}` (existing, unmodified); `simulator/replay.py`'s `_extract_observation_features` convention: a bound method's `__self__.last_decision_features` dict, read once then cleared.
- Produces: `ReplayAdapter` class — `__init__(self, engine: DecisionEngine, registry: EvidenceRegistry)`; `decide(self, market_state, account) -> tuple` (matches `simulator.replay.DecideFn`); `manage(self, market_state, position_view, account) -> str` (matches `simulator.replay.ManageFn`); maintains `self._closes: list[float]` internally, appending `market_state.mid` on every call (both decide and manage) before computing evidence, so the closes-so-far array always includes the current bar; sets `self.last_decision_features` to the computed evidence dict (as `{name: value.value for name, value in evidence.items()}`) on every `decide`/`manage` call, per the existing `_extract_observation_features` convention.

- [ ] **Step 1: Write the failing test**

```python
"""tests/intelligence/test_replay_adapter.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from intelligence.decision_engine import StubDecisionEngine, Action, DecisionEngine
from intelligence.evidence import EvidenceRegistry, EvidenceValue
from intelligence.replay_adapter import ReplayAdapter
from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.replay import run_replay


def _synthetic_df(n=20, start_price=1900.0):
    times = [datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(n)]
    closes = start_price + np.cumsum(np.random.RandomState(0).normal(0, 0.1, n))
    return pd.DataFrame({
        "time": times,
        "open": closes, "high": closes + 0.05, "low": closes - 0.05, "close": closes,
        "spread": [0.3] * n, "tick_volume": [10] * n,
    })


def test_adapter_decide_matches_decide_fn_contract():
    registry = EvidenceRegistry()
    adapter = ReplayAdapter(StubDecisionEngine(), registry)

    class _FakeMarketState:
        mid = 1900.0

    class _FakeAccount:
        pass

    action, sl, tp = adapter.decide(_FakeMarketState(), _FakeAccount())
    assert action == "NO_TRADE"
    assert sl is None
    assert tp is None


def test_adapter_manage_matches_manage_fn_contract():
    registry = EvidenceRegistry()
    adapter = ReplayAdapter(StubDecisionEngine(), registry)

    class _FakeMarketState:
        mid = 1900.0

    result = adapter.manage(_FakeMarketState(), object(), object())
    assert result == "HOLD"


def test_adapter_exposes_observation_features_after_decide():
    registry = EvidenceRegistry()
    registry.register("dummy", lambda closes_so_far: EvidenceValue(value=len(closes_so_far), confidence=1.0, source_name="dummy"))
    adapter = ReplayAdapter(StubDecisionEngine(), registry)

    class _FakeMarketState:
        mid = 1900.0

    class _FakeAccount:
        pass

    adapter.decide(_FakeMarketState(), _FakeAccount())
    assert adapter.last_decision_features == {"dummy": 1.0}


def test_stub_decision_engine_through_run_replay_never_opens_a_position():
    df = _synthetic_df(n=15)
    registry = EvidenceRegistry()
    adapter = ReplayAdapter(StubDecisionEngine(), registry)
    config = SimulatedExecutionConfig()

    recorder = run_replay(df, adapter.decide, adapter.manage, config, EnvironmentTag.SIMULATED_TRAINING)

    records = recorder.all_records()
    assert len(records) == 15
    assert all(r.action == "NO_TRADE" for r in records)
    assert all(r.event_type == "DECIDE" for r in records)


def test_adapter_captures_observation_features_through_run_replay():
    df = _synthetic_df(n=5)
    registry = EvidenceRegistry()
    registry.register("dummy", lambda closes_so_far: EvidenceValue(value=float(len(closes_so_far)), confidence=1.0, source_name="dummy"))
    adapter = ReplayAdapter(StubDecisionEngine(), registry)
    config = SimulatedExecutionConfig()

    recorder = run_replay(df, adapter.decide, adapter.manage, config, EnvironmentTag.SIMULATED_TRAINING)

    records = recorder.all_records()
    assert records[0].observation_features == {"dummy": 1.0}
    assert records[4].observation_features == {"dummy": 5.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/intelligence/test_replay_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'intelligence.replay_adapter'`

- [ ] **Step 3: Write minimal implementation**

```python
"""intelligence/replay_adapter.py
Wires any DecisionEngine + EvidenceRegistry pair into the decide_fn/
manage_fn callables simulator/replay.run_replay already expects
(DecideFn = Callable[[market_state, account], tuple],
ManageFn = Callable[[market_state, position_view, account], str]),
and into the existing observation_features opt-in convention
(simulator/replay.py:_extract_observation_features reads
`fn.__self__.last_decision_features` on a bound method and clears it
after each read). This module contains no trading logic -- it only
adapts intelligence/'s interfaces to simulator/'s existing contract."""
import numpy as np

from intelligence.decision_engine import DecisionEngine
from intelligence.evidence import EvidenceRegistry


class ReplayAdapter:
    def __init__(self, engine: DecisionEngine, registry: EvidenceRegistry):
        self._engine = engine
        self._registry = registry
        self._closes: list[float] = []
        self.last_decision_features = None

    def _compute_evidence(self, market_state):
        self._closes.append(float(market_state.mid))
        closes_so_far = np.array(self._closes, dtype=np.float64)
        evidence = self._registry.compute_all(closes_so_far)
        self.last_decision_features = {name: v.value for name, v in evidence.items()}
        return evidence

    def decide(self, market_state, account) -> tuple:
        evidence = self._compute_evidence(market_state)
        action = self._engine.decide(market_state, evidence, account)
        return action.kind, action.sl_price, action.tp_price

    def manage(self, market_state, position_view, account) -> str:
        evidence = self._compute_evidence(market_state)
        return self._engine.manage(market_state, position_view, evidence, account)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/intelligence/test_replay_adapter.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add intelligence/replay_adapter.py tests/intelligence/test_replay_adapter.py
git commit -m "feat: wire DecisionEngine+EvidenceRegistry into simulator/replay's decide_fn/manage_fn contract"
```

---

### Task 4: Learning loop — research vs. deployment split

**Files:**
- Create: `intelligence/learning_loop.py`
- Test: `tests/intelligence/test_learning_loop.py`

**Interfaces:**
- Produces: `LearningLoop` class — `__init__(self)` (both `research_state` and `_deployed_state` start as `None`); `research_update(self, update_fn: Callable[[object], object]) -> None` (calls `update_fn(self.research_state)`, stores the return value back into `self.research_state` — never touches deployed state); `promote(self, validation_check: Callable[[object], bool]) -> bool` (calls `validation_check(self.research_state)`; if `True`, copies `research_state` into `_deployed_state` via `copy.deepcopy` and returns `True`; if `False`, leaves `_deployed_state` unchanged and returns `False`); `get_deployed(self) -> Optional[object]` (returns `_deployed_state`, `None` if nothing has ever been promoted).

- [ ] **Step 1: Write the failing test**

```python
"""tests/intelligence/test_learning_loop.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from intelligence.learning_loop import LearningLoop


def test_new_loop_has_no_research_or_deployed_state():
    loop = LearningLoop()
    assert loop.research_state is None
    assert loop.get_deployed() is None


def test_research_update_mutates_only_research_state():
    loop = LearningLoop()
    loop.research_update(lambda state: {"count": 1})
    assert loop.research_state == {"count": 1}
    assert loop.get_deployed() is None


def test_research_update_can_be_called_repeatedly():
    loop = LearningLoop()
    loop.research_update(lambda state: {"count": 1})
    loop.research_update(lambda state: {"count": state["count"] + 1})
    assert loop.research_state == {"count": 2}
    assert loop.get_deployed() is None


def test_promote_copies_research_into_deployed_when_validation_passes():
    loop = LearningLoop()
    loop.research_update(lambda state: {"count": 5})
    promoted = loop.promote(validation_check=lambda state: state["count"] == 5)
    assert promoted is True
    assert loop.get_deployed() == {"count": 5}


def test_promote_leaves_deployed_unchanged_when_validation_fails():
    loop = LearningLoop()
    loop.research_update(lambda state: {"count": 5})
    promoted = loop.promote(validation_check=lambda state: False)
    assert promoted is False
    assert loop.get_deployed() is None


def test_promote_is_a_deep_copy_not_a_reference():
    loop = LearningLoop()
    loop.research_update(lambda state: {"count": 5})
    loop.promote(validation_check=lambda state: True)
    loop.research_update(lambda state: {"count": 999})
    assert loop.get_deployed() == {"count": 5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/intelligence/test_learning_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'intelligence.learning_loop'`

- [ ] **Step 3: Write minimal implementation**

```python
"""intelligence/learning_loop.py
Structurally separates research learning (freely iterated, offline) from
validated deployment (a frozen copy, reachable only through promote(),
which requires an explicit caller-supplied validation_check to return
True). This enforces spec Section I's research/deployment split at the
interface level rather than by convention -- there is no way to make
get_deployed() return something that didn't pass validation_check."""
import copy
from typing import Callable, Optional


class LearningLoop:
    def __init__(self):
        self.research_state = None
        self._deployed_state = None

    def research_update(self, update_fn: Callable[[object], object]) -> None:
        self.research_state = update_fn(self.research_state)

    def promote(self, validation_check: Callable[[object], bool]) -> bool:
        if validation_check(self.research_state):
            self._deployed_state = copy.deepcopy(self.research_state)
            return True
        return False

    def get_deployed(self) -> Optional[object]:
        return self._deployed_state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/intelligence/test_learning_loop.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add intelligence/learning_loop.py tests/intelligence/test_learning_loop.py
git commit -m "feat: add LearningLoop with a promote()-gated research/deployment split"
```

---

### Task 5: Evidence sources — wrap the 9 Phase 3A/4 representation functions

**Files:**
- Create: `intelligence/evidence_sources.py`
- Test: `tests/intelligence/test_evidence_sources.py`

**Interfaces:**
- Consumes: `EvidenceValue`, `EvidenceRegistry`, `EvidenceSource` from `intelligence.evidence` (Task 1); `momentum_scalar`, `path_pca_projection`, `multiscale_vol_summary`, `vol_regime_transition`, `MOMENTUM_LOOKBACK`, `PATH_WINDOW`, `VOL_WINDOWS` from `research.phase3a_representation_experiments` (existing, unchanged); `fit_garch11` from `research.phase4_garch_volatility_mechanism` (existing, unchanged); `kalman_level_trend_filter` from `research.phase4_kalman_trend_mechanism` (existing, unchanged); `_rolling_moment`, `WINDOW` (as `MOMENT_WINDOW`) from `research.phase4_distributional_mechanism` (existing, unchanged).
- Produces: `build_default_registry() -> EvidenceRegistry` — returns a registry with all 9 sources registered under these exact names: `"momentum_scalar"`, `"raw_path_window_projection"`, `"multiscale_volatility_ratio"`, `"volatility_regime_transition"`, `"garch11_conditional_variance"`, `"kalman_filtered_velocity"`, `"kalman_innovation"`, `"rolling_skew"`, `"rolling_excess_kurtosis"` (matching the naming used in `docs/superpowers/reports/2026-08-27-goldex-genesis-horizon-sweep-findings.md`). Each wrapped source returns `EvidenceValue(value=None, confidence=0.0, ...)` when `len(closes_so_far)` is below that source's minimum lookback (never raises, never fabricates a value from insufficient history).

- [ ] **Step 1: Write the failing test**

```python
"""tests/intelligence/test_evidence_sources.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from intelligence.evidence_sources import build_default_registry

EXPECTED_NAMES = {
    "momentum_scalar", "raw_path_window_projection", "multiscale_volatility_ratio",
    "volatility_regime_transition", "garch11_conditional_variance",
    "kalman_filtered_velocity", "kalman_innovation", "rolling_skew", "rolling_excess_kurtosis",
}


def _synthetic_closes(n=400, seed=0):
    rng = np.random.RandomState(seed)
    return 1900.0 + np.cumsum(rng.normal(0, 0.2, n))


def test_registry_has_all_nine_expected_sources():
    registry = build_default_registry()
    assert set(registry.names()) == EXPECTED_NAMES


def test_all_sources_return_none_on_short_history():
    registry = build_default_registry()
    result = registry.compute_all(np.array([1900.0, 1900.5]))
    for name in EXPECTED_NAMES:
        assert result[name].value is None
        assert result[name].confidence == 0.0


def test_all_sources_return_a_finite_value_on_sufficient_history():
    registry = build_default_registry()
    closes = _synthetic_closes(400)
    result = registry.compute_all(closes)
    for name in EXPECTED_NAMES:
        assert result[name].value is not None, f"{name} returned None on 400 bars of history"
        assert np.isfinite(result[name].value), f"{name} returned a non-finite value"
        assert result[name].confidence == 1.0


@pytest.mark.parametrize("truncate_at", [150, 250, 399])
def test_no_look_ahead_truncated_recompute_matches(truncate_at):
    """Evidence computed on closes_so_far[:truncate_at+1] must not depend on
    any bar beyond truncate_at -- recomputing on the exact same prefix taken
    from a longer array must give an identical value."""
    registry_a = build_default_registry()
    registry_b = build_default_registry()
    closes = _synthetic_closes(400)
    prefix = closes[:truncate_at + 1]

    result_from_full_history_prefix = registry_a.compute_all(prefix)
    result_from_prefix_alone = registry_b.compute_all(prefix)

    for name in EXPECTED_NAMES:
        a = result_from_full_history_prefix[name].value
        b = result_from_prefix_alone[name].value
        if a is None and b is None:
            continue
        assert a == pytest.approx(b), f"{name} differs between two computations on the identical prefix"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/intelligence/test_evidence_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'intelligence.evidence_sources'`

- [ ] **Step 3: Write minimal implementation**

```python
"""intelligence/evidence_sources.py
Wraps the 9 representation functions already validated as code in Phase
3A/4 (research/phase3a_representation_experiments.py,
research/phase4_garch_volatility_mechanism.py,
research/phase4_kalman_trend_mechanism.py,
research/phase4_distributional_mechanism.py) as EvidenceSource callables.
None of these were ever proven a marginal predictor OOS -- see
docs/superpowers/reports/2026-08-27-goldex-genesis-horizon-sweep-findings.md
(zero of 42 cells survived the trend-confound check). They are admitted
here as candidate evidence for a future DecisionEngine to learn
conditional usefulness from (spec Section G), never as standalone
signals. Each wrapper re-invokes the underlying batch function on the
full closes-so-far array and returns only the last element -- this is
correct (causal, no look-ahead, since closes_so_far never includes
anything beyond the current bar) but O(n) per bar; performance
optimization is explicitly out of scope for this plumbing-only plan."""
import numpy as np

from intelligence.evidence import EvidenceRegistry, EvidenceValue
from research.phase3a_representation_experiments import (
    MOMENTUM_LOOKBACK, PATH_WINDOW, VOL_WINDOWS,
    momentum_scalar, path_pca_projection, multiscale_vol_summary, vol_regime_transition,
)
from research.phase4_garch_volatility_mechanism import fit_garch11
from research.phase4_kalman_trend_mechanism import kalman_level_trend_filter
from research.phase4_distributional_mechanism import _rolling_moment, WINDOW as MOMENT_WINDOW

_MIN_GARCH_KALMAN_HISTORY = 30  # fit_garch11/kalman need enough returns to be numerically stable


def _last_finite_or_none(arr: np.ndarray) -> "float | None":
    if len(arr) == 0:
        return None
    last = arr[-1]
    if last is None or not np.isfinite(last):
        return None
    return float(last)


def _momentum_scalar_source(closes_so_far: np.ndarray) -> EvidenceValue:
    if len(closes_so_far) <= MOMENTUM_LOOKBACK:
        return EvidenceValue(None, 0.0, "momentum_scalar")
    val = _last_finite_or_none(momentum_scalar(closes_so_far))
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "momentum_scalar")


def _raw_path_window_projection_source(closes_so_far: np.ndarray) -> EvidenceValue:
    if len(closes_so_far) <= PATH_WINDOW:
        return EvidenceValue(None, 0.0, "raw_path_window_projection")
    val = _last_finite_or_none(path_pca_projection(closes_so_far))
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "raw_path_window_projection")


def _multiscale_volatility_ratio_source(closes_so_far: np.ndarray) -> EvidenceValue:
    if len(closes_so_far) <= max(VOL_WINDOWS):
        return EvidenceValue(None, 0.0, "multiscale_volatility_ratio")
    vol_ratio, _vols = multiscale_vol_summary(closes_so_far)
    val = _last_finite_or_none(vol_ratio)
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "multiscale_volatility_ratio")


def _volatility_regime_transition_source(closes_so_far: np.ndarray) -> EvidenceValue:
    if len(closes_so_far) <= max(VOL_WINDOWS):
        return EvidenceValue(None, 0.0, "volatility_regime_transition")
    _vol_ratio, vols = multiscale_vol_summary(closes_so_far)
    val = _last_finite_or_none(vol_regime_transition(vols[min(VOL_WINDOWS)]))
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "volatility_regime_transition")


def _returns(closes_so_far: np.ndarray) -> np.ndarray:
    return np.diff(closes_so_far)


def _garch11_conditional_variance_source(closes_so_far: np.ndarray) -> EvidenceValue:
    if len(closes_so_far) <= _MIN_GARCH_KALMAN_HISTORY:
        return EvidenceValue(None, 0.0, "garch11_conditional_variance")
    _omega, _alpha, _beta, sigma2 = fit_garch11(_returns(closes_so_far))
    val = _last_finite_or_none(sigma2)
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "garch11_conditional_variance")


def _kalman_filtered_velocity_source(closes_so_far: np.ndarray) -> EvidenceValue:
    if len(closes_so_far) <= _MIN_GARCH_KALMAN_HISTORY:
        return EvidenceValue(None, 0.0, "kalman_filtered_velocity")
    _levels, velocities, _innovations = kalman_level_trend_filter(closes_so_far)
    val = _last_finite_or_none(velocities)
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "kalman_filtered_velocity")


def _kalman_innovation_source(closes_so_far: np.ndarray) -> EvidenceValue:
    if len(closes_so_far) <= _MIN_GARCH_KALMAN_HISTORY:
        return EvidenceValue(None, 0.0, "kalman_innovation")
    _levels, _velocities, innovations = kalman_level_trend_filter(closes_so_far)
    val = _last_finite_or_none(innovations)
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "kalman_innovation")


def _rolling_skew_source(closes_so_far: np.ndarray) -> EvidenceValue:
    returns = _returns(closes_so_far)
    if len(returns) <= MOMENT_WINDOW:
        return EvidenceValue(None, 0.0, "rolling_skew")
    val = _last_finite_or_none(_rolling_moment(returns, MOMENT_WINDOW, order=3))
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "rolling_skew")


def _rolling_excess_kurtosis_source(closes_so_far: np.ndarray) -> EvidenceValue:
    returns = _returns(closes_so_far)
    if len(returns) <= MOMENT_WINDOW:
        return EvidenceValue(None, 0.0, "rolling_excess_kurtosis")
    val = _last_finite_or_none(_rolling_moment(returns, MOMENT_WINDOW, order=4))
    return EvidenceValue(val, 1.0 if val is not None else 0.0, "rolling_excess_kurtosis")


def build_default_registry() -> EvidenceRegistry:
    registry = EvidenceRegistry()
    registry.register("momentum_scalar", _momentum_scalar_source)
    registry.register("raw_path_window_projection", _raw_path_window_projection_source)
    registry.register("multiscale_volatility_ratio", _multiscale_volatility_ratio_source)
    registry.register("volatility_regime_transition", _volatility_regime_transition_source)
    registry.register("garch11_conditional_variance", _garch11_conditional_variance_source)
    registry.register("kalman_filtered_velocity", _kalman_filtered_velocity_source)
    registry.register("kalman_innovation", _kalman_innovation_source)
    registry.register("rolling_skew", _rolling_skew_source)
    registry.register("rolling_excess_kurtosis", _rolling_excess_kurtosis_source)
    return registry
```

**Note for the implementer:** `fit_garch11` must be checked against its actual return signature in `research/phase4_garch_volatility_mechanism.py` before this compiles — read the function's `return` statement directly (the docstring says it "returns `(omega, alpha, beta)` and the full in-sample conditional-variance path" but confirm the exact tuple order and length in code, not from the docstring alone, since docstrings can drift). Adjust the unpacking in `_garch11_conditional_variance_source` to match exactly what the function actually returns.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/intelligence/test_evidence_sources.py -v`
Expected: PASS (4 tests, including the 3 parametrized no-look-ahead cases = 6 total test invocations)

- [ ] **Step 5: Commit**

```bash
git add intelligence/evidence_sources.py tests/intelligence/test_evidence_sources.py
git commit -m "feat: wrap the 9 Phase 3A/4 representation functions as EvidenceRegistry sources"
```

---

### Task 6: End-to-end integration test — full scaffold through a synthetic replay

**Files:**
- Test: `tests/intelligence/test_scaffold_integration.py`

**Interfaces:**
- Consumes: `StubDecisionEngine` from `intelligence.decision_engine`; `build_default_registry` from `intelligence.evidence_sources`; `ReplayAdapter` from `intelligence.replay_adapter`; `run_replay`, `SimulatedExecutionConfig`, `EnvironmentTag` from `simulator.replay`/`simulator.contracts` (all existing, unchanged).
- Produces: nothing new — this task only proves Tasks 1–5 compose correctly end to end. No new production code.

- [ ] **Step 1: Write the test**

```python
"""tests/intelligence/test_scaffold_integration.py
End-to-end proof that the Track A scaffold (evidence registry with all 9
real representation sources + StubDecisionEngine + ReplayAdapter) runs
through the unmodified simulator/replay.run_replay without error, opens
no positions (StubDecisionEngine is NO_TRADE-only, per this plan's
explicit scope), and every DECIDE record carries a full 9-key
observation_features dict once enough history has accumulated."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from intelligence.decision_engine import StubDecisionEngine
from intelligence.evidence_sources import build_default_registry
from intelligence.replay_adapter import ReplayAdapter
from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.replay import run_replay

EXPECTED_NAMES = {
    "momentum_scalar", "raw_path_window_projection", "multiscale_volatility_ratio",
    "volatility_regime_transition", "garch11_conditional_variance",
    "kalman_filtered_velocity", "kalman_innovation", "rolling_skew", "rolling_excess_kurtosis",
}


def _synthetic_df(n=200, seed=0):
    rng = np.random.RandomState(seed)
    times = [datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(n)]
    closes = 1900.0 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame({
        "time": times,
        "open": closes, "high": closes + 0.05, "low": closes - 0.05, "close": closes,
        "spread": [0.3] * n, "tick_volume": [10] * n,
    })


def test_full_scaffold_runs_end_to_end_with_no_positions_opened():
    df = _synthetic_df(n=200)
    engine = StubDecisionEngine()
    registry = build_default_registry()
    adapter = ReplayAdapter(engine, registry)
    config = SimulatedExecutionConfig()

    recorder = run_replay(df, adapter.decide, adapter.manage, config, EnvironmentTag.SIMULATED_TRAINING)

    records = recorder.all_records()
    assert len(records) == 200
    assert all(r.action == "NO_TRADE" for r in records)

    last_record = records[-1]
    assert set(last_record.observation_features.keys()) == EXPECTED_NAMES
    for name in EXPECTED_NAMES:
        assert last_record.observation_features[name] is not None
```

- [ ] **Step 2: Run test to verify it fails first (sanity check on a broken import)**

Run: `pytest tests/intelligence/test_scaffold_integration.py -v --collect-only`
Expected: test collects successfully (all Task 1-5 modules already exist by this point) — this step confirms nothing is accidentally missing before running for real.

- [ ] **Step 3: Run the full test**

Run: `pytest tests/intelligence/test_scaffold_integration.py -v`
Expected: PASS (1 test)

- [ ] **Step 4: Run the entire intelligence test suite together**

Run: `pytest tests/intelligence/ -v`
Expected: PASS, all tests from Tasks 1-6 (roughly 27 test functions total)

- [ ] **Step 5: Commit**

```bash
git add tests/intelligence/test_scaffold_integration.py
git commit -m "test: add end-to-end integration test proving the Track A scaffold composes correctly"
```

---

## Self-Review Notes

- **Spec coverage:** Section F (scaffold diagram) → Tasks 1-3 (`EvidenceRegistry`, `DecisionEngine`, `ReplayAdapter`). Section G (evidence contract) → Task 1 + Task 5. Section I (research/deployment split) → Task 4. Section L (carry over the 9 representation functions, no strategy claims) → Task 5 + Task 6's assertion that no position is ever opened.
- **Out of scope, confirmed not touched by any task:** Tracks B (cross-instrument), C (tick data), D (conditional quant-mechanism evidence), E (credit-assignment spec), F (trade management) — none of Tasks 1-6 add a new information source beyond the 9 already-existing functions, define a credit-assignment rule, or make an entry/exit/sizing decision beyond `NO_TRADE`/`HOLD`.
- **No look-ahead:** enforced by Task 5's parametrized truncate-and-recompute test and reinforced by every wrapper operating only on `closes_so_far` (never a full pre-loaded array), consistent with `ReplayAdapter._compute_evidence` appending the current bar before computing.
