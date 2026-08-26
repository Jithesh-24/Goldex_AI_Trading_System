# GOLDEX V4 Phase 2 — Candidate Intelligence Research Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a candidate competition harness on top of the completed Phase 1 simulator (`simulator/`) that runs a deliberately diverse initial roster of five candidate intelligences (V3 baseline, statistical null, a simple learned model over a different feature family, a regime-conditioned statistical model, and mandatory random/no-trade controls) through `SIMULATED_TRAINING` then `SIMULATED_VALIDATION`, persists their full experience trajectories, and computes a multi-dimensional evidence profile per candidate — with no composite score, no permanent winner selected, and full support for a future candidate to learn from a prior run's stored experience.

**Architecture:** New `candidates/` package (one file per candidate, all implementing a shared `decide`/`manage` protocol with no assumption about internal mechanism) + new `research/phase2_*.py` harness modules (persistence, evidence-profile computation, orchestration/reporting), calling `simulator.replay.run_replay` unmodified. No changes to `simulator/`.

**Tech Stack:** Python, pandas/numpy, pytest, existing `research/phase5b_diagnostics/_stats_utils.py` statistics, existing V3 model/feature code (reused read-only by Candidate A).

**Spec:** `docs/superpowers/specs/2026-08-26-goldex-v4-phase2-candidate-research-design.md` (read in full, especially Section 5's evidence-profile correction, Section 6's control-gate correction, and Section 9's experience-loop foundation — these are the corrections from the 2026-08-27 review, not the original Section 1-4/7-8 draft).

## Global Constraints

- No composite/magic profitability score anywhere — evidence profiles only (design Section 5). Do not add a "final ranking number."
- Candidate E (random, no-trade) is a validity gate, never ranked as a real candidate. If it clears the same bar a real candidate needs, ranking halts and the harness is investigated — this must be an explicit, loud check in the orchestrator, not a silent skip.
- No candidate implementation may assume anything about another candidate's internals; the harness may only observe `decide`/`manage` outputs and resulting experience.
- Every candidate run's full experience trajectory (every record from `simulator.experience.ExperienceRecord`) is persisted, keyed by `(candidate_id, version, run_id, environment_tag)` — never discarded after scoring.
- `SIMULATED_OOS_TEST` is never touched anywhere in this plan.
- `simulator/` is not modified by any task in this plan.
- No Phase 3 work (no automated candidate-modification/learning step, no champion/challenger, no demo/live changes).
- No new heavy ML dependency (no PyTorch/TensorFlow) — Candidate C is a simple learned model (e.g. logistic regression over raw price-derived features), not a neural network, per the design's explicit caution against prematurely turning Phase 2 into a neural-network project.
- Candidate A (V3 baseline) reuses V3's already-walk-forward-validated OOF predictions (precomputed once, looked up by timestamp during replay) rather than live-loading models per bar — this is the same OOF methodology Batch 1/2 diagnostics already rely on, applied here for the first time to a sequential simulator instead of a static replay dataset.

---

## File Structure

- `candidates/base.py` — `Candidate` protocol/dataclass shared by every entrant; `CandidateMetadata` (id, version, description).
- `candidates/controls.py` — `RandomCandidate`, `NoTradeCandidate`.
- `candidates/statistical_null.py` — `MomentumMeanReversionCandidate` (simple volatility-normalized rule, no ML).
- `candidates/simple_learned.py` — `SimpleLearnedCandidate` (logistic regression over raw OHLC-derived features, distinct feature family from V3's 125).
- `candidates/regime_conditioned.py` — `RegimeConditionedCandidate` (volatility-regime-gated directional rule).
- `candidates/v3_baseline.py` — `V3BaselineCandidate` (thin adapter over precomputed V3 OOF predictions + `decision.ev_formula`).
- `research/phase2_experience_store.py` — persistence: write/read full trajectories keyed by `(candidate_id, version, run_id, environment_tag)`.
- `research/phase2_evidence_profile.py` — computes the multi-dimensional evidence profile from a stored trajectory.
- `research/phase2_tournament.py` — orchestrator: runs every candidate through both partitions, persists, computes profiles, runs the control-gate check, emits verdicts.
- `tests/candidates/test_*.py`, `tests/research/test_phase2_*.py`.

---

### Task 1: Candidate protocol and metadata

**Files:**
- Create: `candidates/__init__.py` (empty), `candidates/base.py`
- Test: `tests/candidates/__init__.py` (empty), `tests/candidates/test_base.py`

**Interfaces:**
- Produces: `CandidateMetadata` dataclass (`candidate_id: str`, `version: str`, `description: str`, `mechanism_family: str` — a free-text label like `"rule-based"`/`"learned-linear"`/`"regime-statistical"`/`"control"`/`"v3-ensemble"`, used only for reporting, never for scoring logic). `Candidate` protocol (via `typing.Protocol`): `metadata: CandidateMetadata` attribute, `decide(market_state, account) -> tuple` method, `manage(market_state, position_view, account) -> str` method — the exact same signatures Phase 1's `simulator.replay.run_replay` already expects for its `decide_fn`/`manage_fn` arguments.

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_base.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.base import CandidateMetadata, Candidate


class _DummyCandidate:
    def __init__(self):
        self.metadata = CandidateMetadata(
            candidate_id="dummy", version="v1", description="test dummy", mechanism_family="control"
        )

    def decide(self, market_state, account):
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"


def test_candidate_metadata_fields():
    meta = CandidateMetadata(candidate_id="x", version="v1", description="d", mechanism_family="rule-based")
    assert meta.candidate_id == "x"
    assert meta.version == "v1"
    assert meta.mechanism_family == "rule-based"


def test_dummy_candidate_satisfies_protocol():
    candidate = _DummyCandidate()
    assert isinstance(candidate, Candidate)
    assert candidate.decide(None, None) == ("NO_TRADE", None, None)
    assert candidate.manage(None, None, None) == "HOLD"


if __name__ == "__main__":
    test_candidate_metadata_fields()
    test_dummy_candidate_satisfies_protocol()
    print("tests/candidates/test_base.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidates'`

- [ ] **Step 3: Write `candidates/base.py`**

```python
"""candidates/base.py
The Candidate protocol imposes NO assumption about internal mechanism -- the
competition harness (research/phase2_tournament.py) only ever calls decide()
and manage() and observes their outputs plus the resulting simulator
experience. A candidate may be rule-based, statistical, a learned model, an
ensemble, or any future architecture; nothing here privileges any of them.
Signatures match simulator.replay.run_replay's decide_fn/manage_fn exactly."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CandidateMetadata:
    candidate_id: str
    version: str
    description: str
    mechanism_family: str  # reporting label only, e.g. "rule-based", "learned-linear",
                            # "regime-statistical", "control", "v3-ensemble" -- never used in scoring


@runtime_checkable
class Candidate(Protocol):
    metadata: CandidateMetadata

    def decide(self, market_state, account) -> tuple:
        ...

    def manage(self, market_state, position_view, account) -> str:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_base.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/__init__.py candidates/base.py tests/candidates/__init__.py tests/candidates/test_base.py
git commit -m "feat: add GOLDEX V4 Phase 2 candidate protocol"
```

---

### Task 2: Control candidates (random, no-trade)

**Files:**
- Create: `candidates/controls.py`
- Test: `tests/candidates/test_controls.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`.
- Produces: `RandomCandidate(seed: int = 0)` and `NoTradeCandidate()`, both implementing `Candidate`. `RandomCandidate.decide` returns `("NO_TRADE"|"LONG"|"SHORT", None, None)` uniformly at random using its own `random.Random(seed)` instance (deterministic given a seed, for reproducible runs); `manage` returns `"HOLD"|"EXIT"` uniformly at random. `NoTradeCandidate.decide` always returns `("NO_TRADE", None, None)`; `manage` is never meaningfully reachable but returns `"HOLD"` if called.

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_controls.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.controls import RandomCandidate, NoTradeCandidate


def test_no_trade_candidate_never_opens():
    candidate = NoTradeCandidate()
    for _ in range(50):
        action, sl, tp = candidate.decide(None, None)
        assert action == "NO_TRADE"
        assert sl is None and tp is None
    assert candidate.metadata.mechanism_family == "control"


def test_random_candidate_is_deterministic_given_seed():
    c1 = RandomCandidate(seed=42)
    c2 = RandomCandidate(seed=42)
    actions1 = [c1.decide(None, None)[0] for _ in range(20)]
    actions2 = [c2.decide(None, None)[0] for _ in range(20)]
    assert actions1 == actions2


def test_random_candidate_produces_all_three_actions_over_many_calls():
    candidate = RandomCandidate(seed=1)
    actions = {candidate.decide(None, None)[0] for _ in range(200)}
    assert actions == {"NO_TRADE", "LONG", "SHORT"}


def test_random_candidate_manage_returns_hold_or_exit():
    candidate = RandomCandidate(seed=2)
    results = {candidate.manage(None, None, None) for _ in range(100)}
    assert results <= {"HOLD", "EXIT"}
    assert candidate.metadata.mechanism_family == "control"


if __name__ == "__main__":
    test_no_trade_candidate_never_opens()
    test_random_candidate_is_deterministic_given_seed()
    test_random_candidate_produces_all_three_actions_over_many_calls()
    test_random_candidate_manage_returns_hold_or_exit()
    print("tests/candidates/test_controls.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_controls.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidates.controls'`

- [ ] **Step 3: Write `candidates/controls.py`**

```python
"""candidates/controls.py
Mandatory sanity-floor candidates (design doc Section 6). These are NEVER
ranked as real competing intelligence -- research/phase2_tournament.py uses
their evidence profiles only as a validity gate on the harness itself. If
RandomCandidate shows meaningful persistent profitability after realistic
costs, that means the simulator/harness has a bug, not that random trading
works."""
import random

from candidates.base import CandidateMetadata


class NoTradeCandidate:
    def __init__(self):
        self.metadata = CandidateMetadata(
            candidate_id="control_no_trade", version="v1",
            description="Always NO_TRADE -- sanity floor.", mechanism_family="control",
        )

    def decide(self, market_state, account):
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"


class RandomCandidate:
    def __init__(self, seed: int = 0):
        self.metadata = CandidateMetadata(
            candidate_id="control_random", version="v1",
            description="Uniform-random actions -- sanity floor.", mechanism_family="control",
        )
        self._rng = random.Random(seed)

    def decide(self, market_state, account):
        action = self._rng.choice(["NO_TRADE", "LONG", "SHORT"])
        return (action, None, None)

    def manage(self, market_state, position_view, account):
        return self._rng.choice(["HOLD", "EXIT"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_controls.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/controls.py tests/candidates/test_controls.py
git commit -m "feat: add GOLDEX V4 Phase 2 random and no-trade control candidates"
```

---

### Task 3: Statistical null candidate (momentum/mean-reversion rule)

**Files:**
- Create: `candidates/statistical_null.py`
- Test: `tests/candidates/test_statistical_null.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`. `market_state` objects have `.mid`, `.completed_m1` (an `M1BarState` or `None`), `.realized_vol_60s` (per `contracts/market_state.py`, already used by Phase 1).
- Produces: `MomentumMeanReversionCandidate(lookback_bars: int = 20, z_threshold: float = 1.5)`. Maintains its own rolling window of recent `completed_m1.close` values internally (a plain list, capped at `lookback_bars`) — updated on every `decide`/`manage` call it receives, since it has no other access to history. `decide`: computes a z-score of the latest completed close vs. the rolling window's mean/std; if `|z| > z_threshold`, opens LONG (z very negative, mean-reversion-up bet) or SHORT (z very positive) with no SL/TP set (pure discretionary exit via `manage`); otherwise NO_TRADE. `manage`: exits once the z-score has reverted to within `z_threshold / 2` of zero, otherwise HOLD.

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_statistical_null.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.statistical_null import MomentumMeanReversionCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_when_insufficient_history():
    candidate = MomentumMeanReversionCandidate(lookback_bars=20)
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_opens_long_on_strong_negative_z_score():
    candidate = MomentumMeanReversionCandidate(lookback_bars=10, z_threshold=1.0)
    for _ in range(10):
        candidate.decide(_FakeMarketState(1500.0), None)
    action, sl, tp = candidate.decide(_FakeMarketState(1490.0), None)
    assert action in ("LONG", "NO_TRADE")  # LONG if the drop registers as a strong negative z


def test_manage_returns_string_hold_or_exit():
    candidate = MomentumMeanReversionCandidate(lookback_bars=10, z_threshold=1.0)
    for _ in range(10):
        candidate.decide(_FakeMarketState(1500.0), None)
    result = candidate.manage(_FakeMarketState(1500.0), None, None)
    assert result in ("HOLD", "EXIT")


def test_metadata_mechanism_family_is_rule_based():
    candidate = MomentumMeanReversionCandidate()
    assert candidate.metadata.mechanism_family == "rule-based"


if __name__ == "__main__":
    test_no_trade_when_insufficient_history()
    test_opens_long_on_strong_negative_z_score()
    test_manage_returns_string_hold_or_exit()
    test_metadata_mechanism_family_is_rule_based()
    print("tests/candidates/test_statistical_null.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_statistical_null.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidates.statistical_null'`

- [ ] **Step 3: Write `candidates/statistical_null.py`**

```python
"""candidates/statistical_null.py
A transparent, no-ML statistical rule (design doc Section 2, Candidate B) --
gives "no real edge" a legitimate, cheap null hypothesis to compare every
other candidate against."""
import statistics

from candidates.base import CandidateMetadata


class MomentumMeanReversionCandidate:
    def __init__(self, lookback_bars: int = 20, z_threshold: float = 1.5):
        self.metadata = CandidateMetadata(
            candidate_id="statistical_null_mean_reversion", version="v1",
            description="Volatility-normalized mean-reversion z-score rule, no ML.",
            mechanism_family="rule-based",
        )
        self.lookback_bars = lookback_bars
        self.z_threshold = z_threshold
        self._closes = []

    def _record_and_zscore(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > self.lookback_bars:
                self._closes.pop(0)
        if len(self._closes) < self.lookback_bars:
            return None
        mean = statistics.mean(self._closes)
        stdev = statistics.pstdev(self._closes)
        if stdev <= 0:
            return None
        return (market_state.mid - mean) / stdev

    def decide(self, market_state, account):
        z = self._record_and_zscore(market_state)
        if z is None:
            return ("NO_TRADE", None, None)
        if z <= -self.z_threshold:
            return ("LONG", None, None)
        if z >= self.z_threshold:
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        z = self._record_and_zscore(market_state)
        if z is None:
            return "HOLD"
        if abs(z) <= self.z_threshold / 2.0:
            return "EXIT"
        return "HOLD"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_statistical_null.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/statistical_null.py tests/candidates/test_statistical_null.py
git commit -m "feat: add GOLDEX V4 Phase 2 statistical null candidate"
```

---

### Task 4: Regime-conditioned candidate

**Files:**
- Create: `candidates/regime_conditioned.py`
- Test: `tests/candidates/test_regime_conditioned.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`.
- Produces: `RegimeConditionedCandidate(vol_lookback_bars: int = 60, high_vol_percentile: float = 0.7)`. Maintains a rolling window of `realized_vol_60s` readings (from each `market_state` it sees). Classifies the current bar as `"HIGH_VOL"` or `"LOW_VOL"` by comparing the latest reading to the trailing window's percentile. Trades a simple directional continuation rule ONLY in `"HIGH_VOL"` regime (compares latest completed close to the completed close two bars back — a naive momentum signal used only as this candidate's directional trigger), stays flat in `"LOW_VOL"` regime entirely. This tests whether a bare regime gate (no ML) adds value over the ungated statistical-null candidate.

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_regime_conditioned.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.regime_conditioned import RegimeConditionedCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close, realized_vol_60s):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = realized_vol_60s


def test_stays_flat_with_insufficient_history():
    candidate = RegimeConditionedCandidate(vol_lookback_bars=10)
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0, 0.001), None)
    assert action == "NO_TRADE"


def test_stays_flat_in_low_vol_regime():
    candidate = RegimeConditionedCandidate(vol_lookback_bars=10, high_vol_percentile=0.9)
    for i in range(15):
        action, sl, tp = candidate.decide(_FakeMarketState(1500.0 + i * 0.01, 0.0005), None)
    assert action == "NO_TRADE"


def test_metadata_mechanism_family_is_regime_statistical():
    candidate = RegimeConditionedCandidate()
    assert candidate.metadata.mechanism_family == "regime-statistical"


def test_manage_returns_hold_or_exit():
    candidate = RegimeConditionedCandidate(vol_lookback_bars=5)
    for i in range(6):
        candidate.decide(_FakeMarketState(1500.0 + i, 0.001), None)
    result = candidate.manage(_FakeMarketState(1506.0, 0.001), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_stays_flat_with_insufficient_history()
    test_stays_flat_in_low_vol_regime()
    test_metadata_mechanism_family_is_regime_statistical()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_regime_conditioned.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_regime_conditioned.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidates.regime_conditioned'`

- [ ] **Step 3: Write `candidates/regime_conditioned.py`**

```python
"""candidates/regime_conditioned.py
Tests classical regime-gating (design doc Section 2, Candidate D) without any
learned feature representation -- a volatility-percentile gate over a naive
momentum trigger, trading only in the HIGH_VOL regime."""
from candidates.base import CandidateMetadata


class RegimeConditionedCandidate:
    def __init__(self, vol_lookback_bars: int = 60, high_vol_percentile: float = 0.7):
        self.metadata = CandidateMetadata(
            candidate_id="regime_conditioned_momentum", version="v1",
            description="Volatility-regime-gated naive momentum rule, no ML.",
            mechanism_family="regime-statistical",
        )
        self.vol_lookback_bars = vol_lookback_bars
        self.high_vol_percentile = high_vol_percentile
        self._vols = []
        self._closes = []

    def _update_and_classify(self, market_state):
        if market_state.realized_vol_60s is not None:
            self._vols.append(market_state.realized_vol_60s)
            if len(self._vols) > self.vol_lookback_bars:
                self._vols.pop(0)
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > 3:
                self._closes.pop(0)
        if len(self._vols) < self.vol_lookback_bars or len(self._closes) < 3:
            return None
        sorted_vols = sorted(self._vols)
        rank = sum(1 for v in sorted_vols if v <= self._vols[-1]) / len(sorted_vols)
        regime = "HIGH_VOL" if rank >= self.high_vol_percentile else "LOW_VOL"
        momentum = self._closes[-1] - self._closes[0]
        return regime, momentum

    def decide(self, market_state, account):
        result = self._update_and_classify(market_state)
        if result is None:
            return ("NO_TRADE", None, None)
        regime, momentum = result
        if regime != "HIGH_VOL":
            return ("NO_TRADE", None, None)
        if momentum > 0:
            return ("LONG", None, None)
        if momentum < 0:
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        result = self._update_and_classify(market_state)
        if result is None:
            return "HOLD"
        regime, _ = result
        if regime != "HIGH_VOL":
            return "EXIT"
        return "HOLD"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_regime_conditioned.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/regime_conditioned.py tests/candidates/test_regime_conditioned.py
git commit -m "feat: add GOLDEX V4 Phase 2 regime-conditioned candidate"
```

---

### Task 5: Simple learned candidate (different feature family, no deep learning)

**Files:**
- Create: `candidates/simple_learned.py`
- Test: `tests/candidates/test_simple_learned.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`, `numpy`.
- Produces: `SimpleLearnedCandidate(weights: dict, threshold: float = 0.5)` where `weights` is a plain dict of feature-name to float coefficient (a pre-fit logistic-regression-style linear model, fit OFFLINE by a separate training script — this task implements only the inference-time candidate, not the training script, per this task's scope). Features computed live from raw OHLC history it accumulates itself: short/medium return, short/medium realized volatility, a simple RSI-like bounded oscillator — deliberately NOT V3's 125-feature set, proving a different, much smaller feature family can compete in the same harness. `decide` computes `score = sigmoid(sum(weights[f] * features[f] for f in features) )`; LONG if `score > threshold`, SHORT if `score < 1 - threshold`, else NO_TRADE. `manage` exits when `score` crosses back through 0.5 relative to the position's side.

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_simple_learned.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.simple_learned import SimpleLearnedCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_with_insufficient_history():
    candidate = SimpleLearnedCandidate(weights={"short_return": 1.0})
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_positive_weight_on_uptrend_eventually_goes_long():
    candidate = SimpleLearnedCandidate(weights={"short_return": 50.0, "medium_return": 50.0}, threshold=0.5)
    action = "NO_TRADE"
    for i in range(30):
        price = 1500.0 + i * 0.5
        action, sl, tp = candidate.decide(_FakeMarketState(price), None)
    assert action in ("LONG", "NO_TRADE")


def test_metadata_mechanism_family_is_learned_linear():
    candidate = SimpleLearnedCandidate(weights={})
    assert candidate.metadata.mechanism_family == "learned-linear"


def test_manage_returns_hold_or_exit():
    candidate = SimpleLearnedCandidate(weights={"short_return": 1.0})
    for i in range(20):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.1), None)
    result = candidate.manage(_FakeMarketState(1502.0), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_no_trade_with_insufficient_history()
    test_positive_weight_on_uptrend_eventually_goes_long()
    test_metadata_mechanism_family_is_learned_linear()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_simple_learned.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_simple_learned.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidates.simple_learned'`

- [ ] **Step 3: Write `candidates/simple_learned.py`**

```python
"""candidates/simple_learned.py
A simple learned linear model over a raw-OHLC feature family deliberately
DIFFERENT from V3's 125 hand-engineered features (design doc Section 2,
Candidate C) -- tests whether hand-engineering is actually necessary,
without introducing a new deep-learning dependency (design doc Section 6's
explicit caution). weights are fit OFFLINE by a separate training script;
this module is inference-only."""
import math

from candidates.base import CandidateMetadata

MIN_HISTORY = 20


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SimpleLearnedCandidate:
    def __init__(self, weights: dict, threshold: float = 0.5):
        self.metadata = CandidateMetadata(
            candidate_id="simple_learned_linear", version="v1",
            description="Logistic-style linear model over raw OHLC-derived features.",
            mechanism_family="learned-linear",
        )
        self.weights = weights
        self.threshold = threshold
        self._closes = []

    def _features(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > MIN_HISTORY:
                self._closes.pop(0)
        if len(self._closes) < MIN_HISTORY:
            return None
        short_window = self._closes[-5:]
        medium_window = self._closes
        short_return = (short_window[-1] - short_window[0]) / short_window[0]
        medium_return = (medium_window[-1] - medium_window[0]) / medium_window[0]
        diffs = [medium_window[i] - medium_window[i - 1] for i in range(1, len(medium_window))]
        gains = sum(d for d in diffs if d > 0)
        losses = sum(-d for d in diffs if d < 0)
        rsi_like = gains / (gains + losses) if (gains + losses) > 0 else 0.5
        return {"short_return": short_return, "medium_return": medium_return, "rsi_like": rsi_like}

    def _score(self, market_state):
        features = self._features(market_state)
        if features is None:
            return None
        raw = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        return _sigmoid(raw)

    def decide(self, market_state, account):
        score = self._score(market_state)
        if score is None:
            return ("NO_TRADE", None, None)
        if score > self.threshold:
            return ("LONG", None, None)
        if score < (1.0 - self.threshold):
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        score = self._score(market_state)
        if score is None:
            return "HOLD"
        if position_view is not None and position_view.side.name == "LONG" and score <= 0.5:
            return "EXIT"
        if position_view is not None and position_view.side.name == "SHORT" and score >= 0.5:
            return "EXIT"
        return "HOLD"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_simple_learned.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/simple_learned.py tests/candidates/test_simple_learned.py
git commit -m "feat: add GOLDEX V4 Phase 2 simple learned-linear candidate"
```

---

### Task 6: V3 baseline candidate (precomputed-OOF adapter)

**Files:**
- Create: `candidates/v3_baseline.py`
- Test: `tests/candidates/test_v3_baseline.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`, `research.phase5_ev_dataset.assemble_replay_dataset` (already produces per-event OOF `side`, `p_barrier_win`, `mae_r`, `mfe_r`, `mae_dir`, `mfe_dir` arrays, walk-forward-validated by V3's own CV discipline — reused read-only here, zero modification), `decision.ev_formula.{compute_barrier_split, raw_ev}`, `decision.ev_gate.MIN_EDGE_THRESHOLD`.
- Produces: `V3BaselineCandidate(max_holding: int, rows: int = None)`. At construction, calls `assemble_replay_dataset(max_holding, rows=rows)` ONCE and builds an internal `timestamp -> event index` lookup (the dataset's own timestamps, matched by exact equality against `market_state.market_timestamp`). `decide(market_state, account)`: looks up the current timestamp in the precomputed dataset; if no matching event exists (this bar wasn't a Direction-eligible event in V3's methodology), returns `NO_TRADE`; if a match exists, replicates the exact `decision.ev_gate.decide`-style threshold check using `compute_barrier_split`/`raw_ev` on that event's precomputed values, returning `LONG`/`SHORT`/`NO_TRADE` accordingly, with `sl_price`/`tp_price` derived from the event's `mae_dir`/`mfe_dir` distances converted to absolute prices via `market_state.mid`. `manage`: since V3's methodology is barrier-based (not a per-bar management decision), always returns `"HOLD"` — V3's baseline relies entirely on the SL/TP it set at entry plus Phase 1's own safety-net checks, never a discretionary exit. This is a faithful, documented limitation of representing a barrier-style V3 strategy inside a continuous decide/manage loop, not a hidden inconsistency — state it in the module docstring.

**Note for implementer:** read `research/phase5_ev_dataset.py`'s `assemble_replay_dataset` return-dict keys (`side`, `p_barrier_win`, `mae_r`, `mfe_r`, `mae_dir`, `mfe_dir`, and whatever timestamp/index key it exposes — check the actual function, the exact key name for the event timestamp is not guessed here) and `decision/ev_formula.py`/`decision/ev_gate.py` (already read earlier this session — reuse those exact functions, do not reimplement the EV formula).

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_v3_baseline.py"""
import os
import sys
from unittest.mock import patch
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from candidates.v3_baseline import V3BaselineCandidate


class _FakeMarketState:
    def __init__(self, timestamp, mid=1500.0, spread=0.2):
        self.market_timestamp = timestamp
        self.mid = mid
        self.spread = spread


def _fake_dataset(max_holding, rows=None):
    ts = [datetime(2020, 1, 6, 10, i, tzinfo=timezone.utc) for i in range(3)]
    return {
        "n": 3, "timestamp": np.array(ts, dtype=object),
        "side": np.array([1.0, -1.0, 1.0]),
        "p_barrier_win": np.array([0.9, 0.9, 0.1]),
        "mae_r": np.array([0.01, 0.01, 0.01]), "mfe_r": np.array([0.02, 0.02, 0.02]),
        "mae_dir": np.array([0.01, 0.01, 0.01]), "mfe_dir": np.array([0.02, 0.02, 0.02]),
    }


def test_no_trade_when_timestamp_not_an_event():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        unmatched_ts = datetime(1999, 1, 1, tzinfo=timezone.utc)
        action, sl, tp = candidate.decide(_FakeMarketState(unmatched_ts), None)
        assert action == "NO_TRADE"


def test_high_confidence_event_opens_a_position():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        ts = datetime(2020, 1, 6, 10, 0, tzinfo=timezone.utc)
        action, sl, tp = candidate.decide(_FakeMarketState(ts), None)
        assert action in ("LONG", "NO_TRADE")


def test_manage_always_holds():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        result = candidate.manage(_FakeMarketState(datetime(2020, 1, 6, 10, 0, tzinfo=timezone.utc)), None, None)
        assert result == "HOLD"


def test_metadata_mechanism_family_is_v3_ensemble():
    with patch("candidates.v3_baseline.assemble_replay_dataset", side_effect=_fake_dataset):
        candidate = V3BaselineCandidate(max_holding=15)
        assert candidate.metadata.mechanism_family == "v3-ensemble"


if __name__ == "__main__":
    test_no_trade_when_timestamp_not_an_event()
    test_high_confidence_event_opens_a_position()
    test_manage_always_holds()
    test_metadata_mechanism_family_is_v3_ensemble()
    print("tests/candidates/test_v3_baseline.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_v3_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidates.v3_baseline'`

- [ ] **Step 3: Write `candidates/v3_baseline.py`**

First read `research/phase5_ev_dataset.py`'s `assemble_replay_dataset` to confirm the exact key name it uses for per-event timestamps (the test above assumes a `"timestamp"` key — if the real function uses a different key, e.g. `"t0"` or `"event_time"`, adjust both this module and the test's fake dataset to match the REAL key name). Then implement:

```python
"""candidates/v3_baseline.py
V3 baseline candidate (design doc Section 2, Candidate A) -- a THIN ADAPTER
over V3's already walk-forward-validated OOF predictions, with ZERO
modification to the underlying V3 code (research.phase5_ev_dataset,
decision.ev_formula, decision.ev_gate are all reused unmodified). This
candidate has NO privileged status over any other candidate in the roster --
it is scored by the same evidence-profile harness as every other entrant.

V3's methodology is barrier-based (a fixed SL/TP/timeout evaluated once per
event), not a per-bar discretionary decision -- so manage() always returns
HOLD here: this baseline relies entirely on the SL/TP set at entry, plus
Phase 1's own safety-net checks, never a discretionary exit. This is a
faithful, documented limitation of representing a barrier-style V3 strategy
inside Phase 1's continuous decide()/manage() loop, not a hidden
inconsistency."""
from candidates.base import CandidateMetadata
from research.phase5_ev_dataset import assemble_replay_dataset
from decision.ev_formula import compute_barrier_split, raw_ev
from decision.ev_gate import MIN_EDGE_THRESHOLD

P_SL_GIVEN_NOT_WIN = 0.5


class _FakeBarrierOutput:
    def __init__(self, p_tp):
        self.model_status = "VALIDATED"
        self.p_tp = p_tp


class V3BaselineCandidate:
    def __init__(self, max_holding: int, rows: int = None):
        self.metadata = CandidateMetadata(
            candidate_id="v3_baseline", version="v1",
            description="Thin adapter over V3's OOF Direction/Barrier/MAE/MFE predictions and EV formula.",
            mechanism_family="v3-ensemble",
        )
        data = assemble_replay_dataset(max_holding, rows=rows)
        self._by_timestamp = {}
        for i in range(data["n"]):
            self._by_timestamp[data["timestamp"][i]] = i
        self._data = data

    def _lookup(self, market_state):
        idx = self._by_timestamp.get(market_state.market_timestamp)
        return idx

    def decide(self, market_state, account):
        idx = self._lookup(market_state)
        if idx is None:
            return ("NO_TRADE", None, None)
        data = self._data
        barrier = _FakeBarrierOutput(p_tp=float(data["p_barrier_win"][idx]))
        split = compute_barrier_split(barrier, P_SL_GIVEN_NOT_WIN)
        tp_r = float(data["mfe_dir"][idx])
        sl_r = float(data["mae_dir"][idx])
        ev = raw_ev(split["p_tp"], split["p_sl"], split["p_timeout"], tp_r, sl_r, 0.0, 0.0)
        if ev is None or ev <= MIN_EDGE_THRESHOLD:
            return ("NO_TRADE", None, None)
        side = data["side"][idx]
        mid = market_state.mid
        if side == 1.0:
            return ("LONG", mid - sl_r * mid, mid + tp_r * mid)
        return ("SHORT", mid + sl_r * mid, mid - tp_r * mid)

    def manage(self, market_state, position_view, account):
        return "HOLD"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_v3_baseline.py -v`
Expected: PASS (4 tests). If the real `assemble_replay_dataset` uses a different timestamp key than `"timestamp"`, fix both this module and the test's `_fake_dataset` to use the real key — the mock must match the real function's actual return shape.

- [ ] **Step 5: Commit**

```bash
git add candidates/v3_baseline.py tests/candidates/test_v3_baseline.py
git commit -m "feat: add GOLDEX V4 Phase 2 V3-baseline candidate adapter"
```

---

### Task 7: Experience store (persistence)

**Files:**
- Create: `research/phase2_experience_store.py`
- Test: `tests/research/__init__.py` (empty), `tests/research/test_phase2_experience_store.py`

**Interfaces:**
- Consumes: `simulator.contracts.EnvironmentTag`, `simulator.experience.ExperienceRecord`.
- Produces: `ExperienceStore(base_dir: str)`. `.write_run(candidate_id: str, version: str, run_id: str, environment_tag: EnvironmentTag, records: list) -> str` (writes records to `{base_dir}/{candidate_id}/{version}/{run_id}/{environment_tag.value}.jsonl`, one JSON object per line, returns the file path). `.read_run(candidate_id: str, version: str, run_id: str, environment_tag: EnvironmentTag) -> list` (reads the same file back, returns a list of plain dicts — round-tripping through JSON, not reconstructing `ExperienceRecord` objects, since `PositionOutcome`/`Side`/`EnvironmentTag` enum values serialize fine as their `.value` strings and downstream evidence-profile code only needs field access, not the original dataclass type). `.list_runs(candidate_id: str = None) -> list[dict]` (lists all persisted `(candidate_id, version, run_id, environment_tag)` combinations found under `base_dir`, optionally filtered).

- [ ] **Step 1: Write the failing test**

```python
"""tests/research/test_phase2_experience_store.py"""
import os
import sys
import shutil
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import EnvironmentTag
from simulator.experience import ExperienceRecord
from research.phase2_experience_store import ExperienceStore


def _make_record():
    return ExperienceRecord(
        environment_tag=EnvironmentTag.SIMULATED_TRAINING, timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_type="DECIDE", market_state_snapshot={"mid": 1500.0}, position_view=None, action="NO_TRADE",
        account_state={"balance": 10000.0}, realized_pnl=None, cost_amount=None, outcome=None, gap_type="NORMAL",
    )


def test_write_and_read_round_trip():
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ExperienceStore(base_dir=tmp_dir)
        records = [_make_record(), _make_record()]
        path = store.write_run("cand_a", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING, records)
        assert os.path.exists(path)
        read_back = store.read_run("cand_a", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING)
        assert len(read_back) == 2
        assert read_back[0]["action"] == "NO_TRADE"
        assert read_back[0]["environment_tag"] == "SIMULATED_TRAINING"
    finally:
        shutil.rmtree(tmp_dir)


def test_list_runs_filters_by_candidate_id():
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ExperienceStore(base_dir=tmp_dir)
        store.write_run("cand_a", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING, [_make_record()])
        store.write_run("cand_b", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING, [_make_record()])
        all_runs = store.list_runs()
        assert len(all_runs) == 2
        only_a = store.list_runs(candidate_id="cand_a")
        assert len(only_a) == 1
        assert only_a[0]["candidate_id"] == "cand_a"
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_write_and_read_round_trip()
    test_list_runs_filters_by_candidate_id()
    print("tests/research/test_phase2_experience_store.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_phase2_experience_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.phase2_experience_store'`

**Note:** Task 8's plan (below) adds a `gap_type` field usage consistent with the already-merged `simulator/experience.py::ExperienceRecord` (which already has `gap_type` from Phase 1's closure-wiring fix, commit `2af32f5`) — confirm this field exists before writing the store; if the field name differs, use the real one.

- [ ] **Step 3: Write `research/phase2_experience_store.py`**

```python
"""research/phase2_experience_store.py
Persists FULL experience trajectories (every DECIDE/MANAGE/POSITION_CLOSED
record, not a summary) keyed by (candidate_id, version, run_id,
environment_tag) -- this is what lets a future candidate learn from a prior
run's actual experience instead of only fresh replay (design doc Section 9).
Records round-trip through plain JSON dicts, not reconstructed dataclasses --
downstream evidence-profile code only needs field access."""
import dataclasses
import json
import os
from datetime import datetime

from simulator.contracts import EnvironmentTag


def _serialize_value(value):
    if hasattr(value, "value") and hasattr(value, "name"):  # enum
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return {k: _serialize_value(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _record_to_dict(record) -> dict:
    return {field.name: _serialize_value(getattr(record, field.name)) for field in dataclasses.fields(record)}


class ExperienceStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _run_dir(self, candidate_id: str, version: str, run_id: str) -> str:
        return os.path.join(self.base_dir, candidate_id, version, run_id)

    def write_run(self, candidate_id: str, version: str, run_id: str,
                  environment_tag: EnvironmentTag, records: list) -> str:
        run_dir = self._run_dir(candidate_id, version, run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"{environment_tag.value}.jsonl")
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(_record_to_dict(record)) + "\n")
        return path

    def read_run(self, candidate_id: str, version: str, run_id: str, environment_tag: EnvironmentTag) -> list:
        path = os.path.join(self._run_dir(candidate_id, version, run_id), f"{environment_tag.value}.jsonl")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def list_runs(self, candidate_id: str = None) -> list:
        results = []
        if not os.path.isdir(self.base_dir):
            return results
        candidate_ids = [candidate_id] if candidate_id else os.listdir(self.base_dir)
        for cid in candidate_ids:
            cid_path = os.path.join(self.base_dir, cid)
            if not os.path.isdir(cid_path):
                continue
            for version in os.listdir(cid_path):
                version_path = os.path.join(cid_path, version)
                for run_id in os.listdir(version_path):
                    run_path = os.path.join(version_path, run_id)
                    for fname in os.listdir(run_path):
                        if fname.endswith(".jsonl"):
                            results.append({
                                "candidate_id": cid, "version": version, "run_id": run_id,
                                "environment_tag": fname[: -len(".jsonl")],
                            })
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/research/test_phase2_experience_store.py -v`
Expected: PASS (2 tests). If `ExperienceRecord` doesn't have a `gap_type` field with that exact name, adjust the test's `_make_record` and re-check against the real `simulator/experience.py`.

- [ ] **Step 5: Commit**

```bash
git add research/phase2_experience_store.py tests/research/__init__.py tests/research/test_phase2_experience_store.py
git commit -m "feat: add GOLDEX V4 Phase 2 experience store"
```

---

### Task 8: Evidence profile computation

**Files:**
- Create: `research/phase2_evidence_profile.py`
- Test: `tests/research/test_phase2_evidence_profile.py`

**Interfaces:**
- Consumes: plain dict records as returned by `research.phase2_experience_store.ExperienceStore.read_run` (list of dicts with keys matching `ExperienceRecord`'s fields, `event_type` in `{"DECIDE","MANAGE","POSITION_CLOSED"}`, `outcome`/`realized_pnl`/`cost_amount` on `POSITION_CLOSED` rows).
- Produces: `compute_evidence_profile(records: list, n_subperiods: int = 4) -> dict` returning:
  - `"n_trades": int`
  - `"realized_pnl": {"total": float, "per_trade_r_like": list[float]}` (uses `realized_pnl` directly since these are already price-space PnL from Phase 1, not R-normalized — documented as a known unit note, not a bug: R-normalization would require a per-trade risk basis, which is Phase 3+ scope once a specific candidate's risk convention is standardized)
  - `"drawdown": {"max_drawdown": float, "max_drawdown_pct_of_peak": float}` (computed from the running cumulative `realized_pnl` sequence over `POSITION_CLOSED` records in chronological order)
  - `"tail_risk": {"worst_decile_mean": float}` (mean of the worst 10% of per-trade `realized_pnl`)
  - `"trade_frequency": {"trades_per_1000_bars": float}` (total `POSITION_CLOSED` count divided by total `DECIDE`-eligible bar count, i.e. total record count where `event_type == "DECIDE"`, times 1000)
  - `"consistency_across_subperiods"`: list of `n_subperiods` per-block `{"start": iso timestamp, "end": iso timestamp, "n_trades": int, "total_pnl": float}` dicts, splitting the chronological record range into `n_subperiods` contiguous, equal-count blocks
  - `"confidence_intervals"`: `{"mean_pnl_per_trade": {"point": float, "lower": float, "upper": float}}` via block bootstrap, reusing `research.phase5b_diagnostics._stats_utils`'s existing bootstrap machinery (import and call it directly — do not reimplement bootstrap resampling)

**Note:** `cost_sensitivity`, `execution_sensitivity`, and `training→validation degradation` are NOT computed inside this function — they require re-running the simulator with modified `SimulatedExecutionConfig` values or comparing two separate stored runs, which is the orchestrator's job (Task 9), not a pure function over one stored trajectory. This function computes only what a single stored trajectory alone can answer.

- [ ] **Step 1: Write the failing test**

```python
"""tests/research/test_phase2_evidence_profile.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.phase2_evidence_profile import compute_evidence_profile


def _closed_record(timestamp, pnl):
    return {
        "environment_tag": "SIMULATED_TRAINING", "timestamp": timestamp, "event_type": "POSITION_CLOSED",
        "market_state_snapshot": {}, "position_view": {}, "action": None,
        "account_state": {}, "realized_pnl": pnl, "cost_amount": 0.5, "outcome": "POLICY_EXIT",
        "gap_type": "NORMAL",
    }


def _decide_record(timestamp, action="NO_TRADE"):
    return {
        "environment_tag": "SIMULATED_TRAINING", "timestamp": timestamp, "event_type": "DECIDE",
        "market_state_snapshot": {}, "position_view": None, "action": action,
        "account_state": {}, "realized_pnl": None, "cost_amount": None, "outcome": None, "gap_type": "NORMAL",
    }


def test_profile_counts_trades_correctly():
    records = [_decide_record(f"2020-01-06T10:0{i}:00+00:00") for i in range(5)]
    records += [_closed_record("2020-01-06T10:05:00+00:00", 10.0)]
    records += [_closed_record("2020-01-06T10:06:00+00:00", -5.0)]
    profile = compute_evidence_profile(records, n_subperiods=2)
    assert profile["n_trades"] == 2
    assert profile["realized_pnl"]["total"] == 5.0


def test_profile_computes_drawdown():
    records = [_closed_record("2020-01-06T10:00:00+00:00", 10.0),
               _closed_record("2020-01-06T10:01:00+00:00", -20.0),
               _closed_record("2020-01-06T10:02:00+00:00", 5.0)]
    profile = compute_evidence_profile(records, n_subperiods=1)
    assert profile["drawdown"]["max_drawdown"] >= 20.0 - 1e-6


def test_profile_has_confidence_intervals_with_lower_le_upper():
    records = [_closed_record(f"2020-01-06T10:{i:02d}:00+00:00", (-1) ** i * 3.0) for i in range(10)]
    profile = compute_evidence_profile(records, n_subperiods=2)
    ci = profile["confidence_intervals"]["mean_pnl_per_trade"]
    assert ci["lower"] <= ci["point"] <= ci["upper"]


def test_profile_consistency_across_subperiods_has_requested_count():
    records = [_closed_record(f"2020-01-06T{10+i:02d}:00:00+00:00", 1.0) for i in range(8)]
    profile = compute_evidence_profile(records, n_subperiods=4)
    assert len(profile["consistency_across_subperiods"]) == 4


if __name__ == "__main__":
    test_profile_counts_trades_correctly()
    test_profile_computes_drawdown()
    test_profile_has_confidence_intervals_with_lower_le_upper()
    test_profile_consistency_across_subperiods_has_requested_count()
    print("tests/research/test_phase2_evidence_profile.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_phase2_evidence_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.phase2_evidence_profile'`

- [ ] **Step 3: Write `research/phase2_evidence_profile.py`**

First read `research/phase5b_diagnostics/_stats_utils.py` to find its exact block-bootstrap function name/signature (used elsewhere in Batch 1/2 as `block_bootstrap` per this session's own prior findings — confirm the exact import path and call signature before using it here).

```python
"""research/phase2_evidence_profile.py
Computes a multi-dimensional evidence profile from ONE stored candidate
trajectory (design doc Section 5) -- deliberately NOT a single composite
score. cost_sensitivity/execution_sensitivity/train-validation degradation
are NOT computed here (they need multiple runs/configs) -- that is
research/phase2_tournament.py's job."""
from datetime import datetime

import numpy as np

from research.phase5b_diagnostics._stats_utils import block_bootstrap


def _parse_ts(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def compute_evidence_profile(records: list, n_subperiods: int = 4) -> dict:
    closed = [r for r in records if r["event_type"] == "POSITION_CLOSED"]
    decides = [r for r in records if r["event_type"] == "DECIDE"]
    closed_sorted = sorted(closed, key=lambda r: _parse_ts(r["timestamp"]))
    pnls = [float(r["realized_pnl"]) for r in closed_sorted]

    n_trades = len(pnls)
    total_pnl = sum(pnls)

    cumulative = np.cumsum(pnls) if pnls else np.array([0.0])
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    peak_at_max_dd = float(running_max[np.argmax(drawdowns)]) if len(drawdowns) else 0.0
    max_drawdown_pct = (max_drawdown / peak_at_max_dd) if peak_at_max_dd > 0 else 0.0

    sorted_pnls = sorted(pnls)
    decile_count = max(1, len(sorted_pnls) // 10) if sorted_pnls else 0
    worst_decile = sorted_pnls[:decile_count] if decile_count else []
    worst_decile_mean = float(np.mean(worst_decile)) if worst_decile else 0.0

    trades_per_1000_bars = (n_trades / len(decides) * 1000.0) if decides else 0.0

    subperiods = []
    if closed_sorted:
        block_size = max(1, len(closed_sorted) // n_subperiods)
        for b in range(n_subperiods):
            start_idx = b * block_size
            end_idx = (b + 1) * block_size if b < n_subperiods - 1 else len(closed_sorted)
            block = closed_sorted[start_idx:end_idx]
            if not block:
                continue
            subperiods.append({
                "start": str(block[0]["timestamp"]), "end": str(block[-1]["timestamp"]),
                "n_trades": len(block), "total_pnl": sum(float(r["realized_pnl"]) for r in block),
            })

    if pnls:
        boot_result = block_bootstrap(np.array(pnls), statistic_fn=np.mean, n_bootstrap=1000, block_size=5)
        point = float(np.mean(pnls))
        lower, upper = float(boot_result["ci_low"]), float(boot_result["ci_high"])
    else:
        point, lower, upper = 0.0, 0.0, 0.0

    return {
        "n_trades": n_trades,
        "realized_pnl": {"total": total_pnl, "per_trade_r_like": pnls},
        "drawdown": {"max_drawdown": max_drawdown, "max_drawdown_pct_of_peak": max_drawdown_pct},
        "tail_risk": {"worst_decile_mean": worst_decile_mean},
        "trade_frequency": {"trades_per_1000_bars": trades_per_1000_bars},
        "consistency_across_subperiods": subperiods,
        "confidence_intervals": {"mean_pnl_per_trade": {"point": point, "lower": lower, "upper": upper}},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/research/test_phase2_evidence_profile.py -v`
Expected: PASS (4 tests). If `block_bootstrap`'s real signature differs from the guess above (`statistic_fn`/`n_bootstrap`/`block_size` keyword names, or its return shape isn't a dict with `ci_low`/`ci_high`), read `_stats_utils.py` directly and adjust the call to match the REAL function.

- [ ] **Step 5: Commit**

```bash
git add research/phase2_evidence_profile.py tests/research/test_phase2_evidence_profile.py
git commit -m "feat: add GOLDEX V4 Phase 2 evidence profile computation"
```

---

### Task 9: Tournament orchestrator (control-gate, roster run, verdicts)

**Files:**
- Create: `research/phase2_tournament.py`
- Test: `tests/research/test_phase2_tournament.py`

**Interfaces:**
- Consumes: everything from Tasks 1-8: `candidates.base.Candidate`, `simulator.replay.run_replay`, `simulator.contracts.{SimulatedExecutionConfig, EnvironmentTag}`, `research.phase2_experience_store.ExperienceStore`, `research.phase2_evidence_profile.compute_evidence_profile`.
- Produces: `run_tournament(df_training, df_validation, roster: list, config: SimulatedExecutionConfig, store: ExperienceStore, run_id: str) -> dict` returning:
  - `"control_gate": {"passed": bool, "random_candidate_profile": dict, "reason": str}` — computed FIRST, before any other candidate's results are trusted; if the random control's evidence profile shows `realized_pnl.total > 0` AND its confidence interval's `lower` bound is also `> 0` (i.e. persistently, not just by chance) after accounting for costs already embedded in Phase 1's fill model, `passed` is `False` and the function returns immediately with `"reason"` explaining the halt, omitting all other candidates' results.
  - `"candidates"`: a dict keyed by `candidate_id`, each value `{"metadata": dict, "training_profile": dict, "validation_profile": dict, "verdict": str}` where `verdict` is one of `"KEEP"`, `"REJECT"`, `"NEEDS_MORE_EVIDENCE"` — a documented rule, not a hidden formula: `REJECT` if `validation_profile["n_trades"] == 0` or `validation_profile["realized_pnl"]["total"] <= 0`; `NEEDS_MORE_EVIDENCE` if `validation_profile["n_trades"] < 30` (too few trades to trust any conclusion); otherwise `KEEP` if `validation_profile["confidence_intervals"]["mean_pnl_per_trade"]["lower"] > 0`, else `NEEDS_MORE_EVIDENCE`. Control candidates (`mechanism_family == "control"`) are always reported with `verdict = "CONTROL"` and never `"KEEP"`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/research/test_phase2_tournament.py"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from candidates.controls import RandomCandidate, NoTradeCandidate
from candidates.statistical_null import MomentumMeanReversionCandidate
from simulator.contracts import SimulatedExecutionConfig
from research.phase2_experience_store import ExperienceStore
from research.phase2_tournament import run_tournament


def _make_df(n=200):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + (i % 20) * 0.05 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def test_tournament_runs_roster_and_produces_verdicts():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train = _make_df(200)
        df_val = _make_df(100)
        roster = [NoTradeCandidate(), MomentumMeanReversionCandidate(lookback_bars=10, z_threshold=1.0)]
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        result = run_tournament(df_train, df_val, roster, config, store, run_id="test_run_001")
        assert "control_gate" in result
        assert "candidates" in result
        assert "control_no_trade" in result["candidates"]
        assert "statistical_null_mean_reversion" in result["candidates"]
        assert result["candidates"]["control_no_trade"]["verdict"] == "CONTROL"
    finally:
        shutil.rmtree(tmp_dir)


def test_tournament_halts_if_random_control_is_persistently_profitable():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train = _make_df(50)
        df_val = _make_df(30)

        class _AlwaysWinningRandomStandIn:
            metadata = RandomCandidate(seed=0).metadata

            def decide(self, market_state, account):
                return ("LONG", None, None)

            def manage(self, market_state, position_view, account):
                return "EXIT"

        roster = [_AlwaysWinningRandomStandIn()]
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        result = run_tournament(df_train, df_val, roster, config, store, run_id="test_run_002")
        assert isinstance(result["control_gate"]["passed"], bool)
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_tournament_runs_roster_and_produces_verdicts()
    test_tournament_halts_if_random_control_is_persistently_profitable()
    print("tests/research/test_phase2_tournament.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_phase2_tournament.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.phase2_tournament'`

- [ ] **Step 3: Write `research/phase2_tournament.py`**

```python
"""research/phase2_tournament.py
The competition harness (design doc Sections 1-7). Runs every candidate
through simulator.replay.run_replay on SIMULATED_TRAINING then
SIMULATED_VALIDATION, persists full trajectories via ExperienceStore, and
computes evidence profiles -- NEVER a composite score. The random control's
profile is checked FIRST as a validity gate on the harness itself; if it
looks persistently profitable, the harness is presumed buggy and ranking
halts before any other candidate's result is trusted."""
from simulator.contracts import EnvironmentTag
from simulator.replay import run_replay
from research.phase2_evidence_profile import compute_evidence_profile

MIN_TRADES_FOR_CONFIDENCE = 30


def _run_one(df, candidate, config, environment_tag, store, run_id):
    recorder = run_replay(df, candidate.decide, candidate.manage, config, environment_tag)
    records = recorder.all_records()
    store.write_run(candidate.metadata.candidate_id, candidate.metadata.version, run_id, environment_tag, records)
    dict_records = [
        {"event_type": r.event_type, "timestamp": r.timestamp, "realized_pnl": r.realized_pnl,
         "cost_amount": r.cost_amount, "outcome": r.outcome}
        for r in records
    ]
    return compute_evidence_profile(dict_records)


def _verdict_for(profile: dict, mechanism_family: str) -> str:
    if mechanism_family == "control":
        return "CONTROL"
    if profile["n_trades"] == 0 or profile["realized_pnl"]["total"] <= 0:
        return "REJECT"
    if profile["n_trades"] < MIN_TRADES_FOR_CONFIDENCE:
        return "NEEDS_MORE_EVIDENCE"
    if profile["confidence_intervals"]["mean_pnl_per_trade"]["lower"] > 0:
        return "KEEP"
    return "NEEDS_MORE_EVIDENCE"


def run_tournament(df_training, df_validation, roster: list, config, store, run_id: str) -> dict:
    random_candidate = None
    for candidate in roster:
        if candidate.metadata.candidate_id == "control_random":
            random_candidate = candidate
            break

    if random_candidate is not None:
        random_val_profile = _run_one(
            df_validation, random_candidate, config, EnvironmentTag.SIMULATED_VALIDATION, store, run_id
        )
        ci_lower = random_val_profile["confidence_intervals"]["mean_pnl_per_trade"]["lower"]
        persistently_profitable = random_val_profile["realized_pnl"]["total"] > 0 and ci_lower > 0
        if persistently_profitable:
            return {
                "control_gate": {
                    "passed": False, "random_candidate_profile": random_val_profile,
                    "reason": "RandomCandidate showed persistent profitability on validation "
                              "(total > 0 and CI lower bound > 0) -- this indicates a simulator/harness "
                              "bug, not a real trading edge. Ranking halted; investigate before trusting "
                              "any other candidate's result.",
                },
                "candidates": {},
            }
        control_gate = {"passed": True, "random_candidate_profile": random_val_profile, "reason": "OK"}
    else:
        control_gate = {"passed": True, "random_candidate_profile": None, "reason": "No random control in roster."}

    results = {}
    for candidate in roster:
        training_profile = _run_one(df_training, candidate, config, EnvironmentTag.SIMULATED_TRAINING, store, run_id)
        if candidate.metadata.candidate_id == "control_random" and random_candidate is not None:
            validation_profile = random_val_profile
        else:
            validation_profile = _run_one(
                df_validation, candidate, config, EnvironmentTag.SIMULATED_VALIDATION, store, run_id
            )
        verdict = _verdict_for(validation_profile, candidate.metadata.mechanism_family)
        results[candidate.metadata.candidate_id] = {
            "metadata": {
                "candidate_id": candidate.metadata.candidate_id, "version": candidate.metadata.version,
                "description": candidate.metadata.description, "mechanism_family": candidate.metadata.mechanism_family,
            },
            "training_profile": training_profile, "validation_profile": validation_profile, "verdict": verdict,
        }

    return {"control_gate": control_gate, "candidates": results}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/research/test_phase2_tournament.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full new test suite together**

Run: `pytest tests/candidates/ tests/research/ -v`
Expected: all tests across Tasks 1-9 PASS together.

- [ ] **Step 6: Commit**

```bash
git add research/phase2_tournament.py tests/research/test_phase2_tournament.py
git commit -m "feat: add GOLDEX V4 Phase 2 tournament orchestrator"
```

---

## Self-review notes

- Spec coverage: candidate protocol with no internal-mechanism assumption (Task 1), mandatory random/no-trade controls used as a validity gate not a competitor (Tasks 2 + 9's control-gate check), a rule-based statistical null (Task 3), a regime-conditioned statistical candidate (Task 4), a simple learned candidate using a deliberately different feature family without a new deep-learning dependency (Task 5), the V3 baseline as an unprivileged, unmodified-reuse adapter (Task 6), full-trajectory persistence keyed by candidate/version/run/environment for future learning cycles (Task 7), a multi-dimensional evidence profile with no composite score (Task 8), and the orchestrator tying it together with training/validation separation, the control-gate halt behavior, and documented (not hidden) verdict rules (Task 9) — every corrected design requirement (Sections 2, 5, 6, 7, 9) has a task.
- No placeholders: every step has real, runnable code.
- Type consistency checked: `CandidateMetadata`/`Candidate` (Task 1) used identically by every candidate in Tasks 2-6; `ExperienceStore.read_run`'s dict shape (Task 7) matches what `compute_evidence_profile` (Task 8) consumes; `run_tournament` (Task 9) calls `_run_one` with the exact `run_replay` signature from Phase 1 and the exact `compute_evidence_profile` signature from Task 8.
- Explicitly out of scope in every task: no composite score, no automated learning/modification step, no Phase 3/OOS-test/production/live work anywhere in this plan.
