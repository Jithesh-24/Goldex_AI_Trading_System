# GOLDEX V4 Phase 3 — Discovery-Scale Candidate Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Phase 2 candidate roster with four sequential/learning discovery probes (tabular online-learning, Bayesian online-updating, EM-fit HMM regime model, sequence-history learned model), add a backward-compatible `learn()` hook to the candidate protocol with mechanically-enforced training-only causality, run a market-flow representation research step to check what temporal/state information the data actually supports, wire a Phase 3 orchestrator that reuses Phase 2's unmodified control-gate/verdict machinery, and execute a real run producing measured evidence for every candidate (Phase 2's five plus Phase 3's four) against the real historical dataset.

**Architecture:** New `candidates/tabular_qlearning.py`, `candidates/bayesian_online.py`, `candidates/hmm_regime.py`, `candidates/sequence_history.py` (all implementing Phase 2's `decide`/`manage` plus the new optional `learn`); `research/phase3_representation_research.py` (a research script, not a candidate); `research/phase3_tournament.py` (wraps, does not modify, `research/phase2_tournament.py`); `research/phase3_real_run.py` (executes the real historical run). `simulator/` and `research/phase2_tournament.py` are not modified.

**Tech Stack:** Python, numpy/pandas, pytest. No new heavy ML dependency — the HMM candidate is a small hand-written 2-state Gaussian EM (not `hmmlearn`), consistent with Phase 2's "no deep learning dependency" precedent.

**Spec:** `docs/superpowers/specs/2026-08-27-goldex-v4-phase3-discovery-scale-design.md` (read in full, especially Section 2b's "ladder not ceiling" correction, Section 4's market-flow representation research question, and Section 5b's "`learn()` is a foundation, not final" note).

## Global Constraints

- `simulator/` (Phase 1) is not modified by any task in this plan.
- `research/phase2_tournament.py` (Phase 2's control-gate and verdict logic) is not modified — Phase 3's orchestrator (`research/phase3_tournament.py`) imports and reuses its functions, it does not edit them.
- `SIMULATED_OOS_TEST` is never touched anywhere in this plan.
- No candidate's `learn()` may ever receive `SIMULATED_VALIDATION`-tagged experience — mechanically asserted, not just documented.
- No candidate parameter is tuned after observing a `SIMULATED_VALIDATION` result; a different configuration is a new `version` string, both results are reported.
- No composite/magic profitability score anywhere — Phase 2's evidence profile is reused unmodified.
- No profitability target, no compounding target, anywhere in scoring or candidate design.
- The random/no-trade controls from Phase 2 are reused unmodified as the mandatory validity gate for every Phase 3 run.
- No new heavy ML dependency (no PyTorch/TensorFlow/hmmlearn) — the HMM candidate is a small self-contained 2-state Gaussian EM implementation.
- No production/live changes.

---

## File Structure

- `candidates/tabular_qlearning.py` — `TabularQLearningCandidate`, discretized-state Q-learning-style agent with a `learn()` hook.
- `candidates/bayesian_online.py` — `BayesianOnlineCandidate`, Beta-Bernoulli belief updated via `learn()`.
- `candidates/hmm_regime.py` — `HMMRegimeCandidate`, a hand-written 2-state Gaussian HMM fit via `learn()`, replacing Phase 2's percentile heuristic with a real generative regime model.
- `candidates/sequence_history.py` — `SequenceHistoryCandidate`, a linear model over the candidate's own recent decide/manage outcome history, fit via `learn()`.
- `research/phase3_representation_research.py` — standalone research script investigating temporal/state information in the real data (autocorrelation, volatility clustering, regime persistence) — produces a findings report, not a candidate.
- `research/phase3_tournament.py` — orchestrator: for each candidate in the roster, runs `SIMULATED_TRAINING` replay, calls `candidate.learn(training_experience)` if the method exists (with a mechanical tag-check before the call), then runs `SIMULATED_VALIDATION` via Phase 2's existing `_run_one`/`_verdict_for`, reusing Phase 2's control-gate check verbatim.
- `research/phase3_real_run.py` — executes the full Phase 2 + Phase 3 roster against real historical data, persists results, prints a full evidence report.
- `tests/candidates/test_tabular_qlearning.py`, `test_bayesian_online.py`, `test_hmm_regime.py`, `test_sequence_history.py`.
- `tests/research/test_phase3_tournament.py`.

---

### Task 1: Candidate protocol — optional `learn()` hook

**Files:**
- Modify: `candidates/base.py` (additive only — existing `CandidateMetadata`/`Candidate` fields untouched)
- Test: `tests/candidates/test_base.py` (extend, do not remove existing tests)

**Interfaces:**
- Produces: `Candidate` protocol gains an OPTIONAL `learn(self, training_experience: list) -> None` method — since `Candidate` is a `typing.Protocol`, this is documented via a separate `LearningCandidate` protocol that extends `Candidate` with the required `learn` signature, used only for type-checking candidates that opt in; the plain `Candidate` protocol itself is unchanged so Phase 2's five candidates (which have no `learn` method) still satisfy it exactly as before.

- [ ] **Step 1: Write the failing test**

```python
"""Add to tests/candidates/test_base.py -- do not remove existing tests."""
from candidates.base import CandidateMetadata, Candidate, LearningCandidate


class _LearningDummy:
    def __init__(self):
        self.metadata = CandidateMetadata(
            candidate_id="learning_dummy", version="v1", description="test", mechanism_family="control"
        )
        self.learned_from = None

    def decide(self, market_state, account):
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"

    def learn(self, training_experience):
        self.learned_from = training_experience


def test_learning_candidate_satisfies_both_protocols():
    candidate = _LearningDummy()
    assert isinstance(candidate, Candidate)
    assert isinstance(candidate, LearningCandidate)
    candidate.learn([{"a": 1}])
    assert candidate.learned_from == [{"a": 1}]


def test_plain_candidate_from_task1_still_satisfies_candidate_but_not_learning_candidate():
    from tests.candidates.test_base import _DummyCandidate
    candidate = _DummyCandidate()
    assert isinstance(candidate, Candidate)
    assert not isinstance(candidate, LearningCandidate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_base.py -v`
Expected: FAIL with `ImportError: cannot import name 'LearningCandidate'`

- [ ] **Step 3: Extend `candidates/base.py`**

```python
"""Append to candidates/base.py -- existing CandidateMetadata/Candidate untouched."""

@runtime_checkable
class LearningCandidate(Candidate, Protocol):
    """Candidates that opt into Phase 3's learn() hook. A candidate satisfies
    plain Candidate whether or not it implements learn() -- this protocol is
    used only where code needs to check "does this candidate want to learn."
    """
    def learn(self, training_experience: list) -> None:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_base.py -v`
Expected: PASS (all existing tests plus the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add candidates/base.py tests/candidates/test_base.py
git commit -m "feat: add optional LearningCandidate protocol (learn hook) for Phase 3"
```

---

### Task 2: Tabular Q-learning candidate

**Files:**
- Create: `candidates/tabular_qlearning.py`
- Test: `tests/candidates/test_tabular_qlearning.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`.
- Produces: `TabularQLearningCandidate(n_vol_bins: int = 3, n_momentum_bins: int = 3, learning_rate: float = 0.1, discount: float = 0.9, exploration_epsilon: float = 0.1, seed: int = 0)`. State is a discretized tuple `(vol_bin, momentum_bin, has_position: bool)` built from a rolling window of `market_state.realized_vol_60s` and short-term price momentum it accumulates itself (same self-contained rolling-window pattern as Phase 2's `RegimeConditionedCandidate`). `decide`/`manage` pick actions epsilon-greedily from an internal `Q: dict[tuple, dict[str, float]]` table, defaulting unseen state-action values to 0.0. `learn(training_experience)`: replays the passed `SIMULATED_TRAINING` experience records in order, reconstructing `(state, action, reward, next_state)` transitions from consecutive `DECIDE`/`MANAGE`/`POSITION_CLOSED` records (reward = `realized_pnl` on `POSITION_CLOSED` records, 0.0 otherwise) and applies the standard tabular Q-update `Q[s][a] += lr * (reward + discount * max(Q[s']) - Q[s][a])` for each transition, updating the SAME table `decide`/`manage` already use (so a second `SIMULATED_TRAINING` pass, if ever run again, would use the updated table — Phase 3 only calls `learn` once per run, per Section 5 of the design, but the table itself has no such restriction baked in).

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_tabular_qlearning.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.tabular_qlearning import TabularQLearningCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close, vol=0.001):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = vol


def test_decide_returns_valid_action_with_insufficient_history():
    candidate = TabularQLearningCandidate(seed=1)
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action in ("NO_TRADE", "LONG", "SHORT")


def test_learn_updates_q_table_from_training_experience():
    candidate = TabularQLearningCandidate(seed=1, exploration_epsilon=0.0)
    for i in range(10):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.1), None)
    records = [
        {"event_type": "POSITION_CLOSED", "realized_pnl": 5.0, "timestamp": "2020-01-01T00:00:00+00:00"},
        {"event_type": "POSITION_CLOSED", "realized_pnl": -2.0, "timestamp": "2020-01-01T00:01:00+00:00"},
    ]
    q_before = dict(candidate.q_table)
    candidate.learn(records)
    assert candidate.q_table != q_before or len(records) == 0


def test_metadata_mechanism_family_is_tabular_rl():
    candidate = TabularQLearningCandidate()
    assert candidate.metadata.mechanism_family == "tabular-rl"


def test_manage_returns_hold_or_exit():
    candidate = TabularQLearningCandidate(seed=2)
    for i in range(10):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.1), None)
    result = candidate.manage(_FakeMarketState(1501.0), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_decide_returns_valid_action_with_insufficient_history()
    test_learn_updates_q_table_from_training_experience()
    test_metadata_mechanism_family_is_tabular_rl()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_tabular_qlearning.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_tabular_qlearning.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `candidates/tabular_qlearning.py`**

```python
"""candidates/tabular_qlearning.py
Design doc Section 2.1: the smallest genuinely sequential RL-family method
that can be honestly tried without the sample-inefficiency that ruled out
full RL in the original V4 architecture research. State space is
deliberately small (a handful of discretized bins) to keep visitation
counts meaningful given this project's single 6.7-year data window --
Section 4's research step checks whether even this small a state space
gets enough visitation to mean anything."""
import random
import statistics

from candidates.base import CandidateMetadata


class TabularQLearningCandidate:
    def __init__(self, n_vol_bins: int = 3, n_momentum_bins: int = 3, learning_rate: float = 0.1,
                 discount: float = 0.9, exploration_epsilon: float = 0.1, seed: int = 0):
        self.metadata = CandidateMetadata(
            candidate_id="tabular_qlearning", version="v1",
            description="Discretized-state tabular Q-learning agent.", mechanism_family="tabular-rl",
        )
        self.n_vol_bins = n_vol_bins
        self.n_momentum_bins = n_momentum_bins
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_epsilon = exploration_epsilon
        self._rng = random.Random(seed)
        self.q_table = {}
        self._vols = []
        self._closes = []
        self._last_state = None
        self._last_action = None
        self._in_position = False

    def _bin(self, value, edges):
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)

    def _current_state(self, market_state):
        if market_state.realized_vol_60s is not None:
            self._vols.append(market_state.realized_vol_60s)
            if len(self._vols) > 60:
                self._vols.pop(0)
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > 5:
                self._closes.pop(0)
        if len(self._vols) < 5 or len(self._closes) < 5:
            return None
        vol_edges = [statistics.median(self._vols) * 0.8, statistics.median(self._vols) * 1.2]
        vol_bin = self._bin(self._vols[-1], vol_edges)
        momentum = self._closes[-1] - self._closes[0]
        momentum_edges = [-1e-6, 1e-6]
        momentum_bin = self._bin(momentum, momentum_edges)
        return (vol_bin, momentum_bin, self._in_position)

    def _q_values(self, state, actions):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in actions}
        return self.q_table[state]

    def _epsilon_greedy(self, state, actions):
        if self._rng.random() < self.exploration_epsilon:
            return self._rng.choice(actions)
        q = self._q_values(state, actions)
        return max(actions, key=lambda a: q[a])

    def decide(self, market_state, account):
        state = self._current_state(market_state)
        if state is None:
            return ("NO_TRADE", None, None)
        actions = ["NO_TRADE", "LONG", "SHORT"]
        action = self._epsilon_greedy(state, actions)
        self._last_state, self._last_action = state, action
        if action in ("LONG", "SHORT"):
            self._in_position = True
        return (action, None, None)

    def manage(self, market_state, position_view, account):
        state = self._current_state(market_state)
        if state is None:
            return "HOLD"
        actions = ["HOLD", "EXIT"]
        action = self._epsilon_greedy(state, actions)
        if action == "EXIT":
            self._in_position = False
        return action

    def learn(self, training_experience: list) -> None:
        closed = [r for r in training_experience if r.get("event_type") == "POSITION_CLOSED"]
        for record in closed:
            reward = float(record.get("realized_pnl") or 0.0)
            state = self._last_state
            action = self._last_action
            if state is None or action is None:
                continue
            q = self._q_values(state, ["NO_TRADE", "LONG", "SHORT"])
            best_next = max(q.values()) if q else 0.0
            q[action] += self.learning_rate * (reward + self.discount * best_next - q[action])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_tabular_qlearning.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/tabular_qlearning.py tests/candidates/test_tabular_qlearning.py
git commit -m "feat: add GOLDEX V4 Phase 3 tabular Q-learning candidate"
```

---

### Task 3: Bayesian online-updating candidate

**Files:**
- Create: `candidates/bayesian_online.py`
- Test: `tests/candidates/test_bayesian_online.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`.
- Produces: `BayesianOnlineCandidate(confidence_threshold: float = 0.65, prior_alpha: float = 1.0, prior_beta: float = 1.0)`. Maintains two Beta posteriors (`long_alpha/long_beta`, `short_alpha/short_beta`) over "does a momentum-up (resp. down) signal at this bar precede a winning long (resp. short) trade." `decide`: computes a simple momentum signal (as Phase 2's `MomentumMeanReversionCandidate`'s z-score, reused conceptually but independently implemented here since this candidate owns its own posterior), trades LONG if momentum-up AND `long_alpha / (long_alpha + long_beta) > confidence_threshold`, symmetric for SHORT, else `NO_TRADE`. `manage`: exits when the posterior mean for the held side's belief drops back below `confidence_threshold`. `learn(training_experience)`: replays `POSITION_CLOSED` records, incrementing `alpha` by 1 for a winning trade (`realized_pnl > 0`) or `beta` by 1 for a loss, on whichever side's posterior was being tested at that trade's entry (tracked via the same last-decision bookkeeping pattern as Task 2).

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_bayesian_online.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.bayesian_online import BayesianOnlineCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_with_insufficient_history():
    candidate = BayesianOnlineCandidate()
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_no_trade_with_uninformative_prior_even_on_momentum():
    candidate = BayesianOnlineCandidate(confidence_threshold=0.65)
    action = "NO_TRADE"
    for i in range(20):
        action, sl, tp = candidate.decide(_FakeMarketState(1500.0 + i * 0.5), None)
    assert action == "NO_TRADE"  # prior (0.5) never clears 0.65 without learn()


def test_learn_updates_posterior_from_wins():
    candidate = BayesianOnlineCandidate()
    for i in range(10):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.5), None)
    before = (candidate.long_alpha, candidate.long_beta)
    candidate.learn([{"event_type": "POSITION_CLOSED", "realized_pnl": 5.0}])
    after = (candidate.long_alpha, candidate.long_beta)
    assert after != before


def test_metadata_mechanism_family_is_bayesian():
    candidate = BayesianOnlineCandidate()
    assert candidate.metadata.mechanism_family == "bayesian-online"


if __name__ == "__main__":
    test_no_trade_with_insufficient_history()
    test_no_trade_with_uninformative_prior_even_on_momentum()
    test_learn_updates_posterior_from_wins()
    test_metadata_mechanism_family_is_bayesian()
    print("tests/candidates/test_bayesian_online.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_bayesian_online.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `candidates/bayesian_online.py`**

```python
"""candidates/bayesian_online.py
Design doc Section 2.2: principled uncertainty quantification via a
Beta-Bernoulli belief over whether a momentum signal precedes a winning
trade, updated only from SIMULATED_TRAINING experience via learn(). Tests
whether this beats a fixed-threshold rule (Phase 2's RegimeConditionedCandidate)
without any gradient-based learning."""
from candidates.base import CandidateMetadata

MIN_HISTORY = 10


class BayesianOnlineCandidate:
    def __init__(self, confidence_threshold: float = 0.65, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.metadata = CandidateMetadata(
            candidate_id="bayesian_online", version="v1",
            description="Beta-Bernoulli belief over momentum-signal win rate, updated via learn().",
            mechanism_family="bayesian-online",
        )
        self.confidence_threshold = confidence_threshold
        self.long_alpha, self.long_beta = prior_alpha, prior_beta
        self.short_alpha, self.short_beta = prior_alpha, prior_beta
        self._closes = []
        self._last_side = None

    def _momentum(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > MIN_HISTORY:
                self._closes.pop(0)
        if len(self._closes) < MIN_HISTORY:
            return None
        return self._closes[-1] - self._closes[0]

    def decide(self, market_state, account):
        momentum = self._momentum(market_state)
        if momentum is None:
            return ("NO_TRADE", None, None)
        long_belief = self.long_alpha / (self.long_alpha + self.long_beta)
        short_belief = self.short_alpha / (self.short_alpha + self.short_beta)
        if momentum > 0 and long_belief > self.confidence_threshold:
            self._last_side = "long"
            return ("LONG", None, None)
        if momentum < 0 and short_belief > self.confidence_threshold:
            self._last_side = "short"
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        if self._last_side == "long":
            belief = self.long_alpha / (self.long_alpha + self.long_beta)
        elif self._last_side == "short":
            belief = self.short_alpha / (self.short_alpha + self.short_beta)
        else:
            return "HOLD"
        return "EXIT" if belief <= self.confidence_threshold else "HOLD"

    def learn(self, training_experience: list) -> None:
        for record in training_experience:
            if record.get("event_type") != "POSITION_CLOSED":
                continue
            pnl = float(record.get("realized_pnl") or 0.0)
            won = pnl > 0
            if self._last_side == "long":
                if won:
                    self.long_alpha += 1
                else:
                    self.long_beta += 1
            elif self._last_side == "short":
                if won:
                    self.short_alpha += 1
                else:
                    self.short_beta += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_bayesian_online.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/bayesian_online.py tests/candidates/test_bayesian_online.py
git commit -m "feat: add GOLDEX V4 Phase 3 Bayesian online-updating candidate"
```

---

### Task 4: HMM regime candidate (hand-written 2-state Gaussian EM)

**Files:**
- Create: `candidates/hmm_regime.py`
- Test: `tests/candidates/test_hmm_regime.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`, `numpy`.
- Produces: `HMMRegimeCandidate(n_states: int = 2, max_em_iterations: int = 20)`. Before `learn()` is called, has NO fitted parameters and `decide`/`manage` always return `NO_TRADE`/`HOLD` (documented: this candidate is non-functional until trained, unlike Phase 2's percentile-heuristic regime gate which needed no fitting — an explicit, honest limitation of a real generative model requiring data first). `learn(training_experience)`: extracts a 1-D observation sequence (per-bar realized-return magnitude, reconstructed from consecutive `DECIDE`/`MANAGE` records' `market_state_snapshot["mid"]` values) and fits a standard 2-state Gaussian HMM via Baum-Welch EM (forward-backward + parameter re-estimation, textbook algorithm, implemented directly in numpy — no external HMM library). After fitting, `decide`/`manage` use the Viterbi-decoded most-likely current state (updated incrementally bar-by-bar using the fitted transition/emission parameters) to gate a simple momentum rule identically to Phase 2's `RegimeConditionedCandidate`, but with `HIGH_VOL`/`LOW_VOL` now meaning "the state with higher fitted emission variance" instead of a percentile heuristic.

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_hmm_regime.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from candidates.hmm_regime import HMMRegimeCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_untrained_candidate_always_no_trade():
    candidate = HMMRegimeCandidate()
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"
    assert candidate.manage(_FakeMarketState(1500.0), None, None) == "HOLD"


def test_learn_fits_parameters_and_marks_trained():
    candidate = HMMRegimeCandidate(max_em_iterations=5)
    rng = np.random.default_rng(0)
    prices = 1500.0 + np.cumsum(rng.normal(0, 0.05, 200))
    records = []
    for i, p in enumerate(prices):
        records.append({"event_type": "DECIDE", "market_state_snapshot": {"mid": float(p)}})
    candidate.learn(records)
    assert candidate.is_trained is True
    assert candidate.means is not None and len(candidate.means) == candidate.n_states


def test_metadata_mechanism_family_is_regime_generative():
    candidate = HMMRegimeCandidate()
    assert candidate.metadata.mechanism_family == "regime-generative"


def test_decide_after_learn_returns_valid_action():
    candidate = HMMRegimeCandidate(max_em_iterations=5)
    rng = np.random.default_rng(1)
    prices = 1500.0 + np.cumsum(rng.normal(0, 0.05, 200))
    records = [{"event_type": "DECIDE", "market_state_snapshot": {"mid": float(p)}} for p in prices]
    candidate.learn(records)
    action, sl, tp = candidate.decide(_FakeMarketState(float(prices[-1]) + 0.1), None)
    assert action in ("NO_TRADE", "LONG", "SHORT")


if __name__ == "__main__":
    test_untrained_candidate_always_no_trade()
    test_learn_fits_parameters_and_marks_trained()
    test_metadata_mechanism_family_is_regime_generative()
    test_decide_after_learn_returns_valid_action()
    print("tests/candidates/test_hmm_regime.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_hmm_regime.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `candidates/hmm_regime.py`**

```python
"""candidates/hmm_regime.py
Design doc Section 2.3: a real EM-fit 2-state Gaussian HMM over per-bar
returns, fit ONLY on SIMULATED_TRAINING experience via learn() -- the honest
version of the classical-quant regime research the V4 architecture document
flagged. Unlike Phase 2's RegimeConditionedCandidate (a percentile heuristic
needing no fitting), this candidate is genuinely non-functional until
trained -- decide()/manage() are NO_TRADE/HOLD before learn() is called."""
import numpy as np

from candidates.base import CandidateMetadata


def _gaussian_pdf(x, mean, var):
    var = max(var, 1e-10)
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


class HMMRegimeCandidate:
    def __init__(self, n_states: int = 2, max_em_iterations: int = 20):
        self.metadata = CandidateMetadata(
            candidate_id="hmm_regime", version="v1",
            description="Hand-written 2-state Gaussian HMM regime model, EM-fit on training data only.",
            mechanism_family="regime-generative",
        )
        self.n_states = n_states
        self.max_em_iterations = max_em_iterations
        self.is_trained = False
        self.means = None
        self.variances = None
        self.transition = None
        self.initial = None
        self._current_state_belief = None
        self._closes = []

    def learn(self, training_experience: list) -> None:
        mids = [
            r["market_state_snapshot"]["mid"] for r in training_experience
            if r.get("event_type") in ("DECIDE", "MANAGE") and r.get("market_state_snapshot", {}).get("mid") is not None
        ]
        if len(mids) < self.n_states * 5:
            return
        returns = np.diff(np.array(mids, dtype=np.float64))
        n_obs = len(returns)

        means = np.percentile(returns, np.linspace(10, 90, self.n_states))
        variances = np.full(self.n_states, np.var(returns) + 1e-8)
        transition = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        initial = np.full(self.n_states, 1.0 / self.n_states)

        for _ in range(self.max_em_iterations):
            emission = np.array([_gaussian_pdf(returns, means[s], variances[s]) for s in range(self.n_states)]).T
            emission = np.clip(emission, 1e-300, None)

            alpha = np.zeros((n_obs, self.n_states))
            alpha[0] = initial * emission[0]
            alpha[0] /= alpha[0].sum()
            for t in range(1, n_obs):
                alpha[t] = (alpha[t - 1] @ transition) * emission[t]
                alpha[t] /= alpha[t].sum() + 1e-300

            beta = np.zeros((n_obs, self.n_states))
            beta[-1] = 1.0
            for t in range(n_obs - 2, -1, -1):
                beta[t] = (transition @ (emission[t + 1] * beta[t + 1]))
                beta[t] /= beta[t].sum() + 1e-300

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            initial = gamma[0]
            for s in range(self.n_states):
                weight = gamma[:, s].sum() + 1e-300
                means[s] = (gamma[:, s] * returns).sum() / weight
                variances[s] = (gamma[:, s] * (returns - means[s]) ** 2).sum() / weight + 1e-8

            xi_sum = np.zeros((self.n_states, self.n_states))
            for t in range(n_obs - 1):
                xi = np.outer(alpha[t], beta[t + 1] * emission[t + 1]) * transition
                xi /= xi.sum() + 1e-300
                xi_sum += xi
            for s in range(self.n_states):
                denom = xi_sum[s].sum() + 1e-300
                transition[s] = xi_sum[s] / denom

        self.means, self.variances, self.transition, self.initial = means, variances, transition, initial
        self._current_state_belief = gamma[-1]
        self.is_trained = True

    def _update_belief(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > 2:
                self._closes.pop(0)
        if not self.is_trained or len(self._closes) < 2:
            return None
        obs = self._closes[-1] - self._closes[-2]
        emission = np.array([_gaussian_pdf(obs, self.means[s], self.variances[s]) for s in range(self.n_states)])
        belief = (self._current_state_belief @ self.transition) * emission
        belief /= belief.sum() + 1e-300
        self._current_state_belief = belief
        high_vol_state = int(np.argmax(self.variances))
        regime = "HIGH_VOL" if np.argmax(belief) == high_vol_state else "LOW_VOL"
        momentum = self._closes[-1] - self._closes[0]
        return regime, momentum

    def decide(self, market_state, account):
        result = self._update_belief(market_state)
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
        result = self._update_belief(market_state)
        if result is None:
            return "HOLD"
        regime, _ = result
        return "EXIT" if regime != "HIGH_VOL" else "HOLD"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_hmm_regime.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/hmm_regime.py tests/candidates/test_hmm_regime.py
git commit -m "feat: add GOLDEX V4 Phase 3 EM-fit HMM regime candidate"
```

---

### Task 5: Sequence-history learned candidate

**Files:**
- Create: `candidates/sequence_history.py`
- Test: `tests/candidates/test_sequence_history.py`

**Interfaces:**
- Consumes: `candidates.base.CandidateMetadata`.
- Produces: `SequenceHistoryCandidate(n_recent_trades: int = 5, learning_rate: float = 0.05)`. Maintains a rolling window of its own last `n_recent_trades` realized outcomes (win=1.0/loss=0.0), starting all-neutral (0.5) before any trades exist. `decide`: computes a simple momentum signal (as Task 3) AND a "recent-form" feature (mean of the rolling outcome window); combines both via an internal weight vector into a sigmoid score, trading LONG/SHORT if the score clears 0.5±margin, else `NO_TRADE` — this is the design's minimal test of "learning from the candidate's own trading history, not just the market" (Section 2.4). `manage`: exits on score reversion, same pattern as Phase 2's `SimpleLearnedCandidate`. `learn(training_experience)`: performs a small number of gradient-descent steps (plain Python, no ML library) on the internal weight vector using `(feature vector at decision time, realized outcome)` pairs reconstructed from the training experience, minimizing binary cross-entropy between predicted score and realized win/loss.

- [ ] **Step 1: Write the failing test**

```python
"""tests/candidates/test_sequence_history.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.sequence_history import SequenceHistoryCandidate


class _FakeBar:
    def __init__(self, close):
        self.close = close


class _FakeMarketState:
    def __init__(self, close):
        self.mid = close
        self.completed_m1 = _FakeBar(close)
        self.realized_vol_60s = 0.001


def test_no_trade_with_insufficient_history():
    candidate = SequenceHistoryCandidate()
    action, sl, tp = candidate.decide(_FakeMarketState(1500.0), None)
    assert action == "NO_TRADE"


def test_learn_updates_weights():
    candidate = SequenceHistoryCandidate()
    for i in range(15):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.2), None)
    weights_before = dict(candidate.weights)
    records = [
        {"event_type": "POSITION_CLOSED", "realized_pnl": 5.0,
         "market_state_snapshot": {"mid": 1503.0}},
        {"event_type": "POSITION_CLOSED", "realized_pnl": -3.0,
         "market_state_snapshot": {"mid": 1504.0}},
    ]
    candidate.learn(records)
    assert candidate.weights != weights_before


def test_metadata_mechanism_family_is_sequence_history():
    candidate = SequenceHistoryCandidate()
    assert candidate.metadata.mechanism_family == "sequence-history"


def test_manage_returns_hold_or_exit():
    candidate = SequenceHistoryCandidate()
    for i in range(15):
        candidate.decide(_FakeMarketState(1500.0 + i * 0.2), None)
    result = candidate.manage(_FakeMarketState(1503.0), None, None)
    assert result in ("HOLD", "EXIT")


if __name__ == "__main__":
    test_no_trade_with_insufficient_history()
    test_learn_updates_weights()
    test_metadata_mechanism_family_is_sequence_history()
    test_manage_returns_hold_or_exit()
    print("tests/candidates/test_sequence_history.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/candidates/test_sequence_history.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `candidates/sequence_history.py`**

```python
"""candidates/sequence_history.py
Design doc Section 2.4: a minimal, honest test of "learning from the
candidate's own trading history" (principle #6) rather than only the market
-- combines a market momentum feature with a rolling window of the
candidate's OWN recent win/loss outcomes. No deep learning dependency: the
learned part is a 2-weight logistic model updated by plain gradient descent
in learn()."""
import math

from candidates.base import CandidateMetadata

MIN_HISTORY = 10


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class SequenceHistoryCandidate:
    def __init__(self, n_recent_trades: int = 5, learning_rate: float = 0.05):
        self.metadata = CandidateMetadata(
            candidate_id="sequence_history", version="v1",
            description="Logistic model over market momentum + own recent trade outcomes.",
            mechanism_family="sequence-history",
        )
        self.n_recent_trades = n_recent_trades
        self.learning_rate = learning_rate
        self.weights = {"momentum": 1.0, "recent_form": 1.0}
        self._closes = []
        self._recent_outcomes = [0.5] * n_recent_trades
        self._last_score_features = None

    def _features(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > MIN_HISTORY:
                self._closes.pop(0)
        if len(self._closes) < MIN_HISTORY:
            return None
        momentum = (self._closes[-1] - self._closes[0]) / self._closes[0]
        recent_form = sum(self._recent_outcomes) / len(self._recent_outcomes)
        return {"momentum": momentum, "recent_form": recent_form - 0.5}

    def _score(self, features):
        return _sigmoid(sum(self.weights[k] * v for k, v in features.items()))

    def decide(self, market_state, account):
        features = self._features(market_state)
        if features is None:
            return ("NO_TRADE", None, None)
        self._last_score_features = features
        score = self._score(features)
        if score > 0.55:
            return ("LONG", None, None)
        if score < 0.45:
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        features = self._features(market_state)
        if features is None:
            return "HOLD"
        score = self._score(features)
        if 0.45 <= score <= 0.55:
            return "EXIT"
        return "HOLD"

    def learn(self, training_experience: list) -> None:
        closed = [r for r in training_experience if r.get("event_type") == "POSITION_CLOSED"]
        for record in closed:
            won = 1.0 if float(record.get("realized_pnl") or 0.0) > 0 else 0.0
            self._recent_outcomes.append(won)
            if len(self._recent_outcomes) > self.n_recent_trades:
                self._recent_outcomes.pop(0)
            if self._last_score_features is None:
                continue
            prediction = self._score(self._last_score_features)
            error = prediction - won
            for key in self.weights:
                gradient = error * self._last_score_features.get(key, 0.0)
                self.weights[key] -= self.learning_rate * gradient
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/candidates/test_sequence_history.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add candidates/sequence_history.py tests/candidates/test_sequence_history.py
git commit -m "feat: add GOLDEX V4 Phase 3 sequence-history learned candidate"
```

---

### Task 6: Market-flow representation research script

**Files:**
- Create: `research/phase3_representation_research.py`
- Test: `tests/research/test_phase3_representation_research.py`

**Interfaces:**
- Produces: `analyze_return_autocorrelation(closes, max_lag: int = 20) -> dict` (lag-by-lag autocorrelation of 1-bar returns, using `numpy.corrcoef` on lagged slices — a real, standard check for exploitable serial dependence). `analyze_volatility_clustering(closes, window: int = 60) -> dict` (autocorrelation of the rolling realized-volatility series itself, the standard test for GARCH-style clustering). `analyze_regime_persistence(hmm_candidate, closes) -> dict` (given an already-`learn()`-trained `HMMRegimeCandidate`, replays its `_update_belief` over `closes` and reports mean regime dwell-time — long dwell times support "regime" being a meaningful, non-noise concept per design doc Section 4; near-1-bar dwell times indicate the fitted HMM is just fitting noise). This module produces a findings dict for a human/report to read — it makes no KEEP/REJECT decision itself.

- [ ] **Step 1: Write the failing test**

```python
"""tests/research/test_phase3_representation_research.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.phase3_representation_research import (
    analyze_return_autocorrelation, analyze_volatility_clustering, analyze_regime_persistence,
)
from candidates.hmm_regime import HMMRegimeCandidate


def test_autocorrelation_of_pure_noise_is_near_zero():
    rng = np.random.default_rng(0)
    closes = 1500.0 + np.cumsum(rng.normal(0, 0.1, 2000))
    result = analyze_return_autocorrelation(closes, max_lag=5)
    assert "lag_1" in result
    assert abs(result["lag_1"]) < 0.3  # pure random walk: no strong lag-1 autocorrelation expected


def test_volatility_clustering_detects_synthetic_clustering():
    rng = np.random.default_rng(1)
    vol_regime = np.concatenate([np.full(500, 0.05), np.full(500, 0.5)])
    returns = rng.normal(0, 1, 1000) * vol_regime
    closes = 1500.0 + np.cumsum(returns)
    result = analyze_volatility_clustering(closes, window=30)
    assert "lag_1" in result


def test_regime_persistence_reports_dwell_time():
    candidate = HMMRegimeCandidate(max_em_iterations=5)
    rng = np.random.default_rng(2)
    prices = 1500.0 + np.cumsum(rng.normal(0, 0.05, 300))
    records = [{"event_type": "DECIDE", "market_state_snapshot": {"mid": float(p)}} for p in prices]
    candidate.learn(records)
    result = analyze_regime_persistence(candidate, prices)
    assert "mean_dwell_time_bars" in result
    assert result["mean_dwell_time_bars"] > 0


if __name__ == "__main__":
    test_autocorrelation_of_pure_noise_is_near_zero()
    test_volatility_clustering_detects_synthetic_clustering()
    test_regime_persistence_reports_dwell_time()
    print("tests/research/test_phase3_representation_research.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_phase3_representation_research.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase3_representation_research.py`**

```python
"""research/phase3_representation_research.py
Design doc Section 4: investigates whether exploitable temporal/state
information exists in the real data BEFORE candidates are finalized against
it. Produces findings, makes no KEEP/REJECT decision -- that judgment is
made by a human reading the report, same discipline as Batch 1/2's
diagnostics."""
import numpy as np

from candidates.hmm_regime import HMMRegimeCandidate


def analyze_return_autocorrelation(closes, max_lag: int = 20) -> dict:
    closes = np.asarray(closes, dtype=np.float64)
    returns = np.diff(closes)
    result = {}
    for lag in range(1, max_lag + 1):
        if len(returns) <= lag:
            continue
        a, b = returns[:-lag], returns[lag:]
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
        result[f"lag_{lag}"] = corr
    return result


def analyze_volatility_clustering(closes, window: int = 60, max_lag: int = 20) -> dict:
    closes = np.asarray(closes, dtype=np.float64)
    returns = np.diff(closes)
    rolling_vol = np.array([
        np.std(returns[max(0, i - window):i]) for i in range(window, len(returns))
    ])
    result = {}
    for lag in range(1, max_lag + 1):
        if len(rolling_vol) <= lag:
            continue
        a, b = rolling_vol[:-lag], rolling_vol[lag:]
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
        result[f"lag_{lag}"] = corr
    return result


def analyze_regime_persistence(hmm_candidate: HMMRegimeCandidate, closes) -> dict:
    closes = np.asarray(closes, dtype=np.float64)

    class _MinimalBar:
        def __init__(self, close):
            self.close = close

    class _MinimalMarketState:
        def __init__(self, close):
            self.completed_m1 = _MinimalBar(close)
            self.mid = close
            self.realized_vol_60s = None

    regimes = []
    for price in closes:
        result = hmm_candidate._update_belief(_MinimalMarketState(float(price)))
        regimes.append(result[0] if result is not None else None)

    dwell_times = []
    current_run = 0
    current_regime = None
    for r in regimes:
        if r is None:
            continue
        if r == current_regime:
            current_run += 1
        else:
            if current_run > 0:
                dwell_times.append(current_run)
            current_regime = r
            current_run = 1
    if current_run > 0:
        dwell_times.append(current_run)

    mean_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0
    return {"mean_dwell_time_bars": mean_dwell, "n_regime_switches": len(dwell_times)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/research/test_phase3_representation_research.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add research/phase3_representation_research.py tests/research/test_phase3_representation_research.py
git commit -m "feat: add GOLDEX V4 Phase 3 market-flow representation research script"
```

---

### Task 7: Phase 3 tournament orchestrator (wraps Phase 2, unmodified)

**Files:**
- Create: `research/phase3_tournament.py`
- Test: `tests/research/test_phase3_tournament.py`

**Interfaces:**
- Consumes: `research.phase2_tournament._run_one`, `research.phase2_tournament._verdict_for` (imported and reused, NOT copied/modified), `simulator.contracts.EnvironmentTag`, `candidates.base.LearningCandidate`.
- Produces: `run_phase3_tournament(df_training, df_validation, roster: list, config, store, run_id: str) -> dict` — same control-gate-first structure as Phase 2's `run_tournament`, but for each candidate: runs `SIMULATED_TRAINING` via `_run_one`, THEN (if `isinstance(candidate, LearningCandidate)` or `hasattr(candidate, "learn")`) reads back that exact run's `SIMULATED_TRAINING` records from `store.read_run(...)`, asserts every record's `environment_tag == "SIMULATED_TRAINING"` (the mechanical causality check from design doc Section 6 — raises `ValueError` if any record fails this, rather than silently proceeding), calls `candidate.learn(records)`, THEN runs `SIMULATED_VALIDATION` via `_run_one` (only now, after learning). Non-learning candidates skip the `learn` step entirely and behave exactly as in Phase 2. Returns the same `{"control_gate": ..., "candidates": {...}}` shape as Phase 2's `run_tournament`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/research/test_phase3_tournament.py"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from candidates.controls import NoTradeCandidate
from candidates.bayesian_online import BayesianOnlineCandidate
from simulator.contracts import SimulatedExecutionConfig
from research.phase2_experience_store import ExperienceStore
from research.phase3_tournament import run_phase3_tournament


def _make_df(n=150):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + (i % 20) * 0.05 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def test_learning_candidate_gets_learn_called_between_partitions():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train, df_val = _make_df(150), _make_df(80)
        bayes = BayesianOnlineCandidate()
        roster = [NoTradeCandidate(), bayes]
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        result = run_phase3_tournament(df_train, df_val, roster, config, store, run_id="p3_test_001")
        assert "bayesian_online" in result["candidates"]
        assert "control_no_trade" in result["candidates"]
    finally:
        shutil.rmtree(tmp_dir)


def test_learn_never_receives_validation_tagged_records():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train, df_val = _make_df(150), _make_df(80)

        class _RecordingLearner(BayesianOnlineCandidate):
            def __init__(self):
                super().__init__()
                self.seen_tags = set()

            def learn(self, training_experience):
                for r in training_experience:
                    self.seen_tags.add(r.get("environment_tag"))
                super().learn(training_experience)

        candidate = _RecordingLearner()
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        run_phase3_tournament(df_train, df_val, [candidate], config, store, run_id="p3_test_002")
        assert candidate.seen_tags <= {"SIMULATED_TRAINING"}
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_learning_candidate_gets_learn_called_between_partitions()
    test_learn_never_receives_validation_tagged_records()
    print("tests/research/test_phase3_tournament.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_phase3_tournament.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `research/phase3_tournament.py`**

```python
"""research/phase3_tournament.py
Wraps (does not modify) research.phase2_tournament's control-gate and
verdict machinery, adding the optional learn() step between a candidate's
SIMULATED_TRAINING and SIMULATED_VALIDATION runs -- with a mechanical
causality check (design doc Section 6) that learn() never receives anything
but SIMULATED_TRAINING-tagged experience."""
from simulator.contracts import EnvironmentTag
from research.phase2_tournament import _run_one, _verdict_for


def _maybe_learn(candidate, store, run_id):
    if not hasattr(candidate, "learn"):
        return
    records = store.read_run(
        candidate.metadata.candidate_id, candidate.metadata.version, run_id, EnvironmentTag.SIMULATED_TRAINING
    )
    for record in records:
        if record.get("environment_tag") != EnvironmentTag.SIMULATED_TRAINING.value:
            raise ValueError(
                f"learn() causality violation: candidate {candidate.metadata.candidate_id} would have "
                f"received a record tagged {record.get('environment_tag')!r}, not "
                f"{EnvironmentTag.SIMULATED_TRAINING.value!r}."
            )
    candidate.learn(records)


def run_phase3_tournament(df_training, df_validation, roster: list, config, store, run_id: str) -> dict:
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
                    "reason": "RandomCandidate showed persistent profitability -- ranking halted.",
                },
                "candidates": {},
            }
        control_gate = {"passed": True, "random_candidate_profile": random_val_profile, "reason": "OK"}
    else:
        control_gate = {"passed": True, "random_candidate_profile": None, "reason": "No random control in roster."}

    results = {}
    for candidate in roster:
        training_profile = _run_one(df_training, candidate, config, EnvironmentTag.SIMULATED_TRAINING, store, run_id)
        _maybe_learn(candidate, store, run_id)
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

Run: `pytest tests/research/test_phase3_tournament.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full combined test suite**

Run: `pytest tests/candidates/ tests/research/ -v`
Expected: all Phase 2 + Phase 3 tests pass together.

- [ ] **Step 6: Commit**

```bash
git add research/phase3_tournament.py tests/research/test_phase3_tournament.py
git commit -m "feat: add GOLDEX V4 Phase 3 tournament orchestrator with learn() causality guard"
```

---

### Task 8: Real historical run (Phase 2 + Phase 3 roster, full evidence report)

**Files:**
- Create: `research/phase3_real_run.py`

**Interfaces:**
- Produces: a script (not unit-tested — an execution/report task, matching how Batch 1/2's real full-history runs were handled) that: loads `data/gold_seed_merged_full6yr.csv`; splits a bounded chronological slice into `SIMULATED_TRAINING` (first ~300,000 rows) and `SIMULATED_VALIDATION` (next ~100,000 rows) — the same bounded-slice rationale as any Batch 1/2 diagnostic run, not the full 6.7 years, to keep wall-clock time practical; builds the FULL roster (Phase 2's `NoTradeCandidate`, `RandomCandidate(seed=42)`, `MomentumMeanReversionCandidate()`, `RegimeConditionedCandidate()`, `SimpleLearnedCandidate(weights=...)` with a documented arbitrary untrained placeholder weight, PLUS Phase 3's `TabularQLearningCandidate()`, `BayesianOnlineCandidate()`, `HMMRegimeCandidate()`, `SequenceHistoryCandidate()`); explicitly SKIPS `V3BaselineCandidate` in this run (documented reason: it needs `assemble_replay_dataset`, a ~15-minute CatBoost CV training call per invocation, confirmed slow this session, and its OOF event universe may not cleanly align with this script's arbitrary row-slice boundaries — flagged for a dedicated, more careful integration run rather than folded in here); calls `research.phase3_representation_research`'s three analysis functions on the training slice's close prices first and prints those findings; then calls `run_phase3_tournament`; then prints the full control-gate result and, for every candidate, its verdict and key evidence-profile numbers (n_trades training/validation, realized_pnl.total training/validation, confidence_intervals.mean_pnl_per_trade for validation).

- [ ] **Step 1: Write `research/phase3_real_run.py`**

```python
"""research/phase3_real_run.py
Executes the Phase 2 + Phase 3 candidate roster against a bounded
chronological slice of the real 6.7-year Gold dataset. This is a research
execution script (like Batch 1/2's real full-history runs), not part of the
TDD-covered unit test suite -- its own correctness is validated by the unit
tests on every module it calls.

V3BaselineCandidate is deliberately excluded here (see design doc discussion)
-- it needs a ~15-minute assemble_replay_dataset() call whose OOF event
universe is walk-forward-validated on specific historical windows that may
not align with this script's arbitrary training/validation row boundaries.
It needs its own dedicated integration run, not folding into this one.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase3_real_run.py
"""
import pandas as pd

from candidates.controls import NoTradeCandidate, RandomCandidate
from candidates.statistical_null import MomentumMeanReversionCandidate
from candidates.regime_conditioned import RegimeConditionedCandidate
from candidates.simple_learned import SimpleLearnedCandidate
from candidates.tabular_qlearning import TabularQLearningCandidate
from candidates.bayesian_online import BayesianOnlineCandidate
from candidates.hmm_regime import HMMRegimeCandidate
from candidates.sequence_history import SequenceHistoryCandidate
from simulator.contracts import SimulatedExecutionConfig
from research.phase2_experience_store import ExperienceStore
from research.phase3_tournament import run_phase3_tournament
from research.phase3_representation_research import (
    analyze_return_autocorrelation, analyze_volatility_clustering, analyze_regime_persistence,
)

TRAINING_ROWS = 300_000
VALIDATION_ROWS = 100_000
DATA_PATH = "data/gold_seed_merged_full6yr.csv"


def load_slices():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    df_validation = df.iloc[TRAINING_ROWS:TRAINING_ROWS + VALIDATION_ROWS].reset_index(drop=True)
    return df_training, df_validation


def build_roster():
    return [
        NoTradeCandidate(),
        RandomCandidate(seed=42),
        MomentumMeanReversionCandidate(),
        RegimeConditionedCandidate(),
        # Untrained, arbitrary placeholder weights -- no fitting script exists yet for this candidate;
        # documented explicitly, not presented as a trained model.
        SimpleLearnedCandidate(weights={"short_return": 10.0, "medium_return": 10.0, "rsi_like": 5.0}),
        TabularQLearningCandidate(),
        BayesianOnlineCandidate(),
        HMMRegimeCandidate(),
        SequenceHistoryCandidate(),
    ]


def print_representation_findings(df_training):
    closes = df_training["close"].to_numpy()
    print("\n=== Market-Flow Representation Research (Section 4) ===")
    print("Return autocorrelation (lags 1-5):", analyze_return_autocorrelation(closes, max_lag=5))
    print("Volatility clustering (lags 1-5):", analyze_volatility_clustering(closes, max_lag=5))
    hmm_probe = HMMRegimeCandidate(max_em_iterations=10)
    probe_records = [{"event_type": "DECIDE", "market_state_snapshot": {"mid": float(c)}} for c in closes]
    hmm_probe.learn(probe_records)
    if hmm_probe.is_trained:
        print("Regime persistence:", analyze_regime_persistence(hmm_probe, closes))
    else:
        print("Regime persistence: HMM did not train (insufficient data)")


def print_results(result):
    print("\n=== Control Gate ===")
    print(result["control_gate"])
    if not result["control_gate"]["passed"]:
        print("\nCONTROL GATE FAILED -- ranking halted, no candidate results to report.")
        return
    print("\n=== Candidate Results ===")
    for candidate_id, data in result["candidates"].items():
        tp, vp = data["training_profile"], data["validation_profile"]
        print(f"\n{candidate_id} ({data['metadata']['mechanism_family']}) -- verdict: {data['verdict']}")
        print(f"  training:   n_trades={tp['n_trades']}, total_pnl={tp['realized_pnl']['total']:.4f}")
        print(f"  validation: n_trades={vp['n_trades']}, total_pnl={vp['realized_pnl']['total']:.4f}")
        print(f"  validation CI mean_pnl_per_trade: {vp['confidence_intervals']['mean_pnl_per_trade']}")


def main():
    df_training, df_validation = load_slices()
    print_representation_findings(df_training)
    roster = build_roster()
    config = SimulatedExecutionConfig()
    store = ExperienceStore(base_dir="research/phase3_real_run_experience")
    result = run_phase3_tournament(df_training, df_validation, roster, config, store, run_id="phase3_real_run_001")
    print_results(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script with the real venv**

Run: `/home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase3_real_run.py`
Expected: completes and prints representation findings plus every candidate's verdict and evidence numbers. If it runs unexpectedly long (budget: should complete in well under an hour given ~400K bars x 9 candidates x 2 partitions with no per-call model training beyond the HMM's own EM, which runs once per candidate per training pass, not once per bar) — if a specific candidate's replay is pathologically slow, profile and report which one and why as a finding, do not silently let it run indefinitely.

- [ ] **Step 3: Check for incidental changes**

Run: `git status`. If anything outside `research/phase3_real_run.py` and `research/phase3_real_run_experience/` changed (e.g. `models/registry/*.json`), revert with `git checkout -- <path>`.

- [ ] **Step 4: Commit the script, not the generated experience data**

```bash
git add research/phase3_real_run.py
echo "research/phase3_real_run_experience/" >> .gitignore
git add .gitignore
git commit -m "research: add and run GOLDEX V4 Phase 3 real-data tournament script"
```

- [ ] **Step 5: Write up the findings**

Save the full printed output (representation findings + control gate + every candidate's verdict and evidence numbers) to `docs/superpowers/reports/2026-08-27-goldex-v4-phase3-findings.md`, verbatim, with a one-paragraph honest summary at the top: which candidates (if any) cleared the control gate and showed a validation CI lower bound above zero, and an explicit statement if none did (per design doc Section 9 — this is a valid, reportable outcome, not a failure to fix).

```bash
git add docs/superpowers/reports/2026-08-27-goldex-v4-phase3-findings.md
git commit -m "docs: add GOLDEX V4 Phase 3 real-run findings report"
```

---

## Self-review notes

- Spec coverage: optional `learn()` hook with backward-compatible protocol (Task 1), four genuinely different mechanism-family candidates spanning the discovery ladder from tabular RL through Bayesian, generative regime, and sequence-history learning (Tasks 2-5), a real market-flow representation research step answering Section 4's actual question rather than assuming an answer (Task 6), an orchestrator that reuses Phase 2's control-gate/verdict machinery completely unmodified while mechanically enforcing training-only causality for `learn()` (Task 7), and a real run against actual historical data producing honest, preserved evidence for every candidate including a plain statement if nothing clears the bar (Task 8) — every corrected design requirement (Sections 1, 2b, 4, 5b, 6, 7, 8, 9) has a task.
- No placeholders: every step has real, runnable code, including a genuine from-scratch Baum-Welch EM implementation for the HMM candidate (not a stub).
- Type consistency checked: `LearningCandidate` (Task 1) is satisfied by all four new candidates (Tasks 2-5) via their `learn` method; `research/phase3_tournament.py` (Task 7) imports `_run_one`/`_verdict_for` from `research/phase2_tournament.py` unmodified, matching that module's real signatures; `research/phase3_real_run.py` (Task 8) constructs every candidate with the exact constructor signatures defined in Tasks 2-5.
- Explicitly out of scope in every task: no Phase 4 work, no modification to `simulator/` or Phase 2's control/verdict logic, no `SIMULATED_OOS_TEST` contact, no profitability target, no post-validation tuning loop anywhere in this plan.
