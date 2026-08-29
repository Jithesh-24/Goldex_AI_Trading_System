# GOLDEX Phase 2: Fast Tier Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans

Governs: "GOLDEX PHASE 2 — ARCHITECTURE APPROVED / PROCEED TO IMPLEMENTATION PLANNING"
mandate. Spec: `docs/superpowers/specs/2026-08-29-goldex-phase2-final-architecture-decision.md`
(Sections 3-15, 19). Builds the Fast Tier only (mandate Section 12's 15-item scope) —
**no Slow Tier**, per the mandate's explicit instruction not to build it without a
concrete demonstrated need. No trained model is deployed; no profitability claim is
made anywhere in this plan.

Grounded in the CURRENT Phase 1 code (post-commit `1654457`), verified by direct
inspection, not recollection — every interface cited below is a real, current
file:line, not an assumption carried over from the prior (unexecuted) Track A plan.

## Global constraints

- `intelligence/` does not exist yet — this plan creates it.
- No modification to `simulator/`, `market/`, `contracts/` unless a genuine Phase 1
  defect is found during implementation; if found, STOP and document rather than
  silently patch (mandate Section 15).
- No Slow Tier, no LLM integration, no discovery mechanism — out of scope per mandate
  Section 12.
- No claim of validated trading signal or profitability anywhere, including test names
  and docstrings.
- Every task ships with a test, following the existing convention: `sys.path.insert(0,
  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))`,
  plain pytest functions, `_make_df`-style synthetic-data helpers where useful.
- `.venv` (numpy/pandas/pydantic/pytest/PyYAML/numba/scikit-learn per
  `requirements.txt`) is the test environment. `scipy` is currently an *undeclared
  transitive dependency* (pulled in by scikit-learn, not in `requirements.txt`) — Task
  1 makes it explicit, since the Bayesian posterior mechanism (Task 3) needs it
  directly, not just transitively.
- Latency budget for anything on the Fast Tier's execution-critical path: informed by
  Phase 1's measured numbers (`build_snapshot` p99 ≈ 2ms, execution/recording in
  low-microseconds) — the Fast Tier's own inference must be measured against this
  scale, not assumed compatible (Task 12).

## Known risk carried into this plan, not resolved by it

`fit_garch11` (`research/phase4_garch_volatility_mechanism.py:66`) and
`kalman_level_trend_filter` (`research/phase4_kalman_trend_mechanism.py:38`) are both
**O(n) full-history recompute on every call** — a naive per-decision wrapper would
refit GARCH's Nelder-Mead optimization from scratch on every bar, which is
incompatible with any short-duration decision loop. Task 5 addresses this directly
with a periodic-refit strategy, not a full rebuild of these functions (out of scope —
they are validated Phase 3A/4 research code, reused unchanged per the mandate's
"identify what can be reused" instruction).

## Tasks

### Task 1 — `requirements.txt`: declare `scipy`

Add `scipy==1.18.1` (the version already resolved transitively, confirmed installed)
to `requirements.txt` as an explicit line. One-line change, no test needed beyond
confirming `.venv/bin/pip check` (or equivalent) shows no conflict — this is
dependency hygiene, following the exact precedent Task 5 of the Phase 1 plan set
(pinning `pyyaml` after it was found silently missing).

### Task 2 — `intelligence/evidence.py`: the upgraded EvidenceSource contract

Create `intelligence/` (with `__init__.py`) and `intelligence/evidence.py`. This is
the "quantitative knowledge/tool interface" (mandate Section 12, item 1) — the richer
contract from the architecture spec's Section 5, not a bare `(value, confidence)`
tuple:

```python
@dataclass
class EvidenceValue:
    value: Optional[float]
    confidence: float
    source_name: str

@dataclass
class EvidenceSourceSpec:
    name: str
    mathematical_formulation: str
    required_inputs: list[str]
    assumptions: str
    known_failure_conditions: str
    compute: Callable[[np.ndarray], EvidenceValue]
    computational_cost_hint: Optional[str] = None  # populated by Task 12's latency
                                                     # instrumentation, not guessed here

class EvidenceRegistry:
    def register(self, spec: EvidenceSourceSpec) -> None: ...
    def names(self) -> list[str]: ...
    def specs(self) -> dict[str, EvidenceSourceSpec]: ...  # for Task 4's applicability
                                                              # representation to consume
    def compute_all(self, closes_so_far: np.ndarray) -> dict[str, EvidenceValue]:
        # exception-isolated per source, exactly as the prior unexecuted Track A
        # plan specified — a broken source returns EvidenceValue(None, 0.0, name),
        # never crashes the registry
        ...
```

Test: `tests/intelligence/test_evidence.py` — dataclass field checks, duplicate-name
rejection, `compute_all` calling every source with the same array, exception
isolation, and a new test the prior plan didn't have: `specs()` returns the full
metadata (mathematical_formulation etc.) unmodified, since Task 4/10 depend on it
being genuinely available, not decorative.

### Task 3 — `intelligence/evidence_sources.py`: wrap the 9 validated functions

`build_default_registry() -> EvidenceRegistry`, wrapping (confirmed current
signatures, no drift from the prior plan's assumptions except where noted):

- `momentum_scalar(closes, lookback=10)` — `research/phase3a_representation_experiments.py:151`
- `path_pca_projection(closes, window=15)` — same file, `:95`
- `multiscale_vol_summary(closes, windows=(10,30,100))` — same file, `:114`, returns
  `(ratio, vols_dict)`, **and** `vol_regime_transition(vols_dict[10], n_bins=3)` —
  `:134` — must be chained (takes the short-window vol array from the first call's
  output, not raw closes) — confirmed this dependency, wrapper must call both in
  sequence, not independently as two unrelated sources.
- `fit_garch11(returns)` — `research/phase4_garch_volatility_mechanism.py:66`,
  returns `((omega, alpha, beta), sigma2)` — confirmed exact shape. Wrapped via
  Task 5's periodic-refit strategy, not called fresh per bar.
- `kalman_level_trend_filter(closes, ...)` — `research/phase4_kalman_trend_mechanism.py:38`,
  returns `(levels, velocities, innovations)` — same periodic-refit treatment as GARCH
  if its O(n) cost proves prohibitive at the latency budget (Task 5/12 determine this
  empirically, not by assumption).
- `_rolling_moment(returns, window, order, center=True)` —
  `research/phase4_distributional_mechanism.py:41`, private but reused (order=3 for
  skew, order=4 for excess kurtosis), `WINDOW=30` at `:31`.

Each wrapper returns `EvidenceValue(None, 0.0, name)` on insufficient history,
otherwise the last finite value from re-invoking the batch function on
`closes_so_far`. Each `EvidenceSourceSpec`'s `mathematical_formulation`/
`assumptions`/`known_failure_conditions` fields are populated from the source
functions' own docstrings (already substantive — e.g. `phase4_garch_volatility_mechanism.py`'s
module docstring documents the from-scratch-implementation limitation) — not invented
prose.

Test: `tests/intelligence/test_evidence_sources.py` — one test per wrapper confirming
correct chaining (especially `multiscale_vol_summary` → `vol_regime_transition`), plus
the causality/no-look-ahead test from the prior plan
(`test_no_look_ahead_truncated_recompute_matches`) — this is the load-bearing
correctness gate for every wrapper and is not weakened relative to the prior plan's
version.

### Task 4 — `intelligence/applicability.py`: conditional applicability representation

Mandate Section 12, item 3 — explicit, not folded silently into Task 5's trust
mechanism. A tool's `known_failure_conditions` (Task 2) is qualitative prose; this
task makes it machine-checkable where possible: a small `ApplicabilityCheck`
per-source, e.g. "requires ≥ N bars of history" (mechanical, derivable from the
wrapped function's own window/lookback parameter) and "unreliable during the first N
bars after a data-quality flag" (checkable against `MarketState.data_quality`/
`market_closed`, already available per Phase 1). This is deliberately minimal — a set
of boolean gates a source's `EvidenceValue` passes through before entering Task 5's
posterior update, not a learned applicability model (that's what Task 5's
context-conditioning already does; this task is the *hard* floor beneath it, e.g. "do
not trust GARCH's output if it hasn't refit since the last 500 bars," derived
mechanically from Task 5's refit cadence).

Test: a source whose applicability check fails contributes `confidence=0.0` to the
downstream posterior regardless of its computed value — verified directly.

### Task 5 — `intelligence/fast_tier.py`: Bayesian adaptive-trust mechanism

The core of the Fast Tier (mandate Section 1: explicitly not a static weighted
average). For each `EvidenceSource`, maintain a Beta-distributed posterior belief
(`scipy.stats.beta`, Task 1) over "this source's directional signal agreed with the
eventual realized outcome," conditioned on a small number of continuous context
buckets derived from the recursive state-space sources themselves (GARCH conditional
variance, Kalman velocity) — not a hardcoded regime label (mandate Section 3,
verified: no `if trend: ... elif range: ...` branch anywhere in this module).

```python
class ToolTrust:
    # per (source_name, context_bucket): Beta(alpha, beta) parameters
    def update(self, source_name: str, context_bucket: int, agreed: bool) -> None: ...
    def posterior_mean(self, source_name: str, context_bucket: int) -> float: ...
    def posterior_uncertainty(self, source_name: str, context_bucket: int) -> float: ...

def context_bucket(evidence: dict[str, EvidenceValue]) -> int:
    # derived from GARCH sigma2 / Kalman velocity magnitude, discretized into a small
    # fixed number of continuous-valued buckets — NOT trend/range/breakout labels
    ...

class FastTierReasoner:
    def hypothesis(self, evidence: dict[str, EvidenceValue], trust: ToolTrust) -> Hypothesis:
        # Hypothesis = net directional belief + aggregate uncertainty + which sources
        # were load-bearing (their applicability-gated, trust-weighted contribution
        # exceeded a floor) — this list feeds Task 7's thesis memory directly
        ...
```

Refit cadence for GARCH/Kalman (the Task 3 risk): `FastTierReasoner` calls the
periodic-refit wrappers at a configurable bar interval (e.g. every 50 bars, not every
bar), caching the fitted parameters between refits and applying them to new
observations incrementally where the underlying function supports it, or accepting
staleness up to the refit interval where it doesn't — measured explicitly in Task 12,
not assumed acceptable.

Test: `tests/intelligence/test_fast_tier.py` — a source that is repeatedly right in
context bucket A and repeatedly wrong in bucket B ends up with a higher posterior mean
in A than B (the direct test of mandate Section 1's "conditional usefulness, not fixed
weight 0.27"); a source given contradictory evidence alongside another source
produces a `Hypothesis` with elevated `aggregate_uncertainty`, not a silently averaged
midpoint (mandate Section 10's contradiction-handling test, made concrete); when every
source's applicability check fails, `Hypothesis` supports genuine abstention
(confidence low enough that Task 6's decision interface returns NO_TRADE), not a
forced trade.

### Task 6 — `intelligence/decision_engine.py`: decision interface

Implements Phase 1's `DecisionEngine`/`DecideFn` seam
(`simulator/replay.py:20-24`, confirmed current signature: `(action, sl_price,
tp_price, size)` 4-tuple, backward-compatible 3-tuple still supported) —
no new interface invented, this plugs into what already exists:

```python
class FastTierDecisionEngine:
    def __init__(self, registry: EvidenceRegistry, trust: ToolTrust, reasoner: FastTierReasoner,
                 ev_cost_gate: Callable, sizing_bootstrap: Callable, sltp_bootstrap: Callable): ...
    def decide(self, market_state, account) -> tuple:
        # evidence = registry.compute_all(...); hypothesis = reasoner.hypothesis(evidence, trust)
        # gated by ev_cost_gate (reuses simulator.cost_model.round_trip_cost_r, confirmed
        # signature simulator/cost_model.py:13, unmodified) -> NO_TRADE|LONG|SHORT + SL/TP/size
        # from the K/L bootstrap (Task 11)
        ...
    def manage(self, market_state, position_view, account) -> str:
        # Task 8's continuous reassessment
        ...
```

Test: `tests/intelligence/test_decision_engine.py` — integration test running this
engine through `simulator.replay.run_replay` on synthetic data (matching the existing
`_make_df` convention), confirming the DECIDE/MANAGE contract is honored exactly and
that a rejected entry (Phase 1's `INSUFFICIENT_MARGIN`/`INVALID_SL_WRONG_SIDE`/
`INVALID_TP_WRONG_SIDE` — confirmed rejection reasons, `simulator/engine.py:23`) is
correctly excluded from Task 9's credit assignment.

### Task 7 — `intelligence/thesis.py`: thesis memory

Mandate Section 6/12 item 7. Retains, only while a position is open, the specific
`(source_name, context_bucket, contribution)` tuples that were load-bearing at entry
(from Task 5's `Hypothesis.load_bearing_sources`), discarded at exit — no persistence
beyond the position's lifetime, enforced by construction (the object lives inside
`FastTierDecisionEngine`'s per-position state, not a module-level dict that could leak
across positions).

Test: thesis is `None` when flat, populated at entry with exactly the load-bearing
sources, cleared immediately after exit — verified across an open→hold→close sequence
via `run_replay`.

### Task 8 — continuous position reassessment (extends Task 6's `manage()`)

Mandate Section 6/12 item 8. At each MANAGE step: re-evaluate the same sources
recorded in Task 7's thesis; if their current `EvidenceValue`s have moved against the
stored thesis direction by more than a configurable threshold (thesis-invalidation,
mandate Section 8/10 of the architecture spec), return `"EXIT"`; otherwise `"HOLD"`.
The static SL/TP set at entry (Task 11's bootstrap) remains the safety floor
underneath this — this method can trigger an earlier exit, never override or loosen
the SL/TP Phase 1's engine enforces (mandate Section 15's explicit "safety constraints
remain underneath the intelligence").

Test: a position whose thesis sources reverse direction mid-hold triggers `"EXIT"`
before the static SL/TP would have; a position whose thesis sources remain consistent
holds through to a normal SL/TP-driven or forced close.

### Task 9 — `intelligence/credit_assignment.py`: trade credit assignment

Mandate Section 7 — explicitly calls out getting this right given a specifically-named
"Phase 3's previous credit-assignment bug." **I could not find a specific prior
credit-assignment bug documented anywhere in this repo's git history or docs** — flag
this to the user directly rather than fabricate one; the discipline below is built to
the mandate's general correctness requirement (never crediting an outcome to a tool/
decision that did not cause it) regardless.

- Entry decisions credited with realized net PnL (`execution_cost_total`-adjusted,
  confirmed field `simulator/contracts.py:110` `Position.execution_cost_total`) over
  the actual realized holding period.
- Exit decisions credited separately from entry (Task 7/8's thesis-invalidation
  signal gets its own credit, not conflated with entry's).
- Rejected entries (Task 6's confirmed rejection reasons) excluded entirely — this is
  mechanically enforced by reading `ExperienceRecord.rejection_reason`
  (`simulator/experience.py:14`, confirmed field, added in Phase 1's fix wave)
  and skipping any record where it's non-None.
- Feeds `ToolTrust.update()` (Task 5) only for the specific `(source_name,
  context_bucket)` pairs that were load-bearing for the specific decision being
  credited (Task 7's thesis) — this is THE explicit test for "must not incorrectly
  attribute a later outcome to the wrong market observation/action," built as a named
  test, not an incidental property.

Test: `tests/intelligence/test_credit_assignment.py` — a synthetic multi-trade
sequence with known ground-truth outcomes verifies each trade's credit lands on
exactly the sources that were load-bearing for it and not on unrelated sources active
at the same time but not part of that trade's thesis (the direct "wrong attribution"
regression test); a rejected-entry record contributes zero credit anywhere, verified
explicitly.

### Task 10 — `intelligence/experience_store.py`: experience memory

Mandate Section 12 item 10. A thin read layer over Phase 1's existing
`ExperienceRecorder.all_records()` (`simulator/experience.py:57-65`, confirmed) —
partitioned by `environment_tag` (Phase 1's existing `write_tag_guard`,
`simulator/experience.py:68`, unmodified), with an explicit accessor that refuses to
read the untouched final OOS partition by name (a hard assertion, not a convention) —
this is the concrete mechanism keeping "protected OOS" protected at the code level,
not just by policy.

Test: attempting to construct an `ExperienceStore` pointed at the OOS partition raises
immediately; a normal TRAINING/RESEARCH_VALIDATION partition read works and returns
records in `decision_id` order.

### Task 11 — analytical SL/TP/sizing bootstrap

Mandate Section 5/12 item (implicit in "independent risk/safety layer," made
explicit here as its own task since K/L are separately named in the architecture
spec). SL/TP: a multiple of the GARCH conditional variance or `realized_vol_60s`
(`contracts/market_state.py`, confirmed field) — whichever is available/fresher per
Task 5's refit cadence. Sizing: reuses Phase 1's existing
`risk_fraction_of_equity` mechanism (`simulator/contracts.py:48`, confirmed,
already caller-overridable via Task 7 of the Phase 1 plan's `size` field) — this task
does not modify `simulator/`, it only decides how `FastTierDecisionEngine` populates
the `size`/`sl_price`/`tp_price` values it passes through the existing seam. Explicitly
NOT a learned sizing head — mandate and architecture spec both defer that until entry/
exit trust is validated (Section L reasoning, unchanged).

Test: SL/TP distance scales with a synthetic volatility input as expected; size stays
within Phase 1's existing margin-rejection bounds (Task 6's `INSUFFICIENT_MARGIN`
path is never silently bypassed by this bootstrap — verified by construction, the
bootstrap only ever proposes a size, Phase 1's `open_position` still enforces the
check).

### Task 12 — latency instrumentation

Mandate Section 8/12 item 12 — explicit, measured, not assumed. Extends Phase 1's
existing measurement discipline (`tests/simulator/test_replay_performance.py`'s
pattern, `time.perf_counter`-based, printed not just asserted) to: observation
latency (already measured in Phase 1, reused), `EvidenceRegistry.compute_all`
latency (isolating the GARCH/Kalman refit-cadence cost specifically — this is where
Task 5's periodic-refit strategy gets empirically validated or invalidated), Task 5's
`FastTierReasoner.hypothesis()` latency, `FastTierDecisionEngine.decide()`
end-to-end latency, order-preparation latency (SL/TP/size computation, Task 11).

Test: `tests/intelligence/test_fast_tier_performance.py` — printed real numbers for
each stage, following Phase 1's `test_replay_performance.py` convention exactly (loose
sanity bounds alongside the printed number, per that file's established pattern).

### Task 13 — Phase 1 simulator integration test

End-to-end: `FastTierDecisionEngine` (all of Tasks 2-11 composed) run through
`simulator.replay.run_replay` on a longer synthetic dataset (following
`test_scaffold_integration.py`'s pattern from the prior unexecuted Track A plan, but
now actually exercised against a real reasoning implementation, not a
`StubDecisionEngine`). Confirms the whole assembled Fast Tier honors every Phase 1
contract: no-look-ahead (reuses Phase 1's existing leakage-test patterns against this
new caller), rejection handling (Task 6/9), thesis lifecycle (Task 7/8), credit
assignment (Task 9) — all in one composed run, not just per-module.

### Task 14 — strict causal-memory boundary tests

Mandate Section 12 item 14, made an explicit standalone task rather than folded into
Task 5/7/9's individual tests, because this is the single highest-risk property in the
whole design (architecture spec's Section 21: "Bayesian posterior misspecification"
risk, and more generally — a trust update leaking future information would silently
invalidate every other test in this plan). Direct tests: `ToolTrust.update()` for
decision *t* can only be called with outcomes from decisions closed strictly before
*t* (a poisoned-future-outcome test, following Phase 1's `test_leakage_extended.py`
poisoning/truncation dual-pattern exactly); `context_bucket()` never reads beyond
`closes_so_far`'s current index (reuses the existing no-look-ahead truncated-recompute
test pattern from Task 3).

### Task 15 — whole-branch review

After Tasks 1-14: run the full retained + new test suite, re-run the
decision/candidates/learning import-cleanliness grep from Phase 1 against the now-
larger tree (confirm `intelligence/` doesn't reintroduce a dependency on archived V3/V4
code), confirm zero modification to `simulator/`/`market/`/`contracts/` unless a
documented, flagged defect was found and explicitly approved. Produce the mandate's
deliverable report for this implementation phase.

## Execution order

1 (trivial, first) → 2 → 3 → 4 (needs 2's spec shape) → 5 (needs 2/3/4) → 6 (needs 5)
→ 7/8 (needs 6, sequential since 8 extends 6's `manage()` using 7's thesis) → 9 (needs
6/7's rejection-reason and thesis output) → 10 (independent, can run parallel to 5-9)
→ 11 (needs 6's seam to plug into) → 12 (needs 2-11 to have something to measure) → 13
(needs everything) → 14 (can start as soon as 5/7 exist, doesn't block 6-13) → 15
(last, always).

## What this plan does not do

No Slow Tier, no LLM integration, no discovery log, no learned gating-network
escalation, no full MoE/RL/sequence-model — all explicitly deferred per the
architecture spec's Section 20 and this mandate's Section 12. No modification to
Phase 1's `simulator/`, `market/`, or `contracts/`. No training run against real
market data — every test in this plan uses synthetic data. No profitability claim.
