# GOLDEX Phase 2: Fast Tier — Final Report

Governs: "GOLDEX PHASE 2 — ARCHITECTURE APPROVED / PROCEED TO IMPLEMENTATION PLANNING"
mandate. Plan: `docs/superpowers/plans/2026-08-29-goldex-phase2-fast-tier-implementation-plan.md`.
Spec: `docs/superpowers/specs/2026-08-29-goldex-phase2-final-architecture-decision.md`.
Range: `464c247..312f9ed` (17 commits — 14 tasks, each with its own task-scoped review, plus
one final whole-branch review and its fix wave).

No trained model, no live orders, no profitability claim anywhere. The Slow Tier was not
built, per the mandate's explicit instruction.

## What was built

A `intelligence/` package implementing the Fast Tier — the execution-critical half of the
two-tier architecture the mandate approved:

- **`evidence.py`/`evidence_sources.py`** — the `EvidenceSource` tool contract (name,
  mathematical formulation, assumptions, known failure conditions, a `compute` callable, a
  `computational_cost_hint`, and — added during the final review's fix wave — an
  `is_directional` classification), wrapping 9 already-validated Phase 3A/4 representation
  functions unchanged.
- **`applicability.py`** — mechanical hard-floor gates independent of any learned trust:
  insufficient history, or `MarketState.data_quality=INVALID`/`market_closed=True`, force a
  source's confidence to 0.0 regardless of what it computed.
- **`fast_tier.py`** — the reasoning core: `ToolTrust` (Beta-distributed, per-tool,
  per-context-bucket posterior belief, updated only from correctly-credited real outcomes),
  `context_bucket()` (a continuous quantity binned into buckets — no hardcoded regime labels
  anywhere, verified), `FastTierReasoner` (combines applicability-gated, trust-weighted
  evidence into a `Hypothesis`: net directional belief, a genuine cross-source disagreement
  term for `aggregate_uncertainty`, and the specific load-bearing sources), with periodic
  refit-caching for the two expensive GARCH/Kalman sources.
- **`decision_engine.py`** — `FastTierDecisionEngine`, plugging into Phase 1's unmodified
  `DecideFn`/`ManageFn` seam unchanged: turns a `Hypothesis` into NO_TRADE/LONG/SHORT +
  SL/TP/size via an EV/cost gate reusing Phase 1's `round_trip_cost_r`.
- **`thesis.py`** + continuous reassessment in `manage()`** — per-position memory of which
  tools justified entry, retained only while open; `manage()` re-evaluates the same evidence
  and can exit earlier than a static SL/TP on thesis invalidation, strictly additive on top of
  Phase 1's SL/TP/liquidation checks (independently verified never to bypass/delay them).
- **`credit_assignment.py`** — attributes a closed trade's outcome to exactly the tools
  load-bearing for that trade's specific thesis, correctly signed (fixed in the final review —
  see below), excludes rejected entries entirely.
- **`experience_store.py`** — a read guard that hard-raises on any attempt to read the real,
  established untouched-OOS partition tag.
- **`bootstrap.py`** — analytical (non-learned) SL/TP (volatility-scaled) and sizing (reuses
  Phase 1's existing risk-fraction formula byte-for-byte) — explicitly not a learned head.

Every task shipped a real test; several tasks required fix rounds after their own task-scoped
review caught genuine issues (a duplicated-logic finding, a discontinuous uncertainty formula,
a vacuous test assertion, a mathematically wrong root-cause claim, a test that proved
commutativity instead of causality). All of those were resolved before the task closed.

## The whole-branch review — what individual task review could not see

After all 14 tasks passed their own review, a final whole-branch review (Opus) found two
**Critical** cross-task defects invisible to any single task's review, both now fixed:

- **A structural long-only bias.** `net_directional_belief` originally treated every
  evidence source's sign as a directional vote via `copysign`. Two of the 9 sources
  (`multiscale_vol_ratio`, `garch_conditional_variance`) are non-negative by mathematical
  construction — variance cannot be negative — so they voted LONG on every single sample,
  forever, and Bayesian trust learning could never correct this (a Beta posterior mean is
  always strictly between 0 and 1, never exactly 0). Measured: 78.5% of samples on symmetric,
  zero-drift synthetic data would have triggered LONG vs. 7.3% SHORT. **Fixed**: sources are
  now explicitly classified directional/non-directional; only directional sources vote.
  Post-fix measurement: 40.7%/38.7% LONG/SHORT on the same symmetric data — genuinely
  unbiased.
- **Credit assignment rewarding dissent.** A trade's win/loss was applied uniformly to every
  load-bearing source regardless of whether that specific source's own vote agreed with the
  direction actually taken — a source that voted SHORT while the trade went LONG and won was
  credited as if it had been right. **Fixed**: each source's own signed contribution is now
  compared against the trade's actual direction before crediting; a dissenting source now
  provably receives the opposite credit outcome from an agreeing source on the same trade,
  verified in both directions and for both LONG and SHORT trades.

Four **Important** findings were also fixed: `manage()`'s real per-bar cost (it also calls the
reasoner, so held-position bars pay the same cost as decisions) is now documented and the
latency test bounds were tightened from 10-25x slack to 2-3x (a real 10x regression would no
longer pass silently); `computational_cost_hint` (built in an early task specifically for
this) is now populated on all 9 sources; the composed integration test now exercises the
`POLICY_EXIT` (thesis-invalidation exit) and stale-market-data-NO_TRADE paths it previously
missed.

## Known, deliberately unresolved limitation

One Important finding was **parked**, not fixed, after the fix wave and its one allowed
re-review (per process — no second fix wave): `context_bucket()`'s reachability fix removed
the *mathematical* impossibility (buckets were structurally capped at the upper 60% of the
range) but the recalibrated scale constant is still ~5x too wide against the real evidence
registry's actual measured input distribution — in practice, on real synthetic-replay data,
96% of decisions land in a single context bucket. **This means the Fast Tier's core
architectural promise — that trust is learned conditionally, per market context — is
substantially weaker in practice than the design intends; it currently functions closer to
unconditional trust learning.** This is a calibration problem, not a correctness or safety
problem (no test fails, nothing crashes, no bias is introduced), and the right fix requires
either real historical data or a deliberate calibration pass this synthetic-data-only plan
didn't attempt — the same category of "documented, deliberately deferred" default as
`load_bearing_floor` and `refit_interval`. **This must be addressed before any training or
validation work treats conditional trust learning as functioning.**

A related, smaller consequence: 4 of the 9 wrapped sources (`multiscale_vol_ratio`,
`vol_regime_transition`, `rolling_skew`, `rolling_excess_kurtosis`) are now fully inert —
computed every decision (a real, non-trivial share of latency) but contributing to neither
votes, trust, nor context, since they're non-directional and `context_bucket` only reads two
specific source names. Not a bug; a real cost with no current benefit, worth either giving
them a genuine role or dropping from the hot path in a future pass.

## Tests

**211 passed, 0 failed** on `tests/intelligence tests/simulator` at final state (independently
re-confirmed by the final re-review, not just the implementer's own report). Environment:
`.venv`, `requirements.txt` (now includes `scipy`, added in Task 1 after the prior Phase 1
session's pyyaml gap taught the lesson to pin explicitly).

## Architecture integrity (independently verified)

Zero modification to `simulator/`, `market/`, or `contracts/` anywhere across all 14 tasks
— confirmed by diff-stat inspection of the whole range. Zero import-path dependency
reintroduced from `intelligence/` back into any V3/V4 decision/candidates/learning code —
confirmed by grep. No hardcoded regime labels anywhere in `context_bucket()`,
`credit_assignment.py`, or `bootstrap.py` — confirmed by direct code reading. No profitability
claim anywhere.

## What's ready for Phase 3 / whatever comes next

- A working, tested `FastTierDecisionEngine` that plugs into Phase 1's unmodified simulator
  and MT5 seam identically.
- A real, corrected credit-assignment and trust-learning loop, with the directional bias
  removed.
- Real, measured (if unenforced-until-now) latency numbers: `decide()` p99 ≈ 40-60ms,
  `compute_all()` p99 ≈ 440-475ms — both well above Phase 1's ~2ms per-bar budget. **This
  performance gap was never resolved in this plan** — it was measured, documented, and its
  root cause identified (6 of 9 sources recompute fresh over an unbounded, growing history
  array every call; only GARCH/Kalman are refit-cached), but fixing it requires either
  windowing/incremental computation for the remaining sources or widening Phase 1's `DecideFn`
  seam to hand the engine a bounded history window — the latter would require touching
  `simulator/replay.py`, which this plan was explicitly forbidden from doing without a
  separately authorized follow-up.

## What is NOT ready

- **Conditional trust learning (the context-bucket calibration) is not functioning as
  designed** — see above. Any Phase 3 work that assumes "the Fast Tier learns which tools
  matter in which market conditions" should not proceed until this is recalibrated.
- **Latency is 10-20x over Phase 1's per-bar budget** and unresolved — this Fast Tier is not
  ready for anything resembling real-time or high-frequency decision loops without further
  work.
- **No Slow Tier exists** — discovery, tool-combination proposals, and the LLM-mediated
  reasoning layer from the architecture spec are entirely unbuilt, correctly, per the
  mandate's explicit scope boundary.
- The 4 inert non-directional sources should be resolved (given a role, or dropped) before
  anyone reasons about "9 active evidence sources" as a system property.

Nothing in this report or the underlying code claims a validated trading edge. This is
infrastructure — a corrected, tested reasoning loop ready to be trained and validated against,
not a system that has been trained or validated.
