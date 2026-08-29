# GOLDEX Phase 2: Intelligence Architecture Design

STATUS: design only, not implemented. No code changes in this document.

Governs: "GOLDEX V2 — PHASE 2 AUTHORIZATION" mandate. Builds on the Phase 1 foundation
(`docs/superpowers/reports/2026-08-29-goldex-phase1-nervous-system-report.md`, commit
`1654457`) and the evidence base from the prior architecture-decision spec
(`docs/superpowers/specs/2026-08-28-goldex-autonomous-architecture-decision.md`), the
horizon-sweep and event-time findings, and the 28 prior null hypotheses. Designed
independently of V3/V4 — those are evidence, not requirements.

## A. First-principles problem definition

The problem is not "predict the next candle." It is: **build a system that can decide,
from a stream of market observations plus its own account/position state, whether a
short-duration trading opportunity currently exists, and if so how to take it, manage
it, and exit it — using quantitative knowledge conditionally, not by predetermined
rule.** The system must learn this from experience it generates itself against
chronological historical data, then transfer unchanged to live MT5 data via the
identical `MarketState` interface Phase 1 already validated.

The evidence constrains the problem sharply: 28 prior hypotheses — specialist models,
5 representation families, a 6-horizon × 7-representation sweep, event-time
conditioning — found zero *marginal* signal in single-instrument M1 features. This
does not mean Gold is unlearnable. It means the search space that has actually been
tested is exhausted and null. Two axes remain genuinely untested: **conditional/joint
structure** (does mechanism X matter only given state Y — never tested, all 28
hypotheses were marginal) and **information sources outside single-instrument M1 bars**
(cross-instrument, tick-level, options). Phase 2's architecture must be built to
exploit exactly these two untested axes, not to re-run marginal search with a bigger
model.

## B. What the AI should perceive

At each decision point: the current `MarketState` (price, spread, activity, data
quality — Phase 1's neutral observation layer), the current `AccountState` and
`PositionView` if a position is open, and the output of every registered
`EvidenceSource` evaluated on the causal history up to that point. It should NOT
perceive raw future data (enforced by Phase 1's leakage tests), a hand-labeled regime
tag (Section E5 below forbids this), or a fixed-horizon forward window.

Whether it should perceive the *sequence* of recent observations (not just the current
snapshot) is a real, evidence-informed question, not assumed. Phase 4's
trajectory-vs-snapshot test found no benefit to sequential structure over a single
snapshot at the horizons tested. That is one data point, not a permanent verdict — it
should be revisited if Track F's trade-management research finds a genuinely
sequential dependency (e.g. optimal exit timing that depends on the shape of the move,
not just its current state). Until then, the default representation is a **stateless
snapshot conditioned on account/position context**, not a sequence model. This is the
single most consequential first-principles decision in this document and it is made
on evidence, not convention.

## C. Internal market representation

A feature vector at decision time, assembled causally (no look-ahead, enforced by the
existing `build_snapshot`/`on_tick` contract): raw `MarketState` fields, plus every
`EvidenceSource`'s `(value, confidence)` output. No hand-built "trend/range/breakout"
label enters this vector (Section E5). If latent structure is useful, it must be
*learned* — either as one more `EvidenceSource` (e.g. an HMM's inferred state, a
Kalman filter's regime-adjacent variable) that the decision layer treats exactly like
any other quantitative mechanism, never as a controlling switch statement.

## D. Quantitative knowledge architecture

Reuse and extend the `EvidenceSource`/`EvidenceRegistry` pattern from the prior
architecture spec (Section G): each mechanism — momentum, mean reversion, GARCH
conditional variance, Kalman trend/velocity, HMM regime, skew/kurtosis, microstructure
proxies, cross-asset correlations, options-derived quantities where legitimately
available — registers as a named `state -> (value, confidence)` function. None of
them vote. None of them are individually required to be marginally predictive to be
admitted (the admission bar from the prior spec stands: MI-vs-shuffled-null as
*evidence-worth-including*, not as a standalone-profitable-strategy test — a source
with weak marginal signal can be legitimately conditionally useful, which is exactly
the untested axis).

The decision layer (Section E) is responsible for learning **which sources matter
under which conditions** — that conditional-usefulness learning is the actual
intelligence problem. This is not new relative to the prior spec; Phase 2 makes it
concrete by picking the mechanism that does this learning.

## E. Decision architecture candidates

Evaluated against: data availability (~6.7 years M1, modest for deep models trained
from scratch), sample efficiency, sequential-learning need (Section B says: not yet
established), interpretability (necessary in a domain where 28/28 prior tests were
null — an opaque model producing a 29th "finding" is unauditable), stability, leakage
risk, and latency (Section R).

1. **Contextual bandit / gated linear-or-additive combiner over the EvidenceRegistry
   output** (recommended starting point, Section V). Treats NO_TRADE/LONG/SHORT as
   arm selection conditioned on the evidence-source vector; the combiner's learned
   weights ARE the "which mechanism matters when" answer, directly interpretable per
   context. Sample-efficient relative to the dataset size, cheap at inference (Section
   R), and structurally cannot become "a voting library" because the weights are
   learned per-context, not fixed.
2. **Gradient-boosted trees (CatBoost — already proven infra, archived not deleted)**
   as a nonlinear escalation if the linear combiner demonstrably underfits real
   conditional structure Track D finds. Still fast at inference, still reasonably
   interpretable (feature importance, SHAP), still far more sample-efficient than deep
   sequence models.
3. **Full trajectory-based RL (policy gradient / actor-critic).** Rejected as the
   *starting* architecture. The strongest justification for RL — that sequential
   structure across a trade's lifetime carries information a snapshot can't capture —
   was directly tested (Phase 4 trajectory-vs-snapshot) and came back null. RL also
   inherits every credit-assignment risk in Section H at maximum severity, and
   modest-sized historical data makes on-policy RL's sample inefficiency a real
   liability, not a theoretical one. Not permanently rejected — revisit if Track F
   proves entry/exit/sizing are jointly, sequentially dependent in a way a snapshot
   combiner cannot express (this is exactly the rejection condition already written
   into the prior architecture spec's Section E).
4. **Transformer / deep sequence models.** Rejected for the same reason as (3), plus:
   no evidence yet justifies their sample and compute cost, and their opacity is a
   direct liability given the 28-null track record — a plausible-looking transformer
   output is *harder* to distinguish from noise than a linear combiner's, not easier.
5. **HMM/regime models as the top-level architecture.** Rejected explicitly — the
   mandate (Section 5) forbids hard-coded regimes, and using an HMM as the *decision*
   architecture (rather than as one evidence source among many) would smuggle exactly
   that assumption back in under a different name.
6. **Bayesian decision system (e.g. Thompson sampling over the combiner's action
   posterior).** Not rejected — this is a legitimate way to formalize (1)'s
   exploration/exploitation tradeoff during the "continuous learning" phase (Section
   O) and should be considered as a refinement of (1), not a competing architecture.

## F. Learning architecture candidates

Given (E1)/(E2) as the decision mechanism: **offline, supervised-style learning on
logged experience with rigorously scoped targets (Section H)**, not online
temporal-difference RL. The target is the realized outcome of the specific decision
being credited (Section H), learned via standard supervised/contextual-bandit methods
(policy-gradient-free — importance-weighted or direct regression/classification on
logged (context, action, outcome) tuples). This is deliberately the least
architecturally ambitious learning method that the decision architecture in (E)
supports, because ambition here should be earned by evidence (Track D/F), not assumed
up front. `LearningLoop`'s research/deployment split (already speced, not yet
implemented) governs how a newly-fit model gets promoted — see Section O.

## G. Experience/memory architecture

Phase 1's `ExperienceRecorder` already records raw facts per decision: full
`MarketState`, full `AccountState`, the action taken, execution result, and (as of the
Phase 1 fix wave) an honest rejection trace when an action was rejected rather than
executed. Phase 2 adds an **offline experience store**: records keyed by `decision_id`,
partitioned by training-run version, kept strictly separate from the untouched final
OOS slice (rows 300,000:400,000, never read by any of the 28 prior hypotheses and not
read here either). No new recording mechanism is needed at the simulator layer — this
is a data-access/versioning layer on top of what Phase 1 already writes.

## H. Credit-assignment solution

This is the most safety-critical open question in the design and the one the mandate
is most explicit about not letting be sloppy (Section 8, Section 15).

- **A LONG/SHORT decision** is credited with the realized net PnL (after
  `execution_cost_total`) of the position it opened, over its actual realized holding
  period — whatever that period turns out to be, since fixed-horizon labeling is
  explicitly rejected (mandate Section 7, Section J of the prior spec).
- **The exit decision (MANAGE→EXIT)** is credited separately from the entry decision
  that opened the position — an early exit that avoided a drawdown and a late exit
  that captured more profit are different decisions with different information
  available at decision time, and conflating their credit would systematically bias
  whichever one is more frequent. This keeps entry-quality and exit-quality separable
  as a *default*, without foreclosing Track F's finding that they should be jointly
  learned — if that evidence arrives, the credit-assignment scheme is revisited, not
  assumed away now.
- **NO_TRADE decisions do NOT get a fabricated counterfactual target.** Constructing
  "what would have happened if we'd traded" is exactly the kind of manufactured label
  that risks leaking the very information the decision was supposed to be made
  without. NO_TRADE is initially learned only implicitly, through the LONG/SHORT
  targets' opportunity cost as expressed by the combiner's own action-value estimates
  — not through a synthetic reward. If this proves too weak a signal, the fix is
  better exploration during training (Section E6), not fabricated counterfactuals.
- **Rejected entries** (Phase 1's margin/SL-TP rejection path) are excluded from
  credit assignment entirely — a rejection is not a trading decision, it's an invalid
  action, and Phase 1 already made this distinguishable in the experience log
  precisely so Phase 2 doesn't have to guess.

## I. Entry architecture

`EvidenceRegistry.compute_all(state)` → combiner (Section E1/E2) → `(action,
confidence)`. This sits behind the `DecisionEngine`/`DecideFn` seam Phase 1 already
validated (`simulator/replay.py`) — no new interface is needed, only an
implementation that currently doesn't exist (`StubDecisionEngine` still returns
NO_TRADE always).

## J. Trade-management architecture

Still explicitly an open research question, not a hardcoded split — this document
does not resolve it, per the prior spec's Section J and the mandate's Section 7. What
Phase 2 commits to *now*: SL/TP/size are fields the entry decision (or the ongoing
MANAGE decision) can populate, with no fixed horizon, no fixed R:R. Whether they end
up jointly learned with entry, hierarchically learned, or partially analytical is
Track F's job to determine before this section can be finalized.

## K. SL/TP architecture

Bootstrap with a volatility-scaled analytical baseline (e.g. a multiple of the
existing `realized_vol_60s`/GARCH conditional variance evidence source) — not because
this is believed to be optimal, but because it is needed to generate the first batch
of realistic experience for the learned combiner to train against at all, and an
under-informed learned SL/TP head trained on too little experience is a known
overfitting trap (Section P). Once enough experience accumulates under the bootstrap
policy, a learned SL/TP head conditioned on the same evidence vector as entry becomes
a legitimate escalation, gated by the same promote()-style validation as any other
model change.

## L. Risk/sizing architecture

Same bootstrap logic as K: start with the existing `risk_fraction_of_equity`
mechanism Phase 1 already has (now overridable per-decision via Task 7's `size`
field), escalate to learned sizing only after entry/exit quality is validated —
sizing errors compound fastest of any of these three (K/L/entry), so it is the
correct one to leave analytical longest.

## M. Information expansion architecture

Ranked by information-value-per-cost, using the prior spec's Section H tracks
unchanged:

- **Track B (cross-instrument: DXY, real yields, silver, other metals)** — highest
  priority. Cheap, historically available, a structurally different information class
  than anything in the 28 null hypotheses, genuinely untested.
- **Track D (conditional/joint MI on existing single-instrument features)** — equally
  high priority and near-zero acquisition cost, since it reuses data already in hand
  and the existing MI-vs-shuffled-null machinery unchanged. This is the most direct
  test of whether the decision architecture in Section E has anything real to learn
  from at all.
- **Track C (tick/bid-ask microstructure)** — higher acquisition cost, justified by
  the horizon-sweep falsification (information source, not horizon, is the open
  question) but should follow B/D, not precede them.
- **Options/Black-Scholes** — genuinely undecided, not inserted on reputation.
  Research whether legitimate, cost-accessible Gold options or implied-volatility
  data exists at all before evaluating relevance; if it doesn't exist at reasonable
  cost, reject explicitly with that finding as the reason, not silently.

## N. Historical training architecture

Reuse Phase 1's chronological, no-look-ahead replay unchanged. Add **Combinatorial
Purged Cross-Validation (CPCV)** for research validation — the upgrade over plain
walk-forward already flagged in the prior spec, needed because trade holding periods
now genuinely vary (Section J), so naive walk-forward splits risk leakage across
overlapping open positions at split boundaries in a way fixed-horizon labeling didn't
have to worry about. CPCV's purge/embargo logic addresses this directly.

## O. Continuous learning architecture

`LearningLoop`'s structural research/deployment split (speced, not yet built):
`research_state` mutates freely on an expanding experience window; `promote()` only
succeeds when a caller-supplied validation check (CPCV performance not degraded
relative to the currently deployed model, no signal of memorization — Section P) is
satisfied. Training happens on an expanding window of TRAINING+RESEARCH VALIDATION
data only; UNTOUCHED OOS is read exactly once, at the point a candidate is believed
ready for DEMO, never iterated against.

## P. Anti-overfitting architecture

- Extend the existing shuffled-null MI discipline from feature-level to model-level:
  a candidate combiner is not trusted until its live (CPCV) performance clears a
  shuffled-label/shuffled-action null by the same statistical bar features already
  have to clear.
- Track D's conditional-MI results should gate how expressive the combiner is allowed
  to become — no jump to (E2)'s nonlinear escalation without conditional-structure
  evidence justifying it first.
- Concept-drift monitoring: track performance decay across rolling forward-time
  slices distinct from the final OOS, to distinguish genuine market adaptation from
  memorization of the training window.
- Reward hacking / simulator exploitation: watch specifically for policies that
  exploit `execution_cost_total`/spread mechanics rather than genuine price
  behavior — a known failure mode the mandate names explicitly (Section 15).

## Q. Validation architecture

Unchanged from the prior spec, restated because it remains non-negotiable: TRAINING
→ RESEARCH VALIDATION (CPCV) → UNTOUCHED OOS → DEMO → LIVE. The OOS split has never
been read by any of the 28 prior hypotheses; it stays unread until a candidate clears
research validation.

## R. Real-time execution architecture

Phase 1 measured `build_snapshot` at p99 ≈ 2ms and execution/experience-recording
latencies at low-microsecond scale. The decision architecture's own inference latency
is the new unknown Phase 2 introduces — this is a direct argument for starting with
(E1)/(E2) (linear/GBT inference: microseconds-to-low-milliseconds) over (E3)/(E4)
(deep models: materially higher and less predictable latency), preserving the latency
budget short-duration trading needs without having measured a hard requirement yet.
Once a candidate model exists, extend Phase 1's measurement discipline
(`test_replay_performance.py`'s pattern) to cover model inference specifically, before
any claim about real-time suitability is made.

## S. MT5 transition architecture

Already substantially solved by Phase 1: `market/mt5_feed.py` → `state_engine.py`
produces the identical `MarketState` shape `simulator/market_state_builder.py`
produces historically (verified by
`test_historical_live_interface_consistency.py`). Whatever `DecisionEngine`
implementation Phase 2 builds consumes `MarketState` the same way regardless of
source — no separate live-model path is needed. Phase 2 does not need to design this;
it needs to not break it.

## T. Failure modes

Overfitting to noise given the 28-null track record (highest risk — a 29th
"finding" from a more expressive model is more likely to be spurious than real,
absent Track D evidence); reward hacking via cost/spread exploitation; leakage
introduced by a poorly-scoped credit-assignment window (Section H) or by a future
`EvidenceSource` that inadvertently reads ahead; catastrophic drawdown from an
under-regularized learned sizing head deployed too early (Section L); undetected
distribution shift between historical training data and live MT5 conditions; silent
degradation of the historical/live `MarketState` parity Phase 1 fixed once already
(caught by its own final-review fix, worth re-testing whenever `MarketState` changes).

## U. Architecture alternatives rejected and why

- Full end-to-end RL over trajectories (E3) — no evidence of sequential-structure
  benefit (Phase 4 trajectory-vs-snapshot null), high sample cost, maximal
  credit-assignment risk.
- Transformer/deep sequence models (E4) — same rejection basis as RL plus opacity
  risk given the null track record.
- Hardcoded regime classifier or HMM-as-top-level-architecture (E5) — explicitly
  forbidden by mandate Section 5; would smuggle back exactly the fixed-category
  assumption the mandate rejects.
- Fixed/weighted voting library over quantitative mechanisms — already rejected in
  the prior spec; structurally equivalent to what the 28 null hypotheses already
  tested, since a static combination of individually-null mechanisms is not evidence
  any combination is meaningful. Nothing in this document reintroduces it.
- Fabricated NO_TRADE counterfactual reward (Section H) — rejected as a leakage risk
  masquerading as a completeness fix.
- "Train until backtest becomes profitable" as the training loop (mandate Section
  10) — explicitly rejected; encourages memorization, has no place in this design.

## V. Recommended architecture

`EvidenceRegistry` (Section D, extended with Track B/D findings as they land) feeding
a per-context gated linear/GBT combiner (Section E1, escalating to E2 only on
Track-D-justified evidence of nonlinear conditional structure) behind the existing
`DecisionEngine`/`DecideFn` seam. SL/TP/sizing bootstrap from analytical
volatility/risk-fraction baselines (K/L) until enough experience justifies learning
them. Offline supervised-style learning (F) on rigorously credited experience (H),
governed by `LearningLoop`'s research/deployment split (O) and validated via CPCV
against a never-touched OOS split (Q). This is a deliberately conservative starting
point relative to what "AI trading system" evokes — that conservatism is earned by
the 28-null evidence trail, not a failure of ambition. It shares its shape with the
prior session's "Architecture G" working hypothesis because the evidence that shaped
that hypothesis hasn't changed; every heavier alternative is rejected for a stated,
falsifiable reason (Section U), not by default.

## W. What should be built first

1. `EvidenceRegistry` + a real `DecisionEngine` implementation (Section E1) — the
   scaffold Track A's unexecuted plan already designed, now backed by a real Phase 1
   `DecideFn` seam to plug into.
2. The offline experience store + credit-assignment implementation (Sections G/H) —
   this must exist and be tested against synthetic known-outcome trajectories before
   any real training happens, per the prior spec's Section I requirement.
3. Analytical SL/TP/sizing bootstrap (K/L) — needed to generate the first batch of
   realistic experience at all.

## X. What can be researched in parallel

Track B (cross-instrument MI), Track D (conditional/joint MI — the highest-value,
lowest-cost open question), the options-data-availability check (Section M), and
Track F (trade-management separability). None of these block W's build items; W does
not block them either.

---

Nothing in this document authorizes implementation. Per mandate Section 20, this is
design only — implementation begins after this design is reviewed.
