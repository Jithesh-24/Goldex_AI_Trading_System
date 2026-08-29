# GOLDEX Phase 2: Intelligence Architecture Design (revised)

STATUS: design only, not implemented. No code changes anywhere in this document.

Supersedes `docs/superpowers/specs/2026-08-29-goldex-phase2-intelligence-architecture-design.md`
(same session). That document was correctly rejected: it answered "what model combines
these features" when the actual question is "what computational process could observe,
reason about, and act on the Gold market." This document answers the second question.
The evidence base (28 null hypotheses, Phase 1 foundation, the prior spec's tracks) is
carried forward unchanged — evidence, not architecture, survives revision.

## 0. What changed and why

The rejected design's architecture was, honestly: engineered features → learned
combiner → trade. That is a predictor with a gate on top, and the mandate is right
that it risks reproducing the V3/V4 pattern under new names. The evidence-based
rejections in that document (no deep RL/Transformer yet, no hardcoded regimes, no
voting) were correct and are kept. What was wrong was treating "which model computes
the output" as the central design question. The central design question is **what
computational process, at decision time, produces the output** — and a process can be
sample-efficient, interpretable, and conservative about model complexity while still
being genuinely agentic: retrieving relevant knowledge, forming a working hypothesis,
tracking whether that hypothesis still holds, and updating its own trust in each
mechanism from outcomes. That is the reframe this document makes.

## 1. What exactly is GOLDEX intelligence?

GOLDEX is a **tool-using reasoning process over a library of quantitative evidence
sources, with persistent memory of its own trust in each tool and of the thesis behind
any open position, operating in a strict causal loop against the Phase 1 environment.**
It is not a predictor with a decision threshold. Concretely, at every decision point it:

1. **Perceives** — reads `MarketState` + `AccountState` + `PositionView` (if open).
2. **Consults its tool library** — evaluates every registered quantitative mechanism
   (Section 5), each returning a value, a confidence, and (new in this revision) enough
   metadata to reason about *why* it fired.
3. **Forms a working hypothesis** — a soft, learned salience/trust layer (Section 6)
   determines which tools' outputs are actually load-bearing right now, not a fixed
   weighted average of all of them always.
4. **Evaluates opportunity net of cost** — the existing EV/cost-gate discipline
   (unchanged mechanically, this part of V3/V4's machinery was sound) applied to the
   hypothesis, not to a raw prediction.
5. **Decides and acts** — NO_TRADE / LONG / SHORT with SL/TP/size (Sections 7, 9).
6. **Tracks the thesis while a position is open** — the specific tool outputs that
   justified entry are retained; MANAGE re-evaluates whether they still hold, not just
   whether price crossed a static SL/TP (Section 8).
7. **Learns** — after exit, updates its trust in the tools that were load-bearing for
   this decision, based on the realized, correctly-credited outcome (Section 9, H
   from the prior document, unchanged).

This is a loop, not a pipeline with one prediction step. The substrate that implements
steps 3-4 (the "reasoning") does not need to be a large neural network to satisfy this
definition — see Section 13 for why it should not be, yet.

## 2. What should it perceive?

Unchanged in substance from the rejected document's Section B, reframed: `MarketState`,
`AccountState`, `PositionView`, and the full tool-library output — but now explicitly
including **the system's own recent history of tool trust and recent decisions** as
part of what it perceives (Section 4), not just raw market data. Whether it perceives a
*sequence* of raw observations is still not assumed (Phase 4's trajectory-vs-snapshot
test is still the only direct evidence, and it's still null) — but Section 6 below
gives a legitimate, evidence-light way to get sequential information into the loop
without a sequence model: state-space evidence sources are themselves recursive memory.

## 3. What should it remember?

Three genuinely distinct kinds of memory, given clearly different temporal-safety
treatment (mandate Section 7's requirement, taken seriously rather than gestured at):

- **Within-trade thesis memory** (Section 8): which tool outputs justified the open
  position, retained only while it's open, discarded at exit. No leakage risk — this
  is causal state about a decision already made, not future information.
- **Cross-decision trust memory** (Section 6): a running estimate of how reliable each
  tool has been *recently*, updated only after a trade closes and only using that
  trade's own correctly-credited outcome (Section 9/H). This is the mechanism that
  lets GOLDEX learn "mechanism X has been useful lately, mechanism Y hasn't" without a
  full model retrain — closer to a bandit's running arm-value estimate than to a
  neural network's weights. Strict rule: trust for decision *t* may only be updated
  using outcomes of decisions closed before *t*. This is the memory-boundary Section 7
  of the mandate demands, made concrete and testable (the same causal-only discipline
  Phase 1's leakage tests already enforce for market data, applied here to the trust
  state).
- **Recursive market state** (state-space evidence sources — Kalman filters, GARCH
  conditional variance): these are already, mathematically, a minimal *world model* —
  a filtered latent estimate of trend/velocity/volatility that persists and updates
  causally. Section 6 explains why this satisfies the "world model" question (Section
  6 of the mandate) without requiring a new deep-learned one.

Raw price history is explicitly NOT remembered as a growing buffer fed wholesale into
the reasoning step — that reintroduces the sequence-model question Section 2 already
declined to assume, and every raw-history access must go through a causal, tested tool
(an `EvidenceSource`), never a bare lookback.

## 4. What should it learn?

Three things, matched to the three memories in Section 3, none of them "predict the
next return":

1. **Trust/salience weights per tool, conditional on context** — this is the actual
   "conditional usefulness" learning problem the mandate keeps returning to (its
   Section 4, Section 8, Section 11). Learned from correctly-credited trade outcomes
   (Section 9).
2. **Thesis-invalidation signals** — which combinations of tool-output changes, while
   a position is open, historically preceded a bad outcome if held vs. a good outcome
   if exited. This is what makes trade management (Section 8) learned rather than a
   static SL/TP.
3. **SL/TP/sizing refinements** — bootstrapped analytically (unchanged from the
   rejected document's Sections K/L reasoning — still correct, still deferred until
   entry/exit trust is validated, to avoid compounding sizing errors on top of an
   unvalidated reasoning layer).

Explicitly not learned yet: a full latent market representation trained end-to-end
(Section 6 explains why this is premature, not permanently wrong), and a NO_TRADE
counterfactual reward (still rejected, same leakage reasoning as before).

## 5. How should quantitative knowledge enter the system?

The `EvidenceSource` contract is upgraded from "value + confidence" to a **formal tool
specification**, directly answering the mandate's Section 11:

```
EvidenceSource:
  name: str
  mathematical_formulation: str  # human-readable, not executable — documentation the
                                  # reasoning layer's trust mechanism can be audited against
  required_inputs: list[str]     # which MarketState/history fields it actually reads
  assumptions: str               # e.g. "assumes locally stationary volatility"
  compute(state) -> (value, confidence)
  computational_cost: measured   # reuse Phase 1's latency-measurement discipline
  known_failure_conditions: str  # e.g. "unreliable during the first 60 bars after a gap"
```

This is not a cosmetic addition. The `known_failure_conditions` and `assumptions`
fields are what let the trust layer (Section 6) reason about *why* a tool might be
unreliable right now, rather than only *whether* it has recently been profitable —
closer to how a human quant desk actually reasons about a model's applicability. The
admission bar is unchanged from the prior spec: MI-vs-shuffled-null as evidence-worth-
including, never as a standalone-profitable-strategy requirement.

## 6. How should the system reason about competing evidence?

This is the section that most needed to change, and where the mandate's push is most
justified.

**Not** a fixed weighted average (voting, rejected twice now for the same structural
reason: it's what the 28 null hypotheses already tested). **Not**, yet, a deep
attention mechanism trained end-to-end (no evidence justifies the sample cost — see
Section 13). Instead: a **learned, context-conditioned trust/salience layer** — for
each tool, a running, decaying estimate of "how reliable has this tool been in
contexts like the current one," itself conditioned on a small number of *soft*,
continuously-valued context variables (not hardcoded regime labels — mandate Section 8
is explicit about this) drawn from the recursive market-state sources in Section 3.

Concretely this can start as a contextual-bandit-style trust update (Thompson
sampling or a simple decayed running mean/variance per tool per soft-context-bucket)
— genuinely a form of reasoning-over-evidence, not a static blend, but still
sample-efficient and auditable. This is the honest resolution of Section 6's
world-model question: **a full learned latent-state world model is not built now**
(no evidence justifies it — Phase 4's trajectory-vs-snapshot null is still the
relevant data point, and a world model trained end-to-end on ~6.7 years of M1 bars is
a large model for a small dataset). **A lightweight one already exists** in the form
of the state-space evidence sources, and the trust layer consumes their output as
context. If Track D or F evidence later shows the trust layer itself needs to be
jointly learned with a richer latent state (rather than conditioned on hand-selected
state-space outputs), that is the evidence-based trigger to build the fuller world
model — not before.

## 7. How should it make an entry decision?

Perceive → consult tools → trust layer weights them into a hypothesis (a small number
of interpretable summary quantities: net directional evidence, confidence, and which
specific tools are currently load-bearing) → EV/cost gate → NO_TRADE/LONG/SHORT with
initial SL/TP/size from the analytical bootstrap (Section 1's Sections K/L logic,
unchanged). The load-bearing tool set from this step is retained as the trade's thesis
(Section 8).

## 8. How should it manage an open position?

**Continuous thesis reassessment, not a static SL/TP watch.** At every MANAGE step,
re-evaluate the same tools that justified entry. If their outputs have moved against
the original thesis in a way the trust layer recognizes as thesis-invalidating (Section
4, item 2), that is itself a decision input — HOLD / MODIFY / EXIT, potentially before
a static SL/TP would ever trigger. The static SL/TP set at entry remains as the safety
floor underneath this (mandate Section 9's explicit requirement: a safety layer must
exist, but must not become the strategy) — the reasoning layer can exit or tighten
earlier, never loosen past the account-safety constraints Phase 1's engine already
enforces (margin/rejection checks, Section 11 of the mandate).

This is genuinely different in kind from "entry=AI, SL/TP=rules": the exit decision is
reasoning over the *same* evidence sources as entry, continuously, with its own
learned trust signal (Section 4, item 2) — not a fixed distance check.

## 9. How should it learn from each trade?

Unchanged from the rejected document's Section H (credit assignment) — that section
was sound and is carried forward exactly:

- LONG/SHORT credited with realized net PnL over the actual realized holding period.
- Exit decisions credited separately from entry decisions.
- No fabricated NO_TRADE counterfactual.
- Rejected entries excluded from credit assignment.

What's new: the credited outcome updates **the trust layer's per-tool, per-context
estimates** (Section 6) and **the thesis-invalidation signal** (Section 4, item 2), not
a single monolithic model's weights. This makes individual updates interpretable
(which tool's trust moved, and why) and keeps each update's blast radius small — a bad
trade degrades trust in the specific tools that were load-bearing for it, not the
entire system's parameters.

## 10. How should it discover new useful mechanisms?

The `EvidenceSource` registry (Section 5) is designed to be extended by future
research (Tracks B/C/D — cross-instrument, tick data, conditional-MI) without touching
the reasoning layer. A newly admitted tool starts with an uninformative trust prior
(Section 6) and earns trust the same way every existing tool does — through credited
outcomes. Nothing about "discovery" requires a different mechanism from ordinary
operation; this is a direct benefit of the trust-layer design over a monolithic model
that would need retraining to incorporate a new feature.

## 11. How should it prevent bad mechanisms from dominating?

Trust decays and is context-conditioned (Section 6) — a tool that stops working in
changed conditions loses influence in *those* conditions specifically, without needing
to be manually removed (mandate Section 4's explicit requirement: don't permanently
remove a mechanism for standalone poor performance, since it may be conditionally
useful elsewhere). A hard floor also applies: any tool whose trust estimate is
statistically indistinguishable from its shuffled-null baseline (the existing MI
discipline, applied here to trust rather than to raw MI) contributes negligibly to the
hypothesis by construction — no manual ranking needed, satisfying mandate Section 4/11.

## 12. How should it adapt without overfitting?

The trust layer's small, interpretable update surface (Section 9) is itself an
anti-overfitting property relative to retraining a monolithic model on an expanding
window — each update is auditable and bounded. On top of that, unchanged from the
rejected document (still correct): CPCV for research validation, the protected
untouched OOS split read exactly once, concept-drift monitoring via rolling
forward-time slices distinct from OOS, and `LearningLoop`'s research/deployment
promote() gate.

## 13. What architecture best supports all of this?

**A tool-using reasoning agent**: `EvidenceRegistry` (Section 5) + a context-
conditioned trust/salience layer (Section 6, implemented initially as a
contextual-bandit-style trust estimator, not a deep network) + explicit thesis-tracking
memory for open positions (Section 8) + the unchanged EV/cost gate and credit-
assignment discipline. Evaluated against the mandate's own criteria in its Section 12:

- **Market perception, quantitative knowledge, conditional reasoning**: directly
  satisfied by the tool library + trust layer design (Sections 5-6), which is the
  actual point of this revision.
- **Sequential decision making**: satisfied at the trade-management level (Section 8's
  continuous reassessment) without requiring a sequence model at the market-
  perception level, where the only direct evidence (Phase 4's trajectory-vs-snapshot
  test) found no benefit.
- **Sample efficiency, ~6.7-year data limitation, latency, interpretability,
  robustness, overfitting resistance**: this is precisely what a contextual-bandit-
  style trust layer over a hand-specified tool library is good at, and precisely what
  deep RL/Transformer/end-to-end world-model architectures are weak at with this
  amount of data. This is not a rejection of ambition — it's matching architectural
  complexity to the evidence and data actually available, revisited as Tracks B/C/D/F
  produce new evidence.
- **Simulator/MT5 compatibility**: unchanged — sits behind the same `DecisionEngine`/
  `DecideFn` seam Phase 1 validated works identically against historical replay and
  live MT5 data.
- **Ability to expand later**: this is the design's strongest property. A richer
  latent world model (Section 6), a jointly-learned entry/exit representation (if
  Track F shows it's needed), or a heavier combiner (if Track D shows real nonlinear
  conditional structure) can each replace one piece of this architecture without
  redesigning the whole loop — the loop itself (perceive → consult tools → reason →
  decide → track thesis → learn) is the stable structure, not any one model choice
  inside it.

## 14. Why is that architecture superior to the alternatives?

Against the mandate's own list (its Section 12, A-M):

- **(A) Learned feature/knowledge combiner, (B) contextual bandit** — this design *is*
  a contextual-bandit-style trust layer, but reframed as reasoning-with-memory-and-
  thesis-tracking rather than a one-shot combiner, which is exactly the distinction the
  mandate is pushing for. Not rejected — refined into something the mandate's own
  Sections 1-9 describe.
- **(C) Mixture-of-experts, (K) neuro-symbolic hybrid** — the tool library + trust
  layer already IS a lightweight form of this (each tool is an "expert" the trust
  layer gates), without the sample cost of a jointly-trained MoE gating network. Not
  needed as a separate architecture.
- **(D) Hierarchical decision architecture** — partially adopted: the soft-context-
  conditioning in Section 6 is a hierarchy (context conditions trust; trust conditions
  action) without hardcoded levels or labels.
- **(E) Learned world/state model + decision policy** — addressed directly in Section
  6: a lightweight version already exists (state-space evidence sources); a fuller
  learned one is deferred to evidence, not rejected outright.
- **(F) Model-based RL, (G) model-free RL** — still not the starting architecture. The
  credit-assignment and thesis-tracking mechanisms in this design (Sections 8-9) give
  most of RL's conceptual benefit (learning from delayed, path-dependent outcomes)
  without RL's sample-inefficiency and instability, given the evidence that sequential
  structure hasn't been shown to matter at the market-perception level (only, so far,
  plausibly at the trade-management level, which this design already handles).
- **(H) Sequence model, (I) Transformer-based market model, (J) memory-augmented deep
  architecture** — not adopted as the market-perception substrate for the reason
  restated in Section 13; the "memory" the mandate's Section 7 actually needs
  (thesis-tracking, trust history) is provided by Sections 3/8 without a deep memory
  network.
- **(L) Tool-using trading agent** — this is what Sections 1-13 describe. Adopted
  directly, at the framing level, implemented with the lightest-weight substrate the
  evidence currently supports.

## 15. What minimum viable version should we build first?

1. The upgraded `EvidenceSource` contract (Section 5) — metadata fields added to the
   existing registry design, wrapping the existing Phase 3A/4 representation
   functions unchanged.
2. A trust/salience layer implementing Section 6 as a decayed running mean/variance
   per tool per soft-context-bucket (simplest legitimate version — Thompson sampling
   is a documented refinement, not a day-one requirement).
3. Thesis-tracking for open positions (Section 8) — which tools were load-bearing at
   entry, re-evaluated at each MANAGE step.
4. The credit-assignment implementation (Section 9, unchanged from the prior spec's
   Section H) — tested against synthetic known-outcome trajectories before any real
   training, per the prior spec's Section I requirement.
5. Analytical SL/TP/sizing bootstrap (unchanged, Sections K/L logic).

This is deliberately close in scope to the rejected document's Section W — the
difference is architectural framing (a reasoning loop with memory and thesis-tracking,
not a flat combiner), not a larger build.

## 16. What can be expanded later?

- A fuller learned world model / jointly-learned latent state (Section 6), gated on
  Track D/F evidence.
- Escalation from a bandit-style trust layer to a lightweight learned gating network
  (still not a full deep model), if the simpler version demonstrably underfits
  real conditional structure Track D finds.
- Track B (cross-instrument), Track C (tick data), the options-availability check —
  unchanged from the rejected document's Section M/X, still parallel, still not
  blocking the minimum viable build.
- Track F (trade-management separability) — directly informs whether Section 8's
  thesis-tracking should eventually be jointly learned with entry rather than
  reasoning over the same tools independently at each step.

## 17. How does it connect to the Phase 1 simulator?

Unchanged: the reasoning loop implements `DecisionEngine`/`DecideFn` exactly as Phase 1
validated (`simulator/replay.py`). Every perception/reasoning/decision/thesis-tracking
step described above happens inside `decide()`/`manage()`; nothing about this design
requires changing `simulator/`, `market/`, or `contracts/` — Phase 1 is not modified
unless a genuine defect is found (mandate Section 20, still respected).

## 18. How does the same intelligence eventually connect to MT5?

Also unchanged, and this is Phase 1's strongest deliverable: `market/mt5_feed.py` →
`state_engine.py` produces the identical `MarketState` shape the historical path
does, verified by `test_historical_live_interface_consistency.py`. The reasoning loop
in this design consumes `MarketState` the same way regardless of source — the trust
layer's state (Section 3) and thesis-tracking (Section 8) are the only genuinely new
persistent state this design introduces, and neither is source-dependent; they persist
in GOLDEX's own memory, not in the market interface. Live order submission remains out
of scope for this phase (mandate Section 16 elsewhere).

---

## What did NOT change from the rejected document, and why that's not evasion

The credit-assignment scheme (Section H there, Section 9 here), the validation
hierarchy (Q), the SL/TP/sizing bootstrap-then-learn sequencing (K/L), the information-
expansion track priorities (M), and the specific rejection of deep RL/Transformers/
hardcoded regimes as the *starting* substrate — none of these were the problem the
mandate identified, and none of them are what "market data → representation →
prediction → gate → trade" describes. The problem was the *framing* of the decision
step as a single combiner rather than a reasoning process with memory and thesis-
tracking. That framing is what changed.

Nothing in this document authorizes implementation. Per mandate Section 18, this is
design only.
