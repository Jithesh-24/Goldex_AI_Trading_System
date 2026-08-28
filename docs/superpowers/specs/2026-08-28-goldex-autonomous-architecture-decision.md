# GOLDEX Autonomous Trading Architecture Decision (revised, working hypothesis)

Supersedes the unrevised draft from the same session. FOMC event-time side-quest
closed as of this document; findings preserved as-is in
`docs/superpowers/reports/2026-08-28-goldex-event-time-test1b-falsification.md`,
no further work on it authorized by this document.

## A. Evidence

28 hypotheses across V3, V4, and the genesis reset (specialist models;
momentum/volatility/GARCH/Kalman/skew-kurtosis representations at a fixed
5-bar horizon; a 6-horizon x 7-representation sweep = 42 cells; event-time
NFP/CPI/FOMC bucketing) — zero surviving marginal signal in M1-derived
single-instrument features. One narrow FOMC-window anomaly (n=780) survives
isolation but was never OOS-tested; shelved by explicit user instruction,
not investigated further here.

## B. What that evidence does and does not prove

**Proves:** the specific search already run — marginal MI of hand-built
scalar features derived from M1 OHLC, at various fixed horizons, on a
single instrument — is exhausted and null. Convergence across 6 model
families and 6 horizons on the same null, via the same trend-confound
mechanism, is real evidence the bottleneck is information, not model
choice.

**Does not prove:** that Gold is unpredictable; that cross-instrument or
tick-level information is useless (neither was tested); that any
particular decision architecture (gated experts, RL, learned combiner,
hierarchical model) would fail — none of those were actually tested,
only marginal statistical prediction was. It also does not prove a
*joint* or *conditional* representation (mechanism X matters only in
market state Y) is useless — every test to date was marginal/unconditional
by construction.

## C. Architecture candidates considered

- Fixed/weighted voting library over quantitative mechanisms — rejected;
  structurally equivalent to what the 28 null hypotheses already tested
  (a static combination of individually-null mechanisms is not evidence
  any combination is meaningful).
- End-to-end learned agent / full RL — no evidence base earns this yet;
  multi-step credit assignment is unsolved (Section I), and the one direct
  test of whether sequential/trajectory structure matters over a single
  snapshot (Phase 4 trajectory-vs-snapshot) came back null, removing RL's
  strongest justification.
- Decomposed gated-expert combiner (entry) + separate exit mechanism +
  rule-based risk/sizing, sharing a persistent state layer — current
  working hypothesis (Section D), explicitly not final.
- Learned persistent state + gate (state layer feeds gate) — plausible
  later, premature without a validated evidence source to build state
  from.
- Hierarchical / mixture architectures — not ruled out; would become
  relevant if Track D or F evidence shows genuinely hierarchical structure
  (e.g. a macro-scale state gating a micro-scale decision).

None of these is locked in. The scaffold (Section F) is required to
support swapping the decision mechanism among any of the above — or one
not yet identified — without touching the simulator, account model, or
execution model.

## D. Why the Architecture-G-shaped hypothesis still exists

Decomposition into auditable per-component pieces (separate entry
signal, exit policy, risk/sizing) remains the best-supported *starting
shape* given how many of the 28 prior results required careful
confound-interrogation to interpret correctly — an opaque, unified
architecture would make the next result impossible to trust either way.
This is a hypothesis about shape only. The mechanism that fills the
entry-decision seam (fixed rule -> learned combiner -> sequential policy
-> hierarchical model) is deliberately left open and is exactly what
Tracks B-F are meant to determine.

## E. Conditions under which this working hypothesis is rejected

- If a conditional/state-dependent signal emerges (Track D) that is only
  expressible as a jointly learned representation (e.g. a shared state
  encoder feeding both entry and exit), decomposing entry and exit into
  independent modules would discard exactly the interaction carrying the
  signal. Reject the decomposition; move toward a shared-state or
  sequential-policy architecture instead.
- If credit assignment (Track E) requires multi-step return propagation
  that a supervised-entry-gate-plus-analytical-exit split cannot express,
  reject the mixed-learning split in Section I.
- If trade-management research (Track F) shows entry, exit, SL/TP, and
  sizing are not analytically separable — e.g. optimal SL depends on the
  same latent state driving the entry signal — reject the "exit is a
  separate mechanism" assumption specifically, independent of the entry
  architecture's fate.

## F. Modular architecture / scaffold design

```
simulator/engine.py, replay.py, AccountState      [KEEP, unmodified]
        |
 MarketObservation (state features, any source, causally constructed)
        |
 EvidenceLayer   <- pluggable registry: each quantitative mechanism
        |            registers as a named EvidenceSource (feature value +
        |            validity/confidence metadata), never as a vote.
        |            New sources plug in without touching anything
        |            downstream of this layer.
        |
 DecisionEngine  <- interface: observe(state, evidence, account) -> Action
        |            Action = NO_TRADE | LONG | SHORT, with entry / SL /
        |            TP / size / exit fields populated where applicable.
        |            This is the ONLY component whose implementation
        |            changes when the decision architecture changes —
        |            starts as a stub/heuristic now, later becomes
        |            whichever mechanism the evidence in B-F supports.
        |
 ExperienceRecorder (decision_id-linked trajectory assembly)   [KEEP]
        |
 LearningLoop (research learning vs. validated deployment, mandate S13)
```

Contract: `DecisionEngine` is the single replaceable seam between market
information and trading action. Everything above and below it — the
simulator, account/execution model, evidence registry mechanics, and
experience recording — is architecture-agnostic and does not change when
the decision mechanism inside `DecisionEngine` changes.

## G. Quantitative-knowledge representation

Each candidate mechanism (momentum, mean reversion, volatility, breakout,
reversal, GARCH, Kalman/state-space, HMM/regime, entropy/information
measures, microstructure, cross-asset relationships, options-derived
quantities where legitimately available) registers as an `EvidenceSource`:
a named, versioned function `state -> (value, validity/confidence)`. No
source votes or decides on its own. The `DecisionEngine` consumes the
full evidence set and is responsible for learning *which sources matter
under which conditions* — that conditional-usefulness learning is the
actual intelligence problem this project is solving, not the existence
of the sources themselves.

Admission bar for a source entering the registry: it must pass the
existing MI-vs-shuffled-null discipline as evidence-worth-including, not
as a standalone-profitable-strategy. This is a real, load-bearing
distinction from the prior research mode: a source with weak/noisy
*marginal* signal can still be legitimately admitted if it is
*conditionally* informative (useful only in specific market states) —
none of the 28 prior hypotheses tested conditional usefulness, only
marginal/unconditional usefulness, so this is genuinely unexplored
ground, not a retest.

Black-Scholes / options-derived information: remains an open question,
not a required or rejected component by default. Track D investigates
whether any Gold-options or implied-volatility data is actually available
and relevant before rendering a verdict either way; if unavailable or
irrelevant, it is rejected explicitly with the evidence for that, not
dismissed on the assumption that spot XAUUSD has no legitimate use for it.

## H. Market-information architecture

Current M1-only inputs (OHLC, bar-aggregated tick_volume, spread series,
timestamp) are exhausted at the marginal level per Section A/B. Two
information tracks run in parallel with the scaffold build, not gating it:

- **Track B — cross-instrument** (DXY, real yields, other metals):
  structurally different information class than anything tested, cheap
  and historically available, never tested in any of the 28 hypotheses.
- **Track C — tick/bid-ask microstructure**: structurally invisible in
  M1 bars by construction (not merely undertested at M1 resolution), higher
  acquisition cost, justified directly by the horizon-sweep falsification
  (genesis reset document, Section 34: horizon/target definition is
  falsified as the missing variable, leaving information source as the
  live open question).

## I. Learning architecture

Deliberately undecided at the mechanism level (supervised / RL /
hierarchical) — that decision depends on what Tracks B-E actually find,
not on a prior commitment. What is decided now: the `LearningLoop`
interface structurally separates *research learning* (offline, on
recorded trajectories, freely iterated) from *validated deployment*
(a frozen, promoted model, only reachable after passing the validation
wall in Section J) — this separation is enforced by the interface, not
left to convention. Credit assignment (attributing a trade's outcome to
the entry decision specifically, vs. path noise afterward, vs. the
unobservable counterfactual for NO_TRADE and early-exit decisions) gets
its own precise, reviewed specification before any trajectory-based
training begins, unconditionally, regardless of which learning mechanism
eventually wins (Track E).

## J. Trade-management architecture

Open research question (Track F), not a hardcoded split of "entry is AI,
SL/TP is rules." The scaffold represents entry / SL / TP / size / exit
as fields the `DecisionEngine` can populate per decision, with no built-in
fixed horizon — no assumption of a 15/45/90-bar window or any other fixed
holding period. Track F explicitly investigates whether the system should
instead learn when an opportunity begins and ends, and whether entry,
exit, SL/TP, and sizing should end up jointly learned, hierarchically
learned, partially analytical, or separately learned. Whichever answer
Track F produces determines the internals of `DecisionEngine`, not a
premise fixed by this document.

Validation wall (mandate S19, unconditional regardless of build speed):
TRAINING -> RESEARCH VALIDATION -> UNTOUCHED OOS -> DEMO -> LIVE. The
untouched OOS split (rows 300,000:400,000) has never been read by any
of the 28 hypotheses and stays unread until a candidate clears research
validation via CPCV.

## K. Parallel research tracks

| Track | Scope | Blocking relationship |
|---|---|---|
| A | Reusable scaffold: EvidenceLayer registry + DecisionEngine interface + ExperienceRecorder wiring, built on the unmodified Phase 1 simulator | Blocks nothing else; buildable immediately |
| B | Cross-instrument MI-vs-null (DXY, real yields, other metals) | Feeds Section D/G; independent of A |
| C | Tick/bid-ask acquisition scoping + first MI-vs-null pass | Feeds Section D/G; independent of A |
| D | Quantitative-knowledge integration: conditional/joint MI tests (mechanism useful *given* market state X), the untested axis every prior hypothesis omitted | Feeds Section E/G; independent of A |
| E | Credit-assignment specification and review | Feeds Section I; independent of A |
| F | Trade-management research: joint-vs-separable entry/exit/sizing, opportunity-duration learning instead of fixed horizons | Feeds Section J; independent of A |

## L. What can be implemented immediately

Track A, in full: the scaffold (Section F), the `EvidenceSource` registry
(populated with the 9 already-validated-as-code representation functions
from Phase 3A/4, explicitly carried over as candidate evidence sources,
not as strategies), `ExperienceRecorder` wiring, and the `LearningLoop`
skeleton with the research/deployment split enforced structurally. This
is plumbing — it does not claim, assume, or require that any validated
trading signal currently exists.

## M. Evidence required before activating autonomous trading

1. At least one evidence source (marginal, per B/C, or conditional, per
   D) clears validation on the untouched OOS split via Combinatorial
   Purged Cross-Validation.
2. A concrete `DecisionEngine` implementation measurably beats the
   existing null/random-baseline control gate (Phase 2, unchanged) on
   research validation.
3. The credit-assignment specification (Track E) has been reviewed and,
   before being trusted on real trajectories, verified against
   known-outcome synthetic trajectories.

None of this currently exists. Track A being built is scaffolding, not a
claim that any of the above conditions are met.

## N. Fastest credible route to a functioning autonomous historical trader

Build Track A now, in parallel with Tracks B-F starting immediately
after. Tracks B and D are the cheapest and highest information-value —
both reuse the existing MI-vs-shuffled-null machinery unchanged, only
conditioning or adding cross-instrument inputs rather than re-running
marginal tests. Whichever track produces the first validated signal
determines what plugs into the `DecisionEngine` seam first; the seam
itself does not depend on which track wins. No track reports back after
every sub-step — each reports only at a genuine decision point (signal
found, or track exhausted with nothing found), consistent with the "no
serial diagnostic loop" instruction governing this phase.
