# GOLDEX Phase 2 — Final Intelligence Architecture Decision

STATUS: design/research only. No code, no training, no live connection. Supersedes both
prior Phase 2 documents in this session (the flat-combiner version and the trust-layer
version) — this document performs the comparison those skipped, and its conclusion
absorbs whichever of the earlier reasoning survives scrutiny.

## 1. Problem definition

Build a computational process that takes (market information, quantitative knowledge,
account/position state, its own past experience, its own current uncertainty) and
produces (understanding → opportunity assessment → action → trade management →
learning), operating causally against Phase 1's environment, transferable unchanged to
MT5. Evidence in hand: 28 null marginal-signal hypotheses across V3/V4/genesis; one
null result (not a prohibition) on raw-trajectory-vs-snapshot benefit at the horizons
tested; a validated Phase 1 substrate with an architecture-neutral `DecisionEngine`
seam. Two axes remain genuinely untested: conditional/joint structure, and information
sources beyond single-instrument M1 bars.

## 2. Required intelligence capabilities

Market perception; quantitative-knowledge access and conditional use; contradiction
handling under uncertainty; dynamic entry, SL/TP, and exit (not fixed-horizon); position
management as continuous reasoning, not a static watch; learning from delayed,
path-dependent outcomes with correct credit assignment; memory (tool trust, open-trade
thesis, at minimum); discovery of new useful combinations without hardcoding them;
resistance to overfitting on ~6.7 years of data; low-enough inference latency for
short-duration trading; expandability without a redesign.

## 3-15. Architecture candidates evaluated

Rather than design one architecture and defend it, each candidate below is scored
against the same 22 criteria from the mandate's Section 4, using what's actually known
(evidence) versus what would need to be assumed (risk). "Yes/Partial/No/Untested"
reflects whether the architecture *can structurally support* the capability, not
whether it's been proven to.

| Criterion | A. Contextual bandit / adaptive trust | B. Learned gating (MoE-lite) | C. Full MoE (jointly trained gate) | D. Bayesian decision system | E. Hierarchical decision | F. Sequence/memory model | G. Tool-using agent (LLM-mediated) | H. Model-based RL | I. Model-free RL | J. World-model + policy | K. Hybrid (A + memory + safety floor) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. Market perception | Yes | Yes | Yes | Yes | Yes | Yes | Yes (via tools) | Yes | Yes | Yes | Yes |
| 2. Conditional reasoning | Partial (context-bucket only) | Yes | Yes | Yes | Yes | Yes | Yes (explicit) | Yes | Partial | Yes | Yes |
| 3. Quant-knowledge usage | Yes (native) | Yes | Yes | Yes | Yes | Partial | Yes (native) | Yes | Yes | Yes | Yes |
| 4. Tool selection | Yes | Yes | Yes | Yes | Yes | No | Yes (native strength) | Yes | Partial | Yes | Yes |
| 5. Contradictory evidence | Partial | Yes | Yes | Yes (native strength) | Yes | Partial | Yes (native strength) | Yes | No | Yes | Yes |
| 6. Uncertainty | Partial | Partial | Partial | Yes (native strength) | Partial | No | Partial (depends on tool confidences) | Yes | No | Yes | Yes |
| 7-10. Short-duration/dynamic entry/SL/TP | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| 11. Position management | Partial (needs Section 8 extension) | Yes | Yes | Yes | Yes | Yes | Yes | Yes (native strength) | Yes | Yes | Yes |
| 12. Learning from experience | Yes | Yes | Yes | Yes | Yes | Yes | Partial (depends on numerical layer) | Yes | Yes | Yes | Yes |
| 13. Memory | Partial (trust only, unless extended) | No (unless added) | No (unless added) | Partial | Partial | Yes (native strength) | Yes (native strength) | Yes | Partial | Yes (native strength) | Yes (explicit) |
| 14. Adaptation | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| 15. Discovery of new mechanisms | Yes | Yes | Yes | Yes | Partial | No | Yes (native strength) | Partial | No | Partial | Yes |
| 16. Failure detection | Partial | Partial | Partial | Yes (native strength) | Partial | No | Yes (via reasoning) | Partial | No | Partial | Yes |
| 17. Overfitting resistance | Yes (native strength) | Partial | No (weak, many params) | Yes | Partial | No | Yes (if numerical layer stays small) | No | No | No | Yes |
| 18. Historical simulation compat. | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| 19. MT5 real-time compat. | Yes | Yes | Yes | Yes | Yes | Yes | Partial (LLM latency risk) | Yes | Yes | Yes | Yes |
| 20. Low inference latency | Yes (native strength) | Yes | Yes | Yes | Yes | Partial | No (native weakness) | Partial | Yes | Partial | Yes |
| 21. Fits 6.7yr dataset | Yes (native strength) | Partial | No (native weakness) | Yes | Partial | No (native weakness) | Yes (numerical layer small; LLM pretrained) | No (native weakness) | No (native weakness) | No (native weakness) | Yes |
| 22. Expandability | Yes | Yes | Yes | Yes | Partial | Partial | Yes (native strength) | Yes | Yes | Yes | Yes |

**Reading the table honestly, not selectively:**

- **A (adaptive trust alone)** is strong on data-fit, latency, overfitting resistance,
  discovery — and structurally weak on uncertainty representation, contradiction
  handling, and memory beyond tool trust. These are exactly the three things the
  mandate's Section 10 scenario (five tools disagreeing) demands. A alone is
  insufficient — this is the honest correction to the prior document, which treated A
  as close to complete.
- **C (full jointly-trained MoE) and F/H/I/J (sequence models, both RL families,
  learned world model)** all fail criterion 21 for the same underlying reason: they
  each introduce enough trainable parameters, or enough sequential credit-assignment
  depth, that ~6.7 years of M1 bars (and far fewer realized trades) is a genuinely
  small dataset for them. This is not a stylistic preference; it is the same data-size
  argument that sank RL/Transformers in both prior documents, now applied consistently
  across the whole candidate set instead of only to two of them.
- **D (Bayesian decision system)** scores well almost everywhere — it is explicitly
  built for uncertainty and contradictory evidence, is data-efficient, and is fast at
  inference. Its main weakness is that "Bayesian decision system" is a reasoning
  *style*, not a full architecture — it needs to be paired with something that
  supplies tool selection, memory, and thesis-tracking, which is exactly A's strength.
- **G (LLM-mediated tool-using agent)** is genuinely the strongest on contradiction-
  handling, discovery, and explanation — the mandate's Section 10 scenario ("what
  matters, why, whether to wait") is close to a description of what an LLM reasoning
  over structured tool outputs is good at. Its two disqualifying weaknesses for the
  *execution-critical path* are latency (criterion 20 — an LLM call per decision is
  incompatible with short-duration trading's timing requirements, measured, not
  assumed: Phase 1's existing decision-loop components run in single-digit
  microseconds to low milliseconds; an LLM call is orders of magnitude slower) and
  overfitting/hallucination risk if used to directly emit numerical SL/TP/size rather
  than qualitative judgments. Its strength is real and shouldn't be discarded — see
  the recommended architecture below, where it appears in a role its latency profile
  actually fits.

## 16. Evidence supporting/rejecting each — summary

No candidate wins on every criterion; this is expected and not a modeling error — it's
why the recommendation below is compositional rather than a single named architecture
picked off the list. The clearest, evidence-backed exclusions: full jointly-trained
MoE, both RL families, sequence/world models as the *primary* substrate — all rejected
for the same data-size reason (criterion 21), consistent with the prior documents'
individual rejections of RL/Transformers, now generalized rather than special-cased.
The clearest evidence-backed inclusions: A's data-efficiency and native discovery
mechanism, D's native uncertainty handling, G's native contradiction/discovery
reasoning strength (in a latency-appropriate role only).

## 17-18. Recommended architecture and why it is superior

**A composed system, not a single named architecture from the list — because no single
candidate covers the required capability set, and the table above shows that
plainly rather than by assertion:**

```
                         GOLDEX BRAIN
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                      ↓
   MARKET PERCEPTION    QUANT KNOWLEDGE          MEMORY
   (Phase 1 MarketState  (EvidenceSource-as-tool  (tool-trust history +
    + AccountState +      library, richer than     open-position thesis +
    PositionView)         value/confidence —        recent-decision log)
                          Section 4 spec below)
        └─────────────────────┼─────────────────────┘
                              ↓
                    REASONING LAYER (two tiers)
        ┌─────────────────────┴─────────────────────┐
        ↓                                            ↓
  FAST NUMERICAL TIER                        SLOW DELIBERATIVE TIER
  (Bayesian-flavored adaptive-trust          (LLM-mediated tool-using
   combiner, D+A hybrid: per-tool             reasoning, G's role — invoked
   posterior belief + uncertainty,            asynchronously/periodically,
   evaluated every decision tick,             not on the execution-latency
   microsecond-to-low-ms latency)             path: discovers candidate tool
                                               combinations, flags contradiction
                                               patterns worth encoding as new
                                               EvidenceSources or trust-context
                                               features, writes hypotheses the
                                               fast tier can then test cheaply)
        └─────────────────────┬─────────────────────┘
                              ↓
                    EV/COST GATE (unchanged)
                              ↓
              ACTION: NO_TRADE / LONG / SHORT + SL/TP/size
                              ↓
              TRADE MANAGEMENT: continuous thesis reassessment
              (fast tier only, on the same latency budget as entry)
                              ↓
                     SAFETY FLOOR (Phase 1 engine:
                margin/rejection checks — never the strategy)
                              ↓
                          LEARNING
              (credit assignment → fast-tier trust update,
               periodic slow-tier discovery pass over accumulated
               experience, never on the execution path)
```

**Why this over any single candidate:** it puts each architecture family in the role
its own evidence profile supports, rather than picking one family to do everything.
The fast tier (D+A hybrid) is what actually executes every decision — it inherits A's
data-efficiency/latency/overfitting-resistance and D's native uncertainty/contradiction
handling, closing the gap the pure-A trust-layer document left open (Section 10's
scenario is now a Bayesian belief-update over five conflicting posteriors, not an ad
hoc trust average). The slow tier (G) runs off the execution path entirely — periodic,
offline, over the accumulated experience store — which resolves G's disqualifying
latency weakness by simply never putting it in the latency-critical loop, while still
capturing what makes it uniquely good at discovery and explanation (mandate Section
7/11). This is not "add an LLM because it's available" — it's confining the LLM to
exactly the one role (Section 10 above) where the comparison table shows it's actually
the strongest candidate, and excluding it everywhere the table shows it isn't.

The heavier candidates (full MoE, RL, sequence/world models) are not part of the
recommendation. They remain evidence-based future escalations (Section 21) if Track
D/F evidence later shows the current design's expressiveness is genuinely insufficient
— not permanently rejected, per mandate Section 6/14, but not started now for the same
data-size reason that ruled them out in the comparison.

## 4-15 (architecture-specific answers, using the recommended composition)

**Market perception:** unchanged from both prior documents — `MarketState` +
`AccountState` + `PositionView`, no forced sequence representation, causal-only.

**Quantitative knowledge:** the `EvidenceSource` contract from the trust-layer
document is kept and is now load-bearing for both tiers, not cosmetic — the fast
tier consumes `(value, confidence)` as a Bayesian likelihood input; the slow tier
consumes the full `mathematical_formulation`/`assumptions`/`known_failure_conditions`
metadata to reason about *why* a tool might apply, which is precisely the
richer-than-a-feature-vector treatment this review's Section 2 demanded. This
resolves the review's concern directly: the abstraction was never "reduced to a
feature vector" for the slow tier, and the fast tier's numerical use of it is
justified by that tier's own latency requirement, not by convenience.

**Memory:** three kinds, as in the prior document (within-trade thesis, cross-decision
trust, recursive market state via state-space tools) plus a fourth: a **discovery log**
— candidate tool-combination hypotheses the slow tier proposes, each tagged with the
evidence-conditions under which it was proposed, tested cheaply by the fast tier
against held-out research-validation data before being promoted into the trust layer's
context features. This is the concrete mechanism for mandate Section 11's "GOLDEX
discovers A becomes useful when B and C behave a certain way" — discovered by the slow
tier, tested and adopted (or rejected) by the fast tier's existing validation
discipline, never hardcoded by a human.

**Reasoning:** two-tier, as diagrammed above — this is the direct answer to the
review's central question ("what computational architecture should be the brain")
rather than deferring to one candidate's native reasoning style.

**Decision:** fast tier's Bayesian posterior over the tool library, gated by the
unchanged EV/cost check, exactly as before.

**Trade management:** continuous thesis reassessment (unchanged from the trust-layer
document's Section 8) — explicitly first-class per the review's Section 12, running
on the fast tier's latency budget since it's on the execution path.

**Learning:** unchanged credit-assignment discipline (Section H/9 from the prior two
documents, carried forward exactly) feeding the fast tier's per-tool trust posteriors;
the slow tier's discovery pass runs periodically over the accumulated experience
store, off the execution path.

**Discovery:** the discovery log above — new in this document, directly answering the
review's Section 11.

**Risk:** unchanged — Phase 1's engine-level margin/SL-TP rejection is the safety
floor beneath the reasoning layer, never the strategy itself, per the review's
explicit warning in its own Section 12/Section 8 of the diagram request.

**Historical training / validation / real-time / MT5 architecture:** unchanged from
both prior documents — Phase 1's chronological replay, CPCV + protected OOS,
Phase 1's existing latency-measurement discipline extended to the fast tier's
inference cost specifically (the slow tier is never on this path, so it doesn't need
to meet the same latency bar — measure it separately, for its own budget: how often
can a periodic discovery pass reasonably run).

## 19. Minimum viable implementation

1. `EvidenceSource` contract with the full metadata fields (already speced in the
   trust-layer document's Section 5) — built once, used by both tiers.
2. Fast tier: per-tool Bayesian posterior (belief + uncertainty) updated from
   correctly-credited outcomes, context-conditioned on the recursive state-space
   tools' output — this is a concrete, buildable refinement of the trust-layer
   document's Section 6, not a new component; it upgrades "decayed running mean" to
   "explicit posterior with uncertainty," which is what closes criterion 6's gap.
3. Thesis-tracking for open positions (unchanged from the prior document).
4. Credit-assignment implementation, tested against synthetic known-outcome
   trajectories before real training (unchanged).
5. Analytical SL/TP/sizing bootstrap (unchanged).
6. **The slow tier is explicitly NOT in the minimum viable build.** It requires (2-5)
   to exist first, since it operates on their accumulated experience/trust state. It
   is the first item in Section 20's expansion path, not Section 19's build list — this
   keeps the minimum-viable build honest about what genuinely needs to exist before
   anything else, rather than building the more novel/interesting piece first.

## 20. Expansion path

Slow deliberative tier (LLM-mediated discovery, Section 8's diagram) — first
expansion, once the fast tier has accumulated enough experience to give it something
real to reason over. Escalation from Bayesian-posterior fast tier to a learned gating
network (candidate B) if Track D shows conditional structure the posterior form can't
express. Track B/C/D/F research tracks — unchanged priorities from both prior
documents. Full MoE/RL/sequence/world-model architectures remain possible future
escalations, gated on evidence the current composition's expressiveness is
insufficient (Track D/F), not on convenience or trend.

## 21. Major risks

- **Two-tier coordination risk**: the slow tier's discovered hypotheses must be
  validated by the fast tier's existing discipline before promotion — if this gate is
  weak, the slow tier becomes a laundering path for spurious patterns discovered on
  the training window, exactly the memorization risk mandate Section 13/15 warns
  about. This is the single highest-risk new surface this document introduces, and the
  research-validation-before-promotion rule above is the mitigation, not a
  formality — it needs to be built and tested with real teeth, not assumed to work.
- **Bayesian posterior misspecification**: a poorly chosen prior/likelihood model
  for tool trust could be miscalibrated in a way that's hard to detect until live —
  mitigated by the existing shuffled-null discipline extended to trust posteriors
  (unchanged from the trust-layer document's Section 11).
- **Latency creep**: if the fast tier's per-tool Bayesian update becomes
  computationally heavier than the current trust-layer's decayed mean (plausible,
  since posteriors are more expensive than running averages), Phase 1's latency
  budget (Section R/19 of the prior documents) needs re-measurement before this is
  trusted for short-duration trading — flagged explicitly rather than assumed fine.
- **Slow-tier availability**: an LLM-mediated discovery process depends on an external
  or hosted model being available/affordable at whatever frequency it runs — a
  practical dependency risk with no analog in the fast tier, worth flagging even
  though it's not architecturally disqualifying (mitigated by being off the
  execution-critical path entirely).

## 22. Explicit assumptions

That ~6.7 years of M1 data is genuinely insufficient for jointly-trained deep
architectures (criterion 21) — this is an assumption grounded in the null trajectory
result and general small-sample deep-learning practice, not a proven bound; it should
be revisited if Track B/C materially expand the usable dataset (e.g. cross-instrument
data extends the effective sample). That a Bayesian-posterior fast tier is
computationally tractable at the required latency — assumed, not yet measured (see
Section 21's latency-creep risk). That an LLM-mediated slow tier's discoveries can be
validated by the existing CPCV/OOS discipline without a new validation mechanism —
assumed; if discovered "hypotheses" turn out to be qualitatively different from
tool-admission decisions (e.g. propose new tool *combinations* rather than new
individual tools), the validation discipline may need extension, not yet designed
here.

## 23. Explicit unknowns

Whether the fast tier's Bayesian-posterior form is actually more capable than a
simpler bandit-style estimate in practice, or whether the added complexity buys
nothing without real conditional structure (Track D's job to determine). Whether the
slow tier's discovery process produces anything genuinely useful versus mostly noise
that the validation gate correctly rejects most of the time (won't be known until
built and run). Whether Track B/C/D/F evidence, once gathered, favors escalating any
single component (fast-tier expressiveness, sequential trade-management structure,
information sources) over the others — this document does not predict which track
will matter most, deliberately, since predicting that now would be exactly the kind of
premature commitment this review's whole exercise was meant to prevent.

---

Nothing in this document authorizes implementation. Per Section 17 of the review that
requested it, this is design/research only.
