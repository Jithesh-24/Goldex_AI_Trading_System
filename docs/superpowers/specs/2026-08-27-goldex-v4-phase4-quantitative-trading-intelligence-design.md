# GOLDEX V4 — Phase 4 Design: Quantitative Trading Intelligence

Design/research only. No code. No production changes. No Phase 3B. No V3 resumption.

---

## 1. Executive conclusion

Phase 3/3A proved that **single-bar, engineered-or-raw M1 representations, fed to linear/tabular/tree/boosted models, contain no exploitable structure for a 5-bar forward-return objective** on this data. That is a narrow, real, well-controlled result — it does not prove Gold has no edge, and it does not tell us anything about trajectory-level, multi-mechanism, or account-aware decision-making, because nothing in Phase 3/3A tested any of those.

The correct Phase 4 is **not** "try a bigger model on the same framing." The evidence says the framing itself — single-bar snapshot → single scalar target → single model — is likely exhausted at M1. Phase 4's job is to build the **architecture** that lets GOLDEX combine multiple heterogeneous, individually-weak evidence sources (quantitative mechanisms) into a state representation, and make a *sequential* decision (not a single forecast) using that state, learned from simulated trajectories rather than single-bar labels. This is an architecture proposal, not a proof that this architecture will find an edge — external research (Section 3) is unambiguous that most published edges at this horizon are thin, cost-sensitive, and hard-won, and this document says so plainly rather than promising otherwise.

**Recommendation: build the architecture (opportunity/decision layer + mechanism layer + experience/learning layer as described below) as a thin, extensible skeleton first, validate the skeleton is honest (no leakage, no premature promotion) with the cheapest possible experiment, and only then invest in any specific mechanism or model.** Do not chase the product-shaped output list in the mandate (SL/TP/confidence/reasoning) until the underlying decision engine has evidence behind it — that's a serialization concern, not an architecture concern, and premature commitment to it constrains the design for no current benefit.

---

## 2. What Phase 3/3A actually proved (and did not)

**Proved:**
- 7 discovery candidates (rule-based, regime-statistical, linear, tabular-RL, Bayesian-online, HMM, sequence-logistic) using hand-engineered scalar features found no edge on real M1 XAUUSD data, honestly re-verified after a real credit-assignment bug fix (Phase 3).
- Raw price-path windows, multi-scale volatility, and volatility-regime-transition representations — tested against a proper chronological split with shuffled-label null controls, using both a shallow tree and a boosted ensemble — showed **no** improvement over Phase 3's original scalar representation for 5-bar forward return (Phase 3A, independently re-verified by review).
- The `ExperienceRecord` observation-recording gap is now closed additively; a real per-decision feature vector can be captured without touching Phase 1/2 verdict machinery.

**Did NOT prove:**
- That no representation at any horizon or trajectory length contains signal — every experiment used one fixed 5-bar horizon and single-bar-snapshot features. Multi-bar sequence models, trajectory-level credit assignment, and other horizons were never tested (explicitly flagged as "untested, not ruled out" in the 3A report).
- That Gold itself has no edge — only that these specific representation/model combinations at this specific horizon do not.
- Anything about account-aware decision-making, position management, dynamic exit, or combining multiple mechanisms — Phase 3/3A tested pure prediction, not the sequential opportunity-capture loop this project actually wants.

This distinction is the entire justification for Phase 4 existing rather than declaring the project over.

---

## 3. External research findings

(Full sourced briefing available on request; condensed and load-bearing points below. Each point tagged by type: **[EXT]** externally supported, **[REPO]** repository evidence, **[INF]** engineering inference, **[HYP]** untested hypothesis.)

- **[EXT]** Professional quant research treats signal discovery as hypothesis-generate → falsify → interrogate *why* something worked/failed, not curve-fit-until-profitable — this matches the discipline Phase 3/3A already followed (control gates, null controls, honest negative results) and should continue into Phase 4.
- **[EXT]** Jane Street's own public description is pragmatic and non-prescriptive: "large-scale ML, domain expertise, or pen-and-paper math depending on the problem" — no single privileged mechanism. This directly supports the mandate's "don't preselect the brain" instruction; it is not itself evidence Jane Street's specific techniques transfer here (their public material is genuinely thin — recruiting-page level, not technical papers).
- **[EXT]** Lopez de Prado's central finding: most financial-ML failures trace to bad data handling and invalid testing (leakage via standard k-fold under serial correlation), not bad models. Phase 3/3A already avoided this (chronological splits, no shuffling, no validation peeking) — Phase 4 must hold this line as complexity increases, when it gets easier to accidentally break.
- **[EXT]** Intraday predictability, where it exists at all, is driven by microstructure friction (order flow imbalance, spread, depth), not fundamentals, and even well-designed ML studies report tiny raw out-of-sample R² (~0.24%) yielding modest Sharpe (~0.73) *after* costs. **[INF]** This sets a realistic expectation ceiling: GOLDEX should not be designed around finding a large, robust edge — it should be designed to find a small one and not destroy it with costs or overfitting, and to survive finding none.
- **[EXT]** Reinforcement learning in trading has two well-documented failure modes directly relevant here: reward hacking (optimizing a proxy objective that diverges from real profitability) and a severe sim-to-real gap where frictionless simulators teach policies that exploit zero-friction assumptions that don't survive real execution. **[INF]** Given Phase 1's simulator already models spread/slippage/liquidation realistically, the sim-to-real risk is lower than a naive RL setup, but reward-hacking risk is real and must be designed against explicitly (Section 10).
- **[EXT]** Self-adaptive/continual-learning models show lower error and faster shock-recovery than static periodic retraining in non-stationary financial series — this favors an online/continual learning posture over "train once, freeze, deploy" for Phase 4's eventual live system, but does not mandate full online RL.
- **[EXT]** ML-produced probabilities are commonly miscalibrated (confidence ≠ true frequency), and ensembles are the most practical industry-scale uncertainty-quantification technique versus exotic Bayesian methods. **[INF]** This matters directly for the mandate's "confidence" output — a confidence score is worthless (and dangerous for position sizing) if uncalibrated; calibration must be a designed, tested property, not an assumed byproduct of model outputs.
- **[EXT]** The Probability of Backtest Overfitting (Bailey/Borwein/Lopez de Prado) shows PBO approaches 1 as the number of tested configurations grows, *regardless* of whether any configuration has real predictive power. **[INF]** Phase 3/3A already tested ~15 hypotheses; Phase 4's mechanism-ladder approach (Section 17) risks compounding this multiple-testing problem if not tracked explicitly — every mechanism tried must be logged, and the eventual go/no-go decision must account for how many things were tried, not just the best result seen.
- **[EXT]** Black-Scholes assumes constant volatility (empirically false) and has no direct pricing/hedging role in a spot (non-options) position — its only legitimate contribution in a spot-only context is as a volatility-estimation *lens* (e.g., implied volatility from a related options market as a forward-looking risk signal). GARCH-family and stochastic-volatility models (which BS's constant-vol assumption relaxes into) are the directly-applicable machinery for spot volatility forecasting. **No gold-specific literature was found** — this is extrapolated from general equity/FX volatility research, flagged accordingly.
- **[EXT]** Stacking (a meta-model over heterogeneous base-model outputs) is generally more robust to base-model misspecification than simple averaging or Bayesian Model Averaging, and dynamic/regime-weighted ensembles are used specifically for regime-shifting financial series. **[INF]** This is the strongest external support for the mandate's "let the intelligence learn when a mechanism is useful" framing — a *learned combiner* over mechanism outputs, not a hardcoded strategy-selector, is a well-established pattern, not a novel bet.

**What research does NOT support:** treating any single architecture (RL, Transformers, HMMs, Black-Scholes) as a privileged default. The literature converges on "combine several weak, individually-understood sources under a learned or statistically-principled combiner, validated ruthlessly against leakage and overfitting" — which is exactly the mandate's own instinct, now with external grounding rather than assumption.

---

## 4. Architectural problem definition

Phase 3/3A's architecture was: `single bar → engineered/raw scalar features → single model → single scalar prediction`. This is a **prediction** architecture. The project's actual objective (Section 2 of the user's mandate) is an **opportunity-capture decision loop**: observe continuously, maintain belief about current market state, decide LONG/SHORT/NO_TRADE/HOLD/EXIT, manage a position through time, learn from the full trajectory. These are different problems. A prediction architecture, however improved, cannot become a decision loop by adding more models — it needs a different shape: a **state layer** that persists across bars (not a fresh snapshot every decision), a **decision layer** that outputs an action given that state (not a scalar forecast), and an **experience layer** that credits whole trajectories, not single-bar labels, to whatever produced them (extending Phase 3A's fixed per-trade bug fix from "credit the trade" to "credit the sequence of decisions that produced the trade").

The concrete architectural bottleneck is therefore: **GOLDEX has no state layer and no trajectory-level learning mechanism yet.** Section 5 addresses this directly.

---

## 5. Proposed GOLDEX intelligence architecture

```
MARKET DATA (M1 bars, real spread; tick data deferred, Section 15)
        │
        ▼
REPRESENTATION LAYER  ── produces the raw observation (price, volume, spread,
        │                 account state) — thin, no modeling, same causal
        │                 guarantees as today's market_state_builder.py
        ▼
QUANTITATIVE MECHANISM LAYER (Section 6) ── a fixed-interface library of
        │        independent "evidence sources": each mechanism consumes the
        │        representation layer's observation history and emits a
        │        small, typed evidence vector (not a trade signal) — e.g.
        │        "volatility regime: high, persistence: 0.7", "momentum
        │        z-score: 1.2", "distributional skew: -0.3". Mechanisms are
        │        stateless w.r.t. each other and individually auditable.
        ▼
MARKET STATE LAYER ── a persistent, updated-every-bar belief state built
        │        from the mechanism layer's evidence vectors plus recent
        │        history (this is the missing "state layer" from Section 4).
        │        This is where sequence/trajectory information lives.
        ▼
LEARNED INTELLIGENCE LAYER (Section 7) ── a combiner over market state that
        │        learns which mechanisms matter when (the mandate's "Mechanism
        │        A is useful under state X" behavior) — this is the piece
        │        that decides IF an opportunity exists and how strong it is,
        │        not yet what to do about it.
        ▼
OPPORTUNITY / DECISION LAYER (Section 8) ── converts the combiner's belief
        │        into a discrete action (LONG/SHORT/NO_TRADE) plus whatever
        │        continuous outputs are justified (not hardcoded), gated by
        │        an explicit expected-value-after-cost check reusing Phase
        │        2/3's cost/EV machinery.
        ▼
TRADE-MANAGEMENT LAYER (Section 9) ── once a position is open, a separate
        │        (but architecturally similar) decision process re-evaluates
        │        every bar: HOLD or EXIT, using the same market state layer.
        │        This is what makes exits dynamic instead of TP/SL-only.
        ▼
EXECUTION / SIMULATOR (Phase 1, untouched) ── fills, costs, liquidation.
        │
        ▼
EXPERIENCE / LEARNING LAYER (Section 10) ── records full trajectories
        (state sequence → actions → outcome), not single-bar labels;
        feeds back into the Learned Intelligence Layer between training
        passes, never into live/validation data.
```

**Why this is superior to "add more predictive models":** every mechanism, the combiner, and the decision layer are separable, independently testable, and independently falsifiable — a failed mechanism is deleted, not silently propping up an opaque monolith. The market state layer is the one genuinely new piece of infrastructure; everything else is a disciplined arrangement of things already partially built (Phase 1 simulator, Phase 2 verdict/EV machinery, Phase 3's `learn()` hook, Phase 3A's `observation_features` recording). This is deliberately not a rebuild.

---

## 6. Quantitative mechanism layer

A mechanism is a **pure function of observation history → a small typed evidence vector**, never a trade signal, never a strategy. Candidate mechanism families to evaluate (not commitments, per the mandate):

- **Volatility family**: realized vol at multiple scales (already exists from Phase 3A), a GARCH-family conditional-volatility forecast (directly supported by external research as more applicable than Black-Scholes for spot), regime persistence (exists from Phase 3).
- **Momentum/mean-reversion family**: the Phase 3 momentum scalar, statistical deviation from a rolling mean, z-scored versions at multiple windows — already shown individually weak (Phase 3A), but their *value as inputs to a learned combiner conditioned on regime* has never been tested; that is a different claim than "the scalar alone predicts returns," and is worth one cheap test (Section 18) before being dismissed.
- **State-estimation family**: Kalman-filtered price level/trend as a denoising mechanism (evidence, not a target) — genuinely different from raw or engineered scalars, untested in Phase 3A.
- **Distributional family**: skew/kurtosis of recent return distribution, jump-detection (large single-bar moves relative to recent vol) — cheap to compute, untested.
- **Execution/microstructure family**: spread state, tick-volume anomalies — limited by M1 aggregation (no true tick data yet, Section 15), but the spread itself is real historical data already used correctly in Phase 1's cost model, and its *level/dynamics* as an evidence source (wide spread = uncertain/illiquid conditions) is untested and cheap.
- **Options/volatility-surface family**: explicitly **not recommended** for near-term work — Section 16 gives the full assessment; the short version is Black-Scholes has no direct role for spot XAUUSD and no gold options data currently exists in this project, so this family is deferred, not built now.

Each mechanism must ship with: its own unit test, a description of what market condition it's meant to capture, and its own cheap standalone validation (mutual-information-vs-null in the style of Phase 3A) before being added to the state layer — this prevents the "strategy zoo" failure mode explicitly, because a mechanism that fails its own cheap test is never wired in.

---

## 7. Learned intelligence layer

This is the combiner: takes the market state vector (built from however many mechanisms have passed their cheap validation) and learns which mechanisms are informative under which conditions. Per the mandate, do not preselect this. Candidate approaches, to be chosen empirically via the ladder in Section 17, **not committed to here**:

- A stacked/gated ensemble (a learned weighting over mechanism outputs, conditioned on regime) — directly supported by external research (Section 3) as more robust than naive averaging and a natural fit for "mechanism A useful under state X."
- A sequence model (even a small recurrent or windowed-attention model) over the market state history, if the trajectory-level cheap experiment (Section 18) shows sequence information Phase 3A's single-bar tests couldn't see.
- A contextual-bandit-style online learner if the eventual live-adaptation requirement (Section 3's continual-learning finding) dominates the design once a base architecture exists.

**Explicitly not committed to at this stage:** full reinforcement learning. RL is the most expressive option but also the one with the most severe documented failure modes for this exact problem (reward hacking, sim-to-real gap) and the least sample efficiency — it should be considered only after the simpler combiner has been tried and evidence specifically points at needing multi-step credit assignment that a stacked/gated model cannot provide.

---

## 8. Opportunity/decision layer

Converts the combiner's belief (not yet a trade) into LONG/SHORT/NO_TRADE by gating on an explicit expected-value-after-cost check — reusing Phase 2's `ev_cost`/`cost_r` machinery rather than inventing a new one. This is where the "opportunity must be sufficiently attractive after costs and uncertainty" requirement (mandate Section 7) is enforced mechanically, not left to the model's discretion. Continuous outputs (confidence, expected magnitude) are produced *only if* the combiner's calibration has been validated (Section 3's calibration finding) — an uncalibrated confidence number should not ship as a product output; that's an explicit stop condition, not a nice-to-have.

Entry timing granularity stays bar-native (as Phase 1 already supports) rather than sub-bar — no evidence yet justifies building faster-than-bar entry logic, and M1 data cannot support validating it (Section 15).

---

## 9. Trade-management layer

Structurally, this reuses Phase 1's existing `manage()` contract (HOLD/EXIT called every bar while in position) — no simulator change needed. What's new is that `manage()`'s decision should be able to consult the same market state layer used at entry, not a separate, narrower "exit rule." This directly implements the mandate's "continuously reassess an open position" requirement using infrastructure that already exists and is already proven leakage-safe.

---

## 10. Learning/experience architecture

Phase 3A's Section F design analysis already mapped the concrete gap: `ExperienceRecord` now carries `observation_features` per-decision (closed additively, no look-ahead, reviewed), but there is no `decision_id` linking a `DECIDE` record to the `POSITION_CLOSED` record it eventually produced, and no orchestration layer above `run_replay` for multi-epoch/trajectory training. Phase 4's minimum experience-architecture change is exactly that: add the linking key, and build a **new** orchestration module (not a modification to `phase3_tournament.py` or `phase2_tournament.py`) that can assemble full trajectories (state sequence → action sequence → outcome) for whichever learning approach Section 17's ladder ends up justifying. This keeps Phase 1/2's verdict machinery completely untouched while unlocking trajectory-level learning.

Per the mandate: do not assume this means RL. A stacked/gated combiner trained via ordinary supervised learning on trajectory-derived labels (e.g., "was this decision, in hindsight, part of a captured or missed opportunity") is a legitimate, lower-risk starting point; full RL, online learning, or a hybrid are later options gated by evidence, not assumed now.

---

## 11. Account/execution awareness

Account state (balance, equity, drawdown, exposure) is already tracked correctly by Phase 1's `AccountState`. Feeding it into the market state layer as an evidence input is a small, additive step. The explicit guardrail: account state may inform position sizing/risk gating, but **must never be allowed to become the training signal that teaches the model to chase losses** — this is enforced architecturally by keeping account-state features separate from the reward/label used to train the combiner (the combiner learns from market-state → outcome, not from balance → next-action), not by hoping the model learns not to martingale.

---

## 12. Real-time/latency architecture

Out of scope for Phase 4 itself — this is a Phase 6 (XM demo) concern. The one Phase-4-relevant point: the market state layer must be defined so that it can be computed identically from a live MT5 feed as from the historical replay (same function, different data source), the same discipline Phase 1's `market_state_builder.py` already follows (`source="synthetic_replay"` vs. `"mt5_live"` distinction already exists in the `MarketState` contract). No new work needed now beyond keeping this invariant in mind when the mechanism layer is built.

---

## 13. Data requirements

M1 historical data (existing, ~6.7 years) remains sufficient for everything in Phase 4's initial scope (Section 18's cheap experiments, mechanism validation, combiner prototyping). No new data acquisition is justified yet (see Section 15).

---

## 14. Historical simulator's role

Unchanged and central: it remains the authoritative environment for all training/validation, exactly as Phase 1-3A established. Phase 4 adds an orchestration layer above it (Section 10), not a replacement.

---

## 15. Future tick-data role

Per Phase 3A's own conclusion: M1's representational ceiling for single-bar/short-scalar prediction was tested and found empty, but that says nothing about tick-level microstructure, which M1 cannot represent by construction. **Do not acquire tick data now.** It becomes justified only if Phase 4's own cheap experiments (Section 18) find evidence that a mechanism or the combiner is bottlenecked specifically by resolution rather than by architecture (e.g., a spread-dynamics mechanism that would clearly benefit from real bid/ask tick sequences rather than M1-aggregated spread). No such evidence exists yet.

---

## 16. Black-Scholes / options / quant-finance assessment

**Direct verdict: Black-Scholes has no legitimate direct role in spot XAUUSD decision-making**, and building it into GOLDEX now would be cargo-culting a famous model without justification — exactly what the mandate warned against. Its only theoretically legitimate contribution (extracting an implied-volatility signal from a related options market as a forward-looking risk gauge) requires options market data GOLDEX does not currently have access to or ingest, making it a non-starter for Phase 4 regardless of theoretical merit. **What is directly applicable instead**: GARCH-family conditional volatility forecasting and, if evidence later justifies more sophistication, stochastic-volatility models (which are what Black-Scholes's constant-volatility assumption relaxes into) — both operate on spot price history alone and fit naturally into the Volatility mechanism family (Section 6). Recommendation: do not build a Black-Scholes component; do add a GARCH-family volatility-forecast mechanism to the mechanism-layer ladder, evaluated with the same cheap-test discipline as everything else.

---

## 17. Candidate model/mechanism ladder

Ordered by cost, cheapest first — nothing on this ladder is built before the item above it produces evidence justifying the next step:

1. Cheap mechanism validation (Section 18, item 1) — near-zero cost, statistics only.
2. A stacked/gated combiner over whichever mechanisms pass step 1, trained supervised on trajectory-derived labels (Section 10) — moderate cost, no new infrastructure beyond the orchestration layer.
3. A small sequence-window model over market-state history, only if step 2's trajectory experiment (Section 18, item 2) shows sequence information single-bar tests couldn't see — moderate-to-higher cost.
4. Online/continual adaptation of whichever model wins steps 2-3, only if live-deployment planning (Phase 6) specifically requires it — deferred, not part of Phase 4's core scope.
5. RL (contextual bandit → full sequential RL, in that order if pursued at all) — highest cost, highest risk of the documented failure modes in Section 3, gated behind explicit evidence that 2-3 cannot provide adequate multi-step credit assignment. Not committed to.

---

## 18. Cheap-measurement-first research plan (the actual Phase 4 work, before any of Section 17's heavier steps)

1. **Mechanism-vs-null validation**: for each candidate mechanism in Section 6 (starting with GARCH-family volatility, Kalman-filtered trend, distributional skew/jump-detection — the three genuinely untested-in-Phase-3A families), run the same MI-vs-shuffled-null methodology Phase 3A already built and validated, on the training partition only. Cost: hours, reuses existing code pattern.
2. **Trajectory-vs-single-bar information test**: using the new `decision_id`-linked experience records, test whether a short sequence window of market-state vectors carries more information about eventual trade outcome than the single-bar snapshot Phase 3A tested — the one concrete "untested, not ruled out" gap flagged by Phase 3A. Cost: a few hours, no model training, MI/simple-model probe only.
3. **Combiner smoke test**: only if 1 or 2 shows real signal, a small stacked/gated combiner over the surviving mechanisms, evaluated with the same chronological-split-plus-null-control discipline as Phase 3A, not wired into any live/validation path.

If none of 1-2 shows signal beyond null, the honest conclusion is that GOLDEX's currently accessible data (M1, no tick, no options) does not support a learnable opportunity-capture edge at this stage, and Phase 4 should conclude with a STOP recommendation on further model investment — the same evidentiary discipline Phase 3A already demonstrated, not a new failure mode to be feared.

---

## 19. Leakage/causality controls

Every control already proven in Phase 1-3A carries forward unchanged: `market_state_builder.py`'s no-look-ahead construction, `write_tag_guard`'s cross-partition rejection, the causality guard in `phase3_tournament.py`'s `learn()` wrapper, chronological-only splits, shuffled-label null controls for every new statistical claim, and the untouched final OOS boundary. The new `decision_id` linking field (Section 10) must ship with its own no-look-ahead test in the same style as Phase 3A's `test_observation_features_no_lookahead.py` — this is a hard requirement, not optional.

---

## 20. Validation architecture

Unchanged from Phase 2/3: control gate, CI-based verdicts, chronological OOS split, real execution costs. Phase 4 adds trajectory-level experience assembly (Section 10) as a new, separate pre-training step that runs entirely within the training partition — it does not touch or redefine what "validation" means for final verdicts.

---

## 21. Failure modes

- **Multiple-testing inflation**: Section 17's ladder, if not tracked, risks the exact PBO problem external research flags — every mechanism/model tried must be logged (extending the existing SDD ledger discipline), and a "found something" result late in a long ladder should be treated with extra skepticism, not celebrated.
- **Reward misspecification**: if a learned reward (rather than the CI-verdict gate) is ever introduced for the combiner, it must be checked for reward-hacking pathways (e.g., a combiner that learns to trade only in artificially easy synthetic-replay conditions) before any promotion.
- **Calibration drift**: a confidence output, once shipped, must be periodically re-validated against realized frequency — an uncalibrated confidence number silently becoming stale is a known live-system risk (Section 3).
- **Premature interface commitment**: locking in the mandate's full product-output list (SL/TP/confidence/reasoning/status) before the decision engine has evidence behind it would constrain the architecture for no current benefit — explicitly deferred (Section 1).

---

## 22. What we explicitly will NOT build in Phase 4

- No hardcoded strategy or indicator-threshold rule as the final decision path (mandate Section 11, permanent).
- No Black-Scholes/options-pricing component (Section 16).
- No full RL implementation without prior evidence from Sections 17-18.
- No tick-data ingestion (Section 15).
- No live/demo integration (that's Phase 6).
- No pyramiding, hedging, or multi-position logic (mandate Section 12, permanent, matches Phase 1's existing single-position simulator).
- No "$100 to millions" objective anywhere in any reward or evaluation formulation.

---

## 23. Phase 5 handoff

Phase 5 (not detailed here) should be the validation phase for whatever survives Phase 4's cheap-measurement ladder: full chronological OOS testing of the combiner/model that made it through Sections 17-18, using Phase 2's untouched verdict machinery, before any demo integration is considered.

---

## 24. Phase 6 XM demo handoff

Phase 6 requires: the market-state layer computable identically from live MT5 data (Section 12's invariant), a monitoring/calibration-recheck process for any shipped confidence output (Section 21), and explicit position-sizing/risk limits derived from the account-awareness design (Section 11) — none of this is built in Phase 4, but Phase 4's architecture must not preclude it.

---

## 25. Eventual live architecture

Deferred beyond this document's scope — Phase 4 only needs to ensure the architecture in Section 5 doesn't have to be redesigned to eventually run live, which the source-agnostic `MarketState` contract (already existing in `contracts/market_state.py`) already supports.

---

## 26. Concrete implementation boundaries

Phase 4 implementation work (once this design is approved) is scoped to: (a) the `decision_id` linking addition to `ExperienceRecord` plus its no-look-ahead test, (b) the new trajectory-orchestration module above `run_replay` (additive, new file, no changes to `simulator/engine.py` or `research/phase2_tournament.py`), (c) 2-3 new mechanism scripts following Section 18 item 1's cheap-validation pattern, (d) the trajectory-vs-single-bar information test (Section 18 item 2). Nothing else. No combiner training, no new candidate roster entries, no promotion, until this scoped work produces evidence.

---

## 27. Success criteria

- The trajectory-vs-single-bar test (Section 18 item 2) produces a real, null-controlled, non-noise result in either direction — success is getting a clean honest answer, not a positive one.
- At least one of the three untested mechanism families (Section 18 item 1) is validated or ruled out with the same rigor as Phase 3A's representation work.
- No leakage, no validation-set contamination, no Phase 1/2 modification — verified by the same independent sonnet whole-branch review cadence used for Phase 3/3A.

---

## 28. Stop criteria

If Section 18's cheap experiments (items 1-2) show no signal beyond null — the same honest outcome Phase 3A already reached once — Phase 4 should stop at that point and report a clean STOP recommendation for further model/mechanism investment, rather than escalating to Section 17's heavier steps on hope. This mirrors Phase 3A's own demonstrated discipline.

---

## 29. Risks

- Spending further real compute/time only to reconfirm "no additional signal found" — mitigated by doing the cheapest possible tests first (Section 18) before any heavier step.
- Multiple-testing inflation across an increasingly long history of tried mechanisms (Phase 3: 7 candidates, Phase 3A: 4 representations × 2 models, Phase 4: 3+ new mechanisms) — mitigated by explicit tracking (Section 21) and treating any eventual positive result with heightened, not lowered, skepticism.
- Scope creep into RL or a full autonomous system before the underlying decision-loop architecture has any validated signal to act on — mitigated by the ladder's strict ordering (Section 17) and this document's explicit deferral of RL.

---

## 30. Final recommendation

**Proceed with Phase 4, scoped strictly to Section 26's implementation boundary and Section 18's cheap-measurement-first research plan.** Build the trajectory-orchestration and mechanism-validation infrastructure; run the two cheap experiments; report the result honestly, including if it is another clean negative. Do not build a combiner, a strategy, or any product-shaped output until that evidence exists. This is not a guarantee GOLDEX will find an edge — external research is explicit that most real edges at short horizons are thin and hard-won, and Phase 3/3A already show this specific data/representation combination is a hard case. The correct ambition for Phase 4 is not "find the edge" — it is "build the smallest architecture capable of finding one if it exists, and get an honest answer either way."
