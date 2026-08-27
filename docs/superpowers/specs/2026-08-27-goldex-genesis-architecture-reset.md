# GOLDEX Genesis — Architecture Reset / Information & Agent Research

Research only. No code. No new phase. No model training. No new experiments beyond what's needed to answer an architectural question directly (none were needed — this document reasons entirely from the 26 results and infrastructure already established).

---

## 1. Executive conclusion

GOLDEX's infrastructure (simulator, cost model, trajectory recording) is sound and should be kept. GOLDEX's research *process* — pick a representation/model/target, test OOS, reject, repeat — produced 26 honest negative results and must stop, not because the results are wrong but because continuing it is a random walk, not a search. The correct reset is not a new model or a new agent architecture bolted onto the same untested assumptions (5-bar horizon, M1-only, single-instrument). It is: **identify which of the untested variables (horizon, target definition, information source) is actually responsible for 26/26 nulls before committing to any agent architecture**, because an agent architecture cannot manufacture information that isn't present in what it's given, and nothing tested so far isolated *which* missing ingredient — if any — is the blocker. The architecture recommendation in Section 32 is conditioned explicitly on this: build the cheapest possible thing that could distinguish the candidate explanations, then build the agent architecture that fits what's actually found, not the other way around.

## 2. What GOLDEX should actually become

A persistent sequential decision-maker, not a predictor: observe → maintain belief about current state → decide LONG/SHORT/NO_TRADE → manage a position → exit → learn from the complete trajectory → repeat. One position at a time, short-duration-capable but not forced to trade. This reframes the problem from "forecast a scalar" to "act well under uncertainty over time" — a control problem, which is what Sections 8-19 of this document treat it as.

## 3. What V3/V4 taught us

- A rigorous, review-caught, reproducible research discipline works: every negative result in this project was independently verified by a second reviewer, numbers were re-derived from scratch and matched, and a real bug (Phase 3's credit-assignment error) was caught and fixed before being allowed to corrupt a conclusion. This discipline is a genuine asset and should be preserved unconditionally.
- Single-bar snapshot → scalar prediction, at a fixed 5-bar horizon, on M1 XAUUSD, using representations ranging from simple rules to boosted ensembles to GARCH/Kalman/distributional statistics, and even a direct sequence-vs-snapshot trajectory test, produced zero surviving OOS signal — 26 for 26.
- The V3 architecture (Probability/EV Engine, specialist models) and V4 Phase 3/3A/4 converged on the same negative result through different routes, which is stronger evidence than either alone that the *common factor* (M1, single-instrument, short fixed horizon, supervised framing) is where the problem likely sits — not that any one model family was poorly chosen.

## 4. What V3/V4 got fundamentally wrong

Not the diligence — the framing. Every one of the 26 hypotheses was a variant of "predict a fixed-horizon return from a snapshot," even Phase 4's trajectory test (which compared two ways of predicting the same fixed outcome, not a genuinely different decision problem). Nobody varied the horizon itself. Nobody tested whether the *decision formulation* (act now / wait / exit) rather than the *prediction target* was the actual missing ingredient. This is a research-design gap, not a modeling gap, and it's exactly why Section 18's instruction ("prioritize finding the missing variable over adding model count") is the right correction.

## 5. What infrastructure survives (component classification)

| Component | Classification | Why |
|---|---|---|
| Phase 1 simulator (`simulator/engine.py`, `replay.py`) | **KEEP** | Independently proven no-look-ahead, realistic spread/slippage/liquidation, untouched through every phase's review. Nothing about the 26 negative results implicates the simulator itself. |
| Phase 2 candidate protocol + control gate + EV/cost machinery | **KEEP** | The control gate (random-baseline sanity check) and CI-based verdict logic are methodologically sound and independent of what gets fed into them. |
| Phase 3 `learn()` hook | **MODIFY** | The interface concept (optional learning hook) is fine; its single-pass, train-once/learn-once/validate-once orchestration (Section 20) is too thin for anything beyond the simplest online update and needs the extension Phase 4 already began (trajectory assembly). |
| Phase 3A `observation_features` recording | **KEEP** | Additive, look-ahead-proven, and is the exact mechanism any future agent needs to reconstruct "what did the agent know." |
| Phase 4 `decision_id` / trajectory-assembly infrastructure | **KEEP** | Directly implements the sequential/trajectory framing Section 2 requires; already look-ahead-proven and independently reviewed. |
| V3 supervised prediction architecture (Probability/EV Engine, specialist models) | **RETIRE** | Same framing problem as V4 Phase 3/3A (fixed-horizon scalar prediction), already superseded by V4's more rigorous version of the same test, with the same negative outcome. Nothing in V3 answers a question V4 hasn't already answered more carefully. |
| Existing market representation (engineered scalars: momentum, vol, regime bins) | **MODIFY** | Individually null-tested repeatedly; keep as one *input family* to a future gate (cheap, already built, already understood) but stop treating any single one as a candidate prediction target on its own. |
| Existing execution/account model (`AccountState`, spread/slippage config) | **KEEP** | Realistic, already used correctly by every phase; the account-awareness separation principle (Section 13) builds on it directly, not around it. |
| Existing leakage controls (`write_tag_guard`, no-look-ahead tests, chronological splits) | **KEEP, non-negotiable** | This is the one thing that must never be relaxed regardless of what architecture comes next. |
| Existing validation methodology (shuffled-label null, chronological OOS, untouched final split) | **KEEP** | Directly responsible for making 26 negative results *trustworthy* rather than just discouraging; the discipline itself is the asset, independent of the results it has produced so far. |

**UNKNOWN** (genuinely undetermined, not yet testable from existing infrastructure or results): whether M1 resolution itself is a binding constraint (Section 10); whether a materially different horizon changes anything (untested, Section 4); whether Gold specifically (vs. a more liquid/microstructure-rich instrument) is a fundamentally harder search space. These are the actual open questions this reset should resolve next, not architecture choice.

## 6. What must be discarded

The *process*, not the code: stop testing "one more mechanism at the same 5-bar horizon, same M1 resolution, same single-instrument scope" as if the next one might be different. That specific search has been run 26 times with no trend toward success. Also discard: any assumption that a more sophisticated agent architecture (RL, world models, deep sequence models) is inherently more likely to find something a simpler model missed — capacity was already varied (shallow tree → boosted ensemble) with no improvement, and per Section 18's instruction, escalating capacity again without first finding the missing variable would repeat the same mistake at a higher price.

## 7. What "seeing the market" technically means

A **persistent, causally-constructed belief state**, updated every bar, composed of: (a) the raw representation (price, spread, timestamp — exists), (b) whichever quantitative mechanisms have individually passed a cheap validation test (exists as a pattern, several already tested and null), (c) the agent's own recent action/outcome history (exists via `decision_id`/trajectory infrastructure), (d) account state, kept architecturally separate (Section 13). This is not new invention — it is Phase 4's design doc's Market State Layer — the open question is not "what shape should state have" but "does any information source we can build state from actually contain something learnable" (Section 8-10).

## 8. What information GOLDEX currently has

M1 OHLC, tick_volume (bar-aggregated, not true tick arrivals), a historical spread series (real, not synthetic), timestamps, and everything derivable from these (volatility at multiple scales, momentum, GARCH/Kalman-filtered state, distributional moments — all tested, all null so far as marginal predictors of a 5-bar return). No true tick/quote stream, no bid/ask beyond the aggregated spread column, no correlated instruments, no macro/event calendar, no options/volatility-surface data.

## 9. What information is missing

Genuinely missing, not yet tested in any form: true tick-level order arrival/bid-ask dynamics (structurally unrepresentable from M1 by construction); any cross-instrument context (DXY, real yields, other metals) as conditioning information; any explicit session/macro-event context; any horizon other than 5 bars. The last one is the cheapest to test and has never been tried — this is the single most important gap identified in this whole document (also Section 4/18).

## 10. Whether M1 is sufficient

**Not established, and should not be assumed either way.** Every one of the 26 tests used M1-derived features at M1 resolution; none isolated resolution as the experimental variable. It is equally consistent with the evidence that (a) M1 is fine and the horizon/target/formulation was wrong, or (b) M1 genuinely cannot represent the information a short-horizon edge would require, because sub-minute order-flow dynamics are structurally invisible in 1-minute bars regardless of how cleverly they're processed. **Distinguishing evidence**: if a horizon sweep (Section 30) shows real signal emerging at some horizon using only M1 data, that directly falsifies "M1 is the bottleneck" for that horizon. If no horizon shows anything, M1-insufficiency becomes a live hypothesis worth testing directly (which requires acquiring tick data specifically to test, not assuming it would help).

## 11. Quantitative knowledge architecture

Treated as an evidence-source menu feeding a future gate, never as strategies. Each candidate family answers one of: does it extract prediction information (momentum, GARCH), state information (Kalman, HMM/regime), decision information (optimal stopping, contextual bandits), or risk information (drawdown/tail-risk models)? None should be added to the architecture without first passing the same cheap MI-vs-null → OOS-check discipline already used successfully to correctly reject GARCH/Kalman/distributional features in Phase 4. Black-Scholes: no legitimate direct role for spot XAUUSD (no options data, constant-vol assumption is wrong); GARCH/stochastic-volatility remain the correct substitute for volatility information specifically and have already been tested (null, at the 5-bar horizon).

## 12. Strategy-library architecture

**Rejected as a standalone architecture** — a fixed library, whether voting (A) or weighted (B), was already effectively what Phase 3's 7-candidate roster was, and it failed. A vote among individually-null mechanisms is not evidence any combination is meaningful; "51/49 voting" is not intelligence, it's noise dressed as consensus, and the mandate is right to flag this. A quantitative-expert library only becomes a legitimate architecture component (Section 13, Architecture C/D) if paired with a *learned gate* that can also learn to distrust or ignore all of them — i.e., the library is an input space, not a decision mechanism on its own.

## 13. Agent architecture candidates

| # | Architecture | Core idea | Learning mechanism | Market-state rep. | Entry | Exit | SL/TP | Sizing | Uncertainty | Adaptability | Compute | Overfitting risk | Reward-hacking risk | Sim-to-real risk | Interpretability | Complexity | Research value now |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Fixed library + voting | Majority/average of fixed rules | None | None (stateless) | Vote | N/A | Fixed | Fixed | None | None | Trivial | Low | None | Low | High | Low | **Already tested in spirit, rejected** |
| B | Library + learned weighting | Static learned weights over fixed experts | Supervised, offline | None persistent | Weighted vote | N/A | Fixed | Fixed | Weak | Low (weights frozen) | Low | Moderate | Low | Low | Moderate | Low | Marginal improvement over A, still no state |
| C | Experts + learned gating (MoE) | Gate conditions expert weight on context | Supervised, gate + experts | Implicit (gate's conditioning features) | Gated combination | separate | separate | separate | Yes if calibrated | Yes, gate retrains | Low-moderate | Moderate (gating collapse is a documented risk) | Low | Low | Moderate (gate weights inspectable) | Moderate | **Best-supported next step, conditional on Sec. 30** |
| D | Experts + persistent learned market state | Adds a state layer feeding the gate | Supervised/self-supervised state + gate | Explicit, persistent | Context-conditioned | separate | separate | separate | Yes | Yes | Moderate | Higher (more parameters) | Low | Low | Lower | Higher | Plausible later, premature now |
| E | Experts + sequential decision agent (RL/bandit hybrid over experts) | Sequential policy consumes expert outputs as features | RL or bandit | Persistent (if RL) or none (if bandit) | Policy action | Policy or separate stopping-rule | Policy or separate | Policy or separate | Depends | Yes | Moderate-high | High (RL) / moderate (bandit) | Moderate-high (RL) | Moderate (RL, simulator exploitation) | Low | High | Conditional — bandit variant plausible for entry only, RL variant not yet justified |
| F | End-to-end learned agent (no expert library, raw/learned features only) | Agent learns representation and policy jointly | RL or large-scale self-supervised + RL | Fully learned | Policy | Policy | Policy | Policy | Depends | Yes | High | Very high (opaque, hardest to audit) | High | High | Very low | Very high | **Rejected for now** — least interpretable, most data/compute hungry, no evidence base earns this yet |
| G | Hybrid: gated experts (C) for entry + optimal stopping for exit + separate risk/sizing layer | Decomposed by sub-problem, not unified | Mixed (supervised gate, analytical/learned stopping rule, rule-based risk layer) | Persistent, shared across sub-problems | Gate | Stopping-rule | Safety-net + optionally learned | Rule-based, risk-budget driven | Calibrated per-component | Yes, per component | Low-moderate | Lower (each component individually auditable) | Low | Low | Highest of the learned options | Moderate | **Recommended — see Section 32** |

Ranking, evidence-based: **G > C > B > E(bandit variant) > A > D > E(RL variant) > F**. The ranking rewards decomposability and auditability given how many negative results already require careful interrogation, and penalizes architectures whose opacity would make a 27th null result (or worse, a false positive) hard to diagnose.

## 14. Entry architecture

The gated-expert combiner (Architecture C, as one component of G). Its job is exactly "is there a directional opportunity right now, and how confident." Explicitly not the same mechanism as exit (Section 15) — conflating them was never actually tried in V3/V4 (everything was single-shot prediction with no persistent position concept at all), so there's no evidence either way, but Section 13's decomposition argument (mixing two different mathematical problems into one model has real documented costs, none of the observed benefits) argues for keeping them separate until proven otherwise.

## 15. Exit architecture

Optimal stopping theory as the first hypothesis (mathematically legitimate, well-established for the "when to close a position" problem specifically) — with the explicit caveat that classical results assume mean-reverting dynamics, which Gold's own tested return autocorrelation (weak, near-zero at 1-bar) does not confirm. This needs its own cheap validation, not an assumption that the theory transfers. The simulator's existing SL/TP safety-net remains regardless of what the primary exit policy becomes (Section 16).

## 16. SL/TP architecture

Hybrid, decided by evidence rather than fiat: SL/TP remain a **mandatory safety net** (already correctly implemented in Phase 1's simulator, catches liquidation/runaway-loss scenarios no learned policy should be trusted to self-regulate against) regardless of whether the *primary* exit decision becomes dynamic (via the optimal-stopping component). This is not indecision — it's two different functions (safety-net vs. primary-decision) that happen to use similar-sounding mechanics; treating them as one hardcoded thing (Phase pre-V4's implicit assumption) versus one learned thing (a naive RL reading of the mandate) are both premature without the exit-specific research in Section 15.

## 17. Position sizing architecture

Deferred, explicitly, until entry and exit both have validated signal (Section 23/31) — sizing a decision that isn't known to be right optimizes decoration, not substance. When it is addressed: rule-based, risk-budget-driven (e.g., fixed fractional risk conditioned on account drawdown state), kept separate from the market-state signal per Section 13's separation principle, not learned jointly with the entry/exit decision.

## 18. Learning architecture

Mixed by component (Architecture G): the entry gate learns supervised, from trajectory-derived labels (Phase 4's infrastructure already supports this); the exit stopping-rule is initially analytical/statistical (fit from historical trajectories, not deep-learned); risk/sizing is rule-based, not learned at all initially. This deliberately avoids committing to "the learning architecture" as one thing — per Section 9's own instruction, the mandate should not assume RL, and this recommendation explicitly does not use RL for any component in the initial build.

## 19. Experience/credit-assignment architecture

This is flagged as the single hardest unsolved problem in the entire reset, harder than architecture choice. The trajectory infrastructure (Phase 4) records the raw ingredients; it does not yet solve: attributing a win/loss to the *entry* decision specifically (distinct from favorable/unfavorable path noise afterward); judging an early exit against the unobservable counterfactual "what if held longer"; learning from a NO_TRADE decision at all (there's no realized outcome to learn from, only a hypothetical one, which risks the same kind of false-confidence Section 20 warns about if handled carelessly). Recommendation: this deserves its own small, dedicated, code-free research pass — write down the credit-assignment rule precisely, review it for the same kind of subtle bug Phase 3 caught (crediting the wrong trade to the wrong state), before it's ever implemented, not after.

## 20. Historical simulation architecture

Phase 1's simulator remains authoritative and needs no redesign for the recommended architecture (G doesn't require anything beyond what `decide()`/`manage()` and the trajectory infrastructure already provide). What's missing for training an agent with any persistent memory across a full 6.7-year pass: an explicit repeated-exposure protocol so that training over the same historical path multiple times (needed for anything beyond a single-pass supervised fit) doesn't degrade into path memorization — this needs internal held-out sub-periods distinct from the real final OOS boundary, the same pattern Phase 3A/4 already used for their internal splits, generalized to cover repeated-pass training specifically.

## 21. Anti-overfitting architecture

Everything already proven (chronological-only splits, shuffled-label nulls on every statistical claim, untouched final OOS, explicit multiple-testing ledger now at 26+ entries) continues unconditionally. Upgrade: Combinatorial Purged Cross-Validation over simple walk-forward for any future agent-level validation (externally documented to produce a lower Probability of Backtest Overfitting than plain walk-forward). New requirement specific to a persistent agent: a repeated-exposure/memorization guard (Section 20), which single-shot models never needed.

## 22. Real-time architecture

Deferred to Phase 6+ in detail, with one non-negotiable design constraint carried forward now: whatever state/gate/stopping-rule architecture gets built must be computable identically from a live MT5 feed and from historical replay (the existing `MarketState.source` distinction already supports this — don't break it). No latency claim (millisecond or otherwise) should be made until actual MT5 API/broker latency is measured — that measurement does not exist yet anywhere in this project and is a concrete, cheap, honest thing to do before any live-architecture commitment, separate from this research pass.

## 23. Data requirements

Given Section 9/10's analysis: do not acquire new data reflexively. The one action clearly justified *before* any data acquisition decision is the horizon sweep (Section 30) — it is nearly free (reuses existing code/data) and directly informs whether "M1 is insufficient" is even a live hypothesis worth spending on. If the sweep shows nothing at any horizon, tick data acquisition becomes a genuinely justified next question (it addresses a structurally different information class M1 cannot represent by construction) — but should be scoped precisely (bid/ask tick sequence, specifically to test order-flow/microstructure mechanisms, not "get more data in general") rather than an open-ended acquisition.

## 24. Compute requirements

Architecture G (gated experts + analytical stopping rule + rule-based sizing) is cheap — comparable to the existing per-experiment cost pattern (hours, CPU-only), no GPU/large-model requirement. This is a deliberate feature of the recommendation, not a compromise: given 26/26 nulls, spending large compute on an expensive architecture (RL, end-to-end learned agent) before the cheap version has been tried would be the "add model count" mistake Section 18 explicitly warns against, independent of whether the compute is affordable.

## 25. Jane Street / professional-quant lessons that are actually transferable

Publicly, the transferable principles are: treat every hypothesis as falsifiable and log it regardless of outcome (already GOLDEX's practice, and the strongest asset from V3/V4 worth explicitly continuing); prefer several small, independently-understood signals combined systematically over one large opaque model (directly supports Architecture G's decomposition over F's end-to-end approach); take execution quality and cost-awareness as seriously as signal quality (already reflected in Phase 1/2's realistic cost modeling). What is **not** transferable: their scale, their proprietary infrastructure, their multi-asset/multi-market opportunity set, or any specific technique — their public material gives no basis to claim any particular architecture "is how Jane Street would do it," and that claim should never be used as justification for a design choice here.

## 26. Architecture comparison matrix

See Section 13 — the single source of truth for this comparison, evaluated across all the dimensions the mandate requested.

## 27. Strongest reasons each architecture could fail (required challenge section)

**Why the recommended architecture (G) could still fail while looking intelligent:**
1. **The gate could learn to fit noise that happens to correlate with the specific historical path**, producing a beautiful in-sample and even naive-OOS curve while carrying zero real generalizable signal — this is exactly what happened, in miniature, with the trend-confound results in Phase 3A/4 (large marginal statistics that evaporated under a proper OOS check). A gate over several such confounded mechanisms could combine multiple false signals into an even more convincing-looking false composite. **Safeguard**: every mechanism feeding the gate must independently pass the same OOS-check discipline before being added, and the gate itself must be re-validated with a fresh shuffled-label null after being trained, not just the individual inputs.
2. **High apparent trade frequency with a positive-looking average could mask a small number of large realistic-cost-sensitive losses** — a gate confidently trading often looks "intelligent" but transaction costs compound fast; this is exactly the failure mode external research (Section 3 of the Phase 4 design doc) flagged: real edges at short horizons are typically thin and cost-sensitive. **Safeguard**: the EV-cost gate (already built, Phase 2) must remain a hard mechanical filter, never soft-overridden by gate confidence alone.
3. **The credit-assignment problem (Section 19) being subtly wrong could make the gate appear to learn "what works" while actually learning an artifact of the labeling scheme** — Phase 3's actual bug is the concrete precedent. **Safeguard**: credit-assignment logic gets its own dedicated review pass and unit tests before any gate is trained on trajectory-derived labels, exactly as Phase 4's `decision_id` linkage was.
4. **Repeated exposure to the same 6.7 years could let any component with enough capacity memorize the specific historical path** rather than learning generalizable structure, producing perfect-looking backtest performance that means nothing OOS. **Safeguard**: Section 20's repeated-exposure protocol, held-out internal sub-periods, is mandatory before any multi-pass training, not optional.

## 28. What should NOT be built

A new model tournament at the same 5-bar horizon. A full RL agent (Architecture E-RL or F) without prior evidence from the cheap sweep. A world-model/model-based-RL system (no public precedent exists for trading, per external research; would be genuinely unexplored territory and premature). A unified end-to-end architecture that handles entry, exit, and sizing as one opaque model. Tick-data acquisition before the horizon sweep. Any interface commitment (final SL/TP/confidence/reasoning output format) before the underlying decision architecture has any validated signal.

## 29. What should be built (research-scoped, not implementation-scoped — for the next SDD cycle, not this document)

The horizon sweep (Section 30) first. If and only if it shows something: the smallest possible gated-expert combiner (Architecture C/G's entry component) over whichever mechanisms correlate at that horizon, evidence-gated exactly as Phase 3A/4 already do it. In parallel (cheap, independent): the exit-specific optimal-stopping cheap-test. Neither of these is authorized to start from this document alone — this document identifies what's justified, a future SDD design/plan cycle authorizes and scopes the actual work.

## 30. The minimum credible proof-of-concept

**A horizon sweep.** Reuse Phase 3A/4's exact MI-vs-shuffled-null estimator, unchanged, on the training partition only, varying only the forward-return target's horizon (e.g., 1, 5, 15, 30, 60, 120 bars) against the representations already computed (momentum, volatility, GARCH, Kalman, skew/kurtosis). This costs almost nothing (no new infrastructure, reuses every existing script's estimator function) and directly answers whether "5 bars specifically" was the wrong choice all along — the one variable never varied across 26 hypotheses. This is the smallest experiment that could change the entire trajectory of this reset, and should happen before any agent-architecture code is written.

## 31. What evidence must exist before scaling

1. At least one horizon/representation pair beats null OOS (not just marginal MI) — the current hard blocker, unmet.
2. A gate conditioned jointly on multiple mechanisms measurably outperforms the best single mechanism alone, OOS — a distinct, currently untested claim from "does mechanism X work alone."
3. The credit-assignment logic for trajectory-derived labels has been specified precisely and passed its own review, independent of any specific gate implementation.

None of these currently exist. Scaling past a small proof-of-concept without them would repeat the pattern this whole reset exists to stop.

## 32. Recommended architecture

**Architecture G: decomposed gated-experts-for-entry + optimal-stopping-for-exit + rule-based risk/sizing, sharing one persistent market-state layer — but only after the horizon sweep (Section 30) identifies which representation/horizon pair, if any, is worth building a gate around.** This is a conditional recommendation, not an unconditional architecture commitment: the mandate explicitly warns against choosing sophistication for its own sake, and given 26/26 nulls, the honest position is that *no* agent architecture is yet justified by evidence — G is the answer to "if and when evidence justifies building something, what shape should it take," not "build this now regardless."

## 33. Why the recommendation beats the alternatives

Versus A/B (fixed/weighted library): G's gate can learn to abstain and to condition on state, which a static vote or fixed weighting cannot — directly addressing why the strategy-zoo pattern already failed. Versus D (added persistent state layer) and F (fully learned): G stays maximally decomposable and auditable, which matters enormously right now given how many of the 26 results required careful confound-interrogation to interpret correctly — an opaque architecture would make the 27th result impossible to trust either way. Versus E-RL/full RL: G doesn't require the multi-step credit assignment RL exists to solve, and the one direct test of whether multi-step structure matters (Phase 4's trajectory-vs-snapshot) came back null — removing RL's strongest justification. Versus E-bandit alone: a bandit can handle the entry decision but structurally cannot represent the exit problem (no persistent state), which is why G pairs it (or the gated-MoE equivalent) with a *separate* exit mechanism rather than forcing one framework to do both.

## 34. What would falsify the recommendation

If the horizon sweep (Section 30) finds nothing at any horizon: this specifically falsifies "the missing variable was horizon/target definition," and the next honest question becomes about information source (tick data, cross-instrument context) or scope (a different instrument entirely), not about agent architecture — no version of G, or anything more sophisticated, fixes an absence of information. If a gate over multiple mechanisms performs *no better* than the single best mechanism found: this falsifies the specific claim that combining heterogeneous evidence sources adds value here, and argues for keeping whatever single mechanism works (if any) simple rather than building a gate around it. If the credit-assignment specification (Section 19) cannot be made precise and reviewable: this falsifies the trajectory-learning approach for the entry gate specifically, and argues for a simpler, non-trajectory-dependent supervised label instead.

## 35. Proposed roadmap

1. Horizon sweep (Section 30) — cheap, decisive for Section 34's first falsification test.
2. If signal found: smallest gated-expert combiner for entry, evidence-gated at every step, exit-specific optimal-stopping research in parallel.
3. Credit-assignment specification and review (Section 19), independent of and before any trajectory-based training.
4. If both entry and exit show validated signal: Phase-5-equivalent full untouched-OOS validation via CPCV.
5. Phase-6-equivalent: XM demo, with live-vs-simulator latency/spread/slippage reconciliation measured explicitly before any live-viability claim.
6. Live, real-money — only after demo performance matches a tolerance specified before the demo period starts.

If step 1 finds nothing, the roadmap's next real decision point is a data/scope conversation (Section 23), not an architecture conversation — and that should be stated to stakeholders plainly rather than quietly pivoting to a bigger model.
