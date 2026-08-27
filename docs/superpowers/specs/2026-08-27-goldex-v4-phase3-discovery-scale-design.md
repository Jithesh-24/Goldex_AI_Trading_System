# GOLDEX V4 — Phase 3 Design: Discovery-Scale Candidate Research

Status: design only, no code. Phase 1 (simulator) and Phase 2 (candidate competition harness) are approved and unmodified by this document. No production changes.

## 0. What Phase 3 is for

Phase 2 proved the harness works: five structurally different candidates can compete inside the same validated simulator, scored by an evidence profile instead of a composite score, with random/no-trade as a validity gate. Phase 3's job is to widen and deepen that discovery process — not to pick a winner from Phase 2's small starter roster and start optimizing it. Doing that would be exactly the V3-style mistake this whole pivot exists to avoid: a handful of manually chosen models tuned until the backtest looks good. Phase 3 instead expands *what kinds of intelligence get to compete* and *how much they're allowed to learn from experience*, while keeping every validation discipline from Phase 1/2 untouched.

## 1. Self-challenge against the 17 carried-forward principles

Going through each explicitly, because several of them are real constraints on what Phase 3 is allowed to build, not just tone:

1. **Phase 2 candidates as foundation, not final intelligence** — Section 2's roster expansion treats all five existing candidates as still-competing baselines, never privileged, exactly as Phase 2 did with V3.
2. **Not another V3-style manual optimization exercise** — Section 3 explicitly forbids hand-tuning any one candidate's hyperparameters against `SIMULATED_VALIDATION`; validation stays a one-shot honesty check per candidate version, not a tuning loop.
3. **Purpose is discovery through competition** — Section 2's roster additions are chosen to broaden mechanism diversity, not depth within one family.
4. **No pre-assumed correct architecture** — Section 4's research/comparison step precedes any commitment, same as the original V4 architecture research.
5. **Research which mechanism families actually fit the data/environment** — Section 4.
6. **Learn from full sequential/account/execution/regime context** — Section 5's `learn(experience)` hook gives candidates access to everything Phase 1's experience records already capture; nothing new needs inventing, it was already collected in Phase 2, just never consumed for retraining.
7. **Strict chronological causality preserved** — Section 6 makes this mechanically enforced, not a promise: a candidate's `learn()` call only ever receives `SIMULATED_TRAINING`-tagged experience, gated by the same partition-tag guard Phase 2 already built (`write_tag_guard`), extended to a read-side check.
8. **Controls remain mandatory** — Section 7 keeps Phase 2's control-gate unmodified and reused verbatim.
9/10. **No predetermined profitability number, no compounding target** — Section 8 reaffirms Phase 2's evidence-profile-only scoring; no such target appears anywhere in this design.
11. **Discover both when to trade and when not to** — the `NO_TRADE` action and the existing `NoTradeCandidate` control already make "not trading" a first-class, explicitly measurable choice, not an absence of one; Section 5's expanded candidates keep `NO_TRADE` as a real, reachable action for every mechanism family.
12. **One position at a time** — unchanged, enforced by Phase 1's simulator (not something Phase 3 code can violate; no candidate has access to open more than one position).
13. **Continuous market-flow loop** — Phase 1's `decide()`/`manage()` interface already implements OBSERVE→DECIDE→TRADE→MONITOR→EXIT; Phase 3 adds REASSESS via the `learn()` hook (Section 5) without changing this loop's shape.
14. **No fixed horizons or fixed R:R** — none of Section 2's new candidates have a fixed holding period or TP:SL ratio baked in; several (Section 2.3, 2.4) explicitly learn holding behavior from experience.
15. **V3 feature library not a ceiling** — Section 4 treats it as one candidate representation among several researched, same as Phase 2's Candidate C already did.
16. **Improvements must survive unseen chronological data** — Section 8 reuses Phase 2's `SIMULATED_TRAINING`→`SIMULATED_VALIDATION` degradation check unmodified; nothing in Phase 3 shortcuts this.
17. **Report honestly if 6.7 years isn't enough** — Section 9 states this as an explicit, anticipated, non-failure outcome of Phase 3, not something to be avoided by construction.

No principle was found to conflict with the architecture below; the closest tension was #2 vs. needing *some* concrete new candidates to research — resolved by keeping the new candidates as additional discovery attempts, evaluated by the same unmodified evidence-profile process as everything else, never singled out for tuning.

## 2. Roster expansion (research targets, not commitments)

Extends Phase 2's five-candidate roster with mechanism families deliberately chosen to test different hypotheses about what Phase 2's harness hasn't yet exercised — sequential learning, online adaptation, and principled probabilistic regime modeling:

- **2.1 Tabular online-learning candidate**: a small discretized-state Q-learning-style agent (state = a coarse discretization of a handful of existing features + current position status; actions = `NO_TRADE`/`LONG`/`SHORT` at decide-time, `HOLD`/`EXIT` at manage-time). Chosen because it's the simplest genuinely sequential learner that can be trained via repeated passes over `SIMULATED_TRAINING` alone, with a tiny enough state space to avoid the sample-inefficiency that ruled out full RL in the original V4 architecture research — this is the smallest RL-family method that can honestly be tried, not a compromise.
- **2.2 Bayesian online-updating candidate**: maintains a Beta-Bernoulli (or similar conjugate) belief over "does the current feature regime favor long/short/neither," updated incrementally as `SIMULATED_TRAINING` experience accumulates, trading only when posterior confidence clears an explicit, reported threshold. Tests whether principled uncertainty quantification beats a fixed-threshold rule (Phase 2's Candidate D) without any gradient-based learning at all.
- **2.3 Fitted regime model (HMM)**: an actual expectation-maximization-fit Gaussian HMM over realized-volatility/return features (fit once on `SIMULATED_TRAINING` before replay, never refit on `SIMULATED_VALIDATION`), replacing Phase 2's percentile-heuristic regime gate with a real generative regime model — the honest version of the classical-quant research track the V4 architecture document flagged as a candidate worth trying properly.
- **2.4 Sequence-window learned candidate**: a small logistic/linear model (same "no deep learning yet" constraint as Phase 2's Candidate C) but over a genuinely sequential feature — e.g. an exponentially-weighted running summary of the last N decide/manage cycles' own outcomes, not just raw price — testing whether learning from the candidate's *own trading history* (not just the market) adds anything, a direct, minimal instance of principle #6 ("learn from previous actions, trade outcomes").
- **2.5 Everything from Phase 2 carries forward unmodified**: `NoTradeCandidate`, `RandomCandidate`, `MomentumMeanReversionCandidate`, `RegimeConditionedCandidate`, `SimpleLearnedCandidate`, `V3BaselineCandidate` — all still compete, none retired without evidence.

Explicitly NOT added without further justification: any neural sequence model (LSTM/TCN/Transformer) or full policy-gradient/actor-critic RL — both were already identified in the original V4 architecture research as requiring either more data diversity (single 6.7-year history) or a more mature simulator-interaction budget than is worth spending before Section 2.1-2.4 have even been tried and evaluated. This is Section 4's research question to actually settle, not an assumption to carry in from that earlier document unexamined.

## 3. No optimization loop against validation

Every new candidate's parameters (Q-learning's discretization/learning rate, the Bayesian model's prior, the HMM's number of states, the sequence-window candidate's weights) are fixed BEFORE any `SIMULATED_VALIDATION` run, chosen either by a documented, simple heuristic or by fitting/training on `SIMULATED_TRAINING` alone. `SIMULATED_VALIDATION` is evaluated exactly once per candidate version, per Phase 2's existing verdict logic (`KEEP`/`REJECT`/`NEEDS_MORE_EVIDENCE`) — never used to pick a better hyperparameter and re-run. A candidate that wants to try a different configuration does so as a new `version`, and both versions are reported, not silently replaced.

## 4. Research step before any Phase 3 candidate is finalized

Before committing to Section 2's exact candidate list, actually investigate (this is real research work, output as a short findings note alongside the design, not assumed): whether the ~300-400K-bar training slices Phase 2's real run will use (or larger, up to the full 6.7 years) give the tabular Q-learner enough state-visitation to converge at all; whether the HMM's regime assignments are stable across re-fits on different training windows (a proxy for whether "regime" is a meaningful, non-noise concept in this specific dataset); whether the Bayesian candidate's posterior ever moves enough from its prior to matter given the data's known weak base rates (Batch 1's finding: Direction's r≈0.03-0.04). If any candidate's own research step shows it cannot possibly produce a meaningful signal from this data (e.g. a state space too sparse to visit twice), it is documented and demoted to `NEEDS_MORE_EVIDENCE`/dropped before a wasted full run, not run anyway for appearances.

## 5. Candidate protocol extension: an optional `learn` hook

Backward-compatible with Phase 2's `Candidate` protocol (`decide`/`manage` unchanged) — adds an optional third method:

```
learn(training_experience: list[dict]) -> None
```

Called at most once per candidate per tournament run, after a `SIMULATED_TRAINING` replay completes and before that same candidate's `SIMULATED_VALIDATION` replay begins, with the full list of that run's own `SIMULATED_TRAINING`-tagged experience records (from `ExperienceStore`, already-collected, nothing new to instrument). A candidate without a `learn` method (all of Phase 2's original five) is simply never called this way — no change to their behavior. A candidate that implements `learn` uses it to fit/update its internal state (Q-table, Bayesian posterior, HMM parameters, sequence-window weights) using only this training-tagged data.

## 6. Causality enforcement for `learn`

The orchestrator (extending `research/phase2_tournament.py`, not replacing it) passes to `learn()` only records whose `environment_tag` matches `EnvironmentTag.SIMULATED_TRAINING` — enforced the same way Phase 2's `write_tag_guard` already enforces tag consistency on write, extended here as a read-side assertion before the `learn` call. `SIMULATED_VALIDATION` experience is never passed to any candidate's `learn` method under any circumstance; the orchestrator asserts this at the call site, not merely by convention.

## 7. Controls unchanged

Phase 2's control-gate (random candidate must not show persistent post-cost profitability on validation, or the run halts and the harness is presumed buggy) is reused verbatim, now also applied against the expanded roster including the new sequential/learning candidates — a sequential learner's presence doesn't change what "the harness itself might be broken" looks like.

## 8. Scoring unchanged

Phase 2's evidence profile (realized P&L, drawdown, tail risk, cost/execution sensitivity, trade frequency, sub-period consistency, training→validation degradation, regime robustness, CIs) is reused unmodified for every Phase 3 candidate, including the new ones. No composite score is introduced. `KEEP`/`REJECT`/`NEEDS_MORE_EVIDENCE` verdict rules are unchanged.

## 9. The honest-null outcome is a designed-for result, not a failure mode

If, after Section 4's research step and a real run of the expanded roster, no candidate (including the new sequential/learning ones) clears the control-gate baseline with a validation CI lower bound above zero, Phase 3's deliverable is that finding, reported plainly — not a reason to keep adding candidates until one looks good by chance. A documented stopping rule: after the roster in Section 2 has been run once on real data with the process in Section 3-4 followed honestly, the result stands as Phase 3's answer for this data window; further candidate search becomes a new, separately-scoped research question, not an open-ended tuning loop against the same validation slice.

## 2b. Correction (2026-08-27 review): the roster is a starting ladder, not a ceiling

Section 2's five new candidates are discovery probes, not the complete search space. The question Phase 3 asks is not "which of these models did we pick" but "which classes of decision intelligence have sufficient justification to be experimentally tested." Concretely:

- **Progressive discovery ladder**: simple statistical → simple learned → sequential (2.1-2.2) → temporal/regime (2.3) → sequence-history (2.4) → ensemble/MoE → policy-learning/RL → more advanced architectures (sequence models, full RL, transformers). Phase 3 does not jump straight to the most complex end, but it also does not artificially stop at the simple end if Section 4's research finds evidence that richer temporal/state representation is actually needed.
- **Advanced mechanisms are not excluded by default cost**: a neural sequence model or full policy-gradient RL is not ruled out because it's expensive to build — it's ruled out (or admitted) by Section 4's research step actually checking whether the data, simulator, compute budget, and experience structure justify testing it. If that check comes back positive for a specific mechanism, it enters the Section 2 roster as a new candidate, tested by the exact same evidence-profile process as everything else — no preferential treatment for being sophisticated, no exclusion for being expensive without a specific, checked reason.
- **The real research question is market-flow representation, not "which classifier."** Section 4's investigation is explicitly not limited to whether V3's 125 features or Section 2's simple alternatives predict direction — it must check whether exploitable temporal/state information exists in: recent price path, sequential market state, volatility evolution, market microstructure (to the extent our OHLC+spread data supports it), regime transitions, tick/activity behavior where available, the candidate's own previous actions and trade outcomes, account state, execution conditions, and interactions between these — a genuinely open representation question, not a foregone conclusion that hand-engineered features are sufficient or insufficient.

## 5b. Correction: `learn()` is a foundation, not the final learning architecture

Phase 3 ships `learn()` (Section 5) as the first working instance of OBSERVE→ACT→EXPERIENCE→LEARN→MODIFY/UPDATE→ACT AGAIN, not as the permanent shape of GOLDEX's learning mechanism. Phase 3's research explicitly includes checking which forms of experience-based learning are actually useful for which candidate — a single training-pass batch update (as specified in Section 5) may suit the Bayesian/HMM/Q-table candidates but be the wrong shape for a future sequence or RL-family candidate that needs incremental per-episode updates; Phase 3 documents this as an open question for whichever later phase builds toward continuous online learning, rather than assuming `learn()`'s current one-shot-per-run design is final.

## 10. What Phase 3 does NOT do

Does not modify `simulator/` (Phase 1) or `research/phase2_tournament.py`'s core control-gate/verdict logic (Phase 2) beyond the additive `learn`-hook orchestration described in Section 5-6. Does not touch `SIMULATED_OOS_TEST`. Does not implement full RL/neural sequence models without the Section 4 research step justifying them first. Does not retrain or re-tune any candidate against `SIMULATED_VALIDATION` results. Does not set or optimize toward any profitability target. Does not modify production/live code.

---

Awaiting your review before writing the implementation plan.
