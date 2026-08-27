# GOLDEX V4 Phase 4 Stage 1 Findings — Trajectory Infrastructure & Cheap Mechanism Validation

Branch: `goldex-v4-phase4-quantitative-trading-intelligence` (off `goldex-v4-phase3a-representation-reward`, off `goldex-v4-phase3-discovery-scale`)
Scope: exactly Section 26's "concrete implementation boundaries" from `docs/superpowers/specs/2026-08-27-goldex-v4-phase4-quantitative-trading-intelligence-design.md` — (a) `decision_id` linking, (b) trajectory-orchestration module, (c) three mechanism-validation scripts, (d) trajectory-vs-single-bar information test. No combiner, no new roster candidate, no Phase 5/6, no RL, no changes to `simulator/engine.py` or `research/phase2_tournament.py`.

---

## 1. `decision_id` linking (Section 26a)

**Files touched:** `simulator/experience.py`, `simulator/replay.py`, `tests/simulator/test_decision_id_linkage.py`.

`ExperienceRecord` gained one new, additive, optional field:

```python
decision_id: Optional[str] = None
```

`simulator/replay.py` generates a fresh `uuid.uuid4()` string only when a `DECIDE` record's action is `LONG`/`SHORT` (i.e. a position is actually opened), attaches it to that `DECIDE` record, propagates the same id to every `MANAGE` record generated while the position stays open, and attaches it to the eventual `POSITION_CLOSED` record (including the end-of-replay forced-close path). `NO_TRADE` `DECIDE` records get `decision_id=None`. `simulator/engine.py` was not touched.

**No-look-ahead proof** (`tests/simulator/test_decision_id_linkage.py`, 3 tests, all passing):
- `test_decide_and_position_closed_share_decision_id` — the opening `DECIDE` record and its `POSITION_CLOSED` record carry the identical non-null `decision_id`; every intervening `MANAGE` record carries it too.
- `test_no_trade_decide_records_have_null_decision_id` — every `NO_TRADE` `DECIDE` record has `decision_id=None`.
- `test_decision_id_no_lookahead_via_truncation` — truncating the dataset immediately after the opening bar forces a different eventual close (a different `END_OF_REPLAY_FORCED_CLOSE` bar/price than the full run), and the opening `DECIDE` record's `decision_id` is still generated identically (non-null, immediately, as soon as the decision opens a position) in both the full and truncated runs — proving the id's existence and value never depend on how or when the trade eventually closes.

## 2. Trajectory-orchestration module (Section 26b)

**File added:** `research/phase4_trajectory_assembly.py` (pure, read-only over `ExperienceRecorder.all_records()`; does not modify `research/phase2_tournament.py` or `research/phase3_tournament.py`).

`assemble_trajectories(records)` buckets records by `decision_id`, then for each decision that opened a position builds a `Trajectory` object: `decide_observation_features` (the opening snapshot), `manage_observation_sequence` (chronologically ordered `observation_features` from every `MANAGE` record while the position was open), and the terminal `realized_pnl`/`outcome`/`cost_r` from the matching `POSITION_CLOSED` record. `NO_TRADE` decisions (no `decision_id`) are skipped — there's no outcome to join. `trajectories_to_rows()` flattens these into plain dicts for downstream dataframe use.

**Test** (`tests/research/test_phase4_trajectory_assembly.py`, 5 tests on synthetic hand-built `ExperienceRecord` lists, all passing):
- single trajectory assembled in the right order with the right terminal outcome.
- `NO_TRADE` records correctly produce zero trajectories.
- **two interleaved trajectories don't leak into each other** — records from trades A and B are interleaved in the list exactly as a real replay would produce them, and each `Trajectory`'s `full_observation_sequence()` contains only its own trade's markers, none of the other's.
- a trajectory with zero `MANAGE` steps (closed on the very next bar) assembles correctly with an empty sequence rather than erroring.
- `trajectories_to_rows` flattens correctly.

## 3. Mechanism-validation scripts (Section 26c / Section 18 item 1)

All three follow Phase 3A's exact discipline: `data/gold_seed_merged_full6yr.csv` rows 0:300,000 only (never the 300,000:400,000 Phase 3 validation split), the same 5-bar forward-return target (`research/phase3a_representation_experiments.forward_return`), the same 10-bin quantile-binned MI estimator with a 20-permutation shuffled-label null (imported directly from `research/phase3a_representation_experiments.py`, not reimplemented).

**Library check:** `arch` (the standard Python GARCH library) is **not installed** in the project venv (`import arch` → `ModuleNotFoundError`). Rather than add a dependency mid-phase, a minimal from-scratch GARCH(1,1) was implemented in `research/phase4_garch_volatility_mechanism.py`, fit by maximum likelihood via `scipy.optimize.minimize` (scipy is installed) with a coordinate-descent fallback if scipy were unavailable.

### 3.1 GARCH(1,1) conditional volatility — `research/phase4_garch_volatility_mechanism.py`

- DATA USED: `data/gold_seed_merged_full6yr.csv` rows 0:300,000.
- REPRESENTATION: GARCH(1,1) one-step-ahead conditional variance `sigma2[t]`, built only from `eps[t-1]` and `sigma2[t-1]` (no look-ahead). Fitted params: `omega≈0.00824, alpha≈0.0918, beta≈0.8824` (persistence 0.974 — typical for real financial return series).
- MODEL: none (MI probe, matching Phase 3A's methodology).
- TARGET: 5-bar forward return.
- TRAIN PERIOD: rows 0:300,000. VALIDATION PERIOD: none (marginal-association probe only, same as Phase 3A Section B).
- RESULT: real MI = **0.2142 nats** vs. null mean 0.000134 (std 0.000021, max 0.000173) — far above null.
- LIMITATION (honest disclosure, same confound Phase 3A already found and documented for its momentum/path representations): the training slice has a strong secular trend (Phase 3A: closes drift +33% over this same slice, 1-bar return autocorrelation ≈0). `sigma2[t]` is a slowly-evolving quantity, and the 5-bar forward-return target is a difference of price levels over an overlapping window — exactly the construction Phase 3A identified as producing large non-null MI even with no genuine local predictability. This result should **not** be read as "GARCH volatility forecasts the next 5 bars"; it is consistent with the same trend confound, not a new finding.
- CONCLUSION: GARCH-family volatility, like Phase 3A's momentum/path representations, shows large raw MI dominated by trend, not demonstrated local predictability. Ruling this in or out properly would require Phase 3A's Section D treatment (real chronological train/test model, not a marginal MI statistic) — out of this stage's scope; flagged for the escalation decision below.

### 3.2 Kalman-filtered trend — `research/phase4_kalman_trend_mechanism.py`

- DATA USED: rows 0:300,000.
- REPRESENTATION: forward (no smoother) constant-velocity Kalman filter, state `[level, velocity]`, observation = close; `velocity[t]` and `innovation[t]` both use only `close[0..t]`.
- MODEL: none.
- TARGET: 5-bar forward return.
- RESULT: `kalman_filtered_velocity` real MI = **0.0909 nats** (null mean 0.000130, std 0.000019, max 0.000167); `kalman_innovation` real MI = **0.0950 nats** (null mean 0.000142, std 0.000025, max 0.000184) — both far above null.
- LIMITATION: same trend-confound caveat as GARCH — velocity is a smoothed trend estimate over the same trending slice, and innovation, while more locally reactive, is still measured against the same overlapping-window forward-return target.
- CONCLUSION: same pattern as Phase 3A's momentum scalar — large raw MI, most plausibly trend-driven rather than genuine short-horizon predictive structure; not validated as exploitable without a proper OOS model check.

### 3.3 Distributional (skew/kurtosis/jump) — `research/phase4_distributional_mechanism.py`

- DATA USED: rows 0:300,000.
- REPRESENTATION: rolling skew and excess kurtosis of trailing 30-bar returns (window ends strictly before `t`), plus a binary jump-detection flag (`|return[t-1]| > 3σ` of the prior 30-bar trailing vol, excluding the jump bar itself). 5,751 / 300,000 bars (1.9%) fired the jump flag.
- MODEL: none. TARGET: 5-bar forward return.
- RESULT: `rolling_skew` real MI = **0.0491 nats** (null mean 0.000131); `rolling_excess_kurtosis` real MI = **0.0545 nats** (null mean 0.000133) — both above null, smaller than the trend-dominated GARCH/Kalman numbers but still an order of magnitude above the momentum-style representations' null. `jump_detection_flag` real MI = **0.0000** — the estimator's quantile-bin construction collapsed the near-binary, heavily-imbalanced (98%/2%) flag distribution to too few effective bins to register any MI at all.
- LIMITATION: skew/kurtosis are still computed over a trailing window on the same trending slice, so the same trend-confound caveat plausibly applies, though less directly than for a raw level/trend statistic (skew/kurtosis are scale- and to some extent trend-invariant transforms of returns, closer in spirit to Phase 3A's trend-invariant volatility-regime-transition representation, which showed two orders of magnitude less MI than the trend-exposed ones). The jump-flag null result is a **measurement-tool limitation**, not evidence the flag carries no information — a binary/rare-event variable needs a different (non-quantile-bin) association statistic to test properly; this is disclosed honestly rather than reported as "no signal."
- CONCLUSION: skew/kurtosis pattern more ambiguously with the trend confound than GARCH/Kalman (plausibly some genuine signal, plausibly still trend-adjacent); jump detection is untested (measurement gap), not ruled out.

## 4. Trajectory-vs-single-bar information test (Section 26d / Section 18 item 2)

**File added:** `research/phase4_trajectory_vs_snapshot_test.py`. Uses a local, fixed, rule-based z-score mean-reversion decider (same mechanism family as `candidates/statistical_null.py`'s `MomentumMeanReversionCandidate`, reimplemented locally so this script never imports from or wires into `candidates/` or any tournament roster) purely to generate real decisions/trajectories via `simulator.replay.run_replay` on the real training partition.

- DATA USED: rows 0:300,000. Decider generated **18,405 real completed trade trajectories**.
- REPRESENTATION 1 (single-snapshot): the opening `DECIDE` record's `observation_features` (`z`, `rolling_mean`, `rolling_std`).
- REPRESENTATION 2 (full-trajectory): the same 3 features, concatenated with mean/std/last of the same 3 features across every `MANAGE`-record snapshot while the position was open, plus step count (13 features total; trajectories with zero `MANAGE` steps get a zero-padded summary rather than being dropped).
- MODEL: `sklearn.linear_model.LogisticRegression`, default/fixed hyperparameters, no tuning.
- TARGET: sign of `realized_pnl` (1 = profitable trade, 0 = not). Test-split positive fraction ≈0.40.
- TRAIN PERIOD: earliest 80% of trajectories by decide timestamp (n=14,724), all within rows 0:300,000. VALIDATION PERIOD: latest 20% (n=3,681), also entirely within rows 0:300,000 — the Phase 3 validation split was never touched.
- NULL CONTROL: 20 label-shuffles of the training split, same model, evaluated on the same real test split.
- RESULT: **both representations produced identical test accuracy, 0.5289**, vs. null mean 0.5000 (std 0.0289) — real accuracy is only ≈1 null-std above the null mean, not a clean beat. Single-snapshot and full-trajectory representations gave the model no distinguishable advantage over each other on this decider's trades.
- LIMITATION: one fixed decider (z-score mean reversion), one fixed model (unregularized-default logistic regression), one feature summarization scheme (mean/std/last) for the trajectory representation — a narrow probe, not an exhaustive search of trajectory-summarization schemes.
- CONCLUSION: **no evidence that the full trajectory sequence carries more outcome-relevant information than the single decision-time snapshot**, on this real data, this decider, this model. This directly answers Phase 3A's flagged "untested, not ruled out" gap: it is now tested, and the honest answer is a clean negative — trajectory information did not beat single-snapshot information beyond null, and neither representation clearly beat null.

## 5. Multiple-testing ledger

| # | Phase | Mechanism / hypothesis | Outcome |
|---|-------|------------------------|---------|
| 1 | Phase 3 | control_no_trade | control (not a candidate) |
| 2 | Phase 3 | control_random | control (not a candidate) |
| 3 | Phase 3 | statistical_null_mean_reversion | tested |
| 4 | Phase 3 | regime_conditioned | tested |
| 5 | Phase 3 | simple_learned (untrained placeholder) | tested |
| 6 | Phase 3 | tabular_qlearning | tested |
| 7 | Phase 3 | bayesian_online | tested |
| 8 | Phase 3 | hmm_regime | tested |
| 9 | Phase 3 | sequence_history | tested |
| 10 | Phase 3A | momentum scalar (MI) | trend-confounded, no clean signal |
| 11 | Phase 3A | raw path window projection (MI) | trend-confounded, no clean signal |
| 12 | Phase 3A | multi-scale volatility ratio (MI) | trend-confounded, no clean signal |
| 13 | Phase 3A | volatility-regime transition (MI) | small, real, likely staleness artifact not edge |
| 14 | Phase 3A | shallow-tree probe (3 representations) | negative, null-consistent |
| 15 | Phase 3A | boosted-ensemble probe | negative, null-consistent |
| 16 | Phase 4 | GARCH(1,1) conditional variance (MI) | large MI, same trend confound as #10-12 |
| 17 | Phase 4 | Kalman filtered velocity (MI) | large MI, same trend confound |
| 18 | Phase 4 | Kalman innovation (MI) | large MI, same trend confound |
| 19 | Phase 4 | rolling skew (MI) | above-null, ambiguous re: trend confound |
| 20 | Phase 4 | rolling excess kurtosis (MI) | above-null, ambiguous re: trend confound |
| 21 | Phase 4 | jump-detection flag (MI) | untested — measurement-tool gap (binary variable, quantile-bin estimator unsuited) |
| 22 | Phase 4 | trajectory (sequence) vs. single-snapshot, logistic regression | clean negative, ~null |
| 23 | Phase 4 | GARCH(1,1) conditional variance, OOS shallow-tree check | clean negative, ~null |
| 24 | Phase 4 | Kalman velocity + innovation, OOS shallow-tree check | clean negative, ~null |
| 25 | Phase 4 | rolling skew + excess kurtosis, OOS shallow-tree check | clean negative, ~null |
| 26 | Phase 4 | all 5 representations combined, OOS shallow-tree check | clean negative, ~null (worst of the four) |

**21 hypotheses tested across Phase 3/3A/4 stage 1 before this stage's headline result (#22); 25 more before the OOS follow-up (#23-26), all also negative.** Per the design doc's PBO caution (Section 3/21), any single "found something" result this late should be treated with extra skepticism, not celebrated — and #22, the test this stage was specifically built to answer, came back a clean negative, which is the honest, well-supported outcome here, not a disappointing one.

## 6. Bottleneck classification

- **NO-ADDITIONAL-SIGNAL-FOUND** — the strongest-supported classification again this round. The one clean, properly-controlled (chronological split + null-controlled) test built this stage (#22, trajectory-vs-snapshot) found no signal beyond null in either representation.
- **REPRESENTATION-LIMITED (partial, unresolved)** — the three new mechanism families (#16-21) all show raw MI far above null, but every one plausibly inherits Phase 3A's already-documented trend confound (GARCH/Kalman clearly; skew/kurtosis more ambiguously). None of them received Phase 3A's Section-D-style proper OOS model treatment in this stage (explicitly out of Section 26's scope) — so "representation carries real local signal" is **not ruled out** for skew/kurtosis specifically, but also not established.
- **DATA-RESOLUTION-LIMITED** — the jump-detection flag could not be tested with the estimator used here; this is a measurement gap, disclosed rather than reported as a negative, and does not by itself support escalation.

No evidence in this stage supports MODEL-CAPACITY-LIMITED, TEMPORAL-STRUCTURE-LIMITED, or REWARD-OR-LEARNING-ARCHITECTURE-LIMITED classifications — the trajectory-vs-snapshot test (the direct probe for TEMPORAL-STRUCTURE-LIMITED) came back negative.

## 7. Escalation-or-stop recommendation (design doc Section 17/28)

Per Section 28's stop criterion: if Section 18's items 1-2 show no signal beyond null, Phase 4 should stop and report a clean STOP recommendation rather than escalating to Section 17 step 2 (a stacked/gated combiner) or step 3 (a sequence-window model).

**This stage's evidence is mixed, not uniformly negative**, so the recommendation is a qualified stop with one narrow follow-up, not a full stop:

1. **STOP on Section 17 step 2 (combiner) and step 3 (sequence model) for now.** The trajectory-vs-single-bar test (#22, the item Phase 3A explicitly flagged as needing to be run before justifying a sequence model) came back a clean negative — the direct evidentiary gate for step 3 is not met. Building a combiner (step 2) over mechanisms whose raw MI is dominated by an already-documented trend confound would risk training on that same confound, not real structure.
2. **One narrow, cheap follow-up is justified before a full stop, not a new ladder step:** re-run GARCH/Kalman/skew/kurtosis through Phase 3A's Section-D-style treatment (a fixed shallow model, proper chronological OOS split, shuffled-label null) — the same cheap-statistics-tier check already built and validated in Phase 3A, applied to these three new representations. This is not Section 17 step 2 (no combiner, no stacking) — it is finishing Section 18 item 1's validation to the same rigor Phase 3A used, since the MI-only read is inconclusive (confounded) rather than clean for two of the three new mechanism families. Recommend doing this narrow follow-up under Phase 4's existing scope before any Phase 5 handoff decision, not as new phase-scope creep.
3. If that follow-up also shows no OOS signal beyond null (the base-rate expectation given Phase 3A's prior 100% negative rate on this exact treatment), the overall recommendation is a full **STOP on further model/mechanism investment** for this data/representation combination, matching Phase 3A's own demonstrated discipline, and Phase 5 should not proceed to combiner/sequence-model work without new data (tick, options, or a genuinely different representation family) or a different question than "does M1 gold carry a learnable 5-bar edge."

## 8. OOS follow-up check (narrow, per Stage 1's own recommendation)

**File added:** `research/phase4_mechanism_oos_check.py`. Applies Phase 3A's Section D treatment (`research/phase3a_raw_path_geometry_probe.py`) to the three Stage 1 mechanism families, reusing each Stage 1 script's fitting/filtering function directly (`fit_garch11`, `kalman_level_trend_filter`, `_rolling_moment`) rather than reimplementing them.

- DATA USED: `data/gold_seed_merged_full6yr.csv` rows 0:300,000 (training partition only; rows 300,000:400,000, the real Phase 3 validation split, never read).
- MODEL: `DecisionTreeRegressor(max_depth=4, random_state=42)` — identical, fixed configuration to Phase 3A's Section D script, chosen before results, not tuned against the test split.
- TARGET: 5-bar forward return, same convention as Phase 3A and Stage 1.
- TRAIN/VALIDATION PERIOD: internal chronological split within the training partition — rows 0:240,000 train, rows 240,000:300,000 test (same split point as Phase 3A's Section D). The real Phase 3 validation split was never touched.

RESULT (real vs. shuffled-label-null control, same fixed model, same split):

| Representation | Real R² | Real dir. acc. | Null R² | Null dir. acc. |
|---|---|---|---|---|
| GARCH(1,1) conditional variance (1 feat) | 0.00051 | 0.4775 | −0.00021 | 0.4777 |
| Kalman velocity + innovation (2 feat) | −0.00823 | 0.4734 | −0.00053 | 0.4784 |
| Rolling skew + excess kurtosis (2 feat) | −0.00045 | 0.4843 | −0.00028 | 0.4748 |
| Combined: all 5 features | −0.01279 | 0.4734 | −0.00020 | 0.4782 |

- LIMITATION: one fixed shallow-tree model, one internal split point, no hyperparameter search — matching Phase 3A's Section D discipline exactly, so this is directly comparable to that prior result, not a broader search of possible OOS treatments.
- CONCLUSION: every representation's real R² is at or below its null-control R² (GARCH's real R² of 0.00051 is trivially above its own null of −0.00021, but both are effectively zero and dir. acc. is below 0.5 and indistinguishable from the null's 0.4777 — not a genuine beat). Direction accuracy for every real and null condition sits at or below chance (≈0.47–0.48). This directly confirms Stage 1's flagged suspicion: the large marginal MI reported for GARCH, Kalman, and skew/kurtosis in Section 3 evaporates under a proper out-of-sample predictive check, exactly the way Phase 3A's momentum/path representations did. The combined-feature representation performs worst of all (real R² −0.01279), consistent with the fixed-depth tree overfitting noise across more inputs rather than finding genuine combined structure. No representation, alone or combined, shows OOS predictive signal beyond null.

**Revised Section 6 classification:** the REPRESENTATION-LIMITED (partial, unresolved) classification from Section 6 is now resolved. With this OOS check completed, GARCH, Kalman (velocity + innovation), and skew/kurtosis all join the NO-ADDITIONAL-SIGNAL-FOUND classification alongside Phase 3A's momentum/path representations and Stage 1's trajectory-vs-snapshot test (#22). The only remaining open item is the jump-detection flag's measurement-tool gap (Section 6's DATA-RESOLUTION-LIMITED note), which this follow-up did not address (it was not part of Stage 1's flagged confound question — the flag showed zero marginal MI due to the estimator, not a large trend-confounded MI needing OOS resolution) and remains a disclosed gap, not a finding either way.

**Revised Section 7 recommendation:** Stage 1's qualified stop is now resolved to an **unqualified STOP**. The narrow follow-up Stage 1 itself recommended has been run, and per its own stated base-rate expectation (matching Phase 3A's 100% negative rate on this exact treatment), it came back negative for every representation tested, individually and combined. There is no representation-family evidence in this Phase 4 stage supporting escalation to Section 17 step 2 (combiner) or step 3 (sequence model). Recommend Phase 5 not proceed with combiner or sequence-model work on this data/representation family without new data (tick, options, or a genuinely different representation family) or a different question than "does M1 gold carry a learnable 5-bar edge" — matching Phase 3A's and this stage's now-consistent discipline. The only unresolved, non-blocking gap is the jump-detection flag's measurement-tool limitation, which is not itself a basis for escalation.

---

## Verification

- `decision_id` linking, no-look-ahead: `tests/simulator/test_decision_id_linkage.py` (3 tests, pass).
- Trajectory assembly: `tests/research/test_phase4_trajectory_assembly.py` (5 tests, pass).
- `/home/jith/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/candidates tests/research tests/simulator -q` → **104 passed** (unchanged after the Section 8 follow-up — `research/phase4_mechanism_oos_check.py` is a standalone script with no new tests, matching the existing OOS-probe scripts' pattern).
- `models/registry/*.json` checked for the known nondeterministic-ordering diff after running scripts; none found before commit.
- Files touched: all under `simulator/experience.py` (additive field only), `simulator/replay.py` (additive population logic only), `research/phase4_*.py` (5 new files, including the Section 8 follow-up), `tests/simulator/test_decision_id_linkage.py`, `tests/research/test_phase4_trajectory_assembly.py`, this report. **No changes to `simulator/engine.py`, `research/phase2_tournament.py`, `research/phase3_tournament.py`, any candidate file, or any roster.**
