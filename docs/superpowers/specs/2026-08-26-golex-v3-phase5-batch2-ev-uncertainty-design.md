# GOLEX V3 — Phase 5, Batch 2: EV + Uncertainty Root-Cause Investigation — Design

Status: DESIGN ONLY — no code until this doc is reviewed, no code until `writing-plans` runs.

## 0. Objective

Batch 1 established the factual starting point: h=15 is the only trading horizon,
raw base rates are balanced, Direction is weak-but-real, downstream specialists
discriminate better than Direction, Barrier's *global* calibration is good
(slope ~1.0), Barrier's *traded-subset* calibration collapses (slope -0.18),
and 35.8% of h=15 events show a Barrier-vs-MAE/MFE reward/risk contradiction.
None of these facts were assumed going in; they were measured.

Batch 2 does not assume which of these facts (or which unmeasured cause)
explains the ~+0.42R gap between `mean_expected_R ≈ +0.4396` and
`mean_realized_R ≈ +0.0194`. It measures. The central deliverable is an
additive, evidence-backed decomposition of that gap, plus a
KEEP/MODIFY/REJECT/NEEDS_MORE_EVIDENCE verdict per component.

## 1. Non-goals

No new predictive models unless the investigation itself proves an existing
component structurally cannot answer the question being asked of it. No
threshold optimization. No EV formula changes. No production changes. No
Phase 3/4 redesign. No promotion. No registry writes. Reuse cached
OOF/replay artifacts and Batch 1's infrastructure wherever the question is
already answered by them — do not recompute what Batch 1 already measured.

## 2. D7 — Barrier vs MAE/MFE contradiction: is it predictive?

Reuses D4's exact contradiction definition and D4's single
`assemble_replay_dataset(15)` call's arrays (same event, same side, same
horizon, same TP/SL definition — the Phase 5A conditioning contract is not
re-derived, it's inherited unchanged from D4).

For the contradicted (`p_barrier_win >= 0.6 & mfe_r <= mae_r`) vs.
non-contradicted populations:
1. Realized R (via `realized_r_for_direction`, already in
   `phase5_ev_dataset.py`) for both populations, with CI.
2. Win/loss/timeout `touch` distribution for both populations.
3. Direct test: is mean realized R materially different between the two
   populations (a two-sample comparison with CI on the difference)? This is
   the falsifiable "is contradiction itself predictive" test.
4. Contradiction rate broken down by: volatility tercile (reuse Batch 1's
   volatility-regime convention from `audit_edge.py`'s existing tercile
   logic, do not invent a new one), Direction side, Barrier-probability
   decile, and any other already-available quantitative state where it adds
   information (e.g. jump-detection flag, if cheaply available from the
   existing feature fabric without a new fit).
5. Whether contradiction concentrates in particular conditions (a
   descriptive cross-tab of #4's breakdowns against contradiction rate).
6. Whether *excluding* contradicted events from the traded population
   changes realized R — reported as an observation on the existing OOS
   population, explicitly NOT as a proposed live threshold (the 0.6/mae<=mfe
   cutoffs stay exactly as D4 defined them; this only asks "what would this
   population's realized R have looked like without these events," it does
   not tune anything).

No presumption that Barrier or MAE/MFE is the "correct" one — D7 reports
which one's signal more closely tracks realized outcomes in the
contradicted population specifically, as a measured fact.

## 3. D8 — Traced calibration collapse through the pipeline

Walk the h=15 population through every real gate in
`decision/ev_engine.py::evaluate()`'s actual control flow (not a
reconstructed approximation of it — reuse D5's existing per-event
`evaluate()` loop pattern, which already proved behaviorally equivalent to
production's replay path):

1. Full OOS population (`has_oof` true for all specialists — D4's
   `combined` mask).
2. After Direction side resolves (`direction_gate_ok`/`short_gate_ok` —
   trivial at this stage, everything has a side).
3. After the Opportunity fail-closed veto (`probability_take >=
   OPPORTUNITY_MIN_TAKE_PROBABILITY`).
4. After Barrier/MAE/MFE availability (`model_status in _OK`).
5. After the full EV gate (`decide()`'s final NO_TRADE/LONG/SHORT split) —
   the actual traded subset.

At each stage: n, Barrier calibration slope/intercept (reusing
`fit_calibration_slope_intercept` from Batch 1's `_stats_utils.py`), Brier
score, the probability distribution's mean/decile spread, realized outcome
rate, and expected-vs-realized R. This identifies the exact stage where
degradation *begins*, not just that the final stage is degraded.

**Counterfactual isolation**: for each gate individually, compute calibration
on "full population minus only this one gate's filtering" (holding all
other gates at their real, actual behavior) to test whether degradation is
attributable to one gate, or only appears from the *interaction* of
multiple gates — do not claim single-gate attribution if the effect only
appears in combination.

## 4. D9 — EV gap decomposition: Shapley component-swap

Implements exactly the approved methodology, with the user's precise
terminology requirements preserved verbatim (these are not stylistic —
they prevent a reader from mistaking an attribution tool for a live
prediction):

- **C1 — Probability**: `(p_tp, p_sl, p_timeout)` swapped for a
  **"hindsight outcome distribution"** (never called a "true probability")
  — 1 on the event's actual realized outcome, 0 elsewhere.
- **C2 — Payoff/TP geometry**: `tp_r` swapped for the event's realized
  favorable excursion, **Direction-side-conditioned** (the same side the EV
  decision actually used — reuses `phase5_ev_dataset.py`'s existing
  side-conditioned `realized_r_long`/`realized_r_short`-style excursion
  arrays, never an independently-generated or historically-winning side).
- **C3 — SL/MAE geometry**: `sl_r` swapped for the event's realized adverse
  excursion, same Direction-side-conditioning discipline as C2.
- **C4 — Cost**: `cost_r` swapped for zero, labeled explicitly as
  **"zero-cost counterfactual / cost drag"** — this measures cost's EV
  contribution, it does not validate the cost *model's* accuracy (a
  separate, unaddressed question, stated as such in the report).

For all 2⁴=16 subsets of {C1,C2,C3,C4}, compute mean EV (via `raw_ev` from
`decision/ev_formula.py`, unmodified) with exactly that subset's components
swapped to their counterfactual values. Shapley value per component =
average marginal contribution across all orderings (computed exactly, not
sampled — 16 subset evaluations is cheap, pure formula recomputation over
~106,561 events, no refitting). Report:
- Model estimate → counterfactual/hindsight EV → realized R, as the
  three-stage progression the user specified.
- Each Ci's Shapley contribution in R units.
- Verification that ΣC1..C4 equals the formula-level counterfactual gap
  (the Shapley efficiency property) — reported as a check, not assumed.

**Implementation note (self-review finding)**: `research/phase5_ev_dataset.py::assemble_replay_dataset` currently computes per-side `mae_long`/`mfe_long`/`mae_short`/`mfe_short` internally but only returns the touch-outcome-combined `realized_r_long`/`realized_r_short` — not the raw Direction-side-conditioned MAE/MFE excursions C2/C3 need separately. The plan must add two new keys to that function's return dict (`mae_dir`, `mfe_dir` — `np.where(side==1, mae_long, mae_short)` / same for mfe, using its own already-computed local arrays), a strictly additive change to a research file with no behavior change to any existing key, consistent with the precedent of that file's prior Phase 5A modifications.

**C5 — Selection-conditioned payoff difference** (never called "selection
bias," per instruction): fully-hindsight EV (all 4 components swapped) of
the traded population vs. of the full eligible OOS population. Reports
whether the EV gate selects into a genuinely better, worse, or merely
differently-distributed opportunity set — three distinct, stated
possibilities, not collapsed into one verdict.

**C6 — Conditional calibration effect**: strictly separate from D8. D8
answers *where* calibration collapses in the pipeline; C6 answers *how much*
the probability component (C1) changes if refit with a traded-subset-only
Platt calibration instead of the global one. Explicitly labeled as
descriptive evidence only — refitting on the traded subset and observing a
change is not itself proof of what causes the collapse (D8 investigates
cause; C6 only quantifies the magnitude a conditional refit would move).

**Residual**: `(0.4396 - 0.0194) - (ΣC1..C4 + C5)`, reported explicitly.
Given Shapley's efficiency property, C1-C4 alone should sum to the
formula-level gap; a nonzero residual after adding C5 is a genuine finding
(an effect the decomposition doesn't capture — e.g., a real distributional
shift between the OOF replay period and full history), not something to
force to zero.

## 5. D10 — Uncertainty

Bootstrap resampling of existing OOF fold predictions (no new fits — resample
row-wise within each `PurgedWalkForwardCV` fold's already-computed
predictions) to produce a per-event EV confidence interval. Test: do
high-uncertainty events (wide bootstrap EV interval) correlate with larger
`|expected_R - realized_R|`? Does uncertainty add information beyond what's
already in the specialist outputs (a check against redundancy — if
uncertainty is just recovering "low Barrier probability," it's not new
information)? Verdict is evidence-quantified, not assumed useful.

## 6. D11 — Cross-horizon comparison (cheap only)

Applies D7's contradiction-rate/predictiveness check and D9's population-level
EV-gap numbers (mean expected/hindsight/realized only — NOT the full
per-event Shapley loop, which is the expensive part) to h=45/h=90 using
their already-cached OOF artifacts. Purpose is comparison context, not
building a case for trading those horizons.

## 7. Methodological rules (carried verbatim as binding constraints)

Full-history real dataset; OOS/OOF discipline preserved; Direction side is
always the exact side used downstream, never independently regenerated;
hindsight values are attribution tools only, never fed into any tradable
prediction; statistical significance is distinguished from practical effect
size in every write-up; every statistic reports `n`; confidence intervals
used wherever practical (reusing Batch 1's `_stats_utils.py` helpers);
ambiguity is reported honestly, not resolved by assumption; component
interactions are never falsely attributed to a single component (this is
exactly why Shapley averaging is used for D9, and why D8 does the
counterfactual single-gate-holdout check); a residual is reported if D9's
extended attribution doesn't fully reconcile; no threshold optimization
anywhere in this batch.

## 8. Decision framework

Final report classifies each of: Direction, Opportunity, Barrier, MAE, MFE,
Calibration, current EV formula, cost model, contradiction handling,
uncertainty methodology, selection/gating mechanism — as KEEP / MODIFY /
REJECT / NEEDS_MORE_EVIDENCE, each with the specific evidence (D7-D11
result) that justifies the verdict, and explicit answers to the seven
numbered questions in the user's brief (why the gap exists, how much each
component explains, whether contradiction is predictive, where calibration
collapses, whether the EV architecture is sound, whether it's correctable
without new models, and — only if not — exactly what capability is
missing and why).

## 9. Artifacts

`research/phase5b_diagnostics/{d7_contradiction,d8_selection_calibration,
d9_ev_shapley,d10_uncertainty,d11_cross_horizon,run_batch2}.py`, each with
its own test under `tests/`, mirroring Batch 1's file-per-concern
convention exactly. Output: `research/phase5b_diagnostics/output/
batch2_report.json` + `.md`. No registry writes, no production changes.

## 10. Checkpoint

Per instruction: complete Batch 2, run a whole-branch review (same
discipline as Batch 1 — a Critical finding was caught there and would not
have been caught by task-level review alone), independently verify all
results, then stop. No automatic progression to Batch 3.
