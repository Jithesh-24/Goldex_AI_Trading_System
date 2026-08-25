# GOLEX V3 — Phase 5, Batch 1: Diagnostic Foundation — Design

Status: DESIGN ONLY — no code until approved, no code until this doc is reviewed and `writing-plans` runs.

## 0. Purpose and framing

Phase 5A fixed the side-conditioning integration bug. The corrected full-history
replay shows h=15 at mean expected R ≈ +0.4396 vs mean realized R ≈ +0.0194 (a
~22x gap), a 24-long/107,611-short skew, and h=45/h=90 producing zero trades
(Opportunity rejected at both). Before researching any new model (Batch 2/3),
Batch 1 answers one factual question with measurement, not modeling:

> **Is the weakness coming from the market/labels, Direction, the downstream
> specialists (Opportunity/Barrier/MAE/MFE), calibration, or disagreement
> between specialists?**

This attribution is the deliverable. Batch 2 starts from Batch 1's factual
answer instead of guessing at the EV gap's cause.

## 1. Non-goals (explicit, per user instruction)

- No new models, no refitting anything. Every diagnostic reads from existing,
  already-corrected OOF infrastructure (`research.direction_side.compute_direction_oof`,
  `research.phase5_calibration._oof_for_opportunity/_oof_for_barrier/_oof_predicted_mae_mfe`,
  `research.phase5_ev_dataset.assemble_replay_dataset`) — it computes new
  *statistics from* their outputs, nothing more.
- No registry writes — this produces no model, so nothing belongs in
  `models/registry/`.
- No production code touched.
- No commitment to Batch 2/3/4's exact content — those are gated on this
  batch's findings, per the agreed batch structure.
- Full 6.7-year real data throughout (no `rows=` slicing) — these are the
  numbers that matter for the attribution question; a slice would risk
  answering a different, smaller question.

## 2. The six diagnostics (all three horizons: h=15/45/90)

Each is its own file under `research/phase5b_diagnostics/`, its own test,
independently reviewable — mirroring how Phase 5A itself was structured.

### D1 — Direction quality

For each horizon: `compute_direction_oof(max_holding=h)` (full history) →
`t0_nz, p_direction_raw, p_direction_cal, side, has_oof`. True label
`y = (touch[has_oof] == 1)` recomputed the same way `_oof_for_direction`
already does (identical triple-barrier call, `side=None`, symmetric barriers).
Report per horizon: point-biserial correlation (`scipy.stats.pointbiserialr`)
of `p_direction_cal[has_oof]` against `y`, plus the raw probability
distribution's mean/std/decile histogram. This is the point-biserial
computation the Phase 5A report explicitly flagged as deferred — this closes
that gap.

### D2 — Base-rate / directional skew audit (model-free)

For each horizon: `assemble_v3_dataset(max_holding=h)` (full history) →
`triple_barrier_labels(..., side=None)` → raw `touch` distribution
(`touch==1` / `touch==-1` / `touch==0`), overall and broken down by calendar
year (using the bar timestamps already available from `load_raw_m1`). This
measures whether skew is baked into the labels *before any model exists* —
directly answers whether h=15's 24-long/107,611-short replay skew could be
explained by the raw event population alone.

### D3 — Opportunity/Barrier OOF quality

For each horizon: `_oof_for_opportunity(h)` and `_oof_for_barrier(h)` (same
underlying function per the existing alias) over full history. Report:
point-biserial correlation of the OOF probability against the true meta-label,
win-rate vs. the existing 0.4887 baseline, and calibration slope/intercept
(the same Newton's-method logistic fit already used in `research/audit_edge.py`'s
calibration block — reused exactly, not reimplemented differently). Also
MAE/MFE: `_oof_predicted_mae_mfe(h)` → per-quantile coverage (global and,
new for this batch, **broken down by side** — the existing `v3b` registry
entries report global/per-vol-regime coverage from training time, not a
by-side breakdown, so this is genuinely new information, not a duplicate of
what's already in the registry).

### D4 — Cross-specialist consistency (corrected scope)

**Hard constraint, per explicit correction**: every comparison in this
diagnostic must use the *same event*, the *same Direction side*, the *same
horizon*, and the *same TP/SL definition* — recreating Phase 5A's exact bug
(comparing outputs implicitly keyed to different sides) is the one thing this
diagnostic must not do. This is enforced structurally, not by convention: all
three quantities below are pulled from the *same* `research.phase5_ev_dataset.assemble_replay_dataset(h)`
call, which already guarantees a single combined index (`t0_nz[combined]`)
and a single `side` array shared across every specialist for that event —
using anything other than this one dataset's arrays for this diagnostic is
out of scope.

From that one dataset, per event: `p_barrier_win` (Barrier's P(TP before SL)),
`p_opportunity` (Opportunity's P(take)), `mae_r`/`mfe_r` (the OOF-predicted
q75 excursions `decision.ev_cost.candidate_sl_tp` would set as SL/TP
verbatim — production sets `sl_r, tp_r = mae.q75, mfe.q75` directly, it does
not derive a separate distance). Because TP *is* `mfe_r` by construction,
comparing `mfe_r` against itself would be circular, not a real check —
avoided here. Two mechanical, non-subjective, non-circular contradiction
rules instead, each reported as a rate (fraction of events), not a
per-event narrative:

- `contradiction_barrier_vs_reward_risk = (p_barrier_win >= 0.6) & (mfe_r <= mae_r)`
  — Barrier says TP-before-SL is likely, but the independently-fit MAE/MFE
  quantile models say the typical adverse excursion is at least as large as
  the typical favorable one (an unfavorable reward-to-risk ratio at the
  exact SL/TP distances production would actually use) — Barrier and the
  excursion models disagree about how favorable this event's geometry is,
  without either being defined in terms of the other.
- `contradiction_opportunity_vs_barrier = (p_opportunity >= 0.5) & (p_barrier_win < 0.5)`
  — Opportunity and Barrier nominally target the same "TP before SL" event
  (per their own `target_definition` strings) but disagree about its
  direction.

### D5 — Reliability / calibration

Reliability curve (bin by predicted probability, compare to observed rate),
Brier score, Expected Calibration Error, and calibration slope/intercept —
computed: globally per horizon, by long/short side (using D4's shared `side`
array), and **in the traded subset only where a traded subset exists**
(`n_traded > 0`, i.e. h=15 only). h=45/h=90's traded-subset numbers are
reported as the literal string `"N/A (zero trades at this horizon)"`, never
as `0` or a silently-omitted key — a zero-trade horizon has no traded-subset
calibration to measure, and that absence must be visible in the report, not
inferred from a missing field.

### D6 — Long/short conditioning behavior

Using D1/D3's already-computed OOF probability arrays and D4's shared `side`
array: for each specialist, split its OOF probability distribution by
`side == +1` vs `side == -1` and report the point-biserial correlation and
distributional mean/std *separately per side*. This tests whether each
specialist's conditioning on Direction's side actually produces
side-dependent discriminative behavior (as opposed to, say, a model that
technically has `assumed_side` as a feature but never learned to use it
differently for longs vs. shorts) — without requiring new feature-importance
extraction machinery beyond what training already computed.

## 3. Stop-early rule (corrected)

A diagnostic result may be flagged in the running report as **decisive** —
e.g., if D2 shows h=15's raw label base rate is itself ~98% short, that's
strong evidence the skew is market/label-driven, not an artifact — but this
NEVER skips a remaining diagnostic. All six run, for all three horizons,
regardless of how early a decisive-looking result appears. "Decisive" is a
label attached to a finding when the report is assembled, not a control-flow
decision made mid-run.

## 4. Attribution framework (the batch's actual deliverable)

The final report maps D1-D6's results onto the five candidate explanations,
using explicit, stated decision rules — not an unstructured narrative:

| Evidence pattern | Points toward |
|---|---|
| D2's raw label base rate at a horizon closely matches that horizon's replay long/short skew | **Market/labels** — the skew predates any model |
| D1's Direction point-biserial correlation is near zero (or the wrong sign) at a horizon | **Direction** |
| D3's downstream point-biserial correlations are materially weaker than D1's Direction correlation at the same horizon, despite correct side-conditioning | **Downstream specialists** |
| D5's calibration slope/intercept deviates substantially from ideal (slope≈1, intercept≈0), especially in the traded subset vs. the global population | **Calibration** |
| D4's contradiction rates are high (a threshold to be set from the actual observed distribution, not assumed in advance — report the full rate, don't pre-commit to "high") | **Disagreement between specialists** |

These patterns are not mutually exclusive — the honest answer may name more
than one contributing factor, with evidence for each, or may be inconclusive
for a given horizon. The report states plainly which of the five explanations
the evidence supports, for each horizon independently (h=15/45/90 may have
different answers), and flags anywhere the evidence is ambiguous rather than
forcing a single verdict.

## 5. Artifacts

- `research/phase5b_diagnostics/d1_direction_quality.py`
- `research/phase5b_diagnostics/d2_base_rate_audit.py`
- `research/phase5b_diagnostics/d3_specialist_oof_quality.py`
- `research/phase5b_diagnostics/d4_cross_specialist_consistency.py`
- `research/phase5b_diagnostics/d5_calibration_reliability.py`
- `research/phase5b_diagnostics/d6_long_short_conditioning.py`
- `research/phase5b_diagnostics/run_all.py` — orchestrates D1-D6 in order over
  all three horizons, assembles one JSON report, applies the attribution
  framework from §4, and prints/saves the final Batch 1 report.
- Corresponding `tests/test_phase5b_d1..d6_*.py`, using the existing
  `rows=`-slice convention for fast tests; `run_all.py` itself is exercised
  against full history as a real (long-running) research run, not a test.
- One output report: `research/phase5b_diagnostics/output/batch1_report.json`
  plus a human-readable `batch1_report.md` — not a registry entry.

## 6. Non-goals restated

No Batch 2/3/4 work begins until this batch's report is delivered and you
decide whether/how to proceed, per the agreed checkpoint structure.
