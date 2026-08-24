# GOLEX V3 — Phase 5A: Specialist Side-Conditioning Design

Status: DESIGN ONLY — no code changes authorized by this document.
Origin: H15 full-history directional-skew investigation (2026-08-24) proved
an INTEGRATION BUG: Opportunity/Barrier/MAE/MFE each condition on a side
they invent themselves, not on Direction's side, while `decision/ev_engine.py`
consumes them as if they answered "given Direction's side, what happens."

## 1. Target definitions (exact conditional target per specialist)

Let `d ∈ {+1, -1}` be Direction's OOF-predicted side for event `t0`
(`+1` = long, `-1` = short; ties/near-50% are Direction's problem, not
downstream).

- **Direction** (unchanged): `P(touch == +1 | features at t0)` — direction-
  agnostic triple-barrier target, symmetric barriers. Not conditioned on
  anything downstream; it IS the side source.
- **Opportunity**: `P(TP hit before SL | features at t0, direction_side = d)` —
  triple-barrier run with `side = d` (Direction's side, not self-generated),
  asymmetric TP/SL meaningful. Timeout (vertical barrier hit, neither TP nor
  SL touched) is explicitly its own outcome, excluded from the numerator and
  counted in the denominator like SL — i.e. `label = 1` iff TP touched
  first, `label = 0` for both SL-first and timeout, matching
  `triple_barrier_labels`'s existing `touch == favorable` definition. Stated
  as `P(TP before SL | ...)`, not `P(win)`, to keep the semantics literally
  aligned with Barrier's TP/SL/timeout split below and with
  `decision/ev_formula.py`'s `p_tp`/`p_sl`/`p_timeout` inputs.
- **Barrier** (`p_tp`/`p_sl`/`p_timeout` split): identical target family to
  Opportunity — `P(outcome ∈ {TP, SL, timeout} | features at t0, side = d)`.
  Currently `_oof_for_barrier` is literally an alias of
  `_oof_for_opportunity`; that aliasing is fine to keep (same target, same
  side contract) as long as the shared side input is fixed.
- **MAE**: `q75(-worst_excursion_against_d / vol_at_t0 | features at t0, side = d)`.
- **MFE**: `q75(best_excursion_with_d / vol_at_t0 | features at t0, side = d)`.

All four downstream targets share one conditioning variable: Direction's
side `d`. None may compute their own `d`.

## 2. Two-stage vs. single-stage

**Current (two-stage, buggy):** each specialist fits a "primary" classifier
on the same raw touch target as Direction, thresholds its OOF prediction
into a side, feeds that self-generated side into `build_meta`, then fits a
second meta-stage model conditioned on that self-generated side. The primary
stage exists ONLY to manufacture a side — it is never registered, never
served, and (per Phase 4's own comments) was always understood as "an
internal input to the meta target, not itself a registered specialist."

**Proposed (single-stage):** delete the primary stage entirely. Feed
Direction's OOF side directly into `build_meta` (or its replacement) to
build each specialist's target, then fit ONE model per specialist, taking
`assumed_side = d` (and optionally `p_direction`) as an input feature.

Single-stage is strictly simpler, removes an entire class of models that
had no purpose beyond generating the side, and removes the exact bug
(mismatched side source) by construction — there is no second side to
diverge from Direction's.

## 3. Side source: Direction's OOF-predicted side, strictly causal

Use Direction's own purged walk-forward OOF prediction, thresholded at
`p_direction >= 0.5 → +1 else -1` (matching Direction's own live decision
rule in `ev_engine.py`: `direction_gate_ok = probability_long > probability_short`).

This is strictly causal already — Direction's OOF pipeline (`oof_run` +
`PurgedWalkForwardCV`) only ever uses information available before `t0`,
same embargo/purge discipline as every other specialist. No new leakage
surface is introduced by reusing it as an input to Opportunity/Barrier/MAE/MFE's
target construction, provided fold boundaries are shared (see §5).

## 4. Side representation: feature, not separate models per side

**Decision: Approach A — `assumed_side` (and `p_direction`) as input
features to one shared model per specialist**, not separate long/short
models.

Rationale:
- Training data for Opportunity/Barrier/MAE/MFE is already scarce relative
  to Direction (fewer CUSUM-filtered events survive to a candidate pool);
  splitting by side would roughly halve each half's data.
- A single model can share statistical strength across sides for any
  feature effect that is side-invariant (e.g. volatility regime, time of
  day), while `assumed_side` (interaction terms via tree splits) captures
  genuinely side-dependent effects.
- Passing `p_direction` (continuous OOF probability), not just `sign(p_direction)`,
  lets the meta-model learn how Direction's confidence modulates P(win)/
  excursion size — directly answering the earlier-approved amendment to
  investigate Direction's probability rather than discarding it.
- Separate-per-side models remain available as a documented fallback if the
  empirical comparison in §10 shows `assumed_side` is not being used
  effectively (e.g. near-zero feature importance, or the single model's
  calibration is materially worse split by side than the two-model
  alternative). Not built now; not required by the current evidence.

## 5. Training / OOF / calibration / replay mechanics

**Training (Phase 4 retrain):**
1. Run Direction's OOF exactly as today (`_oof_for_direction` / `phase4_direction.py`)
   to get `p_direction[t0]` for every event in the shared CUSUM event pool.
2. For each of Opportunity/Barrier/MAE/MFE: derive `d[t0] = sign(p_direction[t0] - 0.5)`
   (map 0 → coin-flip convention, document explicitly which way ties break —
   recommend `+1` on exact tie, since it cannot occur in practice with a
   continuous probability).
3. Call `build_meta(close, high, low, vol, t0_nz, ..., side=d)` — same
   function, new input. `build_meta`'s signature changes from taking
   `oof_pred` (0/1 classifier output) to taking `d` (already-signed side)
   directly, since there is no longer a primary-stage 0/1 prediction to
   convert.
4. Fit ONE model per specialist on `X_meta = features ∪ {assumed_side: d,
   p_direction}`, target = the specialist's meta-label from step 3, using
   the SAME `PurgedWalkForwardCV` fold boundaries as Direction's own OOF run
   (not independently re-split) — this guarantees `d[t0]` used to build a
   given fold's meta-labels was itself produced out-of-fold with respect to
   that fold, closing any possibility of Direction-side leakage into its own
   conditioned targets.
5. Delete the primary-stage fit entirely (no `prim = oof_run(...)` call, no
   `_oof_for_opportunity`-style two-stage plumbing).

**OOF scoring / calibration (Phase 5 `phase5_calibration.py`):** same
change — replace each specialist's own `prim["oof_pred"]`-derived side with
`d` taken from Direction's OOF run (which must be computed once per horizon
and threaded to all four calibration functions, not recomputed per
specialist — a shared upstream value, not four independent copies).

**Replay / live (`phase5_ev_engine.py`, `decision/ev_engine.py`):
single source of truth invariant, per your explicit requirement**:

> For every downstream specialist consumed by Phase 5, `assumed_side` MUST
> be the exact OOF (or live-inference) Direction side for that event. No
> downstream specialist may generate its own side.
>
> The same Direction model/version that produces the live proposed side
> supplies the side and probability to every downstream specialist.

Concretely: `ev_engine.evaluate()` computes `direction_out` first (already
does), then passes `direction_out.probability_long`/`probability_short`
(or the resulting signed side) as an explicit input to the Opportunity/
Barrier/MAE/MFE inference calls — not as an implicit assumption the trained
model just happens to have learned. This makes the contract enforceable in
code (a missing/None `direction_side` argument becomes a hard failure, not
a silent divergence) and matches how the two are already wired for the gate
logic (`direction_gate_ok`) today. `feature_schema_ids`/`specialist_model_ids`
lineage should record which Direction model version supplied the side, per
event, so a live Direction model swap can be traced through downstream
outputs. `Direction` model version pinning: replay/live must use the SAME
Direction model artifact end-to-end for a given decision — no mixing OOF
Direction (research) with a different live Direction checkpoint.

## 6. Causality / leakage audit

No new leakage introduced:
- Direction's `d` is itself OOF (never sees its own fold at fit time).
- Downstream specialists consuming `d` as a feature/target-input only ever
  consume the OOF `d` for events in folds where `d` was produced
  out-of-fold — enforced by sharing fold boundaries (§5 step 4).
- `build_meta`'s existing triple-barrier call is already causal (uses only
  `t0..t1` forward path data to label `t0`, standard triple-barrier
  semantics, unchanged).
- Live path: Direction inference for the live bar happens before
  Opportunity/Barrier/MAE/MFE inference in the same `evaluate()` call,
  same causality Direction already has live.

No new leakage risk is introduced by this design; it is strictly a
*correctness* fix (both trained on and served the same side signal), not a
leakage fix.

## 7. Phase 4 artifact classification — what's invalid vs. valid

**Origin, confirmed by direct grep:** `phase4_opportunity.py`,
`phase4_barrier.py`, `phase4_mae_quantile.py` all independently build their
own primary-stage side exactly as `phase5_calibration.py` does — this is a
**Phase-4-originated** design choice, not a Phase-5 regression.

- **Direction** (`phase4_direction.py`, all horizons): **VALID, unaffected.**
  Never used `build_meta`; its own training/economic-sanity check already
  uses its own predicted side consistently with itself.
- **Opportunity / Barrier / MAE / MFE persisted candidates** (all horizons,
  `models/registry/opportunity_v3_*`, `barrier_split_v3_*`, `mae_quantile_v3_*`,
  `mfe_quantile_v3_*`): **valid as standalone role validations, invalid as
  Phase-5-EV-engine inputs.** Their Phase 4 log-loss/calibration numbers
  correctly answer "how well does this specialist predict outcomes
  conditioned on a side it invented for itself" — a coherent, self-
  consistent question, and not fabricated or buggy in isolation. They are
  simply **not measuring what `ev_engine.py` currently assumes they
  measure** ("given Direction's side..."). Every registry entry for these
  four specialists must be re-trained under the single-stage/Direction-
  conditioned target before being re-used by Phase 5; existing entries stay
  on disk for audit/comparison but must not back live/replay decisions once
  Phase 5A ships.
- **All Phase 5 replay/backtest results to date** (correction-pass report,
  H15 skew investigation numbers): remain valid as diagnostics of the
  CURRENT (buggy) integration — they correctly describe today's system, and
  should NOT be treated as forecasts of retrained performance. The
  correction-pass PASS verdict stands for what it verified (the 4 targeted
  defects); it says nothing about post-Phase-5A performance.
- **EV formula / EV gate / cost model** (`decision/ev_formula.py`,
  `decision/ev_gate.py`, replay cost model): unaffected, no change proposed
  here, side-symmetric by design and correctly so.

## 8. Direction preserved as upstream side generator

No change to `phase4_direction.py`'s own training methodology, features, or
target. Direction continues to be trained and validated exactly as today;
Phase 5A only changes how its OUTPUT is consumed downstream.

## 9. A vs. A+probability empirical comparison (what to measure, not yet run)

Once single-stage models are retrained (Phase 5B/implementation, out of
scope for this document), compare two feature sets for each of
Opportunity/Barrier/MAE/MFE:
- **A**: `assumed_side` (signed ±1) only.
- **A+probability**: `assumed_side` + continuous `p_direction`.

Compare via: OOF log-loss / quantile pinball loss, feature importance of
`p_direction` (near-zero importance ⇒ drop it, no cost to keeping A alone),
and point-biserial correlation of each specialist's OOF prediction against
its own true label (must be materially positive and side-consistent, unlike
today's uncorrelated-with-Direction result). This comparison is empirical
and deferred to the implementation phase; this document only fixes the
methodology, not the outcome.

## 10. Validation criteria (must hold before Phase 5A is declared done)

1. For every OOF event, `assumed_side` fed to Opportunity/Barrier/MAE/MFE
   equals Direction's OOF side for that exact event (an automated equality
   check across all four specialists' training data, not spot-checked).
2. No specialist training or calibration script contains a `build_meta`
   call (or successor) fed anything other than Direction's side.
3. Point-biserial correlation of Opportunity's OOF `probability_take`
   against its own true win/lose label is materially positive (sanity
   floor, exact threshold TBD at implementation time — today's baseline is
   effectively uncorrelated/anti-correlated with Direction-conditioned
   outcomes).
4. Re-run the H15/H45/H90 full-history replay; the long/short trade-count
   skew either resolves or is shown, with the corrected conditioning in
   place, to be a genuine model preference — not an artifact of mismatched
   sides. Either outcome is acceptable; forcing symmetry is explicitly out
   of scope (per your original investigation mandate).
5. `ev_engine.py`'s live path raises/fails closed if `direction_side` is
   ever missing when calling a downstream specialist (enforces §5's
   single-source-of-truth invariant structurally, not just by convention).

## 11. Migration / replacement of invalid Phase 5 artifacts

1. Retrain Direction: no-op (already valid), but its OOF side array becomes
   a new shared artifact consumed by the other four retrains — persist it
   once per horizon (e.g. `research/artifacts/direction_oof_side_h{H}.parquet`)
   rather than recomputing per specialist.
2. Retrain Opportunity, Barrier, MAE, MFE (all horizons) under the
   single-stage/Direction-conditioned target. New registry entries, new
   version tags — do not overwrite existing `*_candidate_*.json` in place;
   existing entries are kept for audit per §7.
3. Re-run Phase 5 calibration (`phase5_calibration.py`) against the new
   registry entries.
4. Re-run full-history OOS replay (all 3 horizons) against the new
   calibrated outputs; this replaces the correction-pass and H15-skew
   replay numbers as the current-truth baseline going forward.
5. Update `decision/ev_engine.py` to thread `direction_side`/`p_direction`
   explicitly into the Opportunity/Barrier/MAE/MFE inference calls (the
   actual code change this design authorizes for a FUTURE implementation
   plan — not authorized to happen yet).
6. Registry status fields (`rejected`/`candidate`/`validated`) get
   re-evaluated fresh against the new artifacts; old status values for the
   four affected specialists should not be assumed to carry over.

## 12. Explicit non-goals

- Not restoring long-trade frequency. Not optimizing thresholds. Not
  changing the EV formula, cost model, or gate logic. Not touching
  Direction's own methodology. Not writing any code yet — this document is
  the design; a `superpowers:writing-plans` implementation plan is the next
  separate step, only after this design is approved.
