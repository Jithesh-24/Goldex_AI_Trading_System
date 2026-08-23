# GOLEX V3 — Phase 5: Probability / EV Engine — Design

Date: 2026-08-23
Status: approved for planning

## 1. Purpose

Phase 4 produced 7 independent specialist models (Direction, Opportunity,
Regime, MAE quantile, MFE quantile, Barrier probability, Execution/Decay),
each answering a different question about a candidate trading opportunity,
trained and validated independently with mixed OOS outcomes (see
`docs/ARCHITECTURE.md`'s Phase 4 section for full real numbers and
methodology limitations).

Phase 5 combines these specialist outputs into a single, mathematically
defensible, cost-aware Expected Value framework that answers: **does this
market state currently offer a trade worth taking, and if so, long or
short?**

Phase 5 does NOT: build virtual trade management, EOD learning, automatic
retraining/champion promotion, or a final system audit (later phases). It
does NOT modify the production decision/Telegram path — it runs as
research + shadow only, exactly like Phase 4's tick-capture opt-in.

## 2. Non-goals / explicit exclusions

- No `P(win) > threshold -> BUY` heuristic.
- No averaging of specialist probabilities into a fake "confidence score."
- No wiring into `app/engine.py`'s production decision path or Telegram.
- No dynamic/automatic model routing based on recent performance (registry-driven only, per spec §31 of the phase brief).
- No replacement of the current production SL/TP — Phase 5's candidate SL/TP is a research artifact only.
- No full re-derivation of joint specialist relationships beyond what Phase 4's independently-trained models actually support (Phase 4 models were NOT jointly trained; any "conditional" relationship used here is an approximation, documented as such, not a rigorously re-derived joint model).

## 3. Specialist status handling (cross-cutting)

Every specialist output value that participates in the EV calculation must
carry `model_status` from `{VALIDATED, CANDIDATE, DATA_LIMITED,
UNAVAILABLE, STALE, INVALID}`. The EV engine must never convert
`DATA_LIMITED`/`UNAVAILABLE`/`INVALID`/`STALE` into a numeric probability
or payoff value (no substituting 0, 0.5, or any other fabricated number).
If a required specialist for the requested horizon carries one of these
statuses, that side/horizon's decision is forced to `NO_TRADE` with
`decision_reason` naming the missing specialist.

Given Phase 4's real validation outcomes: h15 has every required
specialist `VALIDATED` (Direction, Opportunity, Barrier, MAE, MFE); h45 has
Direction `VALIDATED` but Opportunity `rejected` (treated as
`UNAVAILABLE` for gating — its win-rate gate failed OOS); h90 has
Direction `rejected` and Opportunity `rejected` (both `UNAVAILABLE`).
Execution/Decay is `DATA_LIMITED` at all horizons and is recorded in
lineage but never gates or scales EV in v1. The engine is wired generically
across all 3 horizons; it will naturally produce `NO_TRADE` at h90 and a
degraded (Opportunity-less) decision path at h45, driven purely by status,
not by hardcoded horizon exceptions.

## 4. Specialist output contracts

New module: `contracts/specialist_output.py`. Pydantic models, one per
role, each with `model_status`, `model_id`, `horizon` (where applicable),
and `calibrated: bool` (where applicable):

```
DirectionOutput: probability_long, probability_short, calibrated, model_status, model_id, horizon
OpportunityOutput: probability_take, calibrated, model_status, model_id, horizon
RegimeOutput: regime_state, regime_probabilities (optional), model_status, model_id
MAEOutput: q50, q75, q90, model_status, model_id, horizon   # q95 omitted: Phase 4 did not produce it
MFEOutput: q50, q75, q90, model_status, model_id, horizon
BarrierOutput: p_tp, p_sl, p_timeout, calibrated, model_status, model_id, horizon
ExecutionOutput: drift_60s, drift_120s, model_status, data_limited, model_id
```

All fields except `model_status`/`model_id`/`horizon` are `Optional`, so a
non-`VALIDATED`/`CANDIDATE` status can omit misleading numeric values
entirely rather than populate them with placeholders.

## 5. Calibration

Direction/Opportunity/Barrier probabilities are calibrated using
`decision.calibration.PlattCalibrator` (already exists, already
OOF-fit-only per Phase 4 precedent — Task 9 of Phase 4 already fits this
exact way for Opportunity). Phase 5 adds one calibrator per
role-per-horizon, fit on the OOF predictions Phase 4's `oof_run` calls
already produced (no new data collection, no re-touching the held-out
final evaluation window). Calibration artifacts are versioned JSON under
`models/calibration/<role>_<horizon>_platt.json`, loaded via a new
`decision.calibration_registry.CalibrationRegistry` (mirrors
`decision.router.ModelRouter`'s static, config-driven lookup pattern — no
live recalibration, no champion/challenger).

Regime has no probability to calibrate (state assignment only — its
`regime_probabilities` field, if the HMM exposes posteriors, is reported
raw and labeled uncalibrated). MAE/MFE quantiles are calibrated by
construction (trained via pinball/quantile loss) — no extra calibration
step.

Each calibration artifact records: method (`platt`), dataset size,
calibration period (OOF fold range), and is treated as `STALE` if its
underlying specialist's registry entry has been superseded.

## 6. Probability relationships — avoiding double-counting

Direction and Barrier both encode directional-outcome information and
were trained independently (not jointly), so multiplying their
probabilities together is not mathematically justified. Chosen approach:
**Barrier-primary, Direction-investigated.**

- Barrier's `P(p_tp)/P(p_sl)/P(p_timeout)` (already sums to ≈1 by
  construction, per Phase 4 Task 8's design) is the payoff-distribution
  driver: it directly represents the conditional-outcome probabilities
  needed for EV.
- Direction's calibrated probability always picks which side (long/short)
  to evaluate Barrier's outcome distribution against. Whether it ALSO
  carries independent information usable in the EV sum (beyond
  side-selection) is not assumed either way — it must be investigated
  empirically during implementation, not decided here. Implementation
  step: measure, per horizon, the OOS conditional relationship between
  Direction's calibrated probability and Barrier's realized `p_tp` (e.g.
  does `p_tp` vary systematically across Direction-probability deciles,
  holding side fixed?). If Direction carries information Barrier does not
  already capture (a real, measured conditional dependence), define and
  document a principled correction term (e.g. a small multiplicative or
  additive adjustment fit and validated OOS, not curve-fit to the final
  eval set) before folding it into `EV_side`. If the two are found
  redundant (Direction adds no measurable information once Barrier is
  known), Direction is used for side-selection only, exactly as
  originally proposed, and this finding is documented as evidence-based,
  not assumed.
- Opportunity's `probability_take` (interpreted as `P(take | direction)`
  conceptually, though Phase 4 did not train it as an explicit conditional
  model) is applied as a gating multiplier on the decision, not on the raw
  EV number: if Opportunity is `VALIDATED`/`CANDIDATE` and its calibrated
  `probability_take` falls under a documented minimum, the side is
  rejected regardless of EV. Where Opportunity is `UNAVAILABLE` (h45/h90
  per §3), this gate is skipped and disclosed in `decision_reason`.
- The Direction/Barrier relationship is explicitly an area of approximation
  pending the investigation above — Phase 4's specialists were
  independently trained, so no rigorously re-derived joint model exists a
  priori. Whatever relationship is found (redundant, or a documented
  correction term) is recorded as the resolution, not left implicit.

## 7. MAE/MFE → candidate SL/TP

Chosen approach: **quantile-as-barrier + Barrier-role probability.**
Candidate SL distance = MAE q75 (conservative, not q50); candidate TP
distance = MFE q75. The probability of reaching each candidate level is
read directly from the Barrier role's `p_tp`/`p_sl`/`p_timeout` at the
matching horizon (see §7a for how the SL/timeout split is actually
obtained) — a second probability model is NOT derived from the quantile
curve itself (interpolating a CDF from 3 quantile points is statistically
weak and would silently introduce a second, uncross-checked probability
estimate for the same event). q90 is exposed in the contract for optional
conservative-tail sensitivity analysis (§12) but is not the default
candidate level.

### 7a. Barrier role only produces binary P(win) — real fix required

Investigation during planning found Phase 4's actual Barrier role
(`research/phase4_barrier.py`) trains on `triple_barrier_labels(...,
side=...)`'s binary label (1 = TP-before-SL win, 0 = SL-hit-or-timeout
collapsed together) — this is the same target definition as the
Opportunity role, not a 3-way P(PT)/P(SL)/P(timeout) split. The
EV formula (§9) needs the SL-vs-timeout split. Resolution (**derive
2-way split from raw labels**, chosen over collapsing the formula to
`p_win`-only, to preserve §9's SL/timeout distinction rather than
discard it):

- No core-function change needed: `triple_barrier_labels()` already
  returns a `touch` column (raw -1/0/1: which barrier was actually hit,
  before collapsing to the binary `label`) — `research/phase4_barrier.py`
  computes this via `build_meta()`'s internal call but currently discards
  it after extracting the binary `label`. Phase 5 reads `meta_labels["touch"]`
  directly: `favorable = where(side>=0, 1, -1)`; `sl_hit = (touch ==
  -favorable)`; `timeout_hit = (touch == 0)` — the same 3-way partition
  `label` was collapsed from, at zero extra computation cost.
- Phase 5 trains one additional lightweight classifier — `P(sl | not-win)`
  — restricted to the `not-win` (label=0) subset, using the same
  purged+embargoed OOF methodology as every other Phase 4/5 model. This
  yields `p_sl = p_not_win * P(sl | not-win)` and `p_timeout = p_not_win *
  (1 - P(sl | not-win))`, so `p_tp + p_sl + p_timeout` sums to 1 by
  construction (`p_tp` = Barrier role's existing calibrated `p_win`).
- This new classifier and its registry entry follow the same
  `ModelRegistryEntry` pattern as every Phase 4 role (family:
  `barrier_probability`, a distinct `model_id` suffix, `candidate`/
  `validated`/`rejected` status based on its own OOS log loss vs a
  50/50-prior baseline).

`candidate_sl`/`candidate_tp` are research-only fields — they do NOT
replace `config/models.yaml`'s production SL/TP or any live trading
parameter.

## 8. Cost model

Round-trip transaction cost is read from Phase 2's live bid/ask
`MarketState` at decision time: `cost_R = (current_spread * 2) /
candidate_sl_distance` (round-trip spread expressed in R-multiples of the
candidate SL distance). This is `KNOWN COST`. Slippage is NOT modeled in
v1 (no legitimate data source for it exists yet) — the engine explicitly
labels its cost estimate `known_cost_only: true` in `EVDecision` and does
not fabricate a slippage constant. If spread data is stale or missing,
`model_status` for the cost component is `UNAVAILABLE` and the decision is
forced `NO_TRADE`.

## 9. EV formula

For a given side (long/short) and horizon:

```
EV_side = p_tp * TP_R - p_sl * SL_R - p_timeout * timeout_R - cost_R
```

- `p_tp`, `p_sl`, `p_timeout`: Barrier role's calibrated probabilities for that side/horizon.
- `TP_R` = MFE q75 (candidate TP distance in R-multiples).
- `SL_R` = MAE q75 (candidate SL distance in R-multiples).
- `timeout_R`: provisional proxy = 0.5 * (MFE q50 - MAE q50), used ONLY
  until the direct estimate below is built. Implementation step: using the
  OOF event sets Phase 4's `oof_run` already produced (per horizon), filter
  to events whose triple-barrier label is `0`/timeout, and compute the
  actual realized R at timeout directly from those OOF outcomes (mean, and
  its own quantiles for uncertainty). If this direct OOF-derived estimate
  is available with adequate sample size, it REPLACES the midpoint proxy
  as `timeout_R`; if sample size is inadequate at some horizon, the
  midpoint proxy is kept for that horizon only, explicitly labeled
  `provisional_proxy: true` in `EVDecision`'s lineage, not silently used as
  if final.
- `cost_R`: from §8.

Risk adjustment (**lower-confidence EV bound**, chosen over full
expected-utility/variance modeling — simpler, explainable, matches the
"no false precision" principle and avoids overfitting a utility function
to Phase 4's still-thin evidence):

```
EV_adj = EV_side - k * uncertainty
```

`uncertainty` is a documented, bounded [0,1] score derived from: (a)
fraction of contributing specialists at `CANDIDATE` vs `VALIDATED` status,
(b) calibration sample size relative to a reference size, (c) whether
Opportunity's gate was skipped (§6) at this horizon. `k` is NOT set by
guess or intuition — it must be justified and validated during
implementation: derive candidate `k` values from a principled anchor (e.g.
the point where `EV_adj` crossing zero corresponds to a calibration
error-bar-consistent probability of loss), then validate each candidate
`k` OOS on the walk-forward split used in §13 — measuring whether it
actually separates realized-profitable from realized-unprofitable
decisions in held-out data. The chosen `k` and the OOS evidence for it are
documented together in `docs/ARCHITECTURE.md`; `k` is not tuned per-trade
and not fit against the final evaluation window, but it also must not be
picked a priori without this validation step.

`EV_adj` and its inputs are reported with an explicit range/margin (not a
single over-precise decimal), per the "no false precision" principle.

## 10. Regime conditioning

Regime is NOT used to create per-state EV rules or thresholds in v1 (the
brief explicitly warns against automatically creating separate rules per
HMM state without OOS evidence). It is recorded in `EVDecision`'s lineage
and reported as descriptive context. If a future sensitivity check (during
implementation, using Phase 4's real per-state metrics from Task 6) shows
regime materially predicts calibration drift or EV realization error, that
finding is documented and a conditioning rule may be added — this is
explicitly deferred to be evidence-driven, not designed speculatively.

## 11. NO_TRADE gate & long/short evaluation

`EV_adj` is computed independently for long and short. Decision:

- `NO_TRADE` if any required specialist for the side/horizon is not
  `VALIDATED`/`CANDIDATE`, if MarketState is stale, if cost is
  `UNAVAILABLE`, or if `EV_adj <= min_edge_threshold` for both sides.
- Otherwise, pick the side with higher `EV_adj` (if only one side clears
  the threshold, take that side).

`min_edge_threshold` is a fixed constant derived from the known
transaction-cost floor plus a documented minimum-edge buffer — not a bare
`p_win > 0.6` heuristic, and not curve-fit to any evaluation window.
Long/short symmetry is NOT assumed — both sides are computed from their
own Barrier/MAE/MFE conditional values, and asymmetry (if the real OOS
data shows one) is measured and reported, not designed away.

## 12. Sensitivity analysis

The research simulator (§14) computes `EV_adj` under perturbations:
spread ±X%, candidate SL/TP ±1 quantile step, calibration probability
±1 calibration-error-bar. Opportunities where a small perturbation flips
the sign of `EV_adj` are flagged `fragile` in the simulator's output —
this is a reporting/classification feature, not a gate, in v1.

## 13. OOS validation & baseline

The EV engine itself is validated OOS using the same
purged+embargoed walk-forward split machinery Phase 4 used
(`learning.cv.PurgedWalkForwardCV`), evaluated on a held-out window not
used for any Phase 4 specialist's own final OOS evaluation. Compared
against two baselines: (1) current production decision logic (unchanged,
read-only comparison), (2) a simple single-probability gate
(`P(direction) > 0.55 -> trade`, no cost/no EV) — to demonstrate the EV
engine's added complexity earns a measurable improvement (e.g. cost-aware
realized R vs the simple gate's realized R), not just "the math runs."

## 14. Architecture

```
research/phase5_ev_dataset.py   — assembles historical specialist-output + spread replay dataset
research/phase5_ev_engine.py    — research-only EV simulator: historical replay, sensitivity, OOS report (no Telegram)
contracts/specialist_output.py  — the 7 role contracts (§4)
contracts/ev_decision.py        — EVDecision schema (§15)
decision/calibration_registry.py — versioned calibrator lookup (§5)
decision/ev_engine.py           — live pure function: MarketState + specialist outputs -> EVDecision (shadow-only call site, NOT wired into app/engine.py)
models/calibration/*.json       — calibration artifacts
docs/ARCHITECTURE.md            — new "## Phase 5" section
```

`decision/ev_engine.py`'s live entry point is called from a new, separate
shadow-evaluation path (mirroring `app/shadow.py`'s existing pattern) —
`app/engine.py`'s production decision call sequence, current production
SL/TP, and Telegram signal behavior remain byte-for-byte unchanged, the
same production-safety bar Phase 4 held.

## 15. EVDecision schema (lineage)

```
EVDecision:
  timestamp, direction (long/short/none), decision (NO_TRADE/LONG_CANDIDATE/SHORT_CANDIDATE)
  ev_adj, ev_raw, uncertainty, decision_margin
  candidate_sl, candidate_tp
  cost_r, known_cost_only
  specialist_model_ids: dict[role, model_id]
  calibration_ids: dict[role, calibration_id]
  feature_schema_ids: dict[role, schema_id]
  ev_formula_version, cost_model_version
  regime_state
  timeout_r_provisional_proxy: bool
  decision_reason: str
```

## 16. Testing (per brief §29)

New test files under `tests/`: contract validation, barrier-probability
coherence (sums ≈1, deviation explained), calibration correctness on
known cases, EV math on known probability/payoff fixtures, cost correctly
reduces EV, long/short symmetric-input cases behave as expected,
insufficient-EV correctly yields NO_TRADE, a `DATA_LIMITED`/`UNAVAILABLE`
specialist cannot produce a valid numeric decision, stale MarketState
blocks live decisions, schema-mismatched specialist input is rejected,
sensitivity perturbation changes EV_adj in the correct direction, no
future information enters the calculation (causality test using
Phase 4's existing purge/embargo test pattern), deterministic replay,
live/replay equivalence within tolerance, and a boundary test confirming
`app/engine.py`/Telegram/production SL-TP are unchanged (same `git diff`
technique Phase 4's Task 16 used).

## 17. Performance

Benchmark aggregation + calibration + EV calc + candidate SL/TP + full
decision latency, single-row inference pattern reused from
`tests/test_specialist_inference_performance.py` (p50/p95/p99). No
optimization before correctness, per Phase 4 precedent.

## 18. Documentation

Append a new "## Phase 5: Probability / EV Engine" section to
`docs/ARCHITECTURE.md` covering all of §4-§17 above with real numbers once
implemented, plus a Mermaid diagram of §14's architecture. The EV formula
(§9) is documented explicitly, including the timeout-R approximation and
the fixed `k`/`min_edge_threshold` constants and their justification.

## 19. Known limitations (to be carried into the final Phase 5 report)

- Direction/Barrier's exact conditional relationship (§6) is resolved by
  empirical investigation during implementation, not assumed — the
  resolution found (redundant, or a documented+OOS-validated correction
  term) is reported as evidence, whichever way it turns out.
- `timeout_R` uses a direct OOF-derived estimate where sample size
  supports it (§9); the midpoint proxy is a documented, explicitly-flagged
  fallback only where OOF sample size is inadequate, not the default.
- `k` (uncertainty penalty) is derived from a principled anchor and
  OOS-validated on held-out data (§9), not picked a priori — its
  validation evidence is documented alongside the chosen value.
- Slippage is not modeled (§8) — cost is spread-only, explicitly labeled
  `known_cost_only`.
- Regime conditioning is deferred pending evidence (§10).
- Execution/Decay remains `DATA_LIMITED` and does not participate in EV
  in v1.
- h90 (and Opportunity at h45/h90) will produce `NO_TRADE`/degraded
  decisions by design, driven by Phase 4's real rejected/unavailable
  statuses — not a Phase 5 defect.
