# GOLDEX Phase 2 Hardening — Moderate-Scale Simulator Health Check

**Date:** 2026-09-02
**Script:** `scripts/run_fast_tier_health_check.py`
**Raw metrics:** `docs/superpowers/reports/2026-09-02-goldex-phase2-hardening-simulator-run-metrics.json`

## What this is, and what it is not

This is a **system-health / integration verification** run, not a backtest and not a
profitability result. It exercises the exact same real composition used by
`tests/intelligence/test_full_fast_tier_integration.py` — the real
`build_default_registry()`, real `ToolTrust`, real `FastTierReasoner`, the real EV/cost
gate wrapping `simulator.cost_model.round_trip_cost_r`, real
`analytical_sizing_bootstrap`/`analytical_sltp_bootstrap`, real `FastTierDecisionEngine`,
Phase 1's real unmodified `simulator.replay.run_replay`, and the real `ExperienceStore`
read path — over a longer, deterministic synthetic chronological series than that test's
own 300/400-bar fixtures.

**This report contains no P&L figure, no Sharpe ratio, and no win-rate-as-success
framing.** Every number below answers a single question: *did
`MarketState -> Fast Tier -> Action -> Execution -> Experience` compose correctly at
scale, with zero crashes and a sane distributional shape?* Whether the synthetic series'
trades were profitable is irrelevant to that question and is not reported.

## Run parameters

- Dataset: deterministic synthetic OHLC series (`_make_df`, seed `20260902`), same
  construction as the integration test's fixture (trend + oscillation + Gaussian noise,
  `NOISE_STD=0.35` — the noise level shown in that test to be necessary for
  `realized_vol_60s` to clear the EV/cost gate's `spread/(SL_VOL_MULTIPLIER * mid)`
  threshold at all).
- Scale: **4,000 bars** (~13x the integration test's 300-bar fixture).
  - An initial attempt at 20,000 bars (~65x) was aborted after running for more than
    50 minutes of wall-clock without completing — per-bar cost (GARCH/Kalman refits,
    applicability gating, thesis bookkeeping) does not amortize as cheaply as hoped even
    with Task 3's refit-caching, so 20,000 bars is not a practical size for a single
    synchronous run. 4,000 bars was chosen as a scale that both (a) is a real order of
    magnitude larger than the integration test's fixture and (b) actually completes.
    This is disclosed here as a deviation from the brief's "tens of thousands of bars"
    upper end, not concealed.
- Environment tag: `SIMULATED_TRAINING`.

## Results

| Metric | Value | Read as |
|---|---|---|
| Wall-clock runtime | 1593.2 s (~26.5 min) | Completed synchronously; no hang, no timeout inside the run itself. |
| Exceptions raised | **0** | Zero crashes across 4,000 bars — the required pass condition. |
| Total DECIDE-event records | 2,249 | The reasoner was invoked and returned a well-formed action on every decision bar. |
| Action split | NO_TRADE 2,059 / LONG 182 / SHORT 8 | NO_TRADE rate 91.6% — bounded, not 100% (system does propose trades) and not near-0% (EV/cost gate is not rubber-stamping everything). |
| Rejected entries (LONG/SHORT with a rejection reason) | 0 | No entries reached the engine with an unresolved rejection reason recorded. |
| Closed positions | 190 | Consistent with 182+8=190 opened positions — every opened position was accounted for at close. |
| Exit-reason breakdown | TP_HIT 65, SL_HIT 56, POLICY_EXIT 68, END_OF_REPLAY_FORCED_CLOSE 1 | All four of Phase 1's defined exit paths fired at least once; POLICY_EXIT (68) is on the same order as SL/TP exits (121 combined), so the reasoner's own exit logic is meaningfully load-bearing, not vestigial. |
| Avg trade duration | 9.57 bars | Non-degenerate — not 1 bar (no evidence of instant-flip pathology) and not pinned at the run length (no evidence of "never exits" pathology). |
| Min / max trade duration | 1 / 41 bars | Wide spread; the one 1-bar trade did not skew the average. |
| Per-source applicability-gate rate | 3.4%–7.0% across all 9 sources (see table below) | Every source is gated out only a small minority of the time — none is either always-gated (dead weight) or never-gated (gate not actually discriminating). |
| Context buckets observed | {-1: 170, 2: 3598, 3: 110} per-call; {-1: 9, 2: 708, 3: 32} load-bearing | Multiple distinct buckets occupied in both views — the run does not collapse into a single bucket, consistent with Task 4's recalibration goal. |
| `ended_flat` | `False` (one position force-closed at end of replay) | Expected — `run_replay` force-closes any open position at series end; this is the `END_OF_REPLAY_FORCED_CLOSE=1` row, not a leak or unhandled state. |

### Per-source applicability-gate rate

| Source | Gated-out rate |
|---|---|
| momentum_scalar | 3.35% |
| path_pca_projection | 3.48% |
| multiscale_vol_ratio | 5.67% |
| vol_regime_transition | 6.96% |
| garch_conditional_variance | 4.38% |
| kalman_filtered_velocity | 4.38% |
| kalman_innovation | 4.38% |
| rolling_skew | 4.38% |
| rolling_excess_kurtosis | 4.38% |

## Interpretation (health only)

- **Zero exceptions** across 4,000 bars and 2,249 decision calls is the required pass
  condition for this task, and it held.
- **NO_TRADE rate (91.6%)** is bounded away from both 0% and 100%, indicating the EV/cost
  gate and applicability gating are both active and discriminating rather than either
  rubber-stamping every bar or blocking everything.
- **All four exit paths fired** (TP, SL, POLICY_EXIT, forced end-of-replay close), with
  POLICY_EXIT representing roughly a third of all closes — confirming the reasoner's own
  exit decisioning is exercised at scale, not just the mechanical SL/TP paths.
- **Context buckets are spread** (3 distinct buckets occupied, in both the per-call and
  load-bearing views), which is the direct thing Task 4's bucket-constant recalibration
  was meant to produce — a run that pinned to a single bucket would indicate the
  recalibration didn't take.
- **Per-source gate rates are all in a narrow, non-degenerate band (3.4%–7.0%)** — no
  source is a permanent dead weight (100% gated) or a rubber stamp (0% gated).

None of the above is a claim about whether the strategy would make money; it is a claim
about whether the pipeline's components compose correctly and produce a sane
distributional shape under moderate-scale chronological replay.

## Deviation from brief

The brief describes scale as "one to a few years of synthetic chronological data, or
whatever realistic-scale fixture ... extended in length," and the task dispatch note
for this run specified "low thousands to tens of thousands of bars." An initial attempt
at 20,000 bars did not complete within 50+ minutes of wall-clock and was aborted; 4,000
bars was used instead, which completed in ~26.5 minutes. This is a real constraint on
this hardened Fast Tier's per-bar cost, not a shortcut — it is called out here rather
than silently substituted, and 4,000 bars remains a genuine order-of-magnitude increase
over the integration test's 300-bar fixture with the same real composition.
