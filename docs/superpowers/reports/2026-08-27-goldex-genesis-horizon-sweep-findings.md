# GOLDEX Genesis Reset — Horizon Sweep Findings (2026-08-27)

## Summary

Section 30 of the genesis architecture reset document identified one variable never tested across the 26 prior hypotheses (Phase 3, 3A, 4): the forward-return horizon was fixed at 5 bars in every single one. This report is the result of the horizon sweep, the "minimum credible proof-of-concept" that document specified as the one thing to do before any agent-architecture code is written.

**Top-line result: uniformly confounded, zero surviving cells.** Across 7 representations x 6 horizons (1, 5, 15, 30, 60, 120 bars) = 42 cells, every single cell that cleared its own shuffled-label null was explained by the same trend confound Phase 3A/4 already identified. No cell reached category (c) (large MI not explained by the trend confound). Per the genesis document's Section 34 falsification clause, **this specifically falsifies "horizon/target definition was the missing variable."**

## DATA USED / REPRESENTATION / MODEL / TARGET / TRAIN PERIOD

- **DATA USED:** `data/gold_seed_merged_full6yr.csv`, rows 0:300,000 (training partition only — the Phase 3 validation split, rows 300,000:400,000, was never read).
- **REPRESENTATION:** seven representations, all reused unchanged from already-validated prior-phase code (no reimplementation):
  1. `momentum_scalar` — Phase 3A (`research/phase3a_representation_experiments.py`)
  2. `raw_path_window_projection` — Phase 3A
  3. `multiscale_volatility_ratio` — Phase 3A
  4. `volatility_regime_transition` — Phase 3A (also used here as the trend-invariant confound-diagnostic reference)
  5. `garch11_conditional_variance` — Phase 4 (`fit_garch11`, `research/phase4_garch_volatility_mechanism.py`)
  6. `kalman_filtered_velocity` and `kalman_innovation` — Phase 4 (`kalman_level_trend_filter`, `research/phase4_kalman_trend_mechanism.py`)
  7. `rolling_skew` and `rolling_excess_kurtosis` — Phase 4 (`_rolling_moment`, `research/phase4_distributional_mechanism.py`)
- **MODEL:** none. This is a **marginal-MI sweep**, matching Phase 3A's Section B methodology exactly (`binned_mutual_information` + `mi_with_shuffle_control`, imported unchanged, 20-permutation shuffled-label null per cell). It is intentionally **not** a full OOS predictive-modeling check (that pattern is Phase 3A's Section D / Phase 4's OOS check) — the point of a cheap 42-cell sweep is to identify which cells are even worth a proper OOS check, not to run 42 full OOS checks.
- **TARGET:** `forward_return(closes, horizon=H)` for H in {1, 5, 15, 30, 60, 120} bars, identical construction to Phase 3A/4, only H varied.
- **TRAIN PERIOD:** rows 0:300,000 of the training partition only. Rows 300,000:400,000 (real Phase 3 OOS split) were never touched.

## Method — confound check

Every cell where real MI cleared `null_mi_mean + 3*null_mi_std` was checked against the same trend-confound diagnostic Phase 3A used:
1. Compare real MI to the MI of the trend-invariant `volatility_regime_transition` representation at the same horizon (the reference Phase 3A already showed carries ~2 orders of magnitude less MI than trend-sensitive representations under the same secular-drift confound).
2. Check whether the raw forward-return target itself still shows near-zero return autocorrelation at that horizon (reusing `analyze_return_autocorrelation` from `research/phase3_representation_research.py`).

Classification:
- **(a) null-consistent** — doesn't clear its own shuffled null.
- **(b) large-MI-but-likely-trend-confounded** — clears null, but doesn't beat the regime-transition reference MI at that horizon by a wide margin (5x), or the underlying return autocorrelation at that horizon is still near zero (|autocorr| < 0.05).
- **(c) large-MI-and-not-explained-by-trend-confound** — clears null, beats the regime-transition reference by a wide margin, and coincides with non-trivial raw-return autocorrelation. Only these are candidates for a follow-up OOS check (explicitly **not run** here — out of scope for this script per the genesis document).

## Full results table

| representation | horizon | real MI (nats) | null mean | null std | null max | regime-transition MI (same horizon) | raw-return autocorr at horizon | classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| momentum_scalar | 1 | 0.123348 | 0.000142 | 0.000022 | 0.000174 | 0.000766 | -0.038986 | b |
| raw_path_window_projection | 1 | 0.111070 | 0.000135 | 0.000021 | 0.000170 | 0.000766 | -0.038986 | b |
| multiscale_volatility_ratio | 1 | 0.049109 | 0.000139 | 0.000021 | 0.000189 | 0.000766 | -0.038986 | b |
| volatility_regime_transition | 1 | 0.000766 | 0.000014 | 0.000008 | 0.000037 | 0.000766 | -0.038986 | b (self-reference)* |
| garch11_conditional_variance | 1 | 0.225401 | 0.000124 | 0.000020 | 0.000158 | 0.000766 | -0.038986 | b |
| kalman_filtered_velocity | 1 | 0.093885 | 0.000130 | 0.000024 | 0.000191 | 0.000766 | -0.038986 | b |
| kalman_innovation | 1 | 0.105795 | 0.000128 | 0.000022 | 0.000177 | 0.000766 | -0.038986 | b |
| rolling_skew | 1 | 0.051963 | 0.000137 | 0.000021 | 0.000166 | 0.000766 | -0.038986 | b |
| rolling_excess_kurtosis | 1 | 0.057575 | 0.000129 | 0.000022 | 0.000167 | 0.000766 | -0.038986 | b |
| momentum_scalar | 5 | 0.117292 | 0.000139 | 0.000022 | 0.000190 | 0.000784 | -0.010448 | b |
| raw_path_window_projection | 5 | 0.107669 | 0.000139 | 0.000019 | 0.000188 | 0.000784 | -0.010448 | b |
| multiscale_volatility_ratio | 5 | 0.042171 | 0.000139 | 0.000024 | 0.000194 | 0.000784 | -0.010448 | b |
| volatility_regime_transition | 5 | 0.000784 | 0.000016 | 0.000007 | 0.000034 | 0.000784 | -0.010448 | b (self-reference)* |
| garch11_conditional_variance | 5 | 0.214182 | 0.000134 | 0.000021 | 0.000173 | 0.000784 | -0.010448 | b |
| kalman_filtered_velocity | 5 | 0.090912 | 0.000130 | 0.000019 | 0.000167 | 0.000784 | -0.010448 | b |
| kalman_innovation | 5 | 0.095040 | 0.000142 | 0.000025 | 0.000184 | 0.000784 | -0.010448 | b |
| rolling_skew | 5 | 0.049060 | 0.000131 | 0.000013 | 0.000162 | 0.000784 | -0.010448 | b |
| rolling_excess_kurtosis | 5 | 0.054460 | 0.000133 | 0.000017 | 0.000166 | 0.000784 | -0.010448 | b |
| momentum_scalar | 15 | 0.097740 | 0.000149 | 0.000020 | 0.000186 | 0.000642 | -0.000379 | b |
| raw_path_window_projection | 15 | 0.088652 | 0.000143 | 0.000019 | 0.000195 | 0.000642 | -0.000379 | b |
| multiscale_volatility_ratio | 15 | 0.032481 | 0.000134 | 0.000022 | 0.000173 | 0.000642 | -0.000379 | b |
| volatility_regime_transition | 15 | 0.000642 | 0.000015 | 0.000008 | 0.000041 | 0.000642 | -0.000379 | b (self-reference)* |
| garch11_conditional_variance | 15 | 0.186248 | 0.000139 | 0.000016 | 0.000166 | 0.000642 | -0.000379 | b |
| kalman_filtered_velocity | 15 | 0.080476 | 0.000139 | 0.000015 | 0.000161 | 0.000642 | -0.000379 | b |
| kalman_innovation | 15 | 0.080461 | 0.000142 | 0.000021 | 0.000176 | 0.000642 | -0.000379 | b |
| rolling_skew | 15 | 0.038160 | 0.000134 | 0.000018 | 0.000171 | 0.000642 | -0.000379 | b |
| rolling_excess_kurtosis | 15 | 0.042045 | 0.000134 | 0.000016 | 0.000169 | 0.000642 | -0.000379 | b |
| momentum_scalar | 30 | 0.079385 | 0.000145 | 0.000028 | 0.000218 | 0.000455 | -0.003899 | b |
| raw_path_window_projection | 30 | 0.072494 | 0.000140 | 0.000025 | 0.000197 | 0.000455 | -0.003899 | b |
| multiscale_volatility_ratio | 30 | 0.024876 | 0.000137 | 0.000018 | 0.000162 | 0.000455 | -0.003899 | b |
| volatility_regime_transition | 30 | 0.000455 | 0.000011 | 0.000004 | 0.000020 | 0.000455 | -0.003899 | b (self-reference)* |
| garch11_conditional_variance | 30 | 0.160347 | 0.000134 | 0.000019 | 0.000167 | 0.000455 | -0.003899 | b |
| kalman_filtered_velocity | 30 | 0.070839 | 0.000131 | 0.000021 | 0.000183 | 0.000455 | -0.003899 | b |
| kalman_innovation | 30 | 0.067720 | 0.000136 | 0.000019 | 0.000187 | 0.000455 | -0.003899 | b |
| rolling_skew | 30 | 0.023195 | 0.000134 | 0.000021 | 0.000182 | 0.000455 | -0.003899 | b |
| rolling_excess_kurtosis | 30 | 0.025789 | 0.000139 | 0.000021 | 0.000189 | 0.000455 | -0.003899 | b |
| momentum_scalar | 60 | 0.064383 | 0.000139 | 0.000025 | 0.000190 | 0.000400 | -0.003899 | b |
| raw_path_window_projection | 60 | 0.060048 | 0.000128 | 0.000016 | 0.000164 | 0.000400 | -0.003899 | b |
| multiscale_volatility_ratio | 60 | 0.010849 | 0.000137 | 0.000017 | 0.000172 | 0.000400 | -0.003899 | b |
| volatility_regime_transition | 60 | 0.000400 | 0.000018 | 0.000009 | 0.000037 | 0.000400 | -0.003899 | b (self-reference)* |
| garch11_conditional_variance | 60 | 0.140297 | 0.000138 | 0.000026 | 0.000235 | 0.000400 | -0.003899 | b |
| kalman_filtered_velocity | 60 | 0.062905 | 0.000135 | 0.000023 | 0.000169 | 0.000400 | -0.003899 | b |
| kalman_innovation | 60 | 0.056865 | 0.000143 | 0.000018 | 0.000173 | 0.000400 | -0.003899 | b |
| rolling_skew | 60 | 0.023350 | 0.000127 | 0.000020 | 0.000170 | 0.000400 | -0.003899 | b |
| rolling_excess_kurtosis | 60 | 0.025042 | 0.000128 | 0.000019 | 0.000173 | 0.000400 | -0.003899 | b |
| momentum_scalar | 120 | 0.047054 | 0.000132 | 0.000023 | 0.000174 | 0.000273 | -0.003899 | b |
| raw_path_window_projection | 120 | 0.043768 | 0.000130 | 0.000021 | 0.000186 | 0.000273 | -0.003899 | b |
| multiscale_volatility_ratio | 120 | 0.007031 | 0.000132 | 0.000021 | 0.000181 | 0.000273 | -0.003899 | b |
| volatility_regime_transition | 120 | 0.000273 | 0.000011 | 0.000006 | 0.000028 | 0.000273 | -0.003899 | b (self-reference)* |
| garch11_conditional_variance | 120 | 0.108171 | 0.000137 | 0.000018 | 0.000164 | 0.000273 | -0.003899 | b |
| kalman_filtered_velocity | 120 | 0.045720 | 0.000133 | 0.000015 | 0.000166 | 0.000273 | -0.003899 | b |
| kalman_innovation | 120 | 0.044075 | 0.000134 | 0.000025 | 0.000179 | 0.000273 | -0.003899 | b |
| rolling_skew | 120 | 0.012986 | 0.000132 | 0.000024 | 0.000187 | 0.000273 | -0.003899 | b |
| rolling_excess_kurtosis | 120 | 0.014038 | 0.000126 | 0.000022 | 0.000171 | 0.000273 | -0.003899 | b |

\* The `volatility_regime_transition` rows are the trend-invariant reference itself, so its "margin over regime-transition" test is vacuously false by construction — it's listed as (b) only because it's compared against itself, not because it's independently confounded. Its purpose in this table is as the yardstick the other eight representations are measured against; its own row carries no additional classification content.

**Category counts:** a_null_consistent = 0, b_large_mi_likely_trend_confounded = 54 (48 substantive cells + 6 self-reference regime-transition rows), c_large_mi_not_explained_by_trend_confound = **0**.

## Observations

- Every trend-sensitive representation (momentum, raw path, multiscale vol ratio, GARCH, Kalman velocity/innovation, skew/kurtosis) clears its shuffled null by a wide margin at every horizon tested — but so did every one of these at the fixed 5-bar horizon in Phase 3A/4, and that was already shown to be the trend confound, not real signal.
- The trend-invariant `volatility_regime_transition` reference stays 2+ orders of magnitude smaller than the trend-sensitive representations at every horizon (0.0003–0.0008 nats vs. 0.007–0.23 nats), exactly mirroring the Phase 3A pattern. This is strong internal-consistency evidence that the same confound mechanism, not new local signal, is what's driving the larger numbers here too.
- Real MI for every trend-sensitive representation **decays monotonically as horizon increases** (e.g. momentum_scalar: 0.123 at horizon 1 down to 0.047 at horizon 120; GARCH: 0.225 down to 0.108). This is the expected signature of a fixed-magnitude secular trend diluted by a longer, noisier forward-return window — not the signature of a genuine predictive edge, which would be expected to show a horizon-dependent peak rather than monotone decay from the shortest horizon tested.
- Raw return autocorrelation at every horizon tested stays within [-0.039, -0.0004] — essentially zero, consistent with Phase 3A's original finding (lag_1 = -0.039, decaying to ~0 by lag 9-10) and giving no independent basis to believe local predictability exists at any of the 6 horizons swept.

## RESULT

Zero of the 42 (representation x horizon) cells reached category (c). All 48 substantive cells (excluding the 6 self-reference regime-transition rows) that cleared their shuffled null did so via the same trend-confound mechanism Phase 3A/4 already identified and disclosed. No horizon in {1, 5, 15, 30, 60, 120} produces real, non-trend-confounded marginal signal in any of the 7 representations tested.

## LIMITATION

- Marginal MI only — no OOS predictive-modeling check was run (correctly, per scope: the sweep's job was to identify candidates for that check, and it found none).
- Single fixed-seed, 20-permutation shuffle null per cell (matches prior-phase convention, not independently re-tuned here).
- Single lookback/window choice per representation (the lookbacks themselves were not swept — only the target horizon was, per Section 30's explicit scope).
- From-scratch GARCH(1,1) and constant-velocity Kalman implementations (same disclosed limitation as Phase 4 — no external `arch`/filtering library dependency was added).
- The confound-classification heuristic (5x margin over the regime-transition reference, |autocorr| < 0.05 threshold) is a reasonable operationalization of the same diagnostic Phase 3A used qualitatively, but is itself a threshold choice, not a formal proof that every (b)-labeled cell contains zero real signal — it is the same honest-but-imperfect standard already applied and accepted in Phase 3A/4.

## CONCLUSION

**The horizon sweep is uniformly null/confounded.** No horizon among {1, 5, 15, 30, 60, 120} bars, across any of the 7 representations already validated in Phase 3A/4, shows real marginal signal that survives the trend-confound check. This directly answers the genesis document's Section 30 question — "was 5 bars specifically the wrong choice all along?" — with **no**: the same trend-confound pattern reproduces at every horizon tested, not just 5 bars.

Per Section 34 of the genesis architecture reset document (the falsification clause): **this result specifically falsifies "the missing variable was horizon/target definition."** No version of Architecture G (or anything more sophisticated) fixes an absence of information, and this sweep shows the information was still absent across the horizon dimension, not merely mis-targeted at 5 bars.

Per the roadmap in that same document, the next honest question this result raises is about **information source (tick data, cross-instrument context) or scope (a different instrument entirely) — not about agent architecture.** This report does not recommend or begin that next branch; per the task's explicit scope, no MoE, RL, optimal stopping, strategy-library construction, or new candidate tournament is authorized by this result, and none was started here. The decision of which of those (if any) is worth pursuing next belongs to a future SDD design/plan cycle, not to this script or this report.
