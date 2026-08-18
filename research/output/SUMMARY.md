# Phase 1A -- Empirical Edge Audit -- Summary

Run: 2026-08-18 09:53:46 -> 09:59:00 UTC (314.5s). Script: `research/audit_edge.py`.
Data: `research/output/audit_20260818_095900.json` (also `latest.json`), plus
`threshold_sweep_*.csv`, `feature_importance_*.csv`, `family_ablation_*.csv`, `run.log`.

Model/feature version: exactly the currently deployed 28-feature set, CUSUM_K=2.5,
TB_CFG_DIR(1.0/1.0/45), TB_CFG_TRADE(1.5/1.0/45), horizon_vol_scale=0.45, N_SPLITS=6
(5 yielded after min_train_bars drop), EMBARGO_BARS=90, CatBoost depth=4/lr=0.02/l2=15/
early_stop=100. No production files modified. No models retrained/overwritten.

## Headline findings

1. **Calibration is reasonable up to 0.60, untested above it.** Brier=0.240, logloss=0.673,
   calibration slope=0.926 (near-1 = good), intercept=-0.02. Bins 0.50-0.65 all within
   ~3pp of predicted probability. Bins above 0.65 have almost no OOF samples (n=2377 at
   0.65-0.70, n=1 above) -- the deployed system essentially never generates a real-world
   test of "does 0.75 mean 75%," it just hasn't happened enough to know.

2. **The deployed 0.60 threshold has effectively STOPPED FIRING in honest OOF for the
   most recent ~14 months.** Fold 4 of the meta OOF (test window 2025-08-26 ->
   2026-08-17, i.e. a model that never saw 2025-2026 data during training) never
   produces a probability >= 0.60 in calendar 2025 or 2026 (max observed: 0.598 in 2025,
   0.593 in 2026). Zero sequential trades qualify at the deployed threshold in either
   year under strict walk-forward validation. Yet the live system (fit in-sample on
   100% of data through today) is reportedly firing ~14.5 signals/day at this same
   threshold -- a live/OOF frequency mismatch that is the single most important finding
   of this audit.

3. **This is calibration drift, not a vanished edge.** Relative ranking still works: the
   top-decile-by-probability win rate in 2025 (57.7%) and 2026 (59.5%) is comparable to
   2022-2024 (59-60%), even though the absolute probability scale has drifted down (mean
   predicted p: 0.50 in 2022-2024 -> 0.48 in 2025 -> 0.47 in 2026). A fixed global
   threshold picked once no longer tracks the model's own precision frontier over time.

4. **Primary direction accuracy is weak and monotonically declining across chronological
   folds:** 51.60% -> 51.55% -> 50.88% -> 50.97% -> 50.75% (fold 0 -> 4). All folds beat
   50% but the margin is shrinking over calendar time -- consistent with train.py's own
   documented note that the underlying reversal effect decays 52.0% -> 50.1% over
   2020-2026. Almost all of the deployed system's apparent edge lives in the meta
   (precision-filter) stage, not the primary direction call.

5. **Realized-vs-nominal R accounting gap is large and needs resolution before trusting
   any expectancy number.** Idealized barrier-fill accounting (win=+1.5R, loss=-1.0R
   exactly, matching what a live tick-triggered TP/SL fill should realize) gives
   nominal_raw_R=+0.481 at threshold 0.60, net after spread ~+0.24R -- consistent with
   the pre-audit session's numbers. Bar-close-based accounting (actual M1 close price at
   the barrier-touch bar, which is what this audit's `ret` field measures) gives
   avg_loss_R=-1.48 (not -1.0) and realized_mean_R=+0.0097, net after spread **-0.24R**.
   The gap is real information about intrabar overshoot/gap risk in 1-minute gold bars
   (avg loss overshoots the nominal SL by ~48% of the risk unit on average), but is also
   partly a backtest-resolution artifact since live fills are tick-triggered, not
   bar-close-triggered. The TRUE net edge sign cannot be determined from M1 OHLC data
   alone -- this is a genuine data-resolution limitation, stated plainly rather than
   resolved by assumption.

6. **CUSUM event selection is validated but only mildly informative.** CUSUM-selected
   bars: 51.15% primary accuracy vs 50.36% on a random-bar baseline of equal size
   (+0.8pp). Real, modest, not dramatic.

7. **Feature families are mostly redundant with each other.** Dropping the entire
   base_return family (8 cols) costs -0.25pp primary accuracy, the largest single-family
   effect measured; every other family (volatility, jump, kalman, hurst/fracdiff,
   tick_volume, spread) costs <0.1pp when removed. Several individual features
   (fracdiff_log_price, kalman_residual_z/level_dist) rank very high in CatBoost's raw
   importance yet barely matter when their whole family is dropped -- they are
   correlated substitutes for each other, not uniquely load-bearing.

8. **tick_volume: small, real, unstable contribution -- ablation supports keeping it for
   now, not urgently removing it.** Full-pipeline ablation without tick_volume: primary
   accuracy delta -0.03pp (noise-level), sequential trades at 0.60 drop from 19,165 to
   14,296 (fewer candidates clear the bar without it) but win rate is statistically
   indistinguishable (59.4% vs 59.3%). Permutation importance is small but positive; fold
   CV=0.92 (unstable). Net: no evidence it is actively harmful, weak evidence it helps,
   real evidence its scale mismatch (documented in the prior masterplan) makes its
   long-run reliability suspect.

9. **spread feature: dead weight, confirmed.** Zero CatBoost importance (model never
   splits on it -- expected, given it's a constant for 98.9% of training history).
   Removing it from the feature set costs exactly 0.0pp accuracy. Safe to drop as a
   *feature* without retraining risk; this does not resolve its use as a *cost* input
   (see spread scenarios below, which are a separate question).

10. **Volatility regime does not materially change the edge.** Win rate 58.8% (low-vol)
    / 59.3% (medium) / 59.6% (high-vol) at threshold 0.60 -- flat. No evidence the model
    is a hidden volatility-timing strategy in disguise; no urgent case for regime-gating
    given current data.

11. **SL/TP geometry (1.0/1.5) is structurally reasonable but tight relative to real
    intrabar noise.** TP-first 59.3%, SL-first 36.8%, timeout 3.9% of accepted trades --
    close to the meta win rate, consistent. MAE p90 = 1.88R (i.e. 10% of trades travel
    almost double the SL distance against the position before resolving -- gap/overshoot
    risk, not visible in the idealized barrier check). MFE only exceeds the 1.5R TP
    level in 7.5% of trades even though 59% eventually touch it -- most winners drift to
    TP slowly rather than spiking through it early, median time-to-TP 10.9 bars vs
    time-to-SL 13.5 bars.

12. **Execution delay: real decay, worst in the first few minutes, partially recovers
    later (not monotonic).** M1 resolution only -- true sub-minute timing is not
    supported by this dataset and was not attempted. Whole-bar-shift proxy (accepted
    0.60 trades, same side, re-entered at t0+shift bars): win rate 59.3% (immediate) ->
    58.2% (+1min) -> 57.1% (+3min) -> 57.2% (+5min) -> 56.3% (+10min) -> 56.3% (+15min)
    -> 55.9% (+30min); mean R goes negative by the +3min mark and stays mixed-to-negative
    through +15min, recovering slightly to marginally positive at +30min (likely
    reversion back toward the mean rather than the original signal still being live).
    Practical read: the signal's edge is concentrated in the first 1-2 minutes: a
    Telegram-relay-and-manual-click delay in that range is probably fine, several
    minutes of hesitation measurably erodes it.

## Master table (item 15)

| Test                  | Result                                   | OOS Evidence                | Robust? | Confidence |
|------------------------|-------------------------------------------|------------------------------|---------|------------|
| Primary direction      | 50.75-51.60% acc, declining by fold       | 5/5 folds >50%, monotonic decline | Weak, decaying | Low |
| Meta p_win             | 59.1-60.6% acc across folds               | 5/5 folds, calibration slope 0.93 | Moderate | Medium |
| p_win >= 0.60 (deployed)| 59.3% win rate, but 0 trades in 2025-2026 OOF | Present 2021-2024, absent 2025-2026 | Not robust -- threshold-drift | Low |
| Spread-adjusted         | +0.24R (nominal) or -0.24R (realized)    | Sign depends on accounting method | Unresolved | Low |
| Tick-volume excluded    | -0.03pp primary acc, win rate unchanged  | Full OOF ablation | Neutral/stable | Medium |
| Low volatility          | 58.8% WR, +0.47 nominal R                | n=11,073 sequential | Consistent w/ other regimes | Medium |
| Medium volatility       | 59.3% WR, +0.48 nominal R                | n=3,570 sequential | Consistent | Medium |
| High volatility         | 59.6% WR, +0.49 nominal R                | n=5,090 sequential | Consistent | Medium |
| 2021                    | 55.7% WR, n=415                          | Thin sample, partial-year fold coverage | Weak (small n) | Low |
| 2022                    | 59.7% WR, n=5,467                        | Full year OOF | Robust for that year | Medium |
| 2023                    | 59.8% WR, n=6,105                        | Full year OOF | Robust for that year | Medium |
| 2024                    | 58.7% WR, n=7,178                        | Full year OOF | Robust for that year | Medium |
| 2025                    | 0 trades clear 0.60 in OOF               | Max OOF p=0.598 all year | Threshold miscalibrated for this year | Low |
| 2026                    | 0 trades clear 0.60 in OOF                | Max OOF p=0.593 YTD | Threshold miscalibrated for this year | Low |

## Final verdict (item 16)

**B -- PROMISING BUT INCOMPLETE.**

Numerically: primary direction beats random consistently but weakly (5/5 folds >50%,
margin 0.75-1.6pp, shrinking over time). The meta stage adds real, well-calibrated
precision (slope 0.93) through 2024, and CUSUM event selection is validated against a
random-bar control (+0.8pp). 2022-2024 shows a fold-consistent ~59% win rate with a
believable, if execution-fragile, positive expectancy. But three specific, evidenced
problems prevent calling this a robust edge today: (a) the deployed 0.60 threshold has
gone completely silent in honest OOF for the most recent 14 months due to calibration
drift, not a real loss of predictive ranking, so the live-vs-OOF signal-frequency gap is
unexplained and needs to be resolved before trusting current live signals; (b) the sign
of net-of-cost expectancy depends on an accounting choice (idealized barrier fill vs.
bar-close realized fill) that this M1-only dataset cannot resolve; (c) primary accuracy
is declining fold-over-fold, meaning whatever edge exists is being actively arbitraged
away, not stationary. None of these are data-integrity failures (that would be verdict
E) and none show zero signal (that would be D) -- they are exactly the kind of
calibration/execution/decay gaps verdict B is for.

## Implementation order recommendation (item 17)

1. **Probability calibration** (recalibrate/refresh the threshold against a recent
   rolling window rather than a single fixed global constant fit once) -- must go first,
   directly caused finding #2/#3 above, and every other gate downstream (EV gate,
   quantile heads) inherits this same miscalibration if built on top of it uncorrected.
2. **Data integrity correction** (spread/tick_volume source-tagging from the earlier
   masterplan) -- cheap, already evidenced safe (spread: zero importance, free to fix;
   tick_volume: small/unstable, low regret either way), do alongside #1 since both touch
   `core/features.py`/`core/train.py` in the same pass.
3. **EV gate** -- only after #1, since an EV gate computed on a miscalibrated probability
   just launders the same drift into a dollar-shaped number instead of fixing it.
4. Everything else (MAE/MFE engine, quantile prediction, regime modelling,
   microstructure logging) -- hold. No evidence in this audit that any of them would fix
   the #2/#3 problem, which is a calibration/recency issue, not a missing-feature issue.

This audit does not recommend proceeding to implementation yet on its own -- the
live/OOF frequency mismatch (finding #2) should be understood and explained before any
further build work, since it currently means we do not know whether today's live
signals are trustworthy.
