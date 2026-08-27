# GOLDEX V4 — Phase 3 Findings: Discovery-Scale Candidate Research

Branch: `goldex-v4-phase3-discovery-scale`
Run: `research/phase3_real_run.py`, real historical XAUUSD minute data (no mocked evidence), training rows 300,000 / validation rows 100,000.

## Market-flow representation research (Section 4)

- Return autocorrelation, lags 1-5: all small and negative (-0.039 to -0.005) — a weak, decaying mean-reversion signature at the 1-minute bar level, not a strong exploitable trend.
- Volatility clustering, lags 1-5: very high (0.987-0.998) — strong evidence that volatility itself is highly persistent (calm/active regimes cluster in time), even though raw returns barely autocorrelate.
- Regime persistence: mean dwell time ~5.17 bars, with 58,063 regime switches across the sample — regimes are real but short-lived, switching frequently rather than settling into long stable phases.

This says the raw 1-minute return series carries very little linearly exploitable structure on its own, but volatility/regime state is a much richer, persistent signal — consistent with why regime- and volatility-based candidates were included, and a pointer for any future representation work: encode volatility/regime state directly rather than relying on raw return autocorrelation.

## Control gate

Passed. The random-trading baseline (28,613 validation trades) produced a mean per-trade PnL of -0.349 (CI -0.440 to -0.258) — safely unprofitable, as required for the gate to be meaningful. No candidate is being validated against a broken or falsely-profitable baseline.

## Candidate results (validation set, real data)

| Candidate | Type | Verdict | Validation trades | Validation total PnL | Mean PnL/trade (95% CI) |
|---|---|---|---|---|---|
| control_no_trade | control | CONTROL | 0 | 0.0000 | 0.0 |
| control_random | control | CONTROL | 28,613 | -10000.0000 | -0.349 (-0.440, -0.258) |
| statistical_null_mean_reversion | rule-based | REJECT | 6,239 | -10000.0000 | -1.603 (-2.295, -0.921) |
| regime_conditioned_momentum | regime-statistical | REJECT | 1,789 | -9997.8647 | -5.589 (-9.377, -2.227) |
| simple_learned_linear | learned-linear | REJECT | 119 | -9939.6722 | -83.527 (-193.549, 16.592) |
| tabular_qlearning | tabular-rl | REJECT | 4,017 | -9999.9992 | -2.489 (-3.876, -1.130) |
| bayesian_online | bayesian-online | REJECT | 0 | 0.0000 | 0.0 |
| hmm_regime | regime-generative | REJECT | 7,207 | -10000.0000 | -1.388 (-1.906, -0.903) |
| sequence_history | sequence-history | REJECT | 0 | 0.0000 | 0.0 |

Every non-control candidate was rejected on real historical validation data. Three candidates (`bayesian_online`, `sequence_history`, and `hmm_regime` on training) never opened a validation-set position at all — they learned to abstain rather than lose money, which is itself informative (they detected no exploitable setup rather than forcing trades into noise). The candidates that did trade all lost money with confidence intervals that exclude zero in most cases (except `simple_learned_linear`, whose CI straddles zero but is wide and built on only 119 trades — too few to draw any conclusion from).

## Summary

No discovery-scale candidate — rule-based, regime-statistical, learned-linear, tabular-RL, Bayesian-online, HMM-regime, or sequence-history — showed a robust edge over real historical Gold price data at this scale and representation. This is a complete, honest negative result: the added sequential/learning machinery (the `learn()` hook and the four new candidates) did not surface a profitable strategy, and the market-flow research suggests why — raw return autocorrelation is too weak to exploit directly, while the persistent signal that does exist (volatility/regime clustering) was not enough on its own to produce a profitable trading rule in the forms tested here. Per the phase's own mandate, this negative result is preserved as-is rather than iterated on until something looks profitable. Phase 3 concludes with no new candidate promoted past control-gate rejection.
