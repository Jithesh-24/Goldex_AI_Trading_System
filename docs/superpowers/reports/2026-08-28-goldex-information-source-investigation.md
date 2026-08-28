# GOLDEX — Information Source Investigation

STATUS: RESEARCH ONLY. No code changed. No new model. No architecture chosen.

## 1. Executive conclusion

27/27 hypotheses null on M1 OHLCV is not proof Gold has no edge — it is proof that **the information available in the current dataset, once trend-confound is controlled, does not distinguish direction/magnitude at any tested horizon (1–120 bars)**. The dataset itself has a specific, nameable blind spot: it destroys intrabar order-flow/path information and has no bid/ask, no cross-market context, and no event-time labeling. Literature review found real (if unevenly strong) evidence that some of what M1 destroys carries genuine short-horizon predictive content elsewhere (order-flow imbalance in LOB/futures/crypto microstructure; a real ~1-minute gold→silver lead; large, mechanical event-time volatility). None of this is proven for *retail-latency spot XAUUSD specifically* — it is analogy, not verified fact for this instrument.

**Verdict: C — current data may be insufficient, but this is unproven; the cheapest decisive test is direct (Dukascopy tick/bid-ask reconstruction), not more M1 feature engineering.** See §17–18.

## 2. What GOLDEX currently has

Inventoried directly from `data/gold_seed_merged_full6yr.csv` (2,456,224 rows, 2019-12-02 to 2026-08-07, ~6.7 years, M1 bars):

| Field | Resolution | Notes |
|---|---|---|
| time | 1 minute | bar-open timestamp (bar-close construction confirmed via `market_state_builder.py` in earlier phases) |
| open/high/low/close | 1 minute | mid-price OHLC only — no separate bid/ask OHLC |
| tick_volume | 1 minute | count of price updates in the bar, NOT trade volume, NOT order-flow direction |
| spread | 1 minute | snapshot value, not a spread time-series within the bar; **20.0 in 98.9% of bars** (min 20, max 194, mean 20.05, std 0.82) — effectively a near-constant except during volatility spikes |
| real_volume | 1 minute | differs from tick_volume in 16.8% of rows — semantics of the divergence undocumented by the broker feed; do not assume this is genuine traded volume |
| missing data | none found | zero NaNs across all 8 columns |
| bid/ask | **absent** | no columns at all |
| tick-by-tick | **absent** | only 1-minute aggregates |

No look-ahead risk in this audit — this is static file inspection, not a change to any pipeline.

## 3. What M1 destroys

**Concrete same-candle example.** Consider open=1900.00, high=1901.00, low=1899.50, close=1900.30. At least these distinct intrabar paths produce this identical candle:
- (a) up to 1901.00 immediately, drift down to 1899.50, recover to 1900.30 (one full round trip, high hit first)
- (b) down to 1899.50 immediately, rally straight to 1901.00, pull back to 1900.30 (low hit first — opposite sequencing)
- (c) oscillate 1899.50↔1901.00 six times before settling at 1900.30 (high tick-arrival intensity, most information)
- (d) a single monotonic run from 1900.00 to 1901.00 to 1899.50 to 1900.30 with long pauses between (sparse, low-intensity)

All four are indistinguishable in the OHLC record. Lost, specifically:
- **Sequencing** (did high or low come first — directly relevant to whether the bar was net-buying-then-selling or the reverse)
- **Velocity/acceleration** within the bar
- **Reversal count** within the minute
- **Spread evolution** (did the spread widen before or after the move — the current data has one spread value per bar, not per-tick)
- **Tick arrival intensity** (tick_volume gives a count, not a rate profile)
- **Bid/ask asymmetry** (no bid/ask columns exist at all, at any resolution)

This is **potentially available information**, not proven predictive information — see §4/§11.

## 4. Tick/quote data analysis

External research findings (WebSearch, see full citations below):

- **Dukascopy**: free historical tick data (bid/ask + bid/ask volume) for XAUUSD back to ~2003–2005. Good depth, zero cost. But Dukascopy is a bank/ECN feed — its spread regime, requote behavior, and latency profile differ from XM's retail market-maker feed. Training on Dukascopy and deploying on XM risks a real sim-to-live mismatch that has not been quantified in available sources (flagged as speculative but mechanistically plausible: different liquidity providers = different spread-widening behavior around news, which is exactly the regime where intrabar information would matter most).
- **XM/MT5 History Center**: the most live-realistic source (it's literally the target broker), but broker-side tick history is commonly incomplete — one practitioner source suggests XM gold tick history may only extend to ~2022, a fraction of the current 6.7-year M1 dataset. Confidence: moderate, sourced from forums not vendor docs.
- **Paid vendors** (TickData.com, FirstRateData): research-grade bid/ask tick data exists commercially; cost/access not itemized here, would require a direct vendor quote before any acquisition decision.

**Verdict per source**: Dukascopy is the correct choice for a *first decisive test* (free, sufficient depth, causally usable — timestamps are real arrival times, no look-ahead risk if handled correctly) despite the live-mismatch caveat, because the question right now is "does intrabar information carry signal at all," not "is XM's specific feed reproducible." Buying paid tick data or extracting a decade of XM history is not justified before that cheaper test answers the prior question.

## 5. Bid/ask analysis

No column in current data. Literature: order-flow imbalance (bid/ask-derived) has a documented near-linear relationship to short-horizon price change in limit-order-book and futures/crypto microstructure studies (CSI300 futures, crypto LOBs). This evidence is **not** from retail-latency, non-colocated, OTC market-maker-quoted instruments like spot gold — it is the strongest available analogy, not a proof that transfers directly. One directly relevant paper (arXiv 2605.04004, "Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures") argues OHLCV alone is structurally information-poor for intraday prediction, which is consistent with — not independent confirmation of — GOLDEX's own 27-null result.

## 6. Cross-market analysis

| Candidate | New information? | Available at required latency? | Causal at short horizon? | Live-reproducible? | Worth the complexity? |
|---|---|---|---|---|---|
| DXY | Yes (dollar strength) | Yes (traded continuously) | Weak evidence at minute horizon — mostly event-window driven, intraday lead-lag is thin/noisy per available sources | Yes | Not yet — evidence for minute-horizon *lead* (not just correlation) is thin |
| Silver | Yes (a real, sourced ~1-minute Granger lead: gold leads silver) | Yes | This is gold→silver, i.e. informative about silver, not new information *into* gold | Yes | No — doesn't help gold prediction, wrong direction |
| Treasury yields | Yes (real-rate proxy) | Partial — less liquid intraday, session-hour mismatch (bond market hours ≠ 24hr gold) | Alignment/liquidity mismatch risk explicitly flagged in literature | Partial, with lag/alignment risk | Not yet — look-ahead/alignment risk needs a dedicated methodology before use |
| VIX | Marginal — mechanism mostly documented for equity/index futures, not gold specifically | Yes | Low confidence, largely extrapolated | Yes | No — insufficient direct evidence |
| Equity indices | Marginal, similar to VIX | Yes | Low confidence | Yes | No |

None of these clear the bar for immediate inclusion. DXY is the only one worth a dedicated MI-vs-null test (same methodology as the horizon sweep) before committing effort, and only after same-timestamp alignment is proven leak-free.

## 7. Macro/event analysis

Strongest, most rigorously-sourced finding in this whole investigation: scheduled macro releases (CPI, NFP, FOMC) produce large, mechanical, well-documented volatility spikes at fixed intraday times (canonical academic result: Andersen & Bollerslev, Journal of Finance-line FX high-frequency literature). Practitioner sources report 300–1000+ pip XAUUSD moves around these releases.

Implication: event-time and non-event-time gold dynamics are plausibly **two different regimes mixed together** in every one of the 27 prior null tests. A model (or MI probe) run across both regimes without conditioning on event-time could be diluting genuine event-time signal into noise, or attributing event-noise variance to "no signal" when the two regimes were never separated. This is a real, previously-unexamined confound in the horizon sweep and everything before it.

Recommendation: represent event-time as **explicit context** (a flag/countdown feature), not implicit and not as a hardcoded "news = buy/sell" rule. Whether it should be avoided (no-trade window) or exploited is a decision for later architecture work, not this investigation.

## 8. Quantitative knowledge classification

- **Black-Scholes**: no legitimate direct role. It prices options; GOLDEX trades spot XAUUSD with no options exposure. Its volatility-surface machinery has no natural input or output here. Prior Phase 4 design work already reached this conclusion; this investigation finds nothing to overturn it. Classified: **theoretical/background only**.
- **GARCH/Kalman/skew-kurtosis mechanisms** (already implemented, Phase 4): classified **directly useful with current information** as measurement tools (they were correctly computed and correctly tested null) but **not sources of new information** — they are transformations of the same M1 closes already in hand. This is exactly the §11 distinction below.
- **Order-flow imbalance, bid/ask spread dynamics**: classified **potentially useful with additional information** — cannot be computed or tested at all without tick/bid-ask data.
- **Position sizing / risk mechanics** (Kelly-style, volatility-scaled sizing): classified **risk/execution tools**, orthogonal to this investigation.

## 9. Information hierarchy

| Level | Content | Incremental info vs. Level below | Realistic availability | Live-compatible |
|---|---|---|---|---|
| 0 | Current M1 OHLCV | — (baseline) | Have it, 6.7yr | Yes |
| 1 | Tick/bid-ask/spread-evolution | Sequencing, velocity, spread dynamics, order-flow proxy | Dukascopy (free, ~20yr, ECN-not-XM); XM MT5 history (shallow, incomplete) | Requires live tick feed from XM — needs verification XM's MT5 API delivers this in real time, not yet confirmed |
| 2 | Cross-market (DXY primarily; silver/yields/VIX weakly supported) | Dollar-strength context; alignment risk is real cost | Yes, but latency/session mismatch must be solved first | Yes for DXY; yields/VIX marginal |
| 3 | Macro/event calendar | Regime separation (event vs. non-event) | Yes — economic calendars are free and reliable | Yes |
| 4 | Undiscovered | Unknown | — | — |

Level 3 (macro/event) is the cheapest to add and has the strongest evidence. Level 1 (tick/bid-ask) has the largest plausible incremental information but is the most expensive and carries a live-reproducibility question that must be answered before relying on it.

## 10. Scalping feasibility

M1 data supports minute-to-tens-of-minutes holding periods at best — it structurally cannot support genuine sub-minute (true scalping/millisecond) decisions since intrabar sequencing is destroyed. Tick/bid-ask data is a prerequisite (not sufficient on its own) for anything faster than ~1-minute reassessment; it does not by itself guarantee millisecond-scalping is viable, since retail (non-colocated) execution latency, XM's own order-processing latency, and realistic slippage still bound the achievable holding-time floor — none of which have been measured here. Realistic near-term lower bound to *research* with available/acquirable data: single-digit minutes, not seconds.

## 11. Information vs. features — explicit boundary

Already-tested-and-null: GARCH conditional variance, Kalman velocity/innovation, rolling skew/kurtosis, momentum scalar, path-PCA, multiscale volatility ratio, regime-transition — **all of these are mathematical transformations of the same M1 close series**. They are features, not new information sources, regardless of their apparent sophistication. This is why 27/27 came back null: GOLDEX has been re-deriving different views of one information source, not adding a second one.

Genuinely new information sources, by this test: bid/ask tick sequence (not derivable from M1 closes), DXY/cross-market series (a different underlying market), the macro calendar (external, not derivable from price at all). Silver's ~1-minute Granger lag on gold is *not* new information for predicting gold — it's gold's own past information showing up in a different, correlated instrument.

## 12. Simulator implications (no simulator code touched — requirements only)

To represent a higher-frequency environment the Phase 1 simulator would eventually need: tick-level replay ordering, a bid/ask spread series (not the current single-value-per-bar), a defined execution-latency model, a slippage model conditioned on spread/volatility state, and explicit handling of intrabar order timing (fill against bid or ask, not the current mid-close). None of this is being built now — this is a requirements note for a future SDD cycle, contingent on the acquisition decision in §17.

## 13. XM/live implications

Critical constraint: **do not train on information the live system cannot receive.** Dukascopy tick data, if used for research, must be treated strictly as a research/feasibility probe — never as the basis for parameters or thresholds that would be deployed live against XM's feed, since XM's own tick/bid-ask stream (via its MT5 API) has not been verified to exist, nor its historical depth confirmed, in this investigation. Before any acquisition beyond Dukascopy, the live-availability question ("can XM's MT5 API stream real-time bid/ask/tick to a running system") must be answered directly — this is a factual question about XM, not a research question, and should be checked before further investment in this direction.

## 14. Data acquisition recommendation

Acquire **Dukascopy free historical tick/bid-ask data for XAUUSD**, for a bounded pilot window (not the full 6.7-year history), sufficient to run the same MI-vs-shuffled-null methodology already validated in Phase 3A/4/Genesis, now on genuinely new information (order-flow imbalance / bid-ask-derived features) instead of M1-close transformations. Do not acquire XM's own tick history, DXY, yield, or VIX series yet — those come after this test, and only if it is positive or the event-time separation (§7, free/cheap) is tried first.

## 15. Strongest argument against this recommendation

The sim-to-live mismatch (Dukascopy ECN feed vs. XM retail market-maker feed) could mean that any signal found in Dukascopy ticks is an artifact of that specific liquidity provider's microstructure and would not exist, or would be inverted, in XM's own feed — making the whole exercise a second confound-hunting expedition rather than a real answer. A rebuttal: this is exactly why the *first* test should be a null-hypothesis MI/OOS check (does order-flow-derived information beat the shuffled null at all, on Dukascopy data), not a live-deployment decision — if it's null even on the friendlier ECN data, XM's noisier retail feed is not going to do better, and the acquisition question is closed cheaply. If it's positive, then and only then does the XM-reproducibility question need a dedicated, separate investigation before any live-inspired conclusion.

## 16. Cheapest decisive next test

Pull a bounded Dukascopy XAUUSD tick/bid-ask sample (e.g., one representative multi-month window, not full history), reconstruct 1-minute-equivalent order-flow-imbalance and spread-evolution features from the raw ticks, and run the exact same `binned_mutual_information` + `mi_with_shuffle_control` + trend-confound-reference methodology already validated three times (Phase 3A, Phase 4, Genesis horizon sweep) against forward returns. This directly answers "does information destroyed by M1 aggregation carry any signal at all," at minimal cost (free data, reused methodology, small time window), before any acquisition of paid data, cross-market series, or XM-specific history.

## 17. Final decision

**C — CURRENT DATA MAY BE SUFFICIENT BUT IS UNPROVEN.**

Neither extreme evidence exists: there is no proof M1 OHLCV is fundamentally too poor to ever support the intended trading intelligence, and there is no proof it is sufficient — 27 nulls only show that transformations of the same information source don't help, not that no source could. The literature-supported candidates that would materially change the problem, ranked by evidence strength: (1) macro/event-time regime separation — strongest evidence, cheapest to add, should probably be tried even before tick data since it requires no new acquisition; (2) tick/bid-ask order-flow information — plausible, real analogy evidence, but unverified for this specific instrument/latency regime; (3) cross-market DXY — weak evidence at the relevant horizon, not yet justified.

## 18. What this does NOT authorize

This investigation does not authorize acquiring tick data, building new features, changing the simulator, or starting any model/architecture work (MoE, RL, optimal stopping, strategy library, or another tournament). It identifies two concrete, cheap next tests (event-time regime separation on existing M1 data; Dukascopy tick MI-vs-null pilot) as candidates for a future SDD cycle. Which one (if either) to pursue is the user's decision.

---

### External research citations (via WebSearch, this investigation)
dukascopy.com/wiki; tickstory.com; dukascopy-node.app; mql5.com forum threads 446933/449767 (XM tick history depth claim, unverified beyond forum-level evidence); tickdata.com; firstratedata.com; waylandz.com (vendor comparison); arXiv:2605.04004 ("Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures: A Systematic Falsification Study"); arXiv:2505.17388 (CSI300 order-flow imbalance); sciencedirect.com/S0378426624001894 (retail order flow); phillipnova.com.sg (DXY-gold correlation); World Gold Council (Goldhub); na-businesspress.com/jaf (VIX/gold/silver/oil relationships); ncbi.nlm.nih.gov/PMC5407636 (intraday precious-metals stylized facts, incl. gold→silver ~1-minute Granger lead); public.econ.duke.edu/~boller (Andersen–Bollerslev high-frequency FX volatility literature); vantagemarkets.com (practitioner XAUUSD news-trading pip-move figures).
