# Jane Street Tier Gold AI System — Master Plan
## Last updated: 2026-08-12 17:30 IST

---

## EXECUTIVE SUMMARY

**Goal:** $100 → $1M in 4 months via compounding XAUUSD signals.
**Architecture:** LightGBM ensemble + regime router + 3-layer quality gate.
**Status:** M5 walk-forward DONE (96 features). Tick + macro data READY. Retrain needed.
**Target:** Retrain overnight, signals live by tomorrow morning.

---

## PHASE 1: TICK MATRIX BUILD (NOW — 30 min)

### What exists
- `gold_features_m5.csv`: 32.5M rows × 119 cols (M5 bars, 2019-12 to 2026-08)
- `dukascopy_m1_features.csv`: 702K rows × 10 cols (M5 aggregates from 2.4M M1 bars)
- `macro_daily.csv`: 2479 rows × 8 cols (daily macro features)

### What to do
1. Merge dukascopy_m1_features.csv into gold_features_m5.csv on `time`
2. Merge macro_daily.csv into gold_features_m5.csv on bar DATE
3. Output: gold_features_m5_tick.csv (enriched matrix)

### Features added
| Feature | Source | What it captures |
|---|---|---|
| dk_ticks | Dukascopy M1 | Activity proxy (M1 bars per M5) |
| dk_delta | Dukascopy M1 | Signed flow: (up - dn) / (up + dn) |
| dk_cvd | Dukascopy M1 | Cumulative delta (decaying memory) |
| dk_vol_rel | Dukascopy M1 | Volume burst: window / rolling mean |
| dk_spread_z | Dukascopy M1 | Spread z-score (liquidity stress) |
| dxy_z | Yahoo daily | USD strength z-score |
| dxy_5d_chg | Yahoo daily | USD 5-day momentum |
| tnx_level | Yahoo daily | 10Y yield level |
| tnx_5d_chg | Yahoo daily | Yield 5-day momentum |
| gc_5d_chg | Yahoo daily | Gold futures 5-day momentum |
| gld_5d_chg | Yahoo daily | GLD ETF 5-day momentum |
| eur_5d_chg | Yahoo daily | EUR/USD 5-day momentum |

---

## PHASE 2: RETRAIN (overnight — 8-10 hours)

### Walk-forward configuration
- Matrix: gold_features_m5_tick.csv (enriched with tick + macro)
- Features: ~106 (96 base + 10 tick/macro)
- Seeds: 42, 7, 2026
- Trees: 600 per seed
- Target: direction (up/down)
- Calibration: isotonic regression on OOF probs
- Signal rating: decile-based quality gate
- Regime prior: direction ensemble per regime

### What gets rebuilt
1. Walk-forward OOF predictions (32.5M rows)
2. 3 final models (s42, s7, s2026)
3. calibration.json (88 knots)
4. calibration_by_drr_*.json (8 regime-specific calibrations)
5. signal_rating.json (decile thresholds)
6. regime_dir_prior.json (P(up|regime))
7. features.json (updated feature list)

---

## PHASE 3: JANE STREET TIER UPGRADES (post-retrain)

### P0: Live PnL monitoring + drift detection
**Why:** The 5 SL/0 TP streak was caught by NO sensor. PSI catches 70% of degradations 1-5 days before P&L breaks.

**Components:**
1. PSI (Population Stability Index) per feature — fires if > 0.25
2. CUSUM on prediction residual — fires when cumulative drift too large
3. EWMA PnL control chart — fires when realized PnL crosses lower control limit
4. Champion/challenger A/B — head-to-head after 100 trades

**Combined rule:**
- Any 1 of 3 fires → HALT + alert
- Any 2 of 3 fires → HALT + force-retrain
- All 3 fires → HALT + manual review

**Build time:** 1 day, ~300 LoC

### P1: Cost-aware entry gating
**Why:** Even a good signal is a loser if it fires into a 3× spread.

**Components:**
1. spread-z gate: don't fire when spread > 2σ above mean
2. Pre-news kill: suppress 5 min before FOMC/NFP/CPI
3. Slippage surcharge: subtract estimated slippage from expectancy

**Build time:** 0.5 day

### P2: Purged walk-forward + deflated Sharpe
**Why:** Catches snooping bugs that already shipped.

**Components:**
1. Purged K-fold (no overlap between train/test)
2. Deflated Sharpe ratio (Bailey-Lopez de Prado)
3. Post-condition assertions on EOD loop

**Build time:** 1.5 days

### P3: Vol-targeting sizing overlay
**Why:** Most validated risk overlay in literature (Moreira-Muir 2016).

**Components:**
1. EWMA volatility estimate (λ=0.94)
2. Target volatility = 15% annualized
3. Position size = target_vol / realized_vol

**Build time:** 0.5 day

### P4: XAUUSD/XAGUSD pairs leg
**Why:** Independent P&L stream, diversification.

**Components:**
1. Cointegration test (Engle-Granger)
2. Half-life of mean reversion
3. Z-score entry/exit
4. Monthly re-test

**Build time:** 1.5 days

---

## PHASE 4: ENGINE RESTART (tomorrow morning)

### Pre-flight checklist
- [ ] features.json updated with tick + macro features
- [ ] 3 final models trained on enriched matrix
- [ ] calibration.json fitted on enriched OOF
- [ ] signal_rating.json updated
- [ ] regime_dir_prior.json updated
- [ ] Live PnL monitor active
- [ ] Engine watchdog active
- [ ] EOD retrain cron active

### Engine settings
- Poll interval: 5s (xm_ticker.py at 25ms)
- Signal cooldown: 30s minimum
- Same-idea suppression: active
- Rating gate: ≥ 20 (learned threshold)
- EV gate: > 0
- Regime routing: active (8 regimes)

---

## JANE STREET IDEOLOGY — WHAT WE'RE BUILDING

### What Jane Street has (buildable at retail)
1. ✅ Live PnL monitoring + auto-halt (Build #1)
2. ✅ Cost-aware entry gating (Build #2)
3. ✅ Purged walk-forward + deflated Sharpe (Build #3)
4. ✅ Vol-targeting sizing (Build #4)
5. ✅ Adverse-selection awareness via effective/realized spread (tick data)

### What Jane Street has (NOT buildable at retail)
1. ❌ Co-located exchange feeds (sub-microsecond)
2. ❌ $10B+ working capital
3. ❌ Dark pool access
4. ❌ 100+ PhD research desk

### What we have that Jane Street doesn't
1. ✅ Regime-specialist ensemble (8 regimes, separate calibrations)
2. ✅ Live tick microstructure (25ms from xm_ticker.py)
3. ✅ Cross-asset macro context (DXY, yields, gold futures, GLD, EUR)
4. ✅ Real event calendar (387 events, DST-aware)
5. ✅ Closed-loop learning (outcome rows feed next retrain)

---

## RISK MANAGEMENT

### Position sizing
- Risk per trade: 1-2% of account (Kelly-inspired)
- Max daily loss: 5% (auto-halt)
- Max drawdown: 15% (force retrain)

### Stop-loss
- Dynamic: based on ATR (1.5-2.5× ATR)
- Time-based: close if no movement after 30 min
- Regime-based: tighter in ranging, wider in trending

### Take-profit
- Dynamic: based on ATR (2-4× ATR)
- Partial: scale out at 1:1, trail rest
- Regime-based: wider in trending, tighter in ranging

---

## SUCCESS METRICS

### Daily
- Win rate: > 50% (current: 43.9% best decile)
- Average R:R: > 1.5 (current: 1.4)
- Sharpe ratio: > 2.0 (daily)
- Max drawdown: < 5% daily

### Weekly
- Net PnL: positive every week
- Win rate: > 55%
- Sharpe ratio: > 2.5
- No more than 3 consecutive losses

### Monthly
- Compounding: 5-8% daily = 5-7x monthly
- Target: $100 → $500 in month 1
- Target: $500 → $2500 in month 2
- Target: $2500 → $12500 in month 3
- Target: $12500 → $62500 in month 4

---

## TIMELINE

| Day | Task | Status |
|---|---|---|
| Aug 12 | Dukascopy fetch + tick matrix build | ✅ DONE |
| Aug 12 | Macro injection wired | ✅ DONE |
| Aug 12 | Event calendar fixed | ✅ DONE |
| Aug 12 | Orchestrator prereq fixed | ✅ DONE |
| Aug 12 | Research completed | ✅ DONE |
| Aug 12-13 | Retrain (overnight) | ⏳ STARTING |
| Aug 13 | Engine restart + live signals | ⏳ PENDING |
| Aug 13-14 | P0: Live PnL monitor | ⏳ PENDING |
| Aug 14-15 | P1: Cost-aware gating | ⏳ PENDING |
| Aug 15-17 | P2: Purged walk-forward | ⏳ PENDING |
| Aug 17-18 | P3: Vol-targeting | ⏳ PENDING |
| Aug 18-20 | P4: Pairs leg | ⏳ PENDING |

---

## CITATIONS

1. Cont, Kukanov, Stoikov (2014) — "The Price Impact of Order Book Events" — RFS, 1221 citations
2. Easley, Lopez de Prado, O'Hara (2011) — "The Microstructure of the Flash Crash" — JPM, 406 citations
3. Lee, Ready (1991) — "Inferring Trade Direction from Informed Trades" — JF, 2000+ citations
4. Moreira, Muir (2016) — "Volatility-Managed Portfolios" — JF, 600+ citations
5. Bailey, Lopez de Prado (2014) — "The Deflated Sharpe Ratio" — JRFM
6. Lopez de Prado (2018) — "Advances in Financial ML" — book, foundational
7. Roll (1984) — "A Simple Implicit Measure of the Effective Bid-Ask Spread" — JF, 3000+ citations
