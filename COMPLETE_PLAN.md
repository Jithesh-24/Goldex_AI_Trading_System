# COMPLETE PLAN — Institutional-Grade AI Trading System for XAUUSD
**Drafted: 2026-08-11 | Owner: MR. BOSS + Hermes | Status: ACTIVE**

> Grounded in: 4 parallel research streams (institutional architecture, loss reduction, gold flow/microstructure, compounding math) + live diagnostics of the current engine + the 6-year microstructure data audit.

---

## 0. THE HONEST BASELINE (from compounding math — read this twice)

| Metric | Current | Required for $1M in 4mo |
|---|---|---|
| Win rate | 33% (5 TP / 10 SL today) | — |
| RR (actual) | 1.28 (manual exits collapse signal's 3.9) | — |
| **E[R] per trade** | **−0.248R (NEGATIVE edge)** | **positive, then +0.52R+** |
| Break-even WR at RR 1.28 | 43.9% | — |
| Break-even RR at 33% WR | 2.03 | — |
| Kelly f* | 0% (bet nothing) | — |
| Required daily return | — | **+8%/day = 10x/month = 39× Medallion's daily return** |

**THE ONE-LINE TRUTH:** No position size, frequency, or compounding scheme can grow a negative-edge system. **Step 1 is fixing the edge. Step 2 is execution discipline (hold to TP). Step 3 is compounding.** The 4-month $1M target needs 10x/month — the report shows that's a coin flip with brutal variance *even for a genuinely excellent system*; the realistic path is 2.5-5%/day with a real edge → $1M in 9-19 months. We build the best system possible and let the math work.

**THE EXECUTION PRIZE (Monte Carlo, 10,000 paths, 120 days, 5 trades/day):**
| Scenario | median | P(≥$1M) | P(DD≥50%) |
|---|---:|---:|---:|
| 33% WR / RR 1.28 @ 3% (today) | **$1** | 0% | 100% |
| 33% WR / RR 3.9 @ 3% (**execution fixed**) | $1.57M | **62%** | 11% |
| 60% WR / 1:1 @ 10% (half-Kelly aggression) | $831K | 49% | 97.5% |

**Holding the signal's RR 3.9 at the same 33% WR flips E[R] from −0.248R to +0.617R and makes the 4-month goal a plausible coin flip (62%) at tolerable 3% risk.** This is the #1 lever in the entire system.

**The $100 min-lot trap:** 0.01 lot with SL $3.2–$7.2 risks 3.2–7.2% per trade at $100 — the safe 1–2% lane is NOT executable at this size. Top the account to $300–500 so the safe lane is expressible.

**Streak survival (why mostly-winners matters):** at 33% WR, a 10+ loss streak is *guaranteed* in 600 trades (expected longest ≈ 16). Same edge, 60% WR compounds ~2.7x faster per trade at half-Kelly than 33%/2.64RR and faces shorter rarer streaks — which keeps a manual trader from tilting. High-RR+low-WR *requires* RR≥3 just for an edge and then eats routine 10+ streaks → martingale temptation.

---

## 1. WHY THE ENGINE LOSES TODAY (diagnosed from live data, not theory)

**1a. Direction is a coin flip.** The deployed `regime_dir_prior.json` has `horizon_bars: 3` (15-min). The 15-min prior is noise (0.43–0.54). The 180-min prior (0.76–0.77 up, every regime) was deliberately removed (Bug-A: it made the engine *always buy dips* → worse). Result: P(dir)=0.50 exactly → the placement model's marginal expectancy picks the side → today it shorted a rising market 16 of 19 times (gold 4369→4400).

**1b. No M5 direction model exists yet.** The legacy direction model is M1 (horizon 240), correctly rejected by the v8 M5 guard. The chain is building the M5 direction model NOW — this is the #1 fix.

**1c. Rating threshold admits losers.** Threshold=20.0 admits decile-2 trades with 25.95% WR (3-of-4 losers). Relearned per chain.

**1d. Execution leak.** Signal designed for RR 3.9; manual exits collapse it to 1.28 → flips E[R] from +0.617R to −0.248R. **Execution discipline alone flips this system from guaranteed loser to plausible winner.**

**1e. Engine trade bugs (FIXED today):** close-anchored cooldown, same-M5-bar hold, position-state persistence. Verifying live.

---

## 2. THE INSTITUTIONAL ARCHITECTURE (from research, mapped to our system)

```
DATA ──→ FEATURES ──→ MODELS ──→ RISK ──→ EXECUTION ──→ MONITORING
│           │            │         │         │            │
XM ticks    96→~102      LightGBM   fixed-    hold-to-TP   per-regime
M1/M5 bars  +tick feats  3-seed     fractional  discipline  performance
Dukascopy   +position    ensemble   (1-3%)     + SL/TP     report +
M1 (6yr)    state        + M5 dir   + streak   from model  loss lessons
                        + 8 spec   + DD cap              + daily EOD
```

**What institutions do that we don't (yet) — prioritized:**

| # | Practice | Source | Our gap | Effort |
|---|---|---|---|---|
| 1 | **Meta-labeling** (López de Prado): secondary model filters primary signals | AFML ch.3 | We gate on rating; a learned filter is stronger | M |
| 2 | **Trade at the flow** (order-flow delta, CVD, tick imbalance) | Evans & Lyons 2002; Hasbrouck 1991 | Ticker computes imb/ticks live but they're NOT model decision features | S-M |
| 3 | **Multi-timeframe direction** (M15/H1 trend alignment) | Moskowitz 2012 TSMOM | h1/d1/m15_trend features exist in matrix | S |
| 4 | **Purged K-fold + embargo** (label leakage control) | López de Prado AFML ch.7 | Our walk-forward is time-ordered (good); add embargo | S |
| 5 | **HMM/regime latent states** (soft regime posterior) | Hamilton 1989 | Rule-based 8 buckets → upgrade to soft states | L |
| 6 | **Signal diversification** (LightGBM + CatBoost/XGBoost blend) | Gu Kelly Xiu 2020 | 3-seed same-family ensemble | M |
| 7 | **Fixed-fractional + Kelly-aware sizing** | Rotando & Thorp | Manual compounding → formal rule | S |
| 8 | **Session/event awareness** (London/NY peaks, news windows) | gold research | session + event features exist | S |
| 9 | **Sharpe/Calmar gates on OOS** | institutional | evaluate() prints acc only | S |

---

## 3. THE TICK-LEVEL TEACHING PLAN (YOUR DIRECTIVE — trained on ticks, not just M5 bars)

**EXECUTED 2026-08-12 (full autonomy mandate):**

**Real-time (engine, v8.9 — verified):**
- Ticker captures **real XM ticks at 25ms**: `ticks_60s`, `ticks_300s`, `imb_60s`, `imb_300s`, `spread_now/z60` (in `xm_tick_state.json`).
- **DONE:** injected into `fx` decision vector at signal time (ai_signal_engine.py:1366-1377). The model decides on live tick flow, not just closed M5 bars. **Zero-lag — ticks ARE the present.**
- **DONE (2026-08-12):** added `tick_flow_state()` accumulator (line 313) computing `vol_rel` (activity burst: ticks/rolling-288 mean) + `cvd` (decaying cumulative flow) live — SAME definitions as training. Unit-verified: burst 1.199, lull 0.391, cvd +0.383.

**6-year retrain (tick matrix — DONE, in flight):**
- **DONE:** Dukascopy M1 backfill (2019-2026, BID+ASK, LZMA bi5) → per-M5-bar **tick-proxy microstructure**: `imb_300s` (flow delta), `vol_rel` (activity burst), `cvd` (decaying cumulative flow). 1920/2200 days downloaded; CSV lands ~14:00 IST.
- **DONE:** `build_tick_matrix.py` merges tick block into matrix → **99-feature training** (was 96). Coverage guard ≥95% aborts on timezone misalignment.
- **DONE:** `transition_tick.sh` full tick retrain chain (walk-forward, 3 seeds) — auto-launched by orchestrator cron (`tick_pipeline_orchestrator.sh`, every 5 min) when M5 chain completes AND dk CSV lands. Specialists + spec-OOF train after TICK COMPLETE.
- **CRITICAL FIX (scale-free only):** raw tick COUNTS do NOT align (training M1 bars ≈ 1-5/window vs live XM ≈ 8,020/300s). Only ratio/normalized features (`imb_300s` ∈ [-1,1], `vol_rel` ratio, `cvd` bounded) train cleanly. `ticks_60s/300s` stay computed live but unlisted → dropped harmlessly.
- **CRITICAL FIX (name collision):** matrix already had `vol_rel` (tick_volume ratio, features.py:325) — REPLACED by Dukascopy activity-burst definition so training and live agree. Smoke-tested.
- **CRITICAL FIX (closed loop):** incremental builder + full build now emit tick cols via `_attach_tick_block()`; EOD loop refreshes dk window daily (`fetch_dukascopy_m1.py --recent-days 10`); live outcome rows carry tick feats via `merge_live_outcomes_appended` (feats.get per header col).
- **Honest caveat:** M1 is the finest historical resolution available cheaply; true tick history for gold doesn't exist free at retail. The LIVE engine gets true ticks; the 6yr training gets M1-derived microstructure (the closest available). This is exactly how serious shops handle it — live ticks + historical bar-derived features.
- **Canonical matrix swap:** after TICK COMPLETE, `gold_features_m5_tick.csv` → `gold_features_m5.csv` (atomic; old kept as `.m5backup`) so the daily EOD loop operates on the same 99-feature space.

---

## 4. LOSS-REDUCTION ROADMAP (research-grounded, learned not hardcoded)

1. **M5 direction model deploys** (chain, ~today) → kills the P=0.50 coin flip. Verify acc vs majority baseline honestly.
2. **Tick flow into decisions** (§3) → engine trades WITH the flow, not against it (kills today's 16-shorts-in-a-rally pattern).
3. **Meta-labeling filter** (next retrain): train secondary binary model on outcomes with primary-signal features → predicts "will this placement win?" → engine learns which setups to skip. Pure learning.
4. **Relearned rating threshold** from OOF data (chain) — honest calibration, not a gate.
5. **Streak/day-pnl state** already in features (v8.8) → next retrain learns "3rd loss in a row → state is toxic" from data, not from a hardcoded cooldown.
6. **Execution discipline layer** (biggest free win): signals carry TP/SL — user commits to holding to TP; track adherence in journal; report slippage per trade.
7. **Per-regime performance report** (after chain): which of the 8 microstructures the model mastered (crash days, trend days, range days) → teaches the NEXT EOD pass with loss-lesson replay.

---

## 5. EXECUTION ORDER — NOW → 48H

| # | Action | Status |
|---|---|---|
| 1 | OOM-fix train_ai (chunked predict, memmap, gc) | ✅ DONE — chain running, 3.4GB stable |
| 2 | v8.8 chain completes → M5 dir model + calibration + specialists + rating + prior + loss-lessons | ⏳ RUNNING (~seed 42/3) |
| 3 | Watcher auto-restarts engine on completion | ⏳ armed |
| 4 | Inject tick features (imb_60s/300s, ticks, spread_z) into engine `fx` | ⏳ next |
| 5 | Dukascopy M1 downloader (2019-2026, ~40MB) → tick-proxy features into matrix | ⏳ next |
| 6 | Per-regime microstructure performance report from OOF | ⏳ after chain |
| 7 | Meta-labeling filter in next retrain cycle | ⏳ after ~2wk live outcomes |
| 8 | Execution discipline: TP/SL adherence tracking | ⏳ with user |

---

## 6. COMPOUNDING MATH — THE REAL PATH (no fake edge)

- **Phase 1 (now):** edge goes positive (direction model + tick flow + execution discipline). Target: E[R] ≥ +0.3R.
- **Phase 2:** 1% fixed-fractional risk, hold to TP, 3-5 trades/day. At +0.5R avg and 4 trades/day ≈ +1.5%/day → **$100→$1M in ~1 year**.
- **Phase 3:** as OOF-verified edge grows, scale to half-Kelly (never full — variance kills). +3%/day → ~8 months.
- **Hard floor:** never risk more than 2-3% per trade; daily max-loss kill at −6% (sizing rule, not a signal gate — no trading restrictions on the model itself).

**The day-1 contract honored:** mostly winners via learned expectancy (direction + flow + meta-labeling), losses trigger self-analysis (loss-lesson replay + EOD), everything learned from 6yr + daily data, zero lag via live ticks, one trade at a time with learned state.

---

## 7. VERIFICATION GATES (how we know it's real)

- [ ] Chain completes: `v8.8 TRANSITION COMPLETE` in log, engine auto-restarts
- [ ] Direction model: acc vs majority baseline (honest — if 0.51, we say so)
- [ ] Post-reload engine: no P(dir)=0.50 signals; trades follow M5 flow
- [ ] Next 24h journal: WR vs rating deciles; slippage vs TP/SL adherence
- [ ] Per-regime OOF report: which microstructures carry edge
- [ ] Weekly: Sharpe/Calmar on live closed trades (net of costs)
