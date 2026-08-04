# XAUUSD AI Trading System — v7.8 Complete System Summary
**Written:** 2026-08-04 (post-restart) | **Base:** /home/jith/.hermes/profiles/trading/scripts/

## 1. What this system IS
A machine-learning trading system for XAUUSD on XM MT5 (demo). Every
decision — WHICH side, WHERE the stop goes, WHERE the target goes, WHEN to
trade — is made by gradient-boosted decision-tree models (LightGBM) that
were **trained on 6 years of gold data (2020-03-20 → 2026-08-04)** and are
**retrained daily** from the system's own live outcomes. It is NOT a set of
hand-written trading rules.

## 2. The models (all LightGBM, all 3-seed ensembles)
| Model | Purpose | Files | Trained on |
|---|---|---|---|
| Global placement | P(win \| market + SL/TP geometry + direction) | gold_lgb_model_s{42,7,2026}.txt | 6.39M rows, all regimes |
| 8 regime specialists | One placement model PER regime (STRONG_UP, UP, DOWN, STRONG_DOWN, RANGE_TIGHT, RANGE_WIDE, HIGH_VOL, QUIET_LOW_VOL) | spec_<regime>_s{42,7,2026}.txt (24 files) + regime_specialists.json | 6yr rows bucketed per regime |
| Direction | P(price up over holding horizon) — "when to trade" | direction_s{42,7,2026}.txt | 6yr seed, price-direction label |
| Calibration | raw prob → honest win prob (isotonic knots) | calibration.json (99 knots) | OOF walk-forward |
| Per-dir×RR calibration | win prob per (side × reward/risk) | calibration_by_rr.json (8 curves) | OOF |
| Regime transition | regime → big-move precursors | regime_transition_s{42,7,2026}.txt | 6yr |

## 3. How a live trade decision works (every tick, ~12s)
1. Ticker (xm_ticker.py, the ONLY MT5 connection) streams ticks + M1 bars.
2. Engine computes 102 features: momentum, volatility, oscillators, price
   shape, structure, volume, regime, institutional levels, HTF context
   (H1/D1), session clock, event proximity, order flow.
3. `regime_bin()` classifies the CURRENT regime → router picks that regime's
   3-seed specialist (falls back to global ensemble if absent).
4. `best_placement()` sweeps 24 candidates (6 SL widths × 4 TP ratios × 2
   directions), computes calibrated P(win) × TP/SL geometry → expectancy per
   $ risked.
5. Direction model multiplies: final_exp(BUY)=Exp×P(up), final_exp(SELL)=
   Exp×(1−P(up)). Gate: if no demonstrated price-direction edge → NEUTRAL 0.5
   (currently neutral — placement decides the side, which is correct).
6. Max-expectancy side wins; trade fires with SL + TP attached. TP ratios
   are always >1.0 (reward > risk by construction). SL/TP levels match the
   backtest label EXACTLY (first-touch "TP before SL").
7. Ticker watches the trade: first-touch verdict (TP hit or SL hit) → outcome
   row with the full feature vector → live_outcomes.jsonl.

## 4. The learning loop (EOD, daily 15:00 IST)
`eod_learning_loop.sh` → `retrain_loop.py`:
1. Merge live outcomes into the seed (real XM bars + today's trades).
2. Rebuild matrix (6yr rally + XM, 8-core GNU sort, float32).
3. Retrain placement ensemble (warm-start, 8 threads).
4. Retrain direction model (price-direction label, 8 threads) — honest gate
   refuses to swap if OOS ≤ 0.53.
5. Recalibrate (isotonic, per-dir×RR).
6. Retrain regime-transition model.
7. Retrain 8 regime specialists (streaming buckets, MemoryMax=5G).
8. Atomic swap → engine hot-reloads (mtime watch, no restart needed).
The model learns from its OWN results every day — losing setups lose
probability mass, winning setups gain it. No hardcoded lesson injection.

## 5. Supervision (who keeps it alive)
- `ai-engine.service` (systemd-run) — the engine, PID-locked, survives gateway
  restarts. Watchdog restarts it on crash/stale within 5 min.
- `xm_ticker.py` under Wine + Xvfb :99 — the only MT5 IPC client.
- Watchdog v2 (every 5m): engine freshness (journal ≤120s), ticker state
  (≤60s), Xvfb socket, MT5 process. Silent when healthy.
- camofox-watchdog, self-heal, disk-monitor, system-health: 5-10m cadence.

## 6. Honest answer: is this "real AI" or "a build-up"?
It is REAL machine learning: the trading decisions are outputs of models
trained on 6 years of data and validated out-of-sample — NOT rules a human
typed. The 8 regime specialists, walk-forward calibration, and daily
self-retraining are genuine ML practice.
It is NOT a crystal ball. Gold's 4-hour price direction is statistically
noise (OOS 0.502 ≈ coin flip, confirmed twice) — so the direction model is
kept honestly NEUTRAL rather than pretending. The edge that exists is in
PLACEMENT: choosing SL/TP levels the market doesn't hit first, with TP>SL,
per regime. That's where 6 years of training actually pays.
The "build-up" (watchdogs, pipelines, calibration) exists to keep the models
honest and the system alive — the decision-making core is learned, not
built-up.

## 7. Current state (post-restart 2026-08-04 ~23:51)
- REGIME ROUTER: 8 specialists loaded and active.
- Live BUY @ 4077.97 | SL 4067.85 | TP 4108.33 restored on restart.
- Engine polling XM every ~12s, spread ~$0.20, 62,822 bars.
- Direction: NEUTRAL (honest gate — placement decides side).
- v7.9 same-idea suppression is MARKET-STATE (not time): re-fires only when
  price moves ≥0.5×ATR or regime flips; exp>0 is the real-signal gate.
- v7.10 INCREMENTAL MATRIX: EOD appends only new bars (~30s) instead of the
  full 6.39M-row rebuild (58 CPU-min, timeout at 3600s). Verified on real data.
- Next EOD learning: tomorrow 15:00 IST (uses incremental path).

## 8. Reliability / supervision
- Engine watchdog (every 5m): engine journal freshness ≤120s, ticker state
  ≤60s, Xvfb :99 socket, MT5 process — restarts each layer; silent when healthy.
- self-heal-loop, system-health, disk-monitor, memory-compressor,
  config-backup: 5-120m cadence, all "ok" on 2026-08-04/05.
- Space guard: bounded seed (~60d), rotates append-journals, alerts >30GB.
- Model swaps are atomic + hot-reloaded on mtime change. PID lock = 1 engine.
- EOD retrain: warm-start continuation (not cold), honest OOS gates refuse
  to deploy a model with no demonstrated edge.

## 9. Model choice (research verdict)
LightGBM (gradient-boosted trees) is the correct model for this structured
tabular OHLCV task — the proven best-in-class, beats deep nets on tabular
finance data with far less data + CPU-only training. XGBoost/CatBoost offer
no provable gain here; neural nets need 1000× more data + GPU. Keeping
LightGBM is the evidence-based choice, not novelty-chasing.
