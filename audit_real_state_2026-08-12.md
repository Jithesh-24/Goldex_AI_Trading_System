# Honest State Audit (2026-08-12, ~14:30 IST)

**Context:** user asked "audit everything, are we on the right track to be Jane Street tier, what are we missing?" This file documents REAL disk/process state, not plan claims.

## M5 chain (walk-forward + final models)
- ✅ walk-forward: 32.5M rows × 119 cols, OOF cached 08:29
- ✅ final models: s42 / s7 / s2026 all 600 trees, 96 features
- ✅ calibration: 88 knots, healthy
- ✅ regime dir prior: regenerated Aug 11 (P(up) 0.43-0.54 by regime)
- ✅ signal_rating.json: threshold 20, deciles learned from 31M OOF signals
- ✅ calibration_by_drr_*.json: per-regime calibrations (Aug 10-11)
- ✅ dirmask_spec_*.npy: direction masks per regime

## Tick / microstructure pipeline (the GAPS)
- ❌ dukascopy_m1_features.csv: NOT on disk, fetch never completed
- ❌ gold_features_m5_tick.csv: NOT on disk
- ❌ Tick retrain (walk-forward on tick matrix): NOT STARTED
- 🛑 Engine watchdog: PAUSED (correct — alpha decayed)
- 🛑 EOD retrain cron: PAUSED
- 🟢 Live macro injection: WORKING (xm_macro.py publishes, engine reads macro_state.json)
- 🟢 xm_ticker.py: WORKING (publishes ticks_60s/300s, imb, spread dynamics)
- 🟢 Event calendar: FIXED this session (event_calendar.py with 387 real events)

## Live performance evidence (THE critical data)
- Last 5 trades: 5 SL / 0 TP, -$23.88 cumulative, conf 0.22-0.26
- Rating threshold (20) was met every time → engine was selectively firing
- But realized expectancy was NEGATIVE on those fires
- Conclusion: rating learned from 6yr data is miscalibrated for current regime (Aug 11 23:38, gold 4378→4364)

## signal_rating deciles (the smoking gun)
- Decile 0-10: 10.8% WR, -8.8% exp
- Decile 10-20: 14.5% WR, -0.8% exp  ← not in OOF
- Decile 20-30: 25.95% WR, +35% exp ← threshold floor
- Decile 30-40: 39.6% WR, +27.8% exp ← sweet spot
- Decile 40-50: 43.9% WR, +6.1% exp
- Distribution: 22M signals in decile 2 (20-30) — engine fires a LOT there
- The rating system is the bottleneck: it's been telling the engine "fire" when it shouldn't

## What is NOT broken
- Engine architecture: EV gate, same-idea suppression, rating gate — all proper
- Calibration: healthy in active region (low/mid probs)
- Models: trained on 32.5M rows, 96 features
- Macro injection: end-to-end working
- Tick state: live ticks flowing into engine

## What IS broken or missing
1. **Tick matrix build**: Dukascopy CSV not ingested, tick matrix never built, tick retrain never run
2. **Tick orchestrator**: looks for "v8.8 TRANSITION COMPLETE" string that no script writes
3. **Alpha decayed**: rating system says fire but live results say don't
4. **No live PnL monitoring**: EWMA control chart, drift detection not present
5. **No champion/challenger**: 3 seeds but no A/B
6. **No microstructure features beyond basic tick imb/spread**: VPIN, Kyle's lambda, trade size distribution absent
7. **No news/NLP features**: only structured event flags (no sentiment)
8. **No execution layer**: signal → no slippage model, no spread-aware entry (engine pays spread naively)
9. **Feature drift not monitored**: PSI/KS not computed live

## Files that exist (the system is more complete than plan implied)
- 8 calibration_by_drr_spec_*.json files (per-regime calibrations)
- 8 dirmask_spec_*.npy files (per-regime direction masks)
- direction_ensemble.json, direction_metrics.json, direction_features.json
- 3 direction_s*.txt models
- regime_dir_prior.json (regenerated Aug 11 with v8 trend-first)

## Files that don't exist (the system is LESS complete than plan claimed)
- transition_v88.sh (referenced by orchestrator but missing)
- .chain_complete marker
- dukascopy_m1_features.csv
- gold_features_m5_tick.csv
- transition_tick.log

## Time stamp of fetches this session
- 14:25 IST: Started fetch_dukascopy_m1.py (background, PID 3653401)
- 14:30 IST: Audit complete
- 14:30 IST: Research delegation dispatched (deleg_9f0c0e7e)
