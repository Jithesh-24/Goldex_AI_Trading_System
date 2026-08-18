# XAUUSD AI Trading System — v7 FINAL EDITION Architecture (2026-08-02)

**A complete closed-loop AI trading system for XAUUSD on XM MT5 (demo).
v7 = dual-model ensemble + recency-weighted training + HTF context +
session clock + event proximity + order flow. Pure teaching — no
hardening filters, no gates, no cooldowns (user mandate: 'harness, not
harden').**

```
┌────────────────────────────────────────────────────────────────────────┐
│                         XM MT5 (Wine, Xvfb :99)                        │
│                  ONE IPC client — the TICKER owns it                   │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ 25ms tick stream + copy_rates_from_pos
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  xm_ticker.py  — the ONLY MT5 connection (daemon)                     │
│  • xm_tick_state.json   bid/ask + forming M1 bar + first-touch verdict │
│  • MICROSTRUCTURE (v7): ticks_60s/300s, imb_60s/300s (buy-sell tick    │
│    imbalance), spread_now/mean/std over 60s — REAL order flow          │
│  • xm_live_bars.jsonl   completed real XM M1 bars (true UTC)          │
│  • xm_bars_backfill.csv 2000 bars via copy_rates (works market-closed)│
│  • xm_server_offset.json persisted server↔UTC offset (+3.0h)          │
└──────────────┬──────────────────────────────┬──────────────────────────┘
               ▼                              ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│  merge_seed.py (offline)     │   │  ai_signal_engine.py (live)  │
│  history+backfill+live →     │   │  • FeatureComputer → v7 95   │
│  gold_seed.csv (61k real XM) │   │    features via _feature_block│
├──────────────────────────────┤   │  • best_placement(): sweep   │
│  download_duka_rally.py      │   │    24 candidates × 3-seed     │
│  (one-time, 907k bars)       │   │    ENSEMBLE → calibrated P → │
│  merge_rally_seed.py         │   │    EXP-max → fire Exp>0      │
│  → gold_seed_multi.csv       │   │  • direction_prior(): 3-seed │
│  build_rally_features.py     │   │    direction model P(up)     │
│  → gold_features_rally.csv   │   │    multiplies BUY/SELL Exp   │
│  (4.7M rows, STREAMED cache) │   │  • position-state + tick ms  │
├──────────────────────────────┤   │    injected into outcome row │
│  build_full_matrix.py        │   │  • regime_label(): market    │
│  → gold_features.csv         │   │  • closed loop: outcome →    │
│  (3.09M rows × 95 cols)      │   │    live_outcomes.jsonl       │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               ▼                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  train_ai.py (v7)  — PLACEMENT model, 3-seed ensemble                 │
│    P(win | market + HTF + session + events + order flow + geometry)   │
│    walk-forward OOF → PAVA calibration → learned per-dir floors       │
│    RECENCY-WEIGHTED (exp(−age/120d)) — adapts to current regime       │
│    outputs: models/gold_lgb_model_s{42,7,2026}.txt + calibration.json │
│  train_direction.py (v7) — DIRECTION model, 3-seed ensemble           │
│    P(up-move | market) over next 60 bars — 'when to trade' teacher    │
│    outputs: models/direction_s{42,7,2026}.txt + direction_*.json      │
└────────────────────────────────────────────────────────────────────────┘
```

## Why v7 (final edition)

| v6 (2026-08-01) | v7 (2026-08-02) |
|---|---|
| Single placement model | Dual model: placement (how) × direction (when) |
| 63 features | 95 features |
| No HTF context | H1/D1 trend, M1-vs-H1 vol ratio, H1/D1 range position, distance to H1 levels, HTF alignment |
| No session awareness | Circular hour/dow, minutes to London/NY open/close |
| No event awareness | Minutes to NFP/FOMC/CPI, pre/post event windows |
| No order flow | Bar-level flow (close_loc, body_signed, wick frac, flow_mom) + LIVE tick imbalance/spread dynamics |
| Uniform sampling | Recency-weighted training (exp(−age/120d)) |
| Single model | 3-seed ensemble (variance ↓, calibration ↑) |
| Maxlen 2000 (1.4 days) | Maxlen 32000 (~22 days — H1/D1 need long context) |

## Feature groups (v7, 95 cols in _feature_block)

1. **Returns/momentum** (F1): ret_1..ret_20, ewma cross
2. **Volatility** (F2): atr_14, atr_z, bb_width, ewma_vol, vol_spike
3. **Oscillators** (F3): rsi_14, macd histogram, stoch k/d
4. **Price shape** (F4): body/upper/lower wick, range position
5. **Structure** (F5): trend_ema/slope, hh/hl/ll/lh counts, support/resistance dist
6. **Volume** (F6): vol_z, vol_rel, vol_atr_ratio
7. **Regime** (F7): trend-strength, bb_pctile, atr_pctile, news_candle, news_vol_ratio
8. **Institutional levels** (F8): day/weekly/monthly pivot lines (scale-free dist in ATR)
9. **Scale-free** (F9): close/atr, range/atr, body/atr (replaces raw prices — RAW_PRICE_COLS excluded from model)
10. **Geometry** (F10): sl_dist/tp_dist/rr per direction (LEARNED placement input)
11. **HTF context** (F11, NEW): h1_trend, d1_trend, m1_h1_vol_ratio, m1_d1_vol_ratio, h1_pos, d1_pos, dist_h1_hi/lo, htf_align
12. **Session clock** (F12, NEW): hour_sin/cos, min_to_london/ny/close, dow_sin/cos
13. **Event proximity** (F13, NEW): min_to_event, min_since_event, pre_event, post_event
14. **Order flow** (F14, NEW): close_loc, body_signed, up/dn_wick_frac, flow_mom, close_loc_mom, flow_conviction
15. **Live-only** (v7, cold-start → learned via closed loop): day_pnl, streak, trades_today, ticks_60s, imb_60s, imb_300s, spread_z60

## Direction model (when-to-trade) — how 'no falling knives' is taught

- 1 row/bar (no placement geometry), same market features
- target = did close rise ≥ $0.20 within next 60 bars (realistic horizon)
- engine: `final_exp(BUY) = Exp_buy × P(up)`, `final_exp(SELL) = Exp_sell × (1−P(up))`
- In a downtrend P(up) is low → BUY Exp crushed → SELL naturally wins
- NO gates, NO thresholds — pure learned probability multiplier
- soft-clipped to [0.05, 0.95] so one side is never fully muted

## Recency weighting (why the model adapts without regime logic)

- sample_weight = exp(−age_days / 120) per training row
- 2-week-old bar: 0.89× | 6-month: 0.22× | 3-year: ~0.0001×
- The RANGE regression (v6: −$78) came from 2020–24 patterns diluting 2026
  reality. Recency weighting teaches "recent regime matters most" — no
  hardcoded regime filter.

## Event calendar (data, not rules)

- FOMC 2026: Jan 28, Mar 18, Apr 29, Jun 17, Jul 29, Sep 16, Oct 28, Dec 9
  (confirmed federalreserve.gov, statement @ 19:00 UTC)
- NFP: first Friday monthly @ 13:30 UTC
- CPI: ~12th monthly @ 13:30 UTC
- Feature: minutes-to/from event, pre/post windows. The model LEARNS
  what to do near news — no news filter.

## Memory-safe pipeline (2GB gateway cgroup — the binding constraint)

- **build_rally_features.py**: streams per-period blocks (<400MB peak)
  → gold_features_rally.csv (4.7M rows, CACHED; rebuilt when features.py
  or rally source changes)
- **build_full_matrix.py**: rally subsample (every 3rd bar) + streamed XM
  per-period + GNU external sort (dynamic time-col index — v7 added 27
  cols, hardcoded 59 would silently sort wrong) + chunked float32
  → gold_features.csv (3.09M rows × 95 cols ≈ 1.2GB f32)
- **train_ai.py / train_direction.py**: float32 direct-read, usecols,
  view-based X, free_raw_data=True, num_threads=1, del df + gc
- **Never** materialize the full matrix in RAM (v6 OOM twice; kernel log:
  `oom-kill: CONSTRAINT_MEMCG task=main`)

## Closed loop (learn from ACTUAL outcomes, not just history)

```
trade fires → first-touch verdict (25ms ticker) → outcome + FULL feature
vector → live_outcomes.jsonl → daily 12:00 IST retrain_loop.py:
  merge_seed → rebuild XM matrix → merge live outcome rows → retrain
  placement ensemble → retrain direction model → atomic swap → engine
  hot-reloads (mtime watch). No hardcoded lesson injection — the model
  re-weights itself from real results.
```

- trade_audit.py runs on every loss (root-cause + lesson for the user)
- position-state + tick microstructure ride along in outcome rows; the
  NEXT retrain teaches them (cold-start → learned, no gates)

## Deploy / ops

- Xvfb :99 → xfwm4 → MT5 (Wine, XM, Login 316962850) → xm_ticker.py →
  ai_signal_engine.py
- watchdog cron deae2f8e26a2 restarts ticker/engine within 5 min
- models/: gold_lgb_model_s{42,7,2026}.txt (placement ensemble),
  direction_s{42,7,2026}.txt (direction), ensemble.json,
  direction_ensemble.json, features.json, direction_features.json,
  calibration.json (62 knots + learned floors), metrics.json
- Backtest: backtest_v7.py (mirrors engine logic exactly)
