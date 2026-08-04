# Goldex AI Trading System

Autonomous AI trading system for **XAUUSD (Gold)** on the XM MT5 platform.
Trains on **6 years of data (2020–2026)**, learns **8 market regimes**, routes
each live tick to the current regime's specialist, self-learns from losses,
and retrains nightly on live outcomes. Built to compound a proven edge.

Author: **Jithesh**

---

## Architecture

```
src/  (actually lives under scripts/ on the box)
ai_signal_engine.py     Live engine: pools XM, routes to regime specialist,
                        computes placement EV, fires signals, tracks SL/TP.
xm_ticker.py            MT5→XM price feed (25ms real-time poll, single IPC).
features.py             Feature engine: 102 gold-derived features + regime_bin().
build_full_matrix.py    Feature matrix build / v7.10 incremental daily append.
retrain_loop.py         Nightly EOD learning loop (warm-start continuation).
train_regime_spec.py    8-regime × 3-seed specialist trainer (regime_specialists).
trade_audit.py          Loss self-analysis → 6 root causes → lessons.md.
watchdog.py             Engine/ticker/Xvfb/MT5 supervision + auto-restart.
space_guard.py          Disk audit + journal rotation (bounded growth).
```

## How it learns (closed loop)

1. **Features** — 102 gold-derived features (returns, BB, ATR, OBV, geometry,
   regime, ...) computed from 6 years of M1 XAUUSD data.
2. **Model** — LightGBM gradient-boosted trees (proven best-in-class for
   structured tabular finance). Ensembled across 3 seeds, isotonic-calibrated.
3. **Regime routing** — `regime_bin()` maps market state to 8 bins; the engine
   routes each tick to a specialist trained only on that regime.
4. **Placement label** — first-touch TP-before-SL, learns *where* to place.
   Direction is honestly gated (OOS P(price up) must beat majority by 2% or
   the model stays NEUTRAL — no fake edges).
5. **EOD learning** — nightly at 15:00 IST: append today's live outcomes to the
   matrix (incremental), warm-start retrain, recalibrate, respecialize.
6. **Loss analysis** — `trade_audit.py` classifies every loss and feeds
   lessons back into the loop.

## Reliability

- Watchdog chain: every layer (engine, ticker, Xvfb, MT5) has freshness checks
  and independent auto-restart; silent when healthy.
- Atomic model swaps + hot-reload; PID-locked single engine instance.
- Market-state re-fire suppression (not time-based): re-signals only when the
  regime/direction/TP-zone genuinely changes or price moves ≥0.5×ATR.
- Space guard keeps data growth bounded (seed ~60-day window, journal rotation).

## Notes

- Live model is **gold-only** (102 features). Macro (DXY/VIX/US10Y etc.) is
  backtest/analysis only — MT5 offers no live access to those pairs.
- Large datasets, persistent logs, and trained model weights are **not**
  committed here; they are regenerated on the trading box by the nightly loop.