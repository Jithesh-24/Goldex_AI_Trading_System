"""PHASE 1 — Modular feature pipeline for the AI XAUUSD signal system.
Converts raw OHLCV bars into a feature matrix. NO decision rules here —
pure feature engineering. The model learns what matters.

Feature groups (each a pure function of the bars):
  F1. Returns       — log returns at multiple lags/horizons
  F2. Volatility    — ATR, rolling std, BB width, GARCH-like EWMA vol
  F3. Oscillators   — RSI, MACD hist, stochastic, CCI
  F4. Price shape   — position in range, candle bodies/wicks, streaks
  F5. Structure     — session, day-of-week, hour, daily range position
  F6. Volume        — tick volume z-score, volume/ATR ratio
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ════════════════════════════════════════════════════════════════════
# F2: Volatility
# ════════════════════════════════════════════════════════════════════
def add_atr(df, period=14):
    """ATR (Wilder's smoothing) — volatility of the last `period` bars."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(alpha=1/period, adjust=False).mean()
    return df

def add_bb_width(df, period=20):
    """Bollinger Band width normalized by price — compression/expansion."""
    ma = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std()
    df[f"bb_w_{period}"] = (2 * sd) / ma * 100
    df[f"bb_pos_{period}"] = (df["close"] - (ma - 2*sd)) / (4*sd + 1e-9)
    return df

def add_ewma_vol(df, spans=(10, 30, 60)):
    """EWMA volatility at multiple timescales."""
    ret = df["close"].pct_change()
    for s in spans:
        df[f"vol_ewma_{s}"] = ret.ewm(span=s, adjust=False).std() * np.sqrt(1440)  # annualized-ish
    return df

# ════════════════════════════════════════════════════════════════════
# F3: Oscillators
# ════════════════════════════════════════════════════════════════════
def add_rsi(df, period=14):
    """Wilder RSI."""
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-9)
    df[f"rsi_{period}"] = 100 - 100/(1+rs)
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    """MACD line, signal, histogram."""
    ef = df["close"].ewm(span=fast, adjust=False).mean()
    es = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ef - es
    df["macd_sig"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    return df

def add_stoch(df, k=14, d=3):
    """Stochastic %K/%D."""
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    df["stoch_k"] = (df["close"] - ll) / (hh - ll + 1e-9) * 100
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    return df

# ════════════════════════════════════════════════════════════════════
# F4: Price shape
# ════════════════════════════════════════════════════════════════════
def add_price_shape(df, lookback=20):
    """Position in recent range, candle anatomy, return streaks."""
    ll = df["low"].rolling(lookback).min()
    hh = df["high"].rolling(lookback).max()
    df[f"range_pos_{lookback}"] = (df["close"] - ll) / (hh - ll + 1e-9) * 100
    # Candle anatomy
    df["body"] = df["close"] - df["open"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    rng = df["high"] - df["low"] + 1e-9
    df["body_frac"] = df["body"].abs() / rng
    df["wick_ratio"] = (df["upper_wick"] + df["lower_wick"]) / rng
    # Return streaks
    ret = df["close"].pct_change()
    streak = np.zeros(len(df))
    cnt = 0
    for i in range(1, len(df)):
        if ret.iloc[i] > 0:
            cnt = cnt + 1 if ret.iloc[i-1] > 0 else 1
        elif ret.iloc[i] < 0:
            cnt = cnt - 1 if ret.iloc[i-1] < 0 else -1
        else:
            cnt = 0
        streak[i] = cnt
    df["ret_streak"] = streak
    return df

# ════════════════════════════════════════════════════════════════════
# F5: Structure (time-based + daily context)
# ════════════════════════════════════════════════════════════════════
def add_structure(df):
    """Session, hour, day-of-week, daily range position."""
    t = pd.to_datetime(df["time"])
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    # Session: Asia 0-7, London 7-12, NY 12-17, Late 17-24 (UTC)
    df["session"] = 0  # Asia
    df.loc[t.dt.hour.between(7, 11), "session"] = 1   # London
    df.loc[t.dt.hour.between(12, 16), "session"] = 2  # NY
    df.loc[t.dt.hour >= 17, "session"] = 3            # Late
    # Daily range position: where is close within the day's high-low SO FAR (causal, no lookahead)
    day = t.dt.date
    day_idx = t.groupby(day).cumcount()
    dhi = df.groupby(day)["high"].cummax()
    dlo = df.groupby(day)["low"].cummin()
    df["daily_pos"] = (df["close"] - dlo) / (dhi - dlo + 1e-9) * 100
    # Daily range size normalized (so far)
    df["daily_range_pct"] = (dhi - dlo) / dlo * 100
    return df

# ════════════════════════════════════════════════════════════════════
# F1: Returns
# ════════════════════════════════════════════════════════════════════
def add_returns(df, lags=(1, 2, 3, 5, 10, 15, 30, 60)):
    """Log returns at multiple lookbacks."""
    c = df["close"]
    for l in lags:
        df[f"ret_{l}"] = np.log(c / c.shift(l))
    # Return momentum: short vs long
    df["ret_mom"] = df["ret_5"] - df["ret_60"]
    return df

# ════════════════════════════════════════════════════════════════════
# F6: Volume
# ════════════════════════════════════════════════════════════════════
def add_volume(df, lookback=20):
    """Tick volume z-score and volume-ATR ratio."""
    v = df["tick_volume"].astype(float)
    df["vol_z"] = (v - v.rolling(lookback).mean()) / (v.rolling(lookback).std() + 1e-9)
    df["vol_atr_ratio"] = v / (df.get("atr_14", v.rolling(lookback).mean()) + 1e-9)
    return df

# ════════════════════════════════════════════════════════════════════
# F7: SL/TP geometry awareness — the model SEES the exact stop/target
# levels that will be deployed. Without these, the model predicts
# direction blind while geometry is bolted on after — it can't learn
# "this setup's stop is too tight for this volatility regime".
# (2026-07-31: added after live losses showed floor-bound $4.20 stops
#  identical to the OLD system — model had zero awareness of them.)
# ════════════════════════════════════════════════════════════════════
def add_geometry_awareness(df, sl_dist_buy=None, tp_dist_buy=None, sl_dist_sell=None, tp_dist_sell=None):
    """SL/TP distances + ratios as INPUT FEATURES so the model learns how
    placement interacts with market regime. When explicit distances are given
    (training rows / inference sweep), those are used — the model sees the
    SPECIFIC placement it's being asked about. Otherwise falls back to the
    mid-grid placement (backward compat for legacy calls)."""
    atr = df["atr_14"]
    if "spread" in df.columns:
        spr = df["spread"].astype(float) / 100.0
    else:
        spr = pd.Series(SPREAD, index=df.index)
    if sl_dist_buy is None:
        grid = live_geometry(atr, spr)
        mid = grid[len(grid) // 2]
        sl_dist_buy, tp_dist_buy = mid[0], mid[1]
        sl_dist_sell, tp_dist_sell = mid[0], mid[1]
    df["sl_dist_buy"] = sl_dist_buy
    df["tp_dist_buy"] = tp_dist_buy
    df["sl_dist_sell"] = sl_dist_sell
    df["tp_dist_sell"] = tp_dist_sell
    df["sl_atr_buy"] = np.asarray(sl_dist_buy, dtype=float) / (atr + 1e-9)
    df["sl_atr_sell"] = np.asarray(sl_dist_sell, dtype=float) / (atr + 1e-9)
    df["rr_buy"] = np.asarray(tp_dist_buy, dtype=float) / (np.asarray(sl_dist_buy, dtype=float) + 1e-9)
    df["rr_sell"] = np.asarray(tp_dist_sell, dtype=float) / (np.asarray(sl_dist_sell, dtype=float) + 1e-9)
    return df


# ════════════════════════════════════════════════════════════════════
# TARGET: what the model predicts
# ════════════════════════════════════════════════════════════════════
def add_target(df, horizon=15, threshold=0.0):
    """Binary target: did price rise in next `horizon` minutes?
    threshold = minimum $ move required (0 = any direction change wins)."""
    fwd = df["close"].shift(-horizon)
    df["fwd_return"] = (fwd - df["close"]) / df["close"]
    df["target"] = (df["fwd_return"] > threshold).astype(int)
    return df


# ════════════════════════════════════════════════════════════════════
# F8: MARKET REGIME AWARENESS (v5 2026-08-01)
# Teaches the model WHICH market it's in: trending up / trending down /
# ranging / news-spike / quiet — so it learns the right behavior per
# regime (follow trends, fade ranges, avoid news whipsaws, adapt
# placement to volatility). ALL features are causal (rolling windows
# ending at the current bar — no lookahead). News has no separate feed
# in this pipeline, so news is detected the way it actually hits price:
# volatility spike + volume spike + spread widening + body expansion.
# ════════════════════════════════════════════════════════════════════
def add_regime(df):
    """Regime classifier features — soft continuous signals, model decides."""
    c = df["close"]
    atr = df["atr_14"]

    # EMA trend alignment: (ema20 - ema50) in ATR units → signed trend strength
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    df["trend_ema"] = (ema20 - ema50) / (atr + 1e-9)          # + = up, − = down
    df["trend_slope"] = (ema20 - ema20.shift(5)) / (atr + 1e-9)  # momentum of trend

    # Trend consistency: fraction of last 20 closes above ema50 (0..1)
    df["above_ema50"] = (c > ema50).rolling(20).mean()

    # Range detection: BB width percentile over last 200 bars (low = range)
    bb = df.get("bb_w_20")
    if bb is None:
        ma = c.rolling(20).mean(); sd = c.rolling(20).std()
        bb = (2 * sd) / ma * 100
        df["bb_w_20"] = bb
    df["bb_pctile"] = bb.rolling(200).rank(pct=True)           # low→squeeze/range
    df["atr_pctile"] = atr.rolling(200).rank(pct=True)         # volatility regime

    # Choppiness: |EMA slope| / BB width — high = trend, low = chop/range
    df["trend_quality"] = df["trend_ema"].abs() / (bb / 100 + 1e-9)

    # News-spike proxies (how news manifests in price):
    body = (c - df["open"]).abs()
    df["body_atr"] = body / (atr + 1e-9)                       # candle size vs ATR
    df["news_candle"] = (df["body_atr"] > 2.5).astype(float).rolling(5).mean()  # freq of big candles
    df["vol_spike"] = df.get("vol_z", df["tick_volume"].astype(float).rolling(20).mean())  # vol burst
    df["vol_spike_bin"] = (df["vol_spike"] > 2.0).astype(float)  # hard spike flag (still causal)
    spr = df["spread"].astype(float) if "spread" in df.columns else pd.Series(20, index=df.index)
    df["spread_z"] = (spr - spr.rolling(100).mean()) / (spr.rolling(100).std() + 1e-9)  # spread widening
    # Range fades vs trend follows: where is close in the last 20-bar range?
    ll = df["low"].rolling(20).min(); hh = df["high"].rolling(20).max()
    df["range_pos"] = (c - ll) / (hh - ll + 1e-9)              # 0=bottom 1=top
    return df


# ════════════════════════════════════════════════════════════════════
# F9: INSTITUTIONAL LEVELS (v6 2026-08-01)
# Classic S/R concepts the model should learn: previous-day high/low/close
# (the first levels institutions defend), day-open level, round numbers
# (psychology), and weekly/monthly pivot-style anchors. All expressed in
# ATR units (scale-free) and signed (distance + direction from current
# price) — the model learns "close above prev-day high by 0.8 ATR" etc.
# NOT gates — pure descriptive features.
# ════════════════════════════════════════════════════════════════════
def add_institutional_levels(df):
    """Previous-day H/L/C, day open, round numbers, weekly anchors in ATR units."""
    c = df["close"]
    atr = df["atr_14"]
    t = pd.to_datetime(df["time"])
    day = t.dt.date

    # Daily session aggregates (shifted by one day — no lookahead)
    g = df.groupby(day)
    prev_close = g["close"].last().shift(1)
    prev_high = g["high"].max().shift(1)
    prev_low = g["low"].min().shift(1)
    day_open = g["open"].first()

    # Map back onto the bar series
    day_series = pd.Series(day, index=df.index)
    df["prev_close"] = day_series.map(prev_close)
    df["prev_high"] = day_series.map(prev_high)
    df["prev_low"] = day_series.map(prev_low)
    df["day_open_lvl"] = day_series.map(day_open)

    # Signed distances in ATR units
    df["dist_prev_close"] = (c - df["prev_close"]) / (atr + 1e-9)
    df["dist_prev_high"] = (c - df["prev_high"]) / (atr + 1e-9)
    df["dist_prev_low"] = (c - df["prev_low"]) / (atr + 1e-9)
    df["dist_day_open"] = (c - df["day_open_lvl"]) / (atr + 1e-9)

    # First day of a period has no prior day → neutral (0 = at the level)
    df[["dist_prev_close", "dist_prev_high", "dist_prev_low"]] = \
        df[["dist_prev_close", "dist_prev_high", "dist_prev_low"]].fillna(0.0)

    # Round-number proximity: distance to nearest $50 level (ATR units).
    # Gold psychology clusters at whole hundreds ($4000, $4050, $4100...).
    nearest_50 = np.round(c / 50.0) * 50.0
    df["round50_dist"] = (c - nearest_50) / (atr + 1e-9)

    # Day-of-week edge (gold has session-specific tendencies) — keep as
    # descriptive feature, model decides.
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7.0)

    # Drop raw anchor columns (keep only ATR-relative distances)
    df.drop(columns=["prev_close", "prev_high", "prev_low", "day_open_lvl"], inplace=True)
    return df


# ════════════════════════════════════════════════════════════════════
# F10: SCALE-FREE PRICE NORMALIZATION (v6 2026-08-01)
# Training now spans multiple eras (2020 at $1545, 2024 at $2450, 2026 at
# $4043). Raw price levels would let the model memorize "price ≈ X" instead
# of learning PATTERNS. Replace absolute price features with ratios to
# rolling means (scale-free), and ATR with ATR/price. Everything the model
# sees is then era-agnostic — the same candle pattern teaches the same
# lesson whether gold is $1500 or $5000.
# ════════════════════════════════════════════════════════════════════
def add_scale_free(df):
    """Replace absolute price/ATR features with price-relative ratios."""
    c = df["close"]
    # Long-run anchors (200-bar EMA) — price relative to its own era
    ma100 = c.rolling(100).mean()
    ma200 = c.rolling(200).mean()
    df["close_ma100"] = c / (ma100 + 1e-9)      # ~1.0 when flat, <1 below, >1 above
    df["close_ma200"] = c / (ma200 + 1e-9)
    df["open_ma100"] = df["open"] / (ma100 + 1e-9)
    df["high_ma100"] = df["high"] / (ma100 + 1e-9)
    df["low_ma100"] = df["low"] / (ma100 + 1e-9)
    # ATR as fraction of price (volatility regime is relative, not absolute)
    df["atr_pct"] = df["atr_14"] / (c + 1e-9) * 100.0
    # MACD histogram in ATR units (scale-free)
    df["macd_hist_atr"] = df["macd_hist"] / (df["atr_14"] + 1e-9)
    # Volume in z-units only (absolute tick counts differ across feeds)
    if "tick_volume" in df.columns:
        df["vol_rel"] = df["tick_volume"].astype(float) / (
            df["tick_volume"].astype(float).rolling(100).mean() + 1e-9)
    return df


# ════════════════════════════════════════════════════════════════════
# F11: HIGHER-TIMEFRAME CONTEXT (v7 2026-08-02)
# The M1 model was blind to the big picture. Pros trade WITH the trend:
# H1/D1 momentum, M1-vs-H1 volatility ratio (is this minute fast or slow
# relative to the hour?), position within the H1 range. All features are
# causal (resample → shift → reindex — never look into the future of the
# current H1/D1 bar) and scale-free (ATR units / ratios).
# ════════════════════════════════════════════════════════════════════
def add_htf_context(df):
    """H1/D1 resampled context: trend, volatility ratio, range position."""
    c = df["close"].values
    atr = df["atr_14"].values
    t = pd.to_datetime(df["time"])
    s = df.set_index(t)

    def _htf_features(rule, ema_fast, ema_slow):
        # Resample to H1/D1 using CLOSED bars only (shift(1) → previous
        # completed H1/D1 bar; the current forming bar is never used).
        r = s.resample(rule).agg({"close": "last", "high": "max", "low": "min"})
        r = r.dropna()          # weekends/holidays have no bars — drop BEFORE
                                # rolling so windows span trading days only
        r = r.shift(1)          # causal: only completed higher-TF bars
        ef = r["close"].ewm(span=ema_fast, adjust=False).mean()
        es = r["close"].ewm(span=ema_slow, adjust=False).mean()
        rng14 = (r["high"] - r["low"]).rolling(14).mean().replace(0, np.nan)
        r["trend"] = (ef - es) / rng14
        r["range"] = r["high"] - r["low"]
        return r

    h1 = _htf_features("1h", 12, 48)
    d1 = _htf_features("1D", 5, 20)
    m15 = _htf_features("15min", 12, 48)   # v8 M15 context layer (M5 base)

    # Reindex back to base TF (ffill: the value holds until the next completed
    # HTF bar). ALWAYS extract .values immediately — the reindexed
    # Series carries a DatetimeIndex and any later pandas arithmetic with
    # the RangeIndex'd df would silently union them (length doubling).
    h1_trend = h1["trend"].reindex(s.index, method="ffill").values
    h1_range = h1["range"].reindex(s.index, method="ffill").values
    d1_trend = d1["trend"].reindex(s.index, method="ffill").values
    d1_range = d1["range"].reindex(s.index, method="ffill").values
    m15_trend = m15["trend"].reindex(s.index, method="ffill").values
    m15_range = m15["range"].reindex(s.index, method="ffill").values
    h1_hi = h1["high"].reindex(s.index, method="ffill").values
    h1_lo = h1["low"].reindex(s.index, method="ffill").values
    d1_hi = d1["high"].reindex(s.index, method="ffill").values
    d1_lo = d1["low"].reindex(s.index, method="ffill").values
    m15_hi = m15["high"].reindex(s.index, method="ffill").values
    m15_lo = m15["low"].reindex(s.index, method="ffill").values

    df["h1_trend"] = h1_trend                       # + = H1 uptrend
    df["d1_trend"] = d1_trend                       # + = D1 uptrend
    df["m15_trend"] = m15_trend                     # + = M15 uptrend (v8)
    # base-TF volatility vs HTF volatility — 1.0 = typical, >1 = fast bars
    df["m1_h1_vol_ratio"] = atr / (h1_range / 60 + 1e-9)
    df["m1_d1_vol_ratio"] = atr / (d1_range / 1440 + 1e-9)
    df["m15_m1_vol_ratio"] = m15_range / (atr * 3 + 1e-9)   # v8: M15 vs 3×base-ATR
    # HTF momentum: base close relative to completed HTF high-low range
    df["h1_pos"] = (c - h1_lo) / (h1_hi - h1_lo + 1e-9)   # 0..1
    df["d1_pos"] = (c - d1_lo) / (d1_hi - d1_lo + 1e-9)
    df["m15_pos"] = (c - m15_lo) / (m15_hi - m15_lo + 1e-9)   # v8
    # Distance to H1 high/low in base-ATR units (tradeable reference levels)
    df["dist_h1_hi"] = (h1_hi - c) / (atr + 1e-9)
    df["dist_h1_lo"] = (c - h1_lo) / (atr + 1e-9)
    df["dist_m15_hi"] = (m15_hi - c) / (atr + 1e-9)   # v8
    df["dist_m15_lo"] = (c - m15_lo) / (atr + 1e-9)   # v8
    # HTF confluence: base and M15 trend agree? 1 = aligned up, -1 = aligned down
    df["htf_align"] = np.sign(df["h1_trend"]) * np.sign(df["d1_trend"])
    df["m15_align"] = np.sign(df["m15_trend"]) * np.sign(df["h1_trend"])  # v8
    # First bars of a period have no completed HTF → neutral
    df[["h1_trend", "d1_trend", "m15_trend", "m1_h1_vol_ratio", "m1_d1_vol_ratio",
        "m15_m1_vol_ratio", "h1_pos", "d1_pos", "m15_pos", "dist_h1_hi",
        "dist_h1_lo", "dist_m15_hi", "dist_m15_lo", "htf_align", "m15_align"]] = \
        df[["h1_trend", "d1_trend", "m15_trend", "m1_h1_vol_ratio", "m1_d1_vol_ratio",
            "m15_m1_vol_ratio", "h1_pos", "d1_pos", "m15_pos", "dist_h1_hi",
            "dist_h1_lo", "dist_m15_hi", "dist_m15_lo", "htf_align", "m15_align"]].fillna(0.0)
    return df


# ════════════════════════════════════════════════════════════════════
# F12: SESSION CLOCK (v7 2026-08-02)
# Gold has a session heartbeat: Asia chop, London push, NY close, late
# rollover. Circular hour encoding (sin/cos — 23:59 and 00:01 are close,
# raw hour says 23 vs 0), minutes-to-session-boundary (London 07:00 UTC,
# NY 12:00 UTC, close 21:00 UTC), session phase. Pure description — the
# model learns when it can trade and when to stand aside.
# ════════════════════════════════════════════════════════════════════
def add_session_clock(df):
    t = pd.to_datetime(df["time"])
    hour = t.dt.hour + t.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    # Minutes to/from session boundaries (UTC)
    LONDON_OPEN, NY_OPEN, DAY_CLOSE = 7.0, 12.0, 21.0
    df["min_to_london"] = (LONDON_OPEN - hour) * 60.0
    df["min_to_ny"] = (NY_OPEN - hour) * 60.0
    df["min_to_close"] = (DAY_CLOSE - hour) * 60.0
    # Circular day-of-week (Monday and Friday are close on a 7-day cycle)
    dow = t.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


# ════════════════════════════════════════════════════════════════════
# F13: EVENT PROXIMITY (v7 2026-08-02)
# Scheduled macro events move gold violently for minutes then mean-revert
# (or trend). The model must know an event is IMMINENT to decide "stand
# aside / tight scalp / wide momentum". Minutes-to-next-event is a pure
# time feature — the model learns the behavior. Calendar is DATA (sourced:
# Fed 2026 FOMC dates confirmed; NFP = first Friday 13:30 UTC; CPI ~mid-month
# 13:30 UTC), not a rule.
# ════════════════════════════════════════════════════════════════════
# 2026 FOMC decisions (two-day meetings; statement day): confirmed from
# federalreserve.gov — Jan 28, Mar 18, Apr 29, Jun 17, Jul 29, Sep 16,
# Oct 28, Dec 9. NFP: first Friday each month @ 13:30 UTC. CPI: ~mid-month
# (BLS schedule) @ 13:30 UTC.
FOMC_2026 = [(2026,1,28),(2026,3,18),(2026,4,29),(2026,6,17),(2026,7,29),
             (2026,9,16),(2026,10,28),(2026,12,9)]

def _event_times():
    """List of (unix_ts, kind) for scheduled 2026 macro events."""
    import calendar as _cal
    from datetime import datetime as _dt, timezone as _tz
    ev = []
    for y, m, d in FOMC_2026:
        ev.append((int(_dt(y, m, d, 19, 0, tzinfo=_tz.utc).timestamp()), "fomc"))
    # NFP: first Friday of each month @ 13:30 UTC
    for m in range(1, 13):
        c = _cal.monthcalendar(2026, m)
        fd = next(w[4] for w in c if w[4] != 0)
        ev.append((int(_dt(2026, m, fd, 13, 30, tzinfo=_tz.utc).timestamp()), "nfp"))
    # CPI: BLS releases ~10th-15th; use 12th @ 13:30 UTC (schedule data)
    for m in range(1, 13):
        ev.append((int(_dt(2026, m, 12, 13, 30, tzinfo=_tz.utc).timestamp()), "cpi"))
    ev.sort()
    return ev

_EVENTS = None
def _events():
    global _EVENTS
    if _EVENTS is None:
        _EVENTS = _event_times()
    return _EVENTS

def add_event_proximity(df):
    """Minutes until / since the next major macro event (causal — only PAST
    events inform 'since'; 'until' is pure schedule)."""
    t = pd.to_datetime(df["time"])
    ts = t.values.astype("datetime64[s]").astype(np.int64)
    ev = np.array([e[0] for e in _events()], dtype=np.int64)
    # For each bar: next event index (events >= ts), prev event index
    next_i = np.searchsorted(ev, ts, side="left")
    next_i = np.clip(next_i, 0, len(ev) - 1)
    prev_i = np.clip(next_i - 1, 0, len(ev) - 1)
    df["min_to_event"] = (ev[next_i] - ts) / 60.0
    df["min_since_event"] = (ts - ev[prev_i]) / 60.0
    # Event window flag: within 30 min before / 60 min after (causal)
    df["pre_event"] = (df["min_to_event"] <= 30).astype(float)
    df["post_event"] = (df["min_since_event"] <= 60).astype(float)
    return df


# ════════════════════════════════════════════════════════════════════
# F14: ORDER FLOW / MICROSTRUCTURE (v7 2026-08-02)
# Bar-level order flow proxies computable from OHLC (true tick-level
# imbalance comes live from the ticker and enters via live outcomes →
# closed loop). Close-location, signed body, wick asymmetry — who is in
# control of THIS bar: buyers (close near high, small upper wick) or
# sellers. Rolling versions = flow momentum. All causal, all scale-free.
# ════════════════════════════════════════════════════════════════════
def add_order_flow(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    df["close_loc"] = (c - l) / rng                    # 1 = closed at high
    df["body_signed"] = (c - o) / rng                  # + = bull bar, − = bear
    df["up_wick_frac"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    df["dn_wick_frac"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    # Flow momentum: who controlled the last 10 bars
    df["flow_mom"] = df["body_signed"].rolling(10).mean()
    df["close_loc_mom"] = df["close_loc"].rolling(10).mean() - 0.5
    # Body/range ratio scaled by vol_z → conviction
    if "vol_z" in df.columns:
        df["flow_conviction"] = df["body_signed"] * df["vol_z"].clip(-3, 3)
    df[["close_loc", "body_signed", "up_wick_frac", "dn_wick_frac",
        "flow_mom", "close_loc_mom"]] = \
        df[["close_loc", "body_signed", "up_wick_frac", "dn_wick_frac",
            "flow_mom", "close_loc_mom"]].fillna(0.0)
    return df


# ════════════════════════════════════════════════════════════════════
# F9: STRATEGY PLAYBOOK (v7.3 2026-08-02) — the pro playbook, taught.
# User mandate: "find all the trading strategies, techniques, trade
# management skills — feed them all, fine-tune until real values show."
# Pure teaching: these are classical strategy INPUTS the model learns to
# weight per regime. No gates — just more of the playbook on the table.
#   * ADX          → trend strength (trending vs choppy filter, learned)
#   * CCI          → cycle position (overbought/oversold oscillator)
#   * Squeeze      → Bollinger inside Keltner = coiled spring (volatility
#                    contraction → expansion setups)
#   * Candle FX    → engulfing, doji, hammer, pin bar (reversal patterns)
#   * OBV flow     → volume-confirmed accumulation/distribution
#   * Donchian     → channel breakout flags (20-bar expansion)
# ════════════════════════════════════════════════════════════════════
def add_strategy_playbook(df, n=14):
    """Classical strategy inputs, all scale-free, all learned (never gates)."""
    h, l, c = df["high"], df["low"], df["close"]
    # ── ADX (Wilder) — trend strength ──
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / (atr_s + 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / (atr_s + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx_14"] = dx.ewm(alpha=1.0 / n, adjust=False).mean().fillna(0.0)
    df["di_bias"] = ((plus_di - minus_di) / (plus_di + minus_di + 1e-9)).fillna(0.0)  # -1..1
    # ── CCI — cycle position ──
    tp = (h + l + c) / 3.0
    sma = tp.rolling(20).mean()
    md = (tp - sma).abs().rolling(20).mean()
    df["cci_20"] = ((tp - sma) / (0.015 * md + 1e-9)).clip(-400, 400).fillna(0.0)
    # ── Squeeze: Bollinger(width) vs Keltner — contraction → expansion ──
    mid20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    bb_hi, bb_lo = mid20 + 2 * sd20, mid20 - 2 * sd20
    atr20 = atr_s.rolling(20).mean()
    kel_hi, kel_lo = mid20 + 1.5 * atr20, mid20 - 1.5 * atr20
    df["squeeze"] = (((bb_hi - bb_lo) / (kel_hi - kel_lo + 1e-9)) - 1.0).clip(0, 3).fillna(0.0)
    df["squeeze_bin"] = (df["squeeze"] <= 0.0).astype(float)  # 1 = coiled spring
    # ── Candle patterns (last completed candle) ──
    o, cl = df["open"], df["close"]
    body = (cl - o).abs()
    rng = (h - l).replace(0, 1e-9)
    df["engulf"] = ((body.shift(1) > 0.6 * (h.shift(1) - l.shift(1)).replace(0, 1e-9)) &
                    (body > body.shift(1)) &
                    ((cl > o) != (cl.shift(1) > o.shift(1))) &
                    (np.maximum(cl, o) > np.maximum(cl.shift(1), o.shift(1))) &
                    (np.minimum(cl, o) < np.minimum(cl.shift(1), o.shift(1)))).astype(float)
    df["doji"] = (body / rng < 0.1).astype(float)
    df["hammer"] = ((l - np.minimum(cl, o)) / rng > 0.6).astype(float)
    df["pin"] = (((h - np.maximum(cl, o)) / rng > 0.5) | ((np.minimum(cl, o) - l) / rng > 0.5)).astype(float)
    df["patt_dir"] = np.sign((cl - o).shift(1)).fillna(0.0)  # prior candle bias
    # ── OBV — accumulation/distribution flow ──
    obv = (np.sign(c.diff()).fillna(0.0) * df["tick_volume"].astype(float)).cumsum()
    df["obv_slope"] = (obv.diff(20) / (obv.rolling(20).std() + 1e-9)).clip(-3, 3).fillna(0.0)
    # ── Donchian breakout (20-bar channel) ──
    dc_hi = h.rolling(20).max()
    dc_lo = l.rolling(20).min()
    df["donch_pos"] = ((c - dc_lo) / (dc_hi - dc_lo + 1e-9)).fillna(0.5)
    df["donch_break"] = (c > dc_hi.shift(1)).astype(float) - (c < dc_lo.shift(1)).astype(float)
    for col in ["adx_14", "di_bias", "cci_20", "squeeze", "squeeze_bin",
                "engulf", "doji", "hammer", "pin", "patt_dir",
                "obv_slope", "donch_pos", "donch_break"]:
        df[col] = df[col].fillna(0.0)
    return df


# ════════════════════════════════════════════════════════════════════
# SL/TP GEOMETRY — LEARNED, NOT HARDCODED (v4 2026-08-01).
# The user's directive: "the AI understands and gives me signals based on
# where to keep SL and TP levels." So the model is trained with SL/TP
# placement as INPUT FEATURES over a GRID of candidate geometries. At signal
# time the engine evaluates every candidate (SL width × TP ratio × direction),
# asks the model P(win|market+placement), and fires the placement maximizing
# expected value. There is NO deployed formula — the constants below are
# training-time SEARCH SPACE boundaries (data sampling), not live rules.
# ════════════════════════════════════════════════════════════════════
SPREAD = 0.20          # training fallback spread ($) when column missing
MAX_TARGET_BARS = 60   # trade-realistic horizon

# Geometry search space (sampled per bar at TRAINING time; swept at INFERENCE).
# v6 (2026-08-01): regime-specific trade management — the model needs BOTH
# tight range-scalp placements (TP 1.0) AND wide momentum placements (TP 3.0)
# to learn "different management per market condition". 4 SL widths × 3 TP
# ratios × 2 directions = 24 candidates/bar (trimmed from 40 to fit the
# 7GB/1-core machine while keeping the scalp→momentum spectrum).
SL_MULTS = [0.8, 1.2, 1.8, 2.6, 3.4, 4.5]   # SL distance = mult × ATR
TP_RATIOS = [1.3, 1.8, 2.5, 3.0]       # TP = ratio × (SL+spread) — ALL strictly > 1.0
                                  # (user mandate: reward ALWAYS > risk; a 1.0
                                  # ratio is a coin flip, removed 2026-08-02)
MIN_SL_FLOOR = 0.30               # structural sanity: SL must clear the spread

def live_geometry(atr, spread=SPREAD, direction="BUY", sl_mult=None, tp_ratio=None):
    """Compute SL/TP distances for a specific (sl_mult × tp_ratio) placement.
    Returns (sl_dist, tp_dist). When sl_mult/tp_ratio are None, returns the
    full grid of (sl_dist, tp_dist) pairs for ALL candidates (inference sweep).
    Pure function, vectorized (works on scalars AND Series)."""
    if sl_mult is not None and tp_ratio is not None:
        sl_dist = np.maximum(np.asarray(atr, dtype=float) * sl_mult, MIN_SL_FLOOR)
        # TRUE SL distance includes spread (entry fill AND SL trigger eat spread)
        true_sl = sl_dist + spread
        tp_dist = true_sl * tp_ratio
        return sl_dist, tp_dist
    # Full grid sweep (inference + training sampling)
    grid = []
    for m in SL_MULTS:
        for r in TP_RATIOS:
            sd = np.maximum(np.asarray(atr, dtype=float) * m, MIN_SL_FLOOR)
            ts = sd + spread
            grid.append((float(sd), float(ts * r)))
    return grid

def add_trade_target(df, max_bars=MAX_TARGET_BARS, sl_dist=None, tp_dist=None, direction="BUY"):
    """TRADE-REALISTIC target: does price hit TP before SL, under a SPECIFIC
    SL/TP placement (sl_dist, tp_dist in $, direction-aware trigger sides)?
    target=1 → the trade WINS (TP first). Vectorized over the whole series.
    When sl_dist/tp_dist are None, uses the mid-grid placement.

    v7.3d FIX (2026-08-02): levels now match the backtest/live engine EXACTLY.
    Engine: BUY entry = ask = close + spread; SL = entry - sl_dist - spread
    → bid level = close - sl_dist; TP = entry + (sl_dist+spread)*tr → ask
    level = close + spread + tp_dist. SELL entry = bid = close; SL level =
    close + sl_dist + spread; TP level = close - tp_dist.
    The OLD label used close - sl_dist - SPREAD for BUY SL (0.20 too far) and
    close + tp_dist for BUY TP (0.20 too close) → label WR 31.3% vs real
    28.4% at RR 2.5 — the model learned a fictitious market (SELL-bias +
    negative real EV). Fixed: BUY SL = close - sl_dist (no extra spread),
    BUY TP = close + spread + tp_dist. SELL unchanged (was already correct).
    Timeout: neither level hit within max_bars → extend scan to 4×max_bars
    for first-touch (matches the engine's hold-until-hit), else resolve by
    close vs ENTRY (not vs close)."""
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    if sl_dist is None or tp_dist is None:
        grid = live_geometry(df["atr_14"].values, SPREAD)
        mid = grid[len(grid) // 2]
        sl_dist, tp_dist = mid[0], mid[1]
    targets = np.full(n, np.nan)
    # v8 MFE/MFA: max favorable / max adverse excursion (in ATR units) reached
    # BEFORE the first-touch resolution, per placement. These columns let the
    # placement model learn "in this regime, winners run X ATR favorable and
    # losers dip Y ATR adverse" → SL/TP placement is LEARNED from excursion
    # distributions, never hardcoded. NaN where no forward path exists.
    atr = df["atr_14"].values
    mfe = np.full(n, np.nan)   # max favorable excursion (toward TP), in ATR
    mfa = np.full(n, np.nan)   # max adverse excursion (toward SL), in ATR
    sl_dist_a = np.asarray(sl_dist, dtype=float)
    tp_dist_a = np.asarray(tp_dist, dtype=float)
    scalar_sl = sl_dist_a.ndim == 0
    scalar_tp = tp_dist_a.ndim == 0
    ext = max_bars * 4  # extended first-touch scan for timeouts (v7.3d)
    for i in range(n - 1):
        sd = float(sl_dist_a) if scalar_sl else float(sl_dist_a[i])
        td = float(tp_dist_a) if scalar_tp else float(tp_dist_a[i])
        if direction == "BUY":
            entry = closes[i] + SPREAD
            sl_level = entry - sd - SPREAD     # == closes[i] - sd
            tp_level = entry + td              # == closes[i] + SPREAD + td
            j_end = min(i + 1 + max_bars, n)
            seg_lo = lows[i + 1:j_end]
            seg_hi = highs[i + 1:j_end]
            sl_hit = np.where(seg_lo <= sl_level)[0]
            tp_hit = np.where(seg_hi >= tp_level)[0]
        else:
            entry = closes[i]
            sl_level = entry + sd + SPREAD
            tp_level = entry - td
            j_end = min(i + 1 + max_bars, n)
            seg_lo = lows[i + 1:j_end]
            seg_hi = highs[i + 1:j_end]
            sl_hit = np.where(seg_hi >= sl_level)[0]
            tp_hit = np.where(seg_lo <= tp_level)[0]
        if not sl_hit.size and not tp_hit.size:
            # timeout: extend to 4×max_bars for true first-touch (v7.3d)
            j_end2 = min(i + 1 + ext, n)
            if direction == "BUY":
                seg_lo2 = lows[i + 1:j_end2]
                seg_hi2 = highs[i + 1:j_end2]
                sl_hit = np.where(seg_lo2 <= sl_level)[0]
                tp_hit = np.where(seg_hi2 >= tp_level)[0]
            else:
                seg_lo2 = lows[i + 1:j_end2]
                seg_hi2 = highs[i + 1:j_end2]
                sl_hit = np.where(seg_hi2 >= sl_level)[0]
                tp_hit = np.where(seg_lo2 <= tp_level)[0]
        if sl_hit.size and tp_hit.size:
            targets[i] = 0.0 if sl_hit[0] <= tp_hit[0] else 1.0
        elif sl_hit.size:
            targets[i] = 0.0
        elif tp_hit.size:
            targets[i] = 1.0
        else:
            # truly unresolved: compare final close vs ENTRY (not vs close)
            targets[i] = 1.0 if closes[j_end2 - 1] > entry else 0.0
        # v8 excursion: over the window UP TO the first-touch resolution bar
        # (or the extended window when neither hit), measure max favorable /
        # adverse in ATR units. Defines the LEARNED placement: in this regime,
        # winners run X ATR favorable, losers dip Y ATR adverse → SL/TP from
        # excursion distributions, never hardcoded.
        if sl_hit.size and tp_hit.size:
            j_res = i + 1 + int(min(sl_hit[0], tp_hit[0]))
        elif sl_hit.size:
            j_res = i + 1 + int(sl_hit[0])
        elif tp_hit.size:
            j_res = i + 1 + int(tp_hit[0])
        else:
            j_res = j_end2
        if j_res > i + 1:
            seg_hi_f = highs[i + 1:j_res]
            seg_lo_f = lows[i + 1:j_res]
            if direction == "BUY":
                fav = (seg_hi_f - entry)
                adv = (entry - seg_lo_f)
            else:
                fav = (entry - seg_lo_f)
                adv = (seg_hi_f - entry)
            a = atr[i] if atr[i] > 0 else 1.0
            mfe[i] = float(np.max(fav) / a)
            mfa[i] = float(np.max(adv) / a)
    df["target"] = targets
    df["mfe_atr"] = mfe
    df["mfa_atr"] = mfa
    df["fwd_return"] = df["close"] - df["close"].shift(1)
    return df

# ════════════════════════════════════════════════════════════════════
# LEARNED PLACEMENT DATASET (v4 2026-08-01, v6 multi-era)
# Each bar is expanded into (2 directions × SL_MULTS × TP_RATIOS) rows.
# Every row = market features + geometry features for THAT SPECIFIC
# placement + a direction flag + target = did that exact trade win?
# The model learns P(win | market state, placement, direction).
#
# v6: features are built PER CONTIGUOUS PERIOD (gap > 6h splits). Rolling
# windows (ATR, percentiles, MA anchors) must never cross an era boundary
# — a window spanning 2020's $1545 and 2026's $4043 would poison every
# derived feature. Each period is feature-engineered independently, then
# the placement rows are concatenated. The engine's FeatureComputer only
# sees the recent XM window, so live inference is unaffected.
# ════════════════════════════════════════════════════════════════════
# Absolute-price features the model must NOT see (scale-free versions
# replace them — see add_scale_free). Price LEVEL is era-specific
# ($1545 in 2020 vs $4043 in 2026); the PATTERN is what generalizes.
RAW_PRICE_COLS = {"open", "high", "low", "close", "atr_14", "tick_volume",
                  "real_volume", "macd", "macd_sig", "macd_hist",
                  "body", "upper_wick", "lower_wick", "vol_atr_ratio"}

def _feature_block(df_raw):
    """Feature-engineer one contiguous period. Returns df with market
    features (scale-free), geometry awareness cols, and target."""
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    keep = [c for c in ["time","open","high","low","close","tick_volume","spread","real_volume"] if c in df.columns]
    df = df[keep]

    df = add_returns(df); df = add_atr(df); df = add_bb_width(df)
    df = add_ewma_vol(df); df = add_rsi(df); df = add_macd(df)
    df = add_stoch(df); df = add_price_shape(df); df = add_structure(df)
    df = add_volume(df); df = add_regime(df)
    df = add_institutional_levels(df)
    df = add_scale_free(df)
    df = add_htf_context(df)
    df = add_session_clock(df)
    df = add_event_proximity(df)
    df = add_order_flow(df)
    df = add_strategy_playbook(df)
    return df

def build_placement_dataset(df_raw, max_bars=MAX_TARGET_BARS):
    """Long-format training matrix: one row per (bar, direction, placement).
    Features are built per contiguous period (gap > 6h) so multi-era seeds
    (2020 rally + 2026 XM) never cross era boundaries in rolling windows."""
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Split into contiguous periods — gap > 6h (weekend/week/year breaks)
    t = df["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(df)]
    periods = [df.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)]
    periods = [p for p in periods if len(p) >= 300]  # need warmup bars
    print(f"build_placement_dataset: {len(df)} bars -> {len(periods)} contiguous periods")

    blocks = []
    for p in periods:
        fdf = _feature_block(p).dropna().reset_index(drop=True)
        if len(fdf) < 100:
            continue
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values if "spread" in fdf.columns else np.full(len(fdf), SPREAD)
        market_cols = [c for c in fdf.columns
                       if c not in ("time", "target", "fwd_return") and c not in RAW_PRICE_COLS]
        for direction in ("BUY", "SELL"):
            for m in SL_MULTS:
                for r in TP_RATIOS:
                    sl_dist = np.maximum(atr * m, MIN_SL_FLOOR)
                    tp_dist = (sl_dist + spr) * r
                    tdf = add_trade_target(fdf, max_bars=max_bars, sl_dist=sl_dist, tp_dist=tp_dist, direction=direction)
                    gdf = add_geometry_awareness(
                        fdf, sl_dist_buy=sl_dist, tp_dist_buy=tp_dist,
                        sl_dist_sell=sl_dist, tp_dist_sell=tp_dist)
                    out = fdf[market_cols].copy()
                    # Raw prices kept in CSV for backtest simulation only —
                    # excluded from MODEL features via RAW_PRICE_COLS.
                    out["open"] = fdf["open"].values
                    out["high"] = fdf["high"].values
                    out["low"] = fdf["low"].values
                    out["close"] = fdf["close"].values
                    out["spread"] = fdf["spread"].values
                    out["time"] = fdf["time"].values   # keep for sort (dropped from model feats)
                    for c in ("sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
                              "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"):
                        out[c] = gdf[c].values
                    out["direction"] = 1.0 if direction == "BUY" else 0.0
                    out["target"] = tdf["target"].values
                    out = out.dropna().reset_index(drop=True)
                    blocks.append(out)
    if not blocks:
        raise ValueError("no feature blocks produced")
    final = pd.concat(blocks, ignore_index=True)
    # CRITICAL: sort by time so walk-forward row-splits are time-ordered
    # (rows concatenated per (direction, placement) block otherwise).
    # 40 rows per bar (2 dir × 5 sl × 4 tp) — trainer splits on bar boundaries.
    final = final.sort_values("time").reset_index(drop=True)
    return final

# ════════════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════════════
def build_features(df_raw, horizon=15, dropna=True, trade_target=False):
    """Full pipeline: raw OHLCV -> feature matrix + target."""
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Keep only numeric OHLCV columns — drop string metadata (src, etc.)
    keep = [c for c in ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"] if c in df.columns]
    df = df[keep]

    # All feature groups
    df = add_returns(df)
    df = add_atr(df)
    df = add_bb_width(df)
    df = add_ewma_vol(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_stoch(df)
    df = add_price_shape(df)
    df = add_structure(df)
    df = add_volume(df)
    df = add_regime(df)
    df = add_institutional_levels(df)
    df = add_scale_free(df)
    df = add_htf_context(df)
    df = add_session_clock(df)
    df = add_event_proximity(df)
    df = add_order_flow(df)
    df = add_geometry_awareness(df)
    if trade_target:
        df = add_trade_target(df)
    else:
        df = add_target(df, horizon=horizon)

    # Drop NaN rows (warmup + lookahead tail)
    if dropna:
        df = df.dropna().reset_index(drop=True)
    return df

FEATURE_COLS = None

def get_feature_cols():
    global FEATURE_COLS
    return FEATURE_COLS

if __name__ == "__main__":
    import os, sys
    # v6: prefer the multi-era seed (XM + Dukascopy rallies). Fall back to
    # XM-only seed (merge_seed.py output) if rally data isn't merged yet.
    multi = "/home/jith/.hermes/profiles/trading/scripts/gold_seed_multi.csv"
    path = multi if os.path.exists(multi) else "/home/jith/.hermes/profiles/trading/scripts/gold_seed.csv"
    df = pd.read_csv(path)
    print(f"Raw: {len(df)} bars | {df['time'].iloc[0]} -> {df['time'].iloc[-1]} | src: {path.split('/')[-1]}")
    # LEARNED PLACEMENT: long-format dataset (bar × direction × geometry grid)
    fx = build_placement_dataset(df)
    # v6: model sees ONLY scale-free features (+geometry+direction). Raw price
    # cols stay in the CSV for backtest simulation but are NOT model features.
    FEATURE_COLS = [c for c in fx.columns
                    if c not in ("time", "target", "fwd_return") and c not in RAW_PRICE_COLS]
    print(f"Features: {len(FEATURE_COLS)} columns (incl. direction flag)")
    print(f"Matrix: {fx.shape} rows (bars × 2 dir × {len(SL_MULTS)}×{len(TP_RATIOS)} placements)")
    print(f"Target balance: {fx['target'].value_counts().to_dict()}")
    out = "/home/jith/.hermes/profiles/trading/scripts/gold_features.csv"
    fx.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("\nSample feature columns:")
    print(", ".join(FEATURE_COLS[:15]))
    print(f"... +{len(FEATURE_COLS)-15} more")
    fx.to_csv("/home/jith/.hermes/profiles/trading/scripts/gold_features.csv", index=False)
    print("Saved gold_features.csv")

# ════════════════════════════════════════════════════════════════════
# UNIVERSAL REGIME ROUTING (2026-08-04)
# Single source of truth for WHICH specialist fires. The same pure function
# is used at TRAINING time (to bin every gold_features row into a specialist's
# training set) and at LIVE time (engine picks the specialist from fx). No
# separate classifier model — the bin rule IS deterministic and uses exactly
# the features the model sees, so train/live routing can never diverge.
#
# 8 bins ⇒ every move in 6 years lands somewhere; each bin gets its own
# placement ensemble + forward-return regression head (see train_regime_spec).
# Order of evaluation matters: trend first, then comp/vol overlays override.
# ════════════════════════════════════════════════════════════════════
REGIME_NAMES = [
    "STRONG_UP", "UP", "DOWN", "STRONG_DOWN",
    "RANGE_TIGHT", "RANGE_WIDE", "HIGH_VOL", "QUIET_LOW_VOL",
]
REGIME_KEYS = ["trend_ema", "trend_slope", "bb_pctile", "atr_pctile",
               "vol_spike", "news_candle", "rsi_14", "m1_d1_vol_ratio"]

def regime_bin(fx):
    """Map a feature vector (dict-like with regime cols) -> regime bin name.

    Deterministic rule from the SAME 8 features the model trains on:
      trend_ema, trend_slope, bb_pctile, atr_pctile, vol_spike, news_candle,
      rsi_14, m1_d1_vol_ratio. Used identically by trainer and engine.
    """
    try:
        te = float(fx.get("trend_ema", 0.0))
        ts = float(fx.get("trend_slope", 0.0))
        bb = float(fx.get("bb_pctile", 0.5))
        ap = float(fx.get("atr_pctile", 0.5))
        vs = float(fx.get("vol_spike", 0.0))
        ns = float(fx.get("news_candle", 0.0))
        rsi = float(fx.get("rsi_14", 50.0))
        volr = abs(float(fx.get("m1_d1_vol_ratio", 1.0)))

        # Trend bins FIRST (signed trend_ema in ATR units, slope-aligned).
        # 2026-08-06 FIX: vol overlays previously fired before trend detection,
        # stealing trending bars into HIGH_VOL/QUIET bins -> wrong specialist
        # routing (prior 0.504 vs 0.476 on clean trend bins). Docstring always
        # said "trend first, then comp/vol overlays override" — code now matches.
        if te > 1.2 and ts * te > 0:
            return "STRONG_UP"
        if te > 0.4:
            return "UP"
        if te < -1.2 and ts * te > 0:
            return "STRONG_DOWN"
        if te < -0.4:
            return "DOWN"

        # Volatility overlays for NON-TREND bars (news-shock / quiet trump range)
        if vs > 2.0 or ns > 0.4 or ap > 0.85:
            return "HIGH_VOL"
        if ap < 0.15 and ns < 0.2:
            return "QUIET_LOW_VOL"

        # Range bins by compression width
        if bb < 0.35:
            return "RANGE_TIGHT"
        # Momentum within range: high RSI + wide vol ratio -> range-up bias
        if rsi > 60 and volr > 1.2:
            return "UP"
        if rsi < 40 and volr > 1.2:
            return "DOWN"
        return "RANGE_WIDE"
    except Exception:
        return "RANGE_WIDE"
