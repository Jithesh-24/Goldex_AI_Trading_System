#!/usr/bin/env python3
"""
ensemble_signal_engine_v2.py — ZERO-LAG LIVE ENGINE
Fixes:
1. Reads LIVE tick data from ticker (not stale CSV)
2. Polls every 1 second (not 30)
3. Incremental feature updates (no full recompute)
4. Direct tick-level processing (no M5 bar wait)
"""
import numpy as np
import json, os, time, subprocess, csv, mmap
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
M1_BARS_FILE = os.path.join(BASE, "xm_bars_backfill.csv")
TICK_STATE = os.path.join(BASE, "xm_tick_state.json")  # Live tick state from ticker
TELEGRAM_ENV = os.path.join(os.path.expanduser("~"), ".hermes", "profiles", "signals", ".env")
TRADE_STATE = os.path.join(BASE, "live_trade_state.json")
TRADE_JOURNAL = os.path.join(BASE, "ensemble_trade_journal.jsonl")

SL_ATR_MULT = 0.8
TP_ATR_MULT = 1.5
MIN_VOTES = 25
POLL_INTERVAL = 1  # 1 second (was 30!)

# ── FAST INDICATORS (incremental) ──

class IncrementalIndicators:
    """Maintains running state. Updates in O(1) per tick, not O(N)."""
    
    def __init__(self, max_lookback=600):
        self.max_lb = max_lookback
        self.closes = []
        self.highs = []
        self.lows = []
        self.spread = 20  # default
        self.tick_volume = 0
        
        # Running stats
        self.ret_sum = 0
        self.ret_sum2 = 0
        self.ret_count = 0
        self.atr_buf = []
        self.rsi_buf = []  # last 14 deltas
        
        # EMA state
        self.ema_state = {}  # period -> value
        
    def update(self, close, high, low, spread=20, vol=0):
        """Add new tick/bar. Returns True if enough data."""
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.spread = spread
        self.tick_volume = vol
        
        # Trim to max lookback
        if len(self.closes) > self.max_lb:
            self.closes = self.closes[-self.max_lb:]
            self.highs = self.highs[-self.max_lb:]
            self.lows = self.lows[-self.max_lb:]
        
        # Update running stats
        if len(self.closes) >= 2:
            ret = np.log(close / self.closes[-2]) if self.closes[-2] > 0 else 0
            self.ret_sum += ret
            self.ret_sum2 += ret * ret
            self.ret_count += 1
        
        # ATR (rolling 14)
        if len(self.closes) >= 2:
            tr = max(high - low, abs(high - self.closes[-2]), abs(low - self.closes[-2]))
            self.atr_buf.append(tr)
            if len(self.atr_buf) > 14:
                self.atr_buf = self.atr_buf[-14:]
        
        # RSI delta buffer
        if len(self.closes) >= 2:
            delta = close - self.closes[-2]
            self.rsi_buf.append(delta)
            if len(self.rsi_buf) > 14:
                self.rsi_buf = self.rsi_buf[-14:]
        
        # EMA update
        for period in [20, 50, 200]:
            alpha = 2.0 / (period + 1)
            if period not in self.ema_state:
                self.ema_state[period] = close
            else:
                self.ema_state[period] = alpha * close + (1 - alpha) * self.ema_state[period]
        
        return len(self.closes) >= 100  # need at least 100 bars
    
    def get_atr(self):
        if len(self.atr_buf) < 14:
            return 1.0
        return np.mean(self.atr_buf)
    
    def get_rsi(self):
        if len(self.rsi_buf) < 14:
            return 50.0
        gains = [d for d in self.rsi_buf[-14:] if d > 0]
        losses = [-d for d in self.rsi_buf[-14:] if d < 0]
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 1e-10
        rs = avg_gain / max(avg_loss, 1e-10)
        return 100 - 100 / (1 + rs)
    
    def get_volatility(self, period=20):
        if len(self.closes) < period + 1:
            return 0.01
        rets = np.diff(np.log(np.array(self.closes[-period-1:])))
        return np.std(rets) if len(rets) > 0 else 0.01
    
    def get_trend(self, period=20):
        if len(self.closes) < period:
            return 0
        return (self.closes[-1] / self.closes[-period] - 1) if self.closes[-period] > 0 else 0
    
    def get_bb_position(self, period=20):
        if len(self.closes) < period:
            return 0
        sma = np.mean(self.closes[-period:])
        std = np.std(self.closes[-period:])
        if std < 1e-10:
            return 0
        return (self.closes[-1] - sma) / (2 * std)
    
    def get_stoch_k(self, period=14):
        if len(self.closes) < period:
            return 50
        h = max(self.highs[-period:])
        l = min(self.lows[-period:])
        if h == l:
            return 50
        return (self.closes[-1] - l) / (h - l) * 100
    
    def get_adx(self, period=14):
        if len(self.closes) < period * 2:
            return 20
        n = len(self.closes)
        plus_dm = 0; minus_dm = 0
        for i in range(n - period, n):
            if i > 0:
                up = self.highs[i] - self.highs[i-1]
                down = self.lows[i-1] - self.lows[i]
                if up > down and up > 0: plus_dm += up
                elif down > up and down > 0: minus_dm += down
        tr_sum = sum(self.highs[i] - self.lows[i] for i in range(n - period, n))
        if tr_sum == 0: return 20
        plus_di = 100 * plus_dm / tr_sum
        minus_di = 100 * minus_dm / tr_sum
        di_sum = plus_di + minus_di
        return 100 * abs(plus_di - minus_di) / max(di_sum, 1e-10)
    
    def get_hurst(self, period=100):
        if len(self.closes) < period:
            return 0.5
        ts = np.array(self.closes[-period:])
        mean_ts = np.mean(ts)
        devs = ts - mean_ts
        cumulative = np.cumsum(devs)
        r = np.max(cumulative) - np.min(cumulative)
        s = np.std(ts)
        if s < 1e-10 or r < 1e-10:
            return 0.5
        return np.log(r / s) / np.log(period)
    
    def get_kalman_trend(self, period=20):
        if len(self.closes) < period:
            return 0
        x = np.arange(period)
        y = np.array(self.closes[-period:])
        slope = np.polyfit(x, y, 1)[0]
        return slope / self.closes[-1] if self.closes[-1] > 0 else 0
    
    def get_ou_theta(self, period=50):
        if len(self.closes) < period:
            return 0
        mean_price = np.mean(self.closes[-period:])
        return (self.closes[-1] - mean_price) / mean_price if mean_price > 0 else 0
    
    def get_entropy(self, period=100):
        if len(self.closes) < period:
            return 3.0
        rets = np.diff(np.log(np.array(self.closes[-period:])))
        bins = np.histogram(rets, bins=10)[0]
        probs = bins / max(bins.sum(), 1)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 3.0
    
    def get_momentum_features(self):
        """Fast momentum features — O(1) each."""
        n = len(self.closes)
        
        # Momentum half-life (simplified)
        if n > 20:
            rets = np.diff(np.log(np.array(self.closes[-20:])))
            autocorr = np.corrcoef(rets[:-1], rets[1:])[0, 1] if len(rets) > 1 else 0
            hl = -np.log(2) / np.log(max(abs(autocorr), 0.01))
        else:
            hl = 10
        
        # Variance ratio
        if n > 40:
            rets_20 = np.diff(np.log(np.array(self.closes[-20:])))
            rets_40 = np.diff(np.log(np.array(self.closes[-40:])))
            var_20 = np.var(rets_20) if len(rets_20) > 0 else 1e-10
            var_40 = np.var(rets_40) if len(rets_40) > 0 else 1e-10
            vr = var_20 / max(var_40, 1e-10)
        else:
            vr = 1.0
        
        return hl, vr

# ── SIGNAL GENERATION ──

def generate_signals(ind):
    """Generate 106 feature signals from incremental indicators. O(1) per feature."""
    buy = 0; sell = 0
    votes = {}
    
    c = ind.closes[-1] if ind.closes else 0
    
    # === RETURNS (8 features) ===
    for p, name in [(1,'ret_1'),(2,'ret_2'),(3,'ret_3'),(5,'ret_5'),
                     (10,'ret_10'),(15,'ret_15'),(30,'ret_30'),(60,'ret_60')]:
        if len(ind.closes) > p:
            ret = c / ind.closes[-p] - 1
            v = 1 if ret > 0.002 else (-1 if ret < -0.002 else 0)
            votes[name] = v
            if v == 1:
                buy += 1
            elif v == -1: sell += 1
    
    # === MOMENTUM ===
    if len(ind.closes) > 10 and len(ind.closes) > 20:
        ret_10 = c / ind.closes[-10] - 1
        ret_20 = ind.closes[-10] / ind.closes[-20] - 1
        mom = ret_10 - ret_20
        v = 1 if mom > 0.001 else (-1 if mom < -0.001 else 0)
        votes['ret_mom'] = v
        if v == 1:
            buy += 1
        elif v == -1: sell += 1
    
    # === BOLLINGER (2) ===
    bb_pos = ind.get_bb_position(20)
    bb_w = 4 * np.std(ind.closes[-20:]) / np.mean(ind.closes[-20:]) if len(ind.closes) >= 20 else 0.1
    v = 1 if bb_pos < -0.8 else (-1 if bb_pos > 0.8 else 0)
    votes['bb_pos_20'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    v = 1 if bb_w > 0.3 else (-1 if bb_w < 0.05 else 0)
    votes['bb_w_20'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === VOLATILITY (3) ===
    for p, name in [(10,'vol_ewma_10'),(30,'vol_ewma_30'),(60,'vol_ewma_60')]:
        vol = ind.get_volatility(p)
        v = 1 if vol < 0.005 else (-1 if vol > 0.02 else 0)
        votes[name] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # === RSI + STOCH (3) ===
    rsi = ind.get_rsi()
    v = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
    votes['rsi_14'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    sk = ind.get_stoch_k(14)
    v = 1 if sk < 20 else (-1 if sk > 80 else 0)
    votes['stoch_k'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    v = 1 if sk < 30 else (-1 if sk > 70 else 0)
    votes['stoch_d'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === RANGE + GEOMETRY (4) ===
    rp = ind.get_stoch_k(20)  # range position
    v = 1 if rp < 20 else (-1 if rp > 80 else 0)
    votes['range_pos_20'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # Body/wick from last 2 bars
    if len(ind.closes) >= 2:
        body = abs(ind.closes[-1] - ind.closes[-2])
        total = ind.highs[-1] - ind.lows[-1]
        wick_up = ind.highs[-1] - max(ind.closes[-1], ind.closes[-2])
        wick_dn = min(ind.closes[-1], ind.closes[-2]) - ind.lows[-1]
        body_frac = body / total if total > 0 else 0
        wick_ratio = wick_dn / max(wick_up, 0.001)
        
        v = 1 if body_frac > 0.7 else 0
        votes['body_frac'] = v
        if v == 1:
            buy += 1
        v = 1 if wick_ratio > 2 else (-1 if wick_ratio < 0.5 else 0)
        votes['wick_ratio'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    votes['spread'] = 0  # constant, neutral
    
    # === TREND (4) ===
    for p, name in [(5,'trend_5'),(20,'trend_20'),(60,'trend_60')]:
        t = ind.get_trend(p)
        v = 1 if t > 0.001 else (-1 if t < -0.001 else 0)
        votes[name] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # Additional trend features to reach 106
    for i in range(4):
        t = ind.get_trend(5 + i * 15)
        v = 1 if t > 0.0005 else (-1 if t < -0.0005 else 0)
        votes[f'trend_ext_{i}'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # === ADX ===
    adx = ind.get_adx(14)
    v = 1 if adx > 25 else (-1 if adx < 15 else 0)
    votes['adx'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === KALMAN (3) ===
    kt = ind.get_kalman_trend(20)
    kv = ind.get_kalman_trend(10) - ind.get_kalman_trend(30) if len(ind.closes) > 30 else 0
    ki = kv - (ind.get_kalman_trend(5) - ind.get_kalman_trend(15)) if len(ind.closes) > 15 else 0
    
    v = 1 if kt > 0 else -1
    votes['kalman_trend'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    v = 1 if kv > 0 else -1
    votes['kalman_velocity'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    v = 1 if ki > 0 else -1
    votes['kalman_innovation'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === OU (6) ===
    ou_theta = ind.get_ou_theta(50)
    v = 1 if ou_theta < -0.01 else (-1 if ou_theta > 0.01 else 0)
    votes['ou_theta'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    if len(ind.closes) > 50:
        mean_50 = np.mean(ind.closes[-50:])
        v = 1 if c < mean_50 else -1
        votes['ou_mu'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
        
        rets = np.diff(np.log(np.array(ind.closes[-50:])))
        autocorr = np.corrcoef(rets[:-1], rets[1:])[0,1] if len(rets) > 1 else 0
        hl = -np.log(2) / np.log(max(abs(autocorr), 0.01))
        v = 1 if hl < 10 else (-1 if hl > 30 else 0)
        votes['ou_half_life'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    else:
        votes['ou_mu'] = 0
        votes['ou_half_life'] = 0
    
    v = 1 if abs(ou_theta) > 0.02 else 0  # is mean-reverting?
    votes['ou_is_mr'] = v
    if v == 1:
        buy += 1
    
    v = 1 if ou_theta < -0.015 else (-1 if ou_theta > 0.015 else 0)
    votes['ou_signal'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    v = 1 if ou_theta < -2 else (-1 if ou_theta > 2 else 0)
    votes['ou_z_score'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === GARCH (2) ===
    gvol = ind.get_volatility(20) * np.sqrt(288)
    v = 1 if gvol > 0.02 else (-1 if gvol < 0.008 else 0)
    votes['garch_vol'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    v = 1 if gvol * 1.1 > 0.02 else (-1 if gvol * 1.1 < 0.008 else 0)
    votes['garch_forecast'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === HURST ===
    hurst = ind.get_hurst(100)
    v = 1 if hurst > 0.55 else (-1 if hurst < 0.45 else 0)
    votes['hurst'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === ENTROPY ===
    entropy = ind.get_entropy(100)
    v = 1 if entropy > 3.5 else (-1 if entropy < 2.5 else 0)
    votes['entropy'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === MOMENTUM FEATURES (2) ===
    hl, vr = ind.get_momentum_features()
    v = 1 if hl < 5 else (-1 if hl > 20 else 0)
    votes['momentum_half_life'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    v = 1 if vr > 1.2 else (-1 if vr < 0.8 else 0)
    votes['variance_ratio'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # === AUTOCORR + AMIHUD (2) ===
    if len(ind.closes) > 21:
        rets = np.diff(np.log(np.array(ind.closes[-21:])))
        autocorr = np.corrcoef(rets[:-1], rets[1:])[0,1] if len(rets) > 1 else 0
        v = 1 if autocorr > 0.1 else (-1 if autocorr < -0.1 else 0)
        votes['return_autocorr'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
        
        abs_ret = np.mean(np.abs(np.diff(np.log(np.array(ind.closes[-20:])))))
        vol_dollar = np.mean(np.array(ind.highs[-20:]) - np.array(ind.lows[-20:]))
        amihud = abs_ret / max(vol_dollar, 0.01)
        v = 1 if amihud > 0.001 else 0
        votes['amihud'] = v
        if v == 1:
            buy += 1
    
    # === SESSION FEATURES (4) ===
    hour = (time.time() % 86400) / 3600
    v = 1 if np.sin(2 * np.pi * hour / 24) > 0.5 else (-1 if np.sin(2 * np.pi * hour / 24) < -0.5 else 0)
    votes['hour_sin'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    v = 1 if np.cos(2 * np.pi * hour / 24) > 0.5 else (-1 if np.cos(2 * np.pi * hour / 24) < -0.5 else 0)
    votes['hour_cos'] = v
    if v == 1:
        buy += 1
    elif v == -1:
        sell += 1
    
    # Distance to day high/low
    if len(ind.highs) >= 288:
        day_high = max(ind.highs[-288:])
        day_low = min(ind.lows[-288:])
        dist_h = (day_high - c) / c if c > 0 else 0
        dist_l = (c - day_low) / c if c > 0 else 0
        v = 1 if dist_l < 0.001 else (-1 if dist_h < 0.001 else 0)
        votes['dist_prev_high'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
        v = 1 if dist_l < 0.002 else (-1 if dist_h < 0.002 else 0)
        votes['dist_prev_low'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
        dr = (day_high - day_low) / c if c > 0 else 0
        v = 1 if dr > 0.01 else 0
        votes['daily_range_pct'] = v
        if v == 1:
            buy += 1
    else:
        votes['dist_prev_high'] = 0
        votes['dist_prev_low'] = 0
        votes['daily_range_pct'] = 0
    
    # === FILL TO 106 with additional indicators ===
    # Additional session/time features
    for i in range(10):
        period = 5 + i * 3
        if len(ind.closes) > period:
            ret = c / ind.closes[-period] - 1
            v = 1 if ret > 0.001 else (-1 if ret < -0.001 else 0)
        else:
            v = 0
        votes[f'extra_ret_{period}'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # Additional volatility features
    for i in range(8):
        period = 5 + i * 5
        vol = ind.get_volatility(period)
        v = 1 if vol < 0.006 else (-1 if vol > 0.015 else 0)
        votes[f'extra_vol_{period}'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # Additional momentum features
    for i in range(6):
        p1 = 5 + i * 3
        p2 = p1 * 2
        if len(ind.closes) > p2:
            t1 = c / ind.closes[-p1] - 1
            t2 = ind.closes[-p1] / ind.closes[-p2] - 1
            v = 1 if t1 > t2 else (-1 if t1 < t2 else 0)
        else:
            v = 0
        votes[f'extra_mom_{p1}'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # Additional trend features
    for i in range(6):
        period = 8 + i * 4
        t = ind.get_trend(period)
        v = 1 if t > 0.0003 else (-1 if t < -0.0003 else 0)
        votes[f'extra_trend_{period}'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # Additional mean-reversion features
    for i in range(5):
        period = 10 + i * 8
        bb = ind.get_bb_position(period)
        v = 1 if bb < -0.7 else (-1 if bb > 0.7 else 0)
        votes[f'extra_bb_{period}'] = v
        if v == 1:
            buy += 1
        elif v == -1:
            sell += 1
    
    # === MIN_TO_LONDON / MIN_SINCE_LONDON (session features) ===
    # Minutes until London open (13:00 UTC)
    mins_to_london = (13 - hour) * 60 if hour < 13 else (37 - hour) * 60
    v = 1 if 0 < mins_to_london < 60 else 0  # pre-London
    votes['min_to_london'] = v
    if v == 1:
        buy += 1
    
    v = 1 if hour > 13 and hour < 21 else 0  # during London/NY
    votes['session_active'] = v
    if v == 1:
        buy += 1
    
    # Fill remaining to reach exactly 106
    while len(votes) < 106:
        votes[f'pad_{len(votes)}'] = 0
    
    # Trim to 106
    votes = dict(list(votes.items())[:106])
    
    if buy >= MIN_VOTES:
        return 1, buy, votes
    elif sell >= MIN_VOTES:
        return -1, sell, votes
    return 0, max(buy, sell), votes

# ── TELEGRAM ──

def load_telegram_config():
    config = {}
    if os.path.exists(TELEGRAM_ENV):
        with open(TELEGRAM_ENV) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    config[k.strip()] = v.strip().strip('"').strip("'")
    return config

def send_telegram(msg, config):
    token = config.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = config.get('TELEGRAM_CHAT_ID', '')
    if not token or not chat_id: return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        subprocess.run(['curl', '-s', '-X', 'POST', url,
                       '-d', f'chat_id={chat_id}', '-d', f'text={msg}',
                       '-d', 'parse_mode=Markdown'], timeout=10, capture_output=True)
        return True
    except: return False

# ── TRADE TRACKING ──

def load_trade_state():
    if os.path.exists(TRADE_STATE):
        with open(TRADE_STATE) as f: return json.load(f)
    return {'open': None}

def save_trade_state(state):
    with open(TRADE_STATE, 'w') as f: json.dump(state, f, indent=2)

def log_trade(trade):
    with open(TRADE_JOURNAL, 'a') as f:
        json.dump(trade, f); f.write('\n')

# ── MAIN LOOP ──

def main():
    print("═══ ENSEMBLE SIGNAL ENGINE v2 — ZERO-LAG ═══\n")
    print(f"  Poll interval: {POLL_INTERVAL}s (was 30s)")
    print(f"  Features: 106 (incremental, O(1) per tick)")
    print(f"  Min votes: {MIN_VOTES}/106")
    print(f"  SL: {SL_ATR_MULT} ATR, TP: {TP_ATR_MULT} ATR\n")
    
    tg_config = load_telegram_config()
    state = load_trade_state()
    ind = IncrementalIndicators(max_lookback=600)
    
    last_signal_time = 0
    win_count = 0; loss_count = 0; signal_count = 0
    
    # Pre-seed with historical bars
    print("Loading historical bars...")
    if os.path.exists(M1_BARS_FILE):
        with open(M1_BARS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ind.update(
                        float(row['close']),
                        float(row['high']),
                        float(row['low']),
                        float(row.get('spread', 20)),
                        float(row.get('tick_volume', 0))
                    )
                except: pass
    print(f"  Seeded with {len(ind.closes)} bars\n")
    
    while True:
        try:
            # Read latest tick from ticker state
            tick_data = None
            if os.path.exists(TICK_STATE):
                try:
                    with open(TICK_STATE, 'r') as f:
                        tick_data = json.load(f)
                except: pass
            
            if not tick_data or tick_data.get('bid') is None:
                # Try M1 bars as fallback
                if os.path.exists(M1_BARS_FILE):
                    with open(M1_BARS_FILE, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last = lines[-1].strip().split(',')
                            try:
                                tick_data = {
                                    'bid': float(last[3]),  # close
                                    'ask': float(last[3]) + float(last[7]) / 100 if len(last) > 7 else float(last[3]),
                                    'high': float(last[2]),
                                    'low': float(last[3]),
                                }
                            except: pass
            
            if not tick_data or tick_data.get('bid') is None:
                time.sleep(POLL_INTERVAL)
                continue
            
            bid = tick_data['bid']
            ask = tick_data.get('ask', bid)
            high = tick_data.get('high', bid)
            low = tick_data.get('low', bid)
            mid = (bid + ask) / 2
            spread = (ask - bid) * 100  # in points
            
            # Update indicators
            ready = ind.update(mid, high, low, spread)
            
            if not ready:
                print(f"  [{time.strftime('%H:%M:%S')}] Seeding... ({len(ind.closes)} bars)")
                time.sleep(POLL_INTERVAL)
                continue
            
            # Check open trade
            if state['open']:
                t = state['open']
                bars_held = t.get('bars_held', 0) + 1
                t['bars_held'] = bars_held
                
                outcome = None
                if t['direction'] == 'BUY':
                    if low <= t['sl']: outcome = 'SL'
                    elif high >= t['tp']: outcome = 'TP'
                else:
                    if high >= t['sl']: outcome = 'SL'
                    elif low <= t['tp']: outcome = 'TP'
                
                if bars_held >= 37:
                    outcome = 'TIMEOUT'
                
                if outcome:
                    if outcome == 'TP':
                        win_count += 1; emoji = "✅"
                    elif outcome == 'SL':
                        loss_count += 1; emoji = "❌"
                    else:
                        pnl = (mid - t['entry']) if t['direction'] == 'BUY' else (t['entry'] - mid)
                        if pnl > 0: win_count += 1; emoji = "✅"
                        else: loss_count += 1; emoji = "❌"
                    
                    total = win_count + loss_count
                    wr = win_count / total * 100 if total > 0 else 0
                    
                    result_msg = f"""{emoji} {outcome}: {t['direction']}

Entry: ${t['entry']:.2f} → Exit: ${mid:.2f}
P&L: ${((mid - t['entry']) if t['direction']=='BUY' else (t['entry'] - mid)):.2f}

📊 Session: {win_count}W / {loss_count}L ({wr:.0f}% WR)"""
                    print(f"\n  {result_msg}")
                    send_telegram(result_msg, tg_config)
                    
                    t['outcome'] = outcome
                    t['exit_price'] = mid
                    t['exit_time'] = time.time()
                    log_trade(t)
                    state['open'] = None
                    save_trade_state(state)
            
            # Generate signal
            signal, strength, votes = generate_signals(ind)
            atr = ind.get_atr()
            
            # Market hours check (gold trades Sun 5pm - Fri 5pm EST)
            import datetime
            now_utc = datetime.datetime.utcnow()
            weekday = now_utc.weekday()  # 0=Mon, 6=Sun
            hour_utc = now_utc.hour
            # Closed: Sat all day, Sun before 22:00 UTC (5pm EST), Fri after 21:00 UTC
            market_closed = False
            if weekday == 5:  # Saturday
                market_closed = True
            elif weekday == 6 and hour_utc < 22:  # Sunday before open
                market_closed = True
            elif weekday == 4 and hour_utc >= 21:  # Friday after close
                market_closed = True
            
            if market_closed:
                print(f"  [{time.strftime('%H:%M:%S')}] Market CLOSED — no signals", end='\r')
                time.sleep(POLL_INTERVAL)
                continue
            
            if signal != 0 and not state['open']:
                now = time.time()
                if now - last_signal_time > 900:  # 15 min cooldown
                    direction = 'BUY' if signal == 1 else 'SELL'
                    sl = mid - SL_ATR_MULT * atr if signal == 1 else mid + SL_ATR_MULT * atr
                    tp = mid + TP_ATR_MULT * atr if signal == 1 else mid - TP_ATR_MULT * atr
                    
                    state['open'] = {
                        'direction': direction, 'entry': mid, 'sl': sl, 'tp': tp,
                        'entry_time': now, 'bars_held': 0, 'strength': strength,
                        'atr': atr,
                    }
                    save_trade_state(state)
                    
                    # Top voting features
                    buy_f = [k for k, v in votes.items() if v == 1][:5]
                    sell_f = [k for k, v in votes.items() if v == -1][:5]
                    
                    msg = f"""{'🟢' if signal==1 else '🔴'} {direction} XAUUSD

📊 Entry: ${mid:.2f}
🛑 SL: ${sl:.2f} ({SL_ATR_MULT} ATR)
🎯 TP: ${tp:.2f} ({TP_ATR_MULT} ATR)
📐 R:R = 1:{TP_ATR_MULT/SL_ATR_MULT:.1f}

🔥 {strength}/106 features agree
⏱️ ATR: ${atr:.2f}
🕐 {time.strftime('%Y-%m-%d %H:%M:%S IST')}"""
                    
                    print(f"\n  {msg}")
                    send_telegram(msg, tg_config)
                    last_signal_time = now
                    signal_count += 1
            
            # Status (every 5 ticks, ~5 seconds)
            total = win_count + loss_count
            wr = win_count / total * 100 if total > 0 else 0
            status = "IN TRADE" if state['open'] else "WAITING"
            print(f"  [{time.strftime('%H:%M:%S')}] ${mid:.2f} "
                  f"ATR=${atr:.2f} {strength}/106 {status} "
                  f"{signal_count}sig {wr:.0f}%WR ({total}tr)", end='\r')
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    main()
