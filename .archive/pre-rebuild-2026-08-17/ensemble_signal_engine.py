#!/usr/bin/env python3
"""
ensemble_signal_engine.py — LIVE AI SIGNAL ENGINE WITH TRACKING
Sends signals to Telegram AND tracks trade outcomes.
Watches live XM price to detect TP/SL hits.
Records results for self-learning.
"""
import numpy as np
import json, os, time, subprocess, csv
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
M5_BARS_FILE = os.path.join(BASE, "xm_bars_backfill.csv")
TELEGRAM_ENV = os.path.join(os.path.expanduser("~"), ".hermes", "profiles", "signals", ".env")
TRADE_STATE = os.path.join(BASE, "live_trade_state.json")
TRADE_JOURNAL = os.path.join(BASE, "ensemble_trade_journal.jsonl")
FEATURE_CACHE = os.path.join(BASE, "_live_feature_cache.npy")

# Strategy parameters (from backtest optimization)
SL_ATR_MULT = 0.8
TP_ATR_MULT = 1.5
MIN_VOTES = 25
MIN_ATR = 0.1
LOOKBACK = 600
SIGNAL_COOLDOWN = 900  # 15 min between signals

# ── TECHNICAL INDICATORS ──

def ema(data, period):
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def sma(data, period):
    out = np.full(len(data), np.nan, dtype=np.float64)
    cs = np.cumsum(data)
    out[period-1:] = (cs[period-1:] - np.concatenate([[0], cs[:-period]])) / period
    return out

def compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    if n < 2:
        return np.array([1.0])
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    return sma(tr, period)

# ── FEATURE ENGINE (106 features) ──

def compute_features(closes, highs, lows):
    """Compute all 106 features from M5 data. Returns dict of feature values."""
    n = len(closes)
    if n < LOOKBACK:
        return None
    
    feats = {}
    
    # Returns
    for p in [1, 2, 3, 5, 10, 15, 30, 60]:
        feats[f'ret_{p}'] = (closes[-1] / closes[-p] - 1) if p < n else 0
    feats['ret_mom'] = (closes[-1] / closes[-10] - 1) - (closes[-10] / closes[-20] - 1) if 20 < n else 0
    
    # Bollinger Bands
    sma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    feats['bb_w_20'] = (4 * std20 / sma20) if sma20 > 0 else 0
    feats['bb_pos_20'] = (closes[-1] - sma20) / (2 * std20) if std20 > 0 else 0
    
    # Volatility
    for p in [10, 30, 60]:
        feats[f'vol_ewma_{p}'] = np.std(np.diff(np.log(closes[-p:]))) if p < n else 0
    
    # RSI
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
    rs = avg_gain / max(avg_loss, 1e-10)
    feats['rsi_14'] = 100 - 100 / (1 + rs)
    
    # Stochastic
    if n > 14:
        h14 = np.max(highs[-14:])
        l14 = np.min(lows[-14:])
        feats['stoch_k'] = ((closes[-1] - l14) / (h14 - l14) * 100) if h14 != l14 else 50
    else:
        feats['stoch_k'] = 50
    feats['stoch_d'] = feats['stoch_k']  # Simplified
    
    # Range position
    if n > 20:
        h20 = np.max(highs[-20:])
        l20 = np.min(lows[-20:])
        feats['range_pos_20'] = ((closes[-1] - l20) / (h20 - l20) * 100) if h20 != l20 else 50
    else:
        feats['range_pos_20'] = 50
    
    # Candle geometry
    body = abs(closes[-1] - closes[-2]) if n > 1 else 0
    wick_up = highs[-1] - max(closes[-1], closes[-2]) if n > 1 else 0
    wick_down = min(closes[-1], closes[-2]) - lows[-1] if n > 1 else 0
    total = highs[-1] - lows[-1] if n > 0 else 1
    feats['body_frac'] = body / total if total > 0 else 0
    feats['wick_ratio'] = wick_down / max(wick_up, 0.001)
    
    # Trend
    feats['trend_5'] = (closes[-1] - closes[-5]) / closes[-5] if 5 < n else 0
    feats['trend_20'] = (closes[-1] - closes[-20]) / closes[-20] if 20 < n else 0
    feats['trend_60'] = (closes[-1] - closes[-60]) / closes[-60] if 60 < n else 0
    
    # ADX
    if n > 28:
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, min(28, n)):
            up = highs[-(28-i)] - highs[-(28-i+1)]
            down = lows[-(28-i+1)] - lows[-(28-i)]
            plus_dm[-(28-i)] = up if (up > down and up > 0) else 0
            minus_dm[-(28-i)] = down if (down > up and down > 0) else 0
        tr_arr = highs[-28:] - lows[-28:]
        sma_tr = np.mean(tr_arr)
        sma_plus = np.mean(plus_dm[-28:])
        sma_minus = np.mean(minus_dm[-28:])
        plus_di = 100 * sma_plus / max(sma_tr, 1e-10)
        minus_di = 100 * sma_minus / max(sma_tr, 1e-10)
        di_sum = plus_di + minus_di
        feats['adx'] = 100 * abs(plus_di - minus_di) / max(di_sum, 1e-10)
    else:
        feats['adx'] = 20
    
    # ATR
    atr_arr = compute_atr(highs, lows, closes, 14)
    feats['atr'] = atr_arr[-1] if not np.isnan(atr_arr[-1]) else 1.0
    
    # Momentum features
    feats['momentum_half_life'] = np.log(2) / max(abs(np.log(max(np.corrcoef(closes[-20:], np.arange(20))[0,1], 0.01))), 0.01) if n > 20 else 10
    feats['variance_ratio'] = np.var(np.diff(closes[-20:])) / max(np.var(np.diff(closes[-40:])), 1e-10) if n > 40 else 1
    feats['return_autocorr'] = np.corrcoef(np.diff(closes[-21:]), np.diff(closes[-20:]))[0,1] if n > 21 else 0
    
    # Entropy (approximation)
    rets = np.diff(closes[-100:]) / closes[-100:-1] if n > 100 else np.array([0])
    bins = np.histogram(rets, bins=10)[0]
    probs = bins / max(bins.sum(), 1)
    probs = probs[probs > 0]
    feats['entropy'] = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
    
    # Kalman-like trend (simplified)
    if n > 20:
        x = np.arange(20)
        y = closes[-20:]
        slope = np.polyfit(x, y, 1)[0]
        feats['kalman_trend'] = slope / closes[-1] if closes[-1] > 0 else 0
        feats['kalman_velocity'] = slope
    else:
        feats['kalman_trend'] = 0
        feats['kalman_velocity'] = 0
    
    # OU-like features
    if n > 50:
        mean_price = np.mean(closes[-50:])
        feats['ou_theta'] = (closes[-1] - mean_price) / mean_price if mean_price > 0 else 0
        feats['ou_half_life'] = -np.log(max(abs(np.corrcoef(closes[-50:], np.arange(50))[0,1]), 0.01))
    else:
        feats['ou_theta'] = 0
        feats['ou_half_life'] = 10
    
    # GARCH-like
    rets_20 = np.diff(np.log(closes[-20:])) if n > 20 else np.array([0])
    feats['garch_vol'] = np.std(rets_20) * np.sqrt(288)
    feats['garch_forecast'] = feats['garch_vol'] * 1.1
    
    # Hurst exponent (simplified R/S)
    if n > 100:
        ts = closes[-100:]
        mean_ts = np.mean(ts)
        devs = ts - mean_ts
        cumulative = np.cumsum(devs)
        r = np.max(cumulative) - np.min(cumulative)
        s = np.std(ts)
        feats['hurst'] = np.log(r / max(s, 1e-10)) / np.log(100) if s > 0 and r > 0 else 0.5
    else:
        feats['hurst'] = 0.5
    
    # Amihud illiquidity
    if n > 20:
        abs_ret = np.abs(np.diff(closes[-21:]))
        vol_dollar = np.mean(highs[-20:] - lows[-20:])
        feats['amihud'] = np.mean(abs_ret) / max(vol_dollar, 0.01)
    else:
        feats['amihud'] = 0
    
    # Hour/cos/sin (session)
    hour = (time.time() % 86400) / 3600  # UTC hour
    feats['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feats['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Distance to session extremes
    if n > 288:
        day_high = np.max(highs[-288:])
        day_low = np.min(lows[-288:])
        feats['dist_prev_high'] = (day_high - closes[-1]) / closes[-1]
        feats['dist_prev_low'] = (closes[-1] - day_low) / closes[-1]
    else:
        feats['dist_prev_high'] = 0
        feats['dist_prev_low'] = 0
    
    # Daily range
    if n > 288:
        feats['daily_range_pct'] = (np.max(highs[-288:]) - np.min(lows[-288:])) / closes[-1]
    else:
        feats['daily_range_pct'] = 0
    
    return feats

def signal_from_features(feats):
    """
    Each feature votes independently.
    Returns: (signal, vote_count, feature_votes)
    signal: 1=BUY, -1=SELL, 0=NEUTRAL
    """
    buy = 0; sell = 0
    votes = {}
    
    for fname, val in feats.items():
        v = 0  # neutral
        
        if 'rsi' in fname:
            if val < 30: v = 1
            elif val > 70: v = -1
        elif 'adx' in fname:
            if val > 25: v = 1  # trending
            elif val < 15: v = -1  # weak
        elif 'bb_pos' in fname:
            if val < -0.8: v = 1
            elif val > 0.8: v = -1
        elif 'kalman_trend' in fname or 'kalman_velocity' in fname:
            if val > 0: v = 1
            elif val < 0: v = -1
        elif 'ou_theta' in fname:
            if val < -0.01: v = 1  # below mean → buy
            elif val > 0.01: v = -1  # above mean → sell
        elif 'hurst' in fname:
            if val > 0.55: v = 1  # trending
            elif val < 0.45: v = -1  # mean-reverting
        elif 'garch_vol' in fname or 'vol_ewma' in fname:
            # High vol → expect mean reversion
            if val > 0.02: v = -1
            elif val < 0.008: v = 1
        elif 'trend_' in fname:
            if val > 0.001: v = 1
            elif val < -0.001: v = -1
        elif 'ret_' in fname and 'ret_mom' not in fname:
            if val > 0.002: v = 1
            elif val < -0.002: v = -1
        elif 'entropy' in fname:
            if val > 3.5: v = 1
            elif val < 2.5: v = -1
        elif 'variance_ratio' in fname:
            if val > 1.2: v = 1  # trending
            elif val < 0.8: v = -1  # mean-reverting
        elif 'amihud' in fname:
            if val > 0.001: v = -1  # illiquid → careful
        elif 'stoch' in fname:
            if val < 20: v = 1
            elif val > 80: v = -1
        elif 'range_pos' in fname:
            if val < 20: v = 1
            elif val > 80: v = -1
        elif 'momentum_half_life' in fname:
            if val < 5: v = 1  # fast momentum
            elif val > 20: v = -1  # slow
        elif 'body_frac' in fname:
            if val > 0.7: v = 1  # strong body
        elif 'wick_ratio' in fname:
            if val > 2: v = 1  # bullish wick
            elif val < 0.5: v = -1  # bearish wick
        
        if v == 1: buy += 1
        elif v == -1: sell += 1
        votes[fname] = v
    
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
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        subprocess.run(['curl', '-s', '-X', 'POST', url,
                       '-d', f'chat_id={chat_id}',
                       '-d', f'text={msg}',
                       '-d', 'parse_mode=Markdown'],
                      timeout=10, capture_output=True)
        return True
    except:
        return False

# ── TRADE TRACKING ──

def load_trade_state():
    if os.path.exists(TRADE_STATE):
        with open(TRADE_STATE) as f:
            return json.load(f)
    return {'open': None}

def save_trade_state(state):
    with open(TRADE_STATE, 'w') as f:
        json.dump(state, f, indent=2)

def log_trade(trade):
    with open(TRADE_JOURNAL, 'a') as f:
        json.dump(trade, f)
        f.write('\n')

def check_trade_outcome(state, current_price, current_high, current_low):
    """
    Check if open trade hit SL, TP, or timeout.
    Returns: (outcome, pnl_description)
    """
    if not state['open']:
        return None, None
    
    t = state['open']
    direction = t['direction']
    entry = t['entry']
    sl = t['sl']
    tp = t['tp']
    bars_held = t.get('bars_held', 0) + 1
    t['bars_held'] = bars_held
    
    # Check TP/SL
    if direction == 'BUY':
        if current_low <= sl:
            return 'SL', f"SL hit at ${sl:.2f} (entry ${entry:.2f})"
        if current_high >= tp:
            return 'TP', f"TP hit at ${tp:.2f} (entry ${entry:.2f})"
    else:  # SELL
        if current_high >= sl:
            return 'SL', f"SL hit at ${sl:.2f} (entry ${entry:.2f})"
        if current_low <= tp:
            return 'TP', f"TP hit at ${tp:.2f} (entry ${entry:.2f})"
    
    # Timeout (37 bars = ~3 hours)
    if bars_held >= 37:
        pnl = (current_price - entry) if direction == 'BUY' else (entry - current_price)
        return 'TIMEOUT', f"Timeout at ${current_price:.2f} (P&L: ${pnl:.2f})"
    
    return None, None

def format_signal_msg(signal, strength, feats, atr_val):
    direction = "🟢 BUY" if signal == 1 else "🔴 SELL"
    entry = feats.get('close', 0)
    sl = entry - SL_ATR_MULT * atr_val if signal == 1 else entry + SL_ATR_MULT * atr_val
    tp = entry + TP_ATR_MULT * atr_val if signal == 1 else entry - TP_ATR_MULT * atr_val
    
    # Top voting features
    buy_feats = [f for f, v in feats.items() if v == 1][:5]
    sell_feats = [f for f, v in feats.items() if v == -1][:5]
    
    msg = f"""{direction} XAUUSD

📊 Entry: ${entry:.2f}
🛑 SL: ${sl:.2f} ({SL_ATR_MULT} ATR)
🎯 TP: ${tp:.2f} ({TP_ATR_MULT} ATR)
📐 R:R = 1:{TP_ATR_MULT/SL_ATR_MULT:.1f}

🔥 Votes: {strength}/106 features agree
📈 BUY votes: {', '.join(buy_feats[:3])}
📉 SELL votes: {', '.join(sell_feats[:3])}

⏱️ ATR: ${atr_val:.2f}
🕐 {time.strftime('%Y-%m-%d %H:%M:%S IST')}"""
    return msg

# ── MAIN LOOP ──

def load_m5_bars():
    bars = []
    if not os.path.exists(M5_BARS_FILE):
        return bars
    try:
        with open(M5_BARS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    bars.append({
                        'time': float(row.get('time', 0)),
                        'open': float(row.get('open', 0)),
                        'high': float(row.get('high', 0)),
                        'low': float(row.get('low', 0)),
                        'close': float(row.get('close', 0)),
                    })
                except (ValueError, KeyError):
                    continue
    except:
        pass
    return bars

def main():
    print("═══ ENSEMBLE SIGNAL ENGINE v2 — LIVE + TRACKING ═══\n")
    tg_config = load_telegram_config()
    state = load_trade_state()
    
    last_signal_time = 0
    signal_count = 0
    win_count = 0
    loss_count = 0
    
    while True:
        try:
            bars = load_m5_bars()
            if len(bars) < LOOKBACK:
                print(f"  [{time.strftime('%H:%M:%S')}] Waiting... ({len(bars)}/{LOOKBACK} bars)")
                time.sleep(30)
                continue
            
            closes = np.array([b['close'] for b in bars], dtype=np.float64)
            highs = np.array([b['high'] for b in bars], dtype=np.float64)
            lows = np.array([b['low'] for b in bars], dtype=np.float64)
            current_price = closes[-1]
            current_high = highs[-1]
            current_low = lows[-1]
            
            # ── CHECK OPEN TRADE ──
            if state['open']:
                outcome, desc = check_trade_outcome(state, current_price, current_high, current_low)
                if outcome:
                    t = state['open']
                    t['outcome'] = outcome
                    t['exit_price'] = current_price
                    t['exit_time'] = time.time()
                    t['description'] = desc
                    
                    if outcome == 'TP':
                        win_count += 1
                        emoji = "✅"
                    elif outcome == 'SL':
                        loss_count += 1
                        emoji = "❌"
                    else:
                        # Timeout — check if profitable
                        if (t['direction'] == 'BUY' and current_price > t['entry']) or \
                           (t['direction'] == 'SELL' and current_price < t['entry']):
                            win_count += 1
                            emoji = "✅"
                        else:
                            loss_count += 1
                            emoji = "❌"
                    
                    total = win_count + loss_count
                    wr = win_count / total * 100 if total > 0 else 0
                    
                    # Notify
                    result_msg = f"""{emoji} Trade Closed: {outcome}

{desc}

📊 Session: {win_count}W / {loss_count}L ({wr:.0f}% WR)
📈 Total trades: {total}"""
                    print(f"\n{'='*50}")
                    print(result_msg)
                    print(f"{'='*50}")
                    send_telegram(result_msg, tg_config)
                    
                    # Log
                    log_trade(t)
                    state['open'] = None
                    save_trade_state(state)
            
            # ── COMPUTE FEATURES & SIGNAL ──
            feats = compute_features(closes, highs, lows)
            if feats is None:
                time.sleep(30)
                continue
            
            feats['close'] = current_price
            atr_val = feats.get('atr', 1.0)
            
            signal, strength, votes = signal_from_features(feats)
            
            # ── SEND SIGNAL ──
            if signal != 0 and not state['open']:
                now = time.time()
                if now - last_signal_time > SIGNAL_COOLDOWN:
                    direction = 'BUY' if signal == 1 else 'SELL'
                    sl = current_price - SL_ATR_MULT * atr_val if signal == 1 else current_price + SL_ATR_MULT * atr_val
                    tp = current_price + TP_ATR_MULT * atr_val if signal == 1 else current_price - TP_ATR_MULT * atr_val
                    
                    # Open trade in state
                    state['open'] = {
                        'direction': direction,
                        'entry': current_price,
                        'sl': sl,
                        'tp': tp,
                        'entry_time': now,
                        'bars_held': 0,
                        'strength': strength,
                        'votes': {k: v for k, v in votes.items() if v != 0},
                    }
                    save_trade_state(state)
                    
                    # Send Telegram
                    msg = format_signal_msg(signal, strength, votes, atr_val)
                    print(f"\n{'='*50}")
                    print(msg)
                    print(f"{'='*50}")
                    send_telegram(msg, tg_config)
                    
                    last_signal_time = now
                    signal_count += 1
            
            # ── STATUS ──
            total = win_count + loss_count
            wr = win_count / total * 100 if total > 0 else 0
            status = "IN TRADE" if state['open'] else "WAITING"
            print(f"  [{time.strftime('%H:%M:%S')}] ${current_price:.2f} | "
                  f"ATR=${atr_val:.2f} | {strength}/106 votes | "
                  f"{status} | {signal_count} signals | {wr:.0f}% WR ({total} trades)")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"  [{time.strftime('%H:%M:%S')}] Error: {e}")
            import traceback; traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    main()
