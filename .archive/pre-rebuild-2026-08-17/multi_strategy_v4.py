#!/usr/bin/env python3
"""
multi_strategy_v4.py — 10-STRATEGY ENSEMBLE BEAST
Strategies:
1. Trend (H4/D1 EMA)
2. Mean Reversion (BB + Support/Resistance)
3. Volatility (ATR regime)
4. Momentum (RSI)
5. MACD Crossover
6. Breakout (Donchian Channel)
7. Range Trading (Keltner Channel)
8. Seasonality (monthly patterns)
9. Time-of-day (session patterns)
10. Price Action (pin bars / engulfing)

Requires 3/10 strategies to agree (majority vote).
"""
import numpy as np
import json, os, time
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

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
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    return sma(tr, period)

def rsi(closes, period=14):
    """Relative Strength Index."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = sma(gains, period)
    avg_loss = sma(losses, period)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi_val = 100 - 100 / (1 + rs)
    return np.concatenate([[50], rsi_val])  # Pad first value

def macd(closes, fast=12, slow=26, signal=9):
    """MACD indicator."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def donchian(highs, lows, period=20):
    """Donchian Channel."""
    upper = np.full(len(highs), np.nan)
    lower = np.full(len(lows), np.nan)
    for i in range(period-1, len(highs)):
        upper[i] = np.max(highs[i-period+1:i+1])
        lower[i] = np.min(lows[i-period+1:i+1])
    return upper, lower

def keltner(closes, highs, lows, ema_period=20, atr_mult=1.5):
    """Keltner Channel."""
    mid = ema(closes, ema_period)
    atr_arr = compute_atr(highs, lows, closes, ema_period)
    upper = mid + atr_mult * atr_arr
    lower = mid - atr_mult * atr_arr
    return upper, mid, lower

def main():
    t0 = time.time()
    print("═══ 10-STRATEGY ENSEMBLE BEAST ═══\n")
    
    prices = np.load(f"{BASE}/prices_tail.npy")
    n = len(prices)
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    print(f"Data: {n:,} M5 bars, ${closes.min():.2f}–${closes.max():.2f}")
    
    # ═══ PRECOMPUTE ALL INDICATORS ═══
    print("Precomputing indicators (10 strategies)...")
    
    # ATR
    atr_14 = compute_atr(highs, lows, closes, 14)
    atr_56 = compute_atr(highs, lows, closes, 56)
    vol_ratio = np.where(atr_56 > 0, atr_14 / atr_56, 1.0)
    
    # D1 EMA200
    d1_step = 288
    d1_closes = closes[::d1_step]
    d1_ema200 = ema(d1_closes, 200) if len(d1_closes) > 200 else np.zeros(len(d1_closes))
    
    # H4 EMA20/50
    h4_step = 48
    h4_closes = closes[::h4_step]
    h4_ema20 = ema(h4_closes, 20)
    h4_ema50 = ema(h4_closes, 50)
    
    # H1 Bollinger Bands
    h1_step = 12
    h1_closes = closes[::h1_step]
    h1_sma20 = sma(h1_closes, 20)
    h1_std20 = np.array([np.std(h1_closes[max(0,i-19):i+1]) for i in range(len(h1_closes))])
    h1_bb_upper = h1_sma20 + 2.0 * h1_std20
    h1_bb_lower = h1_sma20 - 2.0 * h1_std20
    
    # RSI (H1)
    h1_rsi = rsi(h1_closes, 14)
    
    # MACD (H1)
    h1_macd, h1_macd_signal, h1_macd_hist = macd(h1_closes, 12, 26, 9)
    
    # Donchian Channel (H1)
    h1_don_upper, h1_don_lower = donchian(
        np.array([np.max(highs[i*h1_step:(i+1)*h1_step]) for i in range(n // h1_step + 1)]),
        np.array([np.min(lows[i*h1_step:(i+1)*h1_step]) for i in range(n // h1_step + 1)]),
        20
    )
    
    # Keltner Channel (H1)
    h1_kc_upper, h1_kc_mid, h1_kc_lower = keltner(h1_closes, 
        np.array([np.max(highs[i*h1_step:(i+1)*h1_step]) for i in range(n // h1_step + 1)]),
        np.array([np.min(lows[i*h1_step:(i+1)*h1_step]) for i in range(n // h1_step + 1)]),
        20, 1.5)
    
    # ADX
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
    sma_plus = sma(plus_dm, 14)
    sma_minus = sma(minus_dm, 14)
    tr_arr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    sma_tr = sma(tr_arr, 14)
    plus_di = np.where(sma_tr > 0, 100 * sma_plus / sma_tr, 0)
    minus_di = np.where(sma_tr > 0, 100 * sma_minus / sma_tr, 0)
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)
    adx = sma(dx, 14)
    
    print(f"  Done ({time.time()-t0:.0f}s)")
    
    # ═══ GENERATE SIGNALS ═══
    print("Generating signals (10 strategies)...")
    
    n_h1 = len(h1_closes)
    signals = np.zeros(n, dtype=np.int8)
    strengths = np.zeros(n, dtype=np.float64)
    strategy_votes = np.zeros((n, 10), dtype=np.int8)  # Track individual votes
    
    for i in range(300, n - 37):
        buy_score = 0; sell_score = 0
        
        # ── S1: TREND (H4/D1 EMA) ──
        d1_idx = min(i // d1_step, len(d1_ema200) - 1)
        h4_idx = min(i // h4_step, len(h4_ema20) - 1)
        d1_bull = closes[i] > d1_ema200[d1_idx]
        h4_bull = h4_ema20[h4_idx] > h4_ema50[h4_idx]
        pullback = abs(closes[i] - h4_ema20[h4_idx]) / closes[i] < 0.003
        if d1_bull and h4_bull and pullback:
            buy_score += 1; strategy_votes[i, 0] = 1
        elif not d1_bull and not h4_bull and pullback:
            sell_score += 1; strategy_votes[i, 0] = -1
        
        # ── S2: MEAN REVERSION (BB + SR) ──
        h1_idx = min(i // h1_step, len(h1_sma20) - 1)
        at_lower = closes[i] < h1_bb_lower[h1_idx]
        at_upper = closes[i] > h1_bb_upper[h1_idx]
        d1_i = min(i // d1_step, len(d1_ema200) - 1)
        support = np.min(lows[max(0, (d1_i-5)*d1_step):d1_i*d1_step]) if d1_i > 5 else closes[i]
        resistance = np.max(highs[max(0, (d1_i-5)*d1_step):d1_i*d1_step]) if d1_i > 5 else closes[i]
        at_support = (closes[i] - support) / closes[i] < 0.002
        at_resistance = (resistance - closes[i]) / closes[i] < 0.002
        if at_support and at_lower:
            buy_score += 1; strategy_votes[i, 1] = 1
        elif at_resistance and at_upper:
            sell_score += 1; strategy_votes[i, 1] = -1
        
        # ── S3: VOLATILITY ──
        vr = vol_ratio[i]
        if vr > 1.5:
            sell_score += 1; strategy_votes[i, 2] = -1
        elif vr < 0.5:
            recent = closes[i] - closes[max(0, i-20)]
            if recent > 0:
                buy_score += 1; strategy_votes[i, 2] = 1
            else:
                sell_score += 1; strategy_votes[i, 2] = -1
        
        # ── S4: RSI ──
        h1_i = min(i // h1_step, len(h1_rsi) - 1)
        rsi_val = h1_rsi[h1_i]
        if rsi_val < 30:
            buy_score += 1; strategy_votes[i, 3] = 1
        elif rsi_val > 70:
            sell_score += 1; strategy_votes[i, 3] = -1
        
        # ── S5: MACD CROSSOVER ──
        h1_i = min(i // h1_step, len(h1_macd_hist) - 1)
        if h1_i > 0:
            if h1_macd_hist[h1_i] > 0 and h1_macd_hist[h1_i-1] <= 0:
                buy_score += 1; strategy_votes[i, 4] = 1
            elif h1_macd_hist[h1_i] < 0 and h1_macd_hist[h1_i-1] >= 0:
                sell_score += 1; strategy_votes[i, 4] = -1
        
        # ── S6: DONCHIAN BREAKOUT ──
        h1_i = min(i // h1_step, len(h1_don_upper) - 1)
        if not np.isnan(h1_don_upper[h1_i]):
            if closes[i] > h1_don_upper[h1_i]:
                buy_score += 1; strategy_votes[i, 5] = 1
            elif closes[i] < h1_don_lower[h1_i]:
                sell_score += 1; strategy_votes[i, 5] = -1
        
        # ── S7: KELTNER CHANNEL ──
        h1_i = min(i // h1_step, len(h1_kc_upper) - 1)
        if not np.isnan(h1_kc_upper[h1_i]):
            if closes[i] < h1_kc_lower[h1_i]:
                buy_score += 1; strategy_votes[i, 6] = 1
            elif closes[i] > h1_kc_upper[h1_i]:
                sell_score += 1; strategy_votes[i, 6] = -1
        
        # ── S8: SEASONALITY ──
        # Gold tends to be bullish in Jan, Aug, Oct; bearish in Jun, Sep
        month = (i // (288 * 30)) % 12 + 1  # Approximate month
        if month in [1, 8, 10]:  # Bullish months
            buy_score += 1; strategy_votes[i, 7] = 1
        elif month in [6, 9]:  # Bearish months
            sell_score += 1; strategy_votes[i, 7] = -1
        
        # ── S9: TIME-OF-DAY ──
        # London open (13:00 UTC) and NY open (18:00 UTC) are strong
        bar_hour = (i % 288) // 12  # Hour of day (0-23)
        if bar_hour in [13, 14, 18, 19]:  # London/NY sessions
            recent = closes[i] - closes[max(0, i-12)]
            if recent > 0:
                buy_score += 1; strategy_votes[i, 8] = 1
            else:
                sell_score += 1; strategy_votes[i, 8] = -1
        
        # ── S10: PRICE ACTION (Pin bars / Engulfing) ──
        body = abs(closes[i] - closes[i-1])
        wick_up = highs[i] - max(closes[i], closes[i-1])
        wick_down = min(closes[i], closes[i-1]) - lows[i]
        
        # Bullish pin bar (long lower wick)
        if wick_down > 2 * body and wick_up < body:
            buy_score += 1; strategy_votes[i, 9] = 1
        # Bearish pin bar (long upper wick)
        elif wick_up > 2 * body and wick_down < body:
            sell_score += 1; strategy_votes[i, 9] = -1
        # Bullish engulfing
        elif closes[i-1] < closes[i-2] and closes[i] > closes[i-1] and body > abs(closes[i-1] - closes[i-2]):
            buy_score += 1; strategy_votes[i, 9] = 1
        # Bearish engulfing
        elif closes[i-1] > closes[i-2] and closes[i] < closes[i-1] and body > abs(closes[i-1] - closes[i-2]):
            sell_score += 1; strategy_votes[i, 9] = -1
        
        # ── ENSEMBLE VOTE (3/10 required) ──
        if buy_score >= 3:
            signals[i] = 1
            strengths[i] = buy_score / 10.0
        elif sell_score >= 3:
            signals[i] = -1
            strengths[i] = sell_score / 10.0
    
    buy_count = (signals == 1).sum()
    sell_count = (signals == -1).sum()
    print(f"  BUY: {buy_count:,} ({buy_count/n*100:.2f}%)")
    print(f"  SELL: {sell_count:,} ({sell_count/n*100:.2f}%)")
    
    # ═══ BACKTEST ═══
    print("\n═══ BACKTESTING ═══")
    
    def bt(sl_m, tp_m, min_str, min_votes=3, regime_filter=True, 
           dynamic_sizing=True, base_risk=20.0, trailing_pct=0.6):
        acc = 1000.0; peak = 1000.0; wins=0; losses=0; pnl=0; dd=0; nn=0
        spread = 0.30
        for i in range(300, n - 37):
            if signals[i] == 0 or strengths[i] < min_str:
                continue
            if atr_14[i] < MIN_ATR:
                continue
            
            # Regime filter
            if regime_filter:
                adx_val = adx[i] if not np.isnan(adx[i]) else 20
                if adx_val < 15:
                    continue
            
            # Dynamic sizing
            current_dd = (peak - acc) / peak if peak > 0 else 0
            if dynamic_sizing:
                if current_dd > 0.3:
                    risk_dollar = base_risk * 0.5
                elif current_dd > 0.1:
                    risk_dollar = base_risk * 0.75
                else:
                    risk_dollar = min(base_risk * 1.5, 50.0)
                risk_dollar *= min(strengths[i] * 2, 1.5)  # Scale by strength
            else:
                risk_dollar = base_risk
            
            entry = closes[i]; d = signals[i]
            sl_d = sl_m * atr_14[i]; tp_d = tp_m * atr_14[i]
            if d == 1: sl_p = entry - sl_d; tp_p = entry + tp_d
            else: sl_p = entry + sl_d; tp_p = entry - tp_d
            
            out = 0; bp = entry
            for j in range(i+1, min(i+37, n)):
                if d == 1:
                    if lows[j] <= sl_p: out = -(entry - sl_p); break
                    bp = max(bp, highs[j])
                    nsl = bp - sl_d * trailing_pct
                    if nsl > sl_p: sl_p = nsl
                    if highs[j] >= tp_p: out = (tp_p - entry); break
                else:
                    if highs[j] >= sl_p: out = -(sl_p - entry); break
                    bp = min(bp, lows[j])
                    nsl = bp + sl_d * trailing_pct
                    if nsl < sl_p: sl_p = nsl
                    if lows[j] <= tp_p: out = (entry - tp_p); break
            
            if out == 0:
                out = (closes[min(i+37, n-1)] - entry) * d
            
            p = risk_dollar * (out / sl_d) - spread * 2
            acc += p; pnl += p; nn += 1
            if p > 0: wins += 1
            else: losses += 1
            if acc > peak: peak = acc
            d2 = (peak - acc) / peak if peak > 0 else 0
            if d2 > dd: dd = d2
            if acc <= 0: break
        
        return {'n': nn, 'wr': wins/max(wins+losses,1), 'pnl': pnl, 'acc': acc, 'dd': dd, 'w': wins, 'l': losses}
    
    MIN_ATR = 0.1
    
    # Test different vote thresholds
    print("\n── Testing vote thresholds ──")
    for min_v in [2, 3, 4, 5]:
        r = bt(0.8, 1.5, 0.3, min_votes=min_v)
        print(f"  {min_v}/10 votes: WR={r['wr']:.1%}, PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}, trades={r['n']}")
    
    # Grid search
    print("\n── Grid search (best config) ──")
    best_pnl = -999999; best = None
    for sl in [0.8, 1.0, 1.5]:
        for tp in [1.5, 2.0, 2.5, 3.0]:
            for mv in [3, 4, 5]:
                r = bt(sl, tp, 0.3, min_votes=mv)
                if r['n'] > 50 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'sl': sl, 'tp': tp, 'mv': mv, **r}
                    print(f"  SL={sl} TP={tp} votes>={mv}: WR={r['wr']:.0%}, "
                          f"PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}, trades={r['n']}")
    
    if best:
        print(f"\n═══ BEST RESULT ═══")
        print(f"  SL: {best['sl']} ATR, TP: {best['tp']} ATR")
        print(f"  Min votes: {best['mv']}/10")
        print(f"  Trades: {best['n']:,}")
        print(f"  Win rate: {best['wr']:.1%}")
        print(f"  P&L: ${best['pnl']:,.2f}")
        print(f"  Final account: ${best['acc']:,.2f}")
        print(f"  Max drawdown: {best['dd']:.1%}")
        print(f"  Return: {(best['acc']/1000-1)*100:.1f}%")
        print(f"  Profit factor: {best['w']/max(best['l'],1):.2f}")
        
        years = n / (288 * 365)
        if best['acc'] > 0:
            ann = (best['acc'] / 1000) ** (1/years) - 1
            print(f"\n  Annual return: {ann:.1%}")
            for s in [100, 500, 1000]:
                for y in [1, 2, 3, 5]:
                    print(f"  ${s} → ${s * (1+ann)**y:,.0f} after {y}yr")
    
    # Individual strategy performance
    print(f"\n═══ INDIVIDUAL STRATEGY PERFORMANCE ═══")
    strat_names = ['Trend', 'MeanRev', 'Volatility', 'RSI', 'MACD', 
                   'Donchian', 'Keltner', 'Seasonality', 'TimeOfDay', 'PriceAction']
    for idx, name in enumerate(strat_names):
        sig = np.zeros(n, dtype=np.int8)
        sig[strategy_votes[:, idx] == 1] = 1
        sig[strategy_votes[:, idx] == -1] = -1
        buy_n = (sig == 1).sum()
        sell_n = (sig == -1).sum()
        if buy_n + sell_n > 100:
            r = bt(1.5, 3.0, 0.0, min_votes=1, regime_filter=False, dynamic_sizing=False)
            print(f"  {name:12s}: {buy_n:5d} buys, {sell_n:5d} sells, "
                  f"WR={r['wr']:.1%}, trades={r['n']}")
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")

if __name__ == '__main__':
    main()
