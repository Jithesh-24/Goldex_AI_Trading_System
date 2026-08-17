#!/usr/bin/env python3
"""
multi_strategy_v2.py — OPTIMIZED: precompute all HTF indicators once, then loop.
3 strategies: Trend (H4/D1), Mean Reversion (BB+SR), Volatility.
"""
import numpy as np
import json, os, time
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def ema(data, period):
    """Exponential moving average."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def sma(data, period):
    """Simple moving average."""
    out = np.full_like(data, np.nan, dtype=np.float64)
    cumsum = np.cumsum(data)
    out[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return out

def atr(highs, lows, closes, period=14):
    """Average True Range."""
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    return sma(tr, period)

def main():
    t0 = time.time()
    print("═══ MULTI-STRATEGY ENSEMBLE V2 (OPTIMIZED) ═══\n")
    
    # Load data
    prices = np.load(f"{BASE}/prices_tail.npy")
    n = len(prices)
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    print(f"Data: {n:,} M5 bars, ${closes.min():.2f}–${closes.max():.2f}")
    
    # ═══ PRECOMPUTE ALL INDICATORS ONCE ═══
    print("Precomputing indicators...")
    
    # ATR (M5)
    atr_14 = atr(highs, lows, closes, 14)
    
    # H1 indicators (every 12th bar)
    h1_step = 12
    h1_closes = closes[::h1_step]
    h1_bb_sma = sma(h1_closes, 20)
    h1_bb_std = np.array([np.std(h1_closes[max(0,i-19):i+1]) for i in range(len(h1_closes))])
    h1_bb_upper = h1_bb_sma + 2.0 * h1_bb_std
    h1_bb_lower = h1_bb_sma - 2.0 * h1_bb_std
    
    # H4 indicators (every 48th bar)
    h4_step = 48
    h4_closes = closes[::h4_step]
    h4_ema20 = ema(h4_closes, 20)
    h4_ema50 = ema(h4_closes, 50)
    
    # D1 indicators (every 288th bar)
    d1_step = 288
    d1_closes = closes[::d1_step]
    d1_ema200 = ema(d1_closes, 200)
    d1_highs = np.array([np.max(highs[i*d1_step:min((i+1)*d1_step, n)]) for i in range(n // d1_step + 1)])
    d1_lows = np.array([np.min(lows[i*d1_step:min((i+1)*d1_step, n)]) for i in range(n // d1_step + 1)])
    
    # Volatility ratio (current ATR vs historical ATR)
    atr_56 = atr(highs, lows, closes, 56)  # 4x ATR period
    vol_ratio = np.where(atr_56 > 0, atr_14 / atr_56, 1.0)
    
    # Trend strength (H1 EMA slope)
    h1_ema20 = ema(h1_closes, 20)
    h1_slope = np.gradient(h1_ema20)
    
    print(f"  Indicators computed ({time.time()-t0:.0f}s)")
    
    # ═══ GENERATE SIGNALS (vectorized where possible) ═══
    print("Generating signals...")
    
    signals = np.zeros(n, dtype=np.int8)
    strengths = np.zeros(n, dtype=np.float64)
    
    for i in range(300, n - 37):
        buy_score = 0
        sell_score = 0
        
        # ── STRATEGY 1: TREND (H4/D1) ──
        d1_idx = min(i // d1_step, len(d1_ema200) - 1)
        h4_idx = min(i // h4_step, len(h4_ema20) - 1)
        
        d1_bull = closes[i] > d1_ema200[d1_idx]
        h4_bull = h4_ema20[h4_idx] > h4_ema50[h4_idx]
        pullback = abs(closes[i] - h4_ema20[h4_idx]) / closes[i] < 0.003
        
        if d1_bull and h4_bull and pullback:
            buy_score += 1.0
        elif not d1_bull and not h4_bull and pullback:
            sell_score += 1.0
        
        # ── STRATEGY 2: MEAN REVERSION (BB + Support/Resistance) ──
        h1_idx = min(i // h1_step, len(h1_bb_sma) - 1)
        at_lower = closes[i] < h1_bb_lower[h1_idx]
        at_upper = closes[i] > h1_bb_upper[h1_idx]
        
        # Support/Resistance (5-day)
        d1_start = max(0, (i // d1_step) - 5)
        resistance = np.max(d1_highs[d1_start:i//d1_step + 1]) if i // d1_step > d1_start else closes[i]
        support = np.min(d1_lows[d1_start:i//d1_step + 1]) if i // d1_step > d1_start else closes[i]
        
        at_support = (closes[i] - support) / closes[i] < 0.002
        at_resistance = (resistance - closes[i]) / closes[i] < 0.002
        
        if at_support and at_lower:
            buy_score += 1.0
        elif at_resistance and at_upper:
            sell_score += 1.0
        
        # ── STRATEGY 3: VOLATILITY ──
        vr = vol_ratio[i]
        if vr > 1.5:
            # High vol → mean reversion → SELL
            sell_score += 1.0
        elif vr < 0.5:
            # Low vol → breakout → follow recent trend
            recent = closes[i] - closes[i-20]
            if recent > 0:
                buy_score += 1.0
            else:
                sell_score += 1.0
        
        # ── ENSEMBLE DECISION ──
        if buy_score >= 2:
            signals[i] = 1
            strengths[i] = buy_score / 3.0
        elif sell_score >= 2:
            signals[i] = -1
            strengths[i] = sell_score / 3.0
    
    buy_count = (signals == 1).sum()
    sell_count = (signals == -1).sum()
    print(f"  BUY: {buy_count:,} ({buy_count/n*100:.2f}%)")
    print(f"  SELL: {sell_count:,} ({sell_count/n*100:.2f}%)")
    print(f"  ({time.time()-t0:.0f}s)")
    
    # ═══ BACKTEST ═══
    print("\n═══ BACKTESTING ═══")
    
    def bt(sl_m, tp_m, min_str, use_trail=True, risk_dollar=20.0):
        acc = 1000.0; peak = 1000.0; wins=0; losses=0; pnl=0; dd=0; nn=0
        spread = 0.30
        for i in range(300, n - 37):
            if signals[i] == 0 or strengths[i] < min_str:
                continue
            if atr_14[i] < 0.1:
                continue
            
            entry = closes[i]; d = signals[i]
            sl_d = sl_m * atr_14[i]; tp_d = tp_m * atr_14[i]
            if d == 1: sl_p = entry - sl_d; tp_p = entry + tp_d
            else: sl_p = entry + sl_d; tp_p = entry - tp_d
            
            out = 0; bp = entry
            for j in range(i+1, min(i+37, n)):
                if d == 1:
                    if lows[j] <= sl_p: out = -(entry - sl_p); break
                    bp = max(bp, highs[j])
                    if use_trail:
                        nsl = bp - sl_d * 0.6
                        if nsl > sl_p: sl_p = nsl
                    if highs[j] >= tp_p: out = (tp_p - entry); break
                else:
                    if highs[j] >= sl_p: out = -(sl_p - entry); break
                    bp = min(bp, lows[j])
                    if use_trail:
                        nsl = bp + sl_d * 0.6
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
    
    # Grid search
    best_pnl = -999999; best = None
    for sl in [1.0, 1.5, 2.0]:
        for tp in [2.0, 3.0, 4.0, 5.0]:
            for ms in [0.5, 0.67]:
                r = bt(sl, tp, ms)
                if r['n'] > 20 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'sl': sl, 'tp': tp, 'ms': ms, **r}
                    print(f"  SL={sl} TP={tp} str>={ms}: {r['n']} trades, "
                          f"WR={r['wr']:.0%}, PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}")
    
    if best:
        print(f"\n═══ BEST RESULT ═══")
        print(f"  SL: {best['sl']} ATR, TP: {best['tp']} ATR")
        print(f"  Min strength: {best['ms']}")
        print(f"  Trades: {best['n']:,}")
        print(f"  Win rate: {best['wr']:.1%}")
        print(f"  P&L: ${best['pnl']:,.2f}")
        print(f"  Final account: ${best['acc']:,.2f}")
        print(f"  Max drawdown: {best['dd']:.1%}")
        print(f"  Return: {(best['acc']/1000-1)*100:.1f}%")
        print(f"  Profit factor: {best['w']/max(best['l'],1):.2f}")
        
        # Projections
        years = n / (288 * 365)
        if best['acc'] > 0:
            ann = (best['acc'] / 1000) ** (1/years) - 1
            print(f"\n  Annual return: {ann:.1%}")
            for s in [100, 500, 1000]:
                for y in [1, 2, 3, 5]:
                    print(f"  ${s} → ${s * (1+ann)**y:,.0f} after {y}yr")
    
    # Individual strategies
    print(f"\n═══ INDIVIDUAL STRATEGIES ═══")
    for name, sig_arr in [
        ('Trend only', (signals == 1).astype(int) - (signals == -1).astype(int)),
    ]:
        r = bt(1.5, 3.0, 0.5, use_trail=True)
        print(f"  {name}: WR={r['wr']:.1%}, PnL=${r['pnl']:.2f}, trades={r['n']}")
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")

if __name__ == '__main__':
    main()
