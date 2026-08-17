#!/usr/bin/env python3
"""
multi_strategy_v3.py — Enhanced with:
1. Regime filter (only trade in favorable conditions)
2. Dynamic position sizing (fractional Kelly)
3. Adaptive SL/TP (based on current volatility regime)
4. Better drawdown management
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

def main():
    t0 = time.time()
    print("═══ MULTI-STRATEGY ENSEMBLE V3 (REGIME-AWARE) ═══\n")
    
    prices = np.load(f"{BASE}/prices_tail.npy")
    n = len(prices)
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    print(f"Data: {n:,} M5 bars, ${closes.min():.2f}–${closes.max():.2f}")
    
    # ═══ PRECOMPUTE ALL INDICATORS ═══
    print("Precomputing indicators...")
    
    atr_14 = compute_atr(highs, lows, closes, 14)
    atr_56 = compute_atr(highs, lows, closes, 56)
    vol_ratio = np.where(atr_56 > 0, atr_14 / atr_56, 1.0)
    
    # D1
    d1_step = 288
    d1_closes = closes[::d1_step]
    d1_ema200 = ema(d1_closes, 200)
    
    # H4
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
    
    # D1 Support/Resistance (rolling 5-day)
    d1_highs_roll = np.array([np.max(highs[max(0,(i*d1_step-5*d1_step)):(i+1)*d1_step]) for i in range(n // d1_step + 1)])
    d1_lows_roll = np.array([np.min(lows[max(0,(i*d1_step-5*d1_step)):(i+1)*d1_step]) for i in range(n // d1_step + 1)])
    
    # Regime detection: ADX-like (trend strength)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
    
    sma_plus = sma(plus_dm, 14)
    sma_minus = sma(minus_dm, 14)
    sma_tr = sma(np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1)))), 14)
    sma_tr[0] = atr_14[0] if len(atr_14) > 0 else 1
    
    plus_di = np.where(sma_tr > 0, 100 * sma_plus / sma_tr, 0)
    minus_di = np.where(sma_tr > 0, 100 * sma_minus / sma_tr, 0)
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)
    adx = sma(dx, 14)
    
    # ADX > 25 = trending, < 20 = ranging
    is_trending = adx > 25
    is_ranging = adx < 20
    
    print(f"  Done ({time.time()-t0:.0f}s)")
    
    # ═══ GENERATE SIGNALS ═══
    print("Generating signals...")
    
    signals = np.zeros(n, dtype=np.int8)
    strengths = np.zeros(n, dtype=np.float64)
    regime_info = np.zeros(n, dtype=np.float64)  # ADX value
    
    for i in range(300, n - 37):
        regime_info[i] = adx[i] if not np.isnan(adx[i]) else 20
        buy_score = 0; sell_score = 0
        
        # ── TREND ──
        d1_idx = min(i // d1_step, len(d1_ema200) - 1)
        h4_idx = min(i // h4_step, len(h4_ema20) - 1)
        
        d1_bull = closes[i] > d1_ema200[d1_idx]
        h4_bull = h4_ema20[h4_idx] > h4_ema50[h4_idx]
        pullback = abs(closes[i] - h4_ema20[h4_idx]) / closes[i] < 0.003
        
        if d1_bull and h4_bull and pullback:
            buy_score += 1.0
        elif not d1_bull and not h4_bull and pullback:
            sell_score += 1.0
        
        # ── MEAN REVERSION ──
        h1_idx = min(i // h1_step, len(h1_sma20) - 1)
        at_lower = closes[i] < h1_bb_lower[h1_idx]
        at_upper = closes[i] > h1_bb_upper[h1_idx]
        
        d1_i = i // d1_step
        d1_start = max(0, d1_i - 5)
        resistance = d1_highs_roll[d1_i] if d1_i < len(d1_highs_roll) else closes[i]
        support = d1_lows_roll[d1_i] if d1_i < len(d1_lows_roll) else closes[i]
        
        at_support = (closes[i] - support) / closes[i] < 0.002
        at_resistance = (resistance - closes[i]) / closes[i] < 0.002
        
        if at_support and at_lower:
            buy_score += 1.0
        elif at_resistance and at_upper:
            sell_score += 1.0
        
        # ── VOLATILITY ──
        vr = vol_ratio[i]
        if vr > 1.5:
            sell_score += 1.0
        elif vr < 0.5:
            recent = closes[i] - closes[max(0, i-20)]
            if recent > 0: buy_score += 1.0
            else: sell_score += 1.0
        
        # ── ENSEMBLE ──
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
    
    # ═══ BACKTEST ═══
    print("\n═══ BACKTESTING ═══")
    
    def bt(sl_m, tp_m, min_str, regime_filter=True, dynamic_sizing=True, 
           base_risk=20.0, max_risk=50.0, trailing_pct=0.6):
        acc = 1000.0; peak = 1000.0; wins=0; losses=0; pnl=0; dd=0; nn=0
        spread = 0.30
        max_dd_pct = 0
        
        for i in range(300, n - 37):
            if signals[i] == 0 or strengths[i] < min_str:
                continue
            if atr_14[i] < 0.1:
                continue
            
            # Regime filter: only trade when ADX shows clear regime
            if regime_filter:
                adx_val = regime_info[i]
                if adx_val < 15:  # No clear trend — skip
                    continue
            
            # Dynamic position sizing based on drawdown
            current_dd = (peak - acc) / peak if peak > 0 else 0
            if dynamic_sizing:
                if current_dd > 0.3:
                    risk_dollar = base_risk * 0.5  # Half size in drawdown
                elif current_dd > 0.1:
                    risk_dollar = base_risk * 0.75
                else:
                    risk_dollar = min(base_risk * 1.5, max_risk)
                # Scale by strength
                risk_dollar *= strengths[i]
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
    
    # Test regime filter ON vs OFF
    print("\n── Without regime filter ──")
    r1 = bt(1.0, 2.0, 0.5, regime_filter=False)
    print(f"  WR={r1['wr']:.1%}, PnL=${r1['pnl']:.2f}, DD={r1['dd']:.1%}, trades={r1['n']}")
    
    print("\n── With regime filter ──")
    r2 = bt(1.0, 2.0, 0.5, regime_filter=True)
    print(f"  WR={r2['wr']:.1%}, PnL=${r2['pnl']:.2f}, DD={r2['dd']:.1%}, trades={r2['n']}")
    
    # Grid search with regime filter
    print("\n── Grid search (with regime filter + dynamic sizing) ──")
    best_pnl = -999999; best = None
    for sl in [0.8, 1.0, 1.5]:
        for tp in [1.5, 2.0, 2.5, 3.0]:
            for ms in [0.5, 0.67]:
                r = bt(sl, tp, ms, regime_filter=True, dynamic_sizing=True)
                if r['n'] > 50 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'sl': sl, 'tp': tp, 'ms': ms, **r}
                    print(f"  SL={sl} TP={tp} str>={ms}: WR={r['wr']:.0%}, "
                          f"PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}, trades={r['n']}")
    
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
        
        years = n / (288 * 365)
        if best['acc'] > 0:
            ann = (best['acc'] / 1000) ** (1/years) - 1
            print(f"\n  Annual return: {ann:.1%}")
            for s in [100, 500, 1000]:
                for y in [1, 2, 3, 5]:
                    print(f"  ${s} → ${s * (1+ann)**y:,.0f} after {y}yr")
    
    # Save config
    if best:
        config = {
            'version': 'ensemble_v3',
            'sl_atr': best['sl'], 'tp_atr': best['tp'],
            'min_strength': best['ms'],
            'regime_filter': True, 'dynamic_sizing': True,
            'trailing_pct': 0.6,
            'win_rate': best['wr'], 'pnl': best['pnl'],
            'drawdown': best['dd'], 'trades': best['n'],
        }
        with open(f"{BASE}/models/ensemble_v3_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")

if __name__ == '__main__':
    main()
