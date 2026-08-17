#!/usr/bin/env python3
"""
feature_ensemble.py — THE REAL BEAST
Every feature IS a strategy. Each generates a signal.
Combined via voting. This is how Renaissance actually works.

106 features × individual logic = 106 strategies
Need 15+/106 to agree = massive diversification
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

def percentile_signal(values, lookback=1000, buy_pct=10, sell_pct=90):
    """Generate signal based on rolling percentile."""
    n = len(values)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(lookback, n):
        window = values[max(0, i-lookback):i]
        p_buy = np.percentile(window, buy_pct)
        p_sell = np.percentile(window, sell_pct)
        if values[i] < p_buy:
            signals[i] = 1  # Buy when below 10th percentile
        elif values[i] > p_sell:
            signals[i] = -1  # Sell when above 90th percentile
    return signals

def zscore_signal(values, lookback=1000, threshold=2.0):
    """Generate signal based on rolling z-score."""
    n = len(values)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(lookback, n):
        window = values[max(0, i-lookback):i]
        mu = np.mean(window)
        std = np.std(window)
        if std > 0:
            z = (values[i] - mu) / std
            if z < -threshold:
                signals[i] = 1  # Buy when oversold
            elif z > threshold:
                signals[i] = -1  # Sell when overbought
    return signals

def crossover_signal(fast, slow):
    """Generate signal from EMA crossover."""
    n = len(fast)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if fast[i] > slow[i] and fast[i-1] <= slow[i-1]:
            signals[i] = 1  # Bullish crossover
        elif fast[i] < slow[i] and fast[i-1] >= slow[i-1]:
            signals[i] = -1  # Bearish crossover
    return signals

def momentum_signal(values, period=20):
    """Generate signal from momentum."""
    n = len(values)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(period, n):
        mom = values[i] - values[i-period]
        if mom > 0:
            signals[i] = 1
        elif mom < 0:
            signals[i] = -1
    return signals

def main():
    t0 = time.time()
    print("═══ FEATURE ENSEMBLE — 106 STRATEGIES ═══\n")
    
    # Load data
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    mmap_rows = 9999963
    fm = json.load(open(f"{BASE}/models/feature_map.json"))
    live_idx = fm['live_indices']
    n_live = len(live_idx)
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(mmap_rows, nf))
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    n = len(closes)
    
    print(f"Data: {n:,} M5 bars × {n_live} features")
    print(f"Each feature = 1 strategy = 1 market view")
    
    # ═══ EXTRACT ALL FEATURES ═══
    print("\nExtracting features...")
    X_all = np.empty((mmap_rows, n_live), dtype=np.float32)
    CHUNK = 500_000
    for cs in range(0, mmap_rows, CHUNK):
        ce = min(cs + CHUNK, mmap_rows)
        for j, fi in enumerate(live_idx):
            X_all[cs:ce, j] = X[cs:ce, fi]
        if cs % 2000000 == 0:
            print(f"  {cs:,}/{mmap_rows:,}")
    print(f"  Done ({time.time()-t0:.0f}s)")
    
    # ═══ GENERATE STRATEGY SIGNALS ═══
    print(f"\nGenerating {n_live} strategy signals...")
    
    # For each feature, generate a signal using the BEST logic for that feature type
    all_signals = np.zeros((n, n_live), dtype=np.int8)
    
    for feat_idx in range(n_live):
        values = X_all[:, feat_idx].astype(np.float64)
        
        # Skip constant features
        if np.std(values) < 1e-10:
            continue
        
        feat_name = fm['live_features'][feat_idx] if feat_idx < len(fm.get('live_features', [])) else f"f{feat_idx}"
        
        # Choose signal logic based on feature name/type
        if 'rsi' in feat_name.lower():
            # RSI: buy < 30, sell > 70
            sig = np.zeros(n, dtype=np.int8)
            sig[values < 30] = 1
            sig[values > 70] = -1
        elif 'macd' in feat_name.lower():
            # MACD: use z-score of histogram
            sig = zscore_signal(values, lookback=500, threshold=1.5)
        elif 'bb' in feat_name.lower() or 'bollinger' in feat_name.lower():
            # Bollinger: percentile-based
            sig = percentile_signal(values, lookback=500, buy_pct=15, sell_pct=85)
        elif 'ema' in feat_name.lower() or 'sma' in feat_name.lower():
            # Moving average: momentum
            sig = momentum_signal(values, period=20)
        elif 'atr' in feat_name.lower() or 'vol' in feat_name.lower():
            # Volatility: mean reversion
            sig = zscore_signal(values, lookback=500, threshold=1.0)
        elif 'adx' in feat_name.lower():
            # ADX: trend strength
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 25] = 1  # Trending = buy
            sig[values < 15] = -1  # Weak = sell
        elif 'hmm' in feat_name.lower() or 'regime' in feat_name.lower():
            # HMM regime: buy in bull, sell in bear
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0.6] = 1
            sig[values < 0.4] = -1
        elif 'kalman' in feat_name.lower():
            # Kalman: velocity direction
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0] = 1
            sig[values < 0] = -1
        elif 'ou' in feat_name.lower():
            # Ornstein-Uhlenbeck: mean reversion
            sig = zscore_signal(values, lookback=500, threshold=1.5)
        elif 'garch' in feat_name.lower():
            # GARCH: volatility regime
            sig = zscore_signal(values, lookback=500, threshold=1.0)
        elif 'hurst' in feat_name.lower():
            # Hurst: trending vs mean-reverting
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0.55] = 1  # Trending
            sig[values < 0.45] = -1  # Mean-reverting
        elif 'amihud' in feat_name.lower():
            # Amihud: liquidity
            sig = zscore_signal(values, lookback=500, threshold=1.5)
        elif 'entropy' in feat_name.lower():
            # Entropy: market efficiency
            sig = zscore_signal(values, lookback=500, threshold=1.5)
        elif 'kelly' in feat_name.lower():
            # Kelly: position sizing
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0.5] = 1
            sig[values < -0.5] = -1
        elif 'corr' in feat_name.lower():
            # Correlation: regime
            sig = zscore_signal(values, lookback=500, threshold=1.5)
        else:
            # Default: z-score
            sig = zscore_signal(values, lookback=500, threshold=2.0)
        
        all_signals[:, feat_idx] = sig
        
        if feat_idx % 20 == 0:
            print(f"  Feature {feat_idx}/{n_live}: {feat_name}")
    
    # Count total signals
    total_buy = (all_signals == 1).sum(axis=1)
    total_sell = (all_signals == -1).sum(axis=1)
    print(f"\n  Avg buy votes: {total_buy.mean():.1f}/{n_live}")
    print(f"  Avg sell votes: {total_sell.mean():.1f}/{n_live}")
    
    # ═══ ENSEMBLE VOTING ═══
    print("\n═══ ENSEMBLE VOTING ═══")
    
    def ensemble_bt(min_votes, sl_m, tp_m, regime_filter=True, dynamic_sizing=True, base_risk=20.0):
        """Backtest with feature ensemble voting."""
        atr_14 = compute_atr(highs, lows, closes, 14)
        
        acc = 1000.0; peak = 1000.0; wins=0; losses=0; pnl=0; dd=0; nn=0
        spread = 0.30
        
        for i in range(300, n - 37):
            buy_votes = (all_signals[i] == 1).sum()
            sell_votes = (all_signals[i] == -1).sum()
            
            if buy_votes >= min_votes:
                signal = 1
                strength = buy_votes / n_live
            elif sell_votes >= min_votes:
                signal = -1
                strength = sell_votes / n_live
            else:
                continue
            
            if atr_14[i] < 0.1:
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
                risk_dollar *= min(strength * 3, 1.5)
            else:
                risk_dollar = base_risk
            
            entry = closes[i]; d = signal
            sl_d = sl_m * atr_14[i]; tp_d = tp_m * atr_14[i]
            if d == 1: sl_p = entry - sl_d; tp_p = entry + tp_d
            else: sl_p = entry + sl_d; tp_p = entry - tp_d
            
            out = 0; bp = entry
            for j in range(i+1, min(i+37, n)):
                if d == 1:
                    if lows[j] <= sl_p: out = -(entry - sl_p); break
                    bp = max(bp, highs[j])
                    nsl = bp - sl_d * 0.6
                    if nsl > sl_p: sl_p = nsl
                    if highs[j] >= tp_p: out = (tp_p - entry); break
                else:
                    if highs[j] >= sl_p: out = -(sl_p - entry); break
                    bp = min(bp, lows[j])
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
    
    # Test different vote thresholds
    print("\n── Testing vote thresholds ──")
    for mv in [5, 10, 15, 20, 25, 30, 35, 40]:
        r = ensemble_bt(mv, 0.8, 1.5)
        if r['n'] > 10:
            print(f"  {mv:2d}/{n_live} votes: WR={r['wr']:.1%}, PnL=${r['pnl']:.2f}, "
                  f"DD={r['dd']:.1%}, trades={r['n']}, PF={r['w']/max(r['l'],1):.2f}")
    
    # Grid search for best config
    print("\n── Grid search ──")
    best_pnl = -999999; best = None
    for mv in [10, 15, 20, 25]:
        for sl in [0.8, 1.0, 1.5]:
            for tp in [1.5, 2.0, 2.5, 3.0]:
                r = ensemble_bt(mv, sl, tp)
                if r['n'] > 50 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'mv': mv, 'sl': sl, 'tp': tp, **r}
                    print(f"  votes>={mv} SL={sl} TP={tp}: WR={r['wr']:.0%}, "
                          f"PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}")
    
    if best:
        print(f"\n═══ BEST RESULT ═══")
        print(f"  Min votes: {best['mv']}/{n_live}")
        print(f"  SL: {best['sl']} ATR, TP: {best['tp']} ATR")
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
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")

if __name__ == '__main__':
    main()
