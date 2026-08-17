#!/usr/bin/env python3
"""
feature_ensemble_v2.py — OPTIMIZED: memmap signals, chunked backtest.
Each of 106 features = 1 strategy. Combined via voting.
"""
import numpy as np
import json, os, time
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

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

def zscore_signal_chunk(values, lookback=500, threshold=2.0):
    """Vectorized z-score signal."""
    n = len(values)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(lookback, n):
        mu = np.mean(values[max(0,i-lookback):i])
        std = np.std(values[max(0,i-lookback):i])
        if std > 1e-10:
            z = (values[i] - mu) / std
            if z < -threshold: signals[i] = 1
            elif z > threshold: signals[i] = -1
    return signals

def main():
    t0 = time.time()
    print("═══ FEATURE ENSEMBLE V2 (OPTIMIZED) ═══\n")
    
    # Load data
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    mmap_rows = 9999963
    fm = json.load(open(f"{BASE}/models/feature_map.json"))
    live_idx = fm['live_indices']
    n_live = len(live_idx)
    live_feats = fm.get('live_features', [f"f{i}" for i in range(n_live)])
    
    # Use last 2M rows for speed
    USE_ROWS = 2_000_000
    START = mmap_rows - USE_ROWS
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(mmap_rows, nf))
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[START:START+USE_ROWS].astype(np.float64)
    highs = prices[START:START+USE_ROWS, 1].astype(np.float64)
    lows = prices[START:START+USE_ROWS, 2].astype(np.float64)
    n = len(closes)
    
    print(f"Using last {USE_ROWS:,} rows ({n:,} bars)")
    print(f"Features: {n_live}")
    
    # Extract features
    print("Extracting features...")
    X_all = np.empty((USE_ROWS, n_live), dtype=np.float32)
    CHUNK = 500_000
    for cs in range(0, USE_ROWS, CHUNK):
        ce = min(cs + CHUNK, USE_ROWS)
        for j, fi in enumerate(live_idx):
            X_all[cs:ce, j] = X[START+cs:START+ce, fi]
    print(f"  Done ({time.time()-t0:.0f}s)")
    
    # Generate signals per feature (CHUNKED to avoid OOM)
    print(f"\nGenerating {n_live} strategy signals...")
    
    # Use memmap for signals
    sig_file = f"{BASE}/_feat_signals.npy"
    all_signals = np.memmap(sig_file, dtype=np.int8, mode='w+', shape=(USE_ROWS, n_live))
    
    for feat_idx in range(n_live):
        values = X_all[:, feat_idx].astype(np.float64)
        if np.std(values) < 1e-10:
            continue
        
        fname = live_feats[feat_idx] if feat_idx < len(live_feats) else f"f{feat_idx}"
        
        # Choose signal logic
        if 'rsi' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values < 30] = 1
            sig[values > 70] = -1
        elif 'macd' in fname.lower():
            sig = zscore_signal_chunk(values, lookback=300, threshold=1.5)
        elif 'bb' in fname.lower():
            sig = zscore_signal_chunk(values, lookback=300, threshold=1.5)
        elif 'atr' in fname.lower() or 'vol' in fname.lower():
            sig = zscore_signal_chunk(values, lookback=300, threshold=1.0)
        elif 'adx' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 25] = 1
            sig[values < 15] = -1
        elif 'hmm' in fname.lower() or 'regime' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0.6] = 1
            sig[values < 0.4] = -1
        elif 'kalman' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0] = 1
            sig[values < 0] = -1
        elif 'hurst' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0.55] = 1
            sig[values < 0.45] = -1
        elif 'entropy' in fname.lower():
            sig = zscore_signal_chunk(values, lookback=300, threshold=1.5)
        elif 'kelly' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0.5] = 1
            sig[values < -0.5] = -1
        else:
            sig = zscore_signal_chunk(values, lookback=300, threshold=2.0)
        
        all_signals[:, feat_idx] = sig
        
        if (feat_idx + 1) % 20 == 0:
            print(f"  {feat_idx+1}/{n_live} features done")
    
    all_signals.flush()
    print(f"  All {n_live} signals generated ({time.time()-t0:.0f}s)")
    
    # ═══ BACKTEST ═══
    print("\n═══ BACKTESTING ═══")
    atr_14 = compute_atr(highs, lows, closes, 14)
    
    def bt(min_votes, sl_m, tp_m, dynamic_sizing=True, base_risk=20.0):
        acc = 1000.0; peak = 1000.0; wins=0; losses=0; pnl=0; dd=0; nn=0
        spread = 0.30
        
        for i in range(500, n - 37):
            buy_v = (all_signals[i] == 1).sum()
            sell_v = (all_signals[i] == -1).sum()
            
            if buy_v >= min_votes:
                signal = 1; strength = buy_v / n_live
            elif sell_v >= min_votes:
                signal = -1; strength = sell_v / n_live
            else:
                continue
            
            if atr_14[i] < 0.1:
                continue
            
            current_dd = (peak - acc) / peak if peak > 0 else 0
            if dynamic_sizing:
                if current_dd > 0.3: risk_dollar = base_risk * 0.5
                elif current_dd > 0.1: risk_dollar = base_risk * 0.75
                else: risk_dollar = min(base_risk * 1.5, 50.0)
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
    
    # Test vote thresholds
    print("\n── Vote thresholds ──")
    for mv in [5, 10, 15, 20, 25, 30]:
        r = bt(mv, 0.8, 1.5)
        if r['n'] > 10:
            print(f"  {mv:2d}/{n_live}: WR={r['wr']:.1%}, PnL=${r['pnl']:.2f}, "
                  f"DD={r['dd']:.1%}, trades={r['n']}, PF={r['w']/max(r['l'],1):.2f}")
    
    # Grid search
    print("\n── Grid search ──")
    best_pnl = -999999; best = None
    for mv in [10, 15, 20, 25, 30]:
        for sl in [0.8, 1.0, 1.5]:
            for tp in [1.5, 2.0, 2.5, 3.0]:
                r = bt(mv, sl, tp)
                if r['n'] > 20 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'mv': mv, 'sl': sl, 'tp': tp, **r}
                    print(f"  v>={mv} SL={sl} TP={tp}: WR={r['wr']:.0%}, "
                          f"PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}, trades={r['n']}")
    
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
        
        years = USE_ROWS / (288 * 365)
        if best['acc'] > 0:
            ann = (best['acc'] / 1000) ** (1/years) - 1
            print(f"\n  Annual return: {ann:.1%}")
            for s in [100, 500, 1000]:
                for y in [1, 2, 3, 5]:
                    print(f"  ${s} → ${s * (1+ann)**y:,.0f} after {y}yr")
    
    # Cleanup
    os.remove(sig_file)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")

if __name__ == '__main__':
    main()
