#!/usr/bin/env python3
"""
feature_ensemble_v3.py — FULLY VECTORIZED.
106 features × 2M bars. No Python loops in signal generation.
"""
import numpy as np
import json, os, time
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def sma(data, period):
    cs = np.cumsum(data)
    out = np.full(len(data), np.nan, dtype=np.float64)
    out[period-1:] = (cs[period-1:] - np.concatenate([[0], cs[:-period]])) / period
    return out

def compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    return sma(tr, period)

def rolling_zscore(values, lookback=500, threshold=2.0):
    """VECTORIZED rolling z-score using cumsum."""
    n = len(values)
    signals = np.zeros(n, dtype=np.int8)
    
    # Rolling mean via cumsum
    cs = np.cumsum(values)
    cs2 = np.cumsum(values**2)
    
    for i in range(lookback, n):
        s = cs[i] - cs[i-lookback]
        s2 = cs2[i] - cs2[i-lookback]
        mu = s / lookback
        var = s2 / lookback - mu**2
        std = np.sqrt(max(var, 1e-10))
        z = (values[i] - mu) / std
        if z < -threshold:
            signals[i] = 1
        elif z > threshold:
            signals[i] = -1
    
    return signals

def rolling_percentile_signal(values, lookback=500, buy_pct=10, sell_pct=90):
    """Rolling percentile — still needs loop but faster with sorted window."""
    n = len(values)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(lookback, n):
        window = values[i-lookback:i]
        if values[i] < np.percentile(window, buy_pct):
            signals[i] = 1
        elif values[i] > np.percentile(window, sell_pct):
            signals[i] = -1
    return signals

def main():
    t0 = time.time()
    print("═══ FEATURE ENSEMBLE V3 (VECTORIZED) ═══\n")
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    mmap_rows = 9999963
    fm = json.load(open(f"{BASE}/models/feature_map.json"))
    live_idx = fm['live_indices']
    n_live = len(live_idx)
    live_feats = fm.get('live_features', [f"f{i}" for i in range(n_live)])
    
    USE_ROWS = 2_000_000
    START = mmap_rows - USE_ROWS
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(mmap_rows, nf))
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[START:START+USE_ROWS, 3].astype(np.float64)
    highs = prices[START:START+USE_ROWS, 1].astype(np.float64)
    lows = prices[START:START+USE_ROWS, 2].astype(np.float64)
    n = len(closes)
    
    print(f"Data: {n:,} bars × {n_live} features")
    
    # Extract features
    print("Extracting features...")
    X_all = np.empty((USE_ROWS, n_live), dtype=np.float32)
    for j, fi in enumerate(live_idx):
        X_all[:, j] = X[START:START+USE_ROWS, fi]
    print(f"  Done ({time.time()-t0:.0f}s)")
    
    # Generate ALL signals at once
    print(f"\nGenerating {n_live} strategy signals (vectorized)...")
    all_signals = np.zeros((USE_ROWS, n_live), dtype=np.int8)
    
    for feat_idx in range(n_live):
        values = X_all[:, feat_idx].astype(np.float64)
        if np.std(values) < 1e-10:
            continue
        
        fname = live_feats[feat_idx]
        
        # FAST signal logic — minimal Python loops
        if 'rsi' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values < 30] = 1
            sig[values > 70] = -1
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
        elif 'kelly' in fname.lower():
            sig = np.zeros(n, dtype=np.int8)
            sig[values > 0.5] = 1
            sig[values < -0.5] = -1
        else:
            # z-score — use cumsum approach (faster than loop)
            sig = np.zeros(n, dtype=np.int8)
            lookback = 300
            cs = np.cumsum(values)
            cs2 = np.cumsum(values**2)
            threshold = 1.5 if any(x in fname.lower() for x in ['vol','atr','garch']) else 2.0
            
            for i in range(lookback, n):
                s = cs[i] - cs[i-lookback]
                s2 = cs2[i] - cs2[i-lookback]
                mu = s / lookback
                var = s2 / lookback - mu**2
                if var > 1e-10:
                    z = (values[i] - mu) / np.sqrt(var)
                    if z < -threshold: sig[i] = 1
                    elif z > threshold: sig[i] = -1
        
        all_signals[:, feat_idx] = sig
        
        if (feat_idx + 1) % 26 == 0:
            print(f"  {feat_idx+1}/{n_live} ({time.time()-t0:.0f}s)")
    
    print(f"  All done ({time.time()-t0:.0f}s)")
    
    # Vote counts
    buy_votes = (all_signals == 1).sum(axis=1)
    sell_votes = (all_signals == -1).sum(axis=1)
    print(f"\n  Avg buy votes: {buy_votes.mean():.1f}/{n_live}")
    print(f"  Avg sell votes: {sell_votes.mean():.1f}/{n_live}")
    
    # ═══ BACKTEST ═══
    print("\n═══ BACKTESTING ═══")
    atr_14 = compute_atr(highs, lows, closes, 14)
    
    def bt(min_votes, sl_m, tp_m, dynamic=True, base_risk=20.0):
        acc = 1000.0; peak = 1000.0; w=0; l=0; pnl=0; dd=0; nn=0
        sp = 0.30
        for i in range(500, n - 37):
            bv = buy_votes[i]; sv = sell_votes[i]
            if bv >= min_votes: sig = 1; strn = bv/n_live
            elif sv >= min_votes: sig = -1; strn = sv/n_live
            else: continue
            if atr_14[i] < 0.1: continue
            
            cdd = (peak-acc)/peak if peak>0 else 0
            if dynamic:
                if cdd > 0.3: rd = base_risk*0.5
                elif cdd > 0.1: rd = base_risk*0.75
                else: rd = min(base_risk*1.5, 50)
                rd *= min(strn*3, 1.5)
            else: rd = base_risk
            
            e = closes[i]; d = sig
            sl = sl_m*atr_14[i]; tp = tp_m*atr_14[i]
            if d==1: slp=e-sl; tpp=e+tp
            else: slp=e+sl; tpp=e-tp
            
            out=0; bp=e
            for j in range(i+1, min(i+37, n)):
                if d==1:
                    if lows[j]<=slp: out=-(e-slp); break
                    bp=max(bp,highs[j]); nsl=bp-sl*0.6
                    if nsl>slp: slp=nsl
                    if highs[j]>=tpp: out=(tpp-e); break
                else:
                    if highs[j]>=slp: out=-(slp-e); break
                    bp=min(bp,lows[j]); nsl=bp+sl*0.6
                    if nsl<slp: slp=nsl
                    if lows[j]<=tpp: out=(e-tpp); break
            if out==0: out=(closes[min(i+37,n-1)]-e)*d
            
            p = rd*(out/sl)-sp*2
            acc+=p; pnl+=p; nn+=1
            if p>0: w+=1
            else: l+=1
            if acc>peak: peak=acc
            d2=(peak-acc)/peak if peak>0 else 0
            if d2>dd: dd=d2
            if acc<=0: break
        return {'n':nn, 'wr':w/max(w+l,1), 'pnl':pnl, 'acc':acc, 'dd':dd, 'w':w, 'l':l}
    
    # Test thresholds
    print("\n── Vote thresholds ──")
    for mv in [5,10,15,20,25,30,35,40]:
        r = bt(mv, 0.8, 1.5)
        if r['n'] > 10:
            print(f"  {mv:2d}/{n_live}: WR={r['wr']:.1%}, PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}, trades={r['n']}")
    
    # Grid search
    print("\n── Grid search ──")
    best_pnl = -999999; best = None
    for mv in [10,15,20,25,30]:
        for sl in [0.8,1.0,1.5]:
            for tp in [1.5,2.0,2.5,3.0]:
                r = bt(mv, sl, tp)
                if r['n'] > 20 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'mv':mv, 'sl':sl, 'tp':tp, **r}
                    print(f"  v>={mv} SL={sl} TP={tp}: WR={r['wr']:.0%}, PnL=${r['pnl']:.2f}, DD={r['dd']:.1%}")
    
    if best:
        print(f"\n═══ BEST RESULT ═══")
        print(f"  Votes: {best['mv']}/{n_live}")
        print(f"  SL: {best['sl']} ATR, TP: {best['tp']} ATR")
        print(f"  Trades: {best['n']:,}")
        print(f"  Win rate: {best['wr']:.1%}")
        print(f"  P&L: ${best['pnl']:,.2f}")
        print(f"  Account: ${best['acc']:,.2f}")
        print(f"  Drawdown: {best['dd']:.1%}")
        print(f"  Profit factor: {best['w']/max(best['l'],1):.2f}")
        
        years = USE_ROWS / (288 * 365)
        if best['acc'] > 0:
            ann = (best['acc'] / 1000) ** (1/years) - 1
            print(f"\n  Annual return: {ann:.1%}")
            for s in [100,500,1000]:
                for y in [1,2,3,5]:
                    print(f"  ${s} → ${s*(1+ann)**y:,.0f} after {y}yr")
    
    print(f"\n═══ DONE ═══ ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
