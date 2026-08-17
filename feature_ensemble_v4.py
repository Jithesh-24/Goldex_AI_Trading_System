#!/usr/bin/env python3
"""
feature_ensemble_v4.py — REALISTIC: 1 trade at a time.
Trade closes → recompute → wait for fresh signal → next trade.
No overlapping. No hardcoded frequency. Market decides.
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

def main():
    t0 = time.time()
    print("═══ FEATURE ENSEMBLE V4 (1-TRADE-AT-A-TIME) ═══\n")
    
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
    
    # Generate signals
    print("Generating signals...")
    all_signals = np.zeros((USE_ROWS, n_live), dtype=np.int8)
    
    for feat_idx in range(n_live):
        values = X_all[:, feat_idx].astype(np.float64)
        if np.std(values) < 1e-10: continue
        fname = live_feats[feat_idx]
        
        sig = np.zeros(n, dtype=np.int8)
        if 'rsi' in fname.lower():
            sig[values < 30] = 1; sig[values > 70] = -1
        elif 'adx' in fname.lower():
            sig[values > 25] = 1; sig[values < 15] = -1
        elif 'hmm' in fname.lower() or 'regime' in fname.lower():
            sig[values > 0.6] = 1; sig[values < 0.4] = -1
        elif 'kalman' in fname.lower():
            sig[values > 0] = 1; sig[values < 0] = -1
        elif 'hurst' in fname.lower():
            sig[values > 0.55] = 1; sig[values < 0.45] = -1
        elif 'kelly' in fname.lower():
            sig[values > 0.5] = 1; sig[values < -0.5] = -1
        else:
            lookback = 300
            cs = np.cumsum(values)
            cs2 = np.cumsum(values**2)
            threshold = 1.5 if any(x in fname.lower() for x in ['vol','atr','garch']) else 2.0
            for i in range(lookback, n):
                s = cs[i] - cs[i-lookback]; s2 = cs2[i] - cs2[i-lookback]
                mu = s / lookback; var = s2 / lookback - mu**2
                if var > 1e-10:
                    z = (values[i] - mu) / np.sqrt(var)
                    if z < -threshold: sig[i] = 1
                    elif z > threshold: sig[i] = -1
        
        all_signals[:, feat_idx] = sig
        
        if (feat_idx + 1) % 26 == 0:
            print(f"  {feat_idx+1}/{n_live} ({time.time()-t0:.0f}s)")
    
    buy_votes = (all_signals == 1).sum(axis=1)
    sell_votes = (all_signals == -1).sum(axis=1)
    print(f"  Done. Avg buy={buy_votes.mean():.1f} sell={sell_votes.mean():.1f}")
    
    atr_14 = compute_atr(highs, lows, closes, 14)
    
    # ═══ REALISTIC BACKTEST: 1 TRADE AT A TIME ═══
    print("\n═══ REALISTIC BACKTEST ═══")
    print("  1 trade at a time → wait for close → next setup\n")
    
    def bt_realistic(min_votes, sl_m, tp_m, max_hold=37, cooldown=0, 
                     dynamic=True, base_risk=20.0):
        """
        Realistic backtest:
        - Enter only when no position is open
        - Wait for TP/SL/timeout
        - After close, optionally cooldown N bars
        - Then look for next fresh signal
        """
        acc = 1000.0; peak = 1000.0
        w = 0; l = 0; pnl = 0; dd = 0; nn = 0
        sp = 0.30
        
        in_position = False
        pos_entry = 0
        pos_dir = 0
        pos_sl = 0
        pos_tp = 0
        pos_sl_dist = 0
        pos_end = 0
        cooldown_until = 0
        
        trade_log = []
        
        for i in range(500, n - 1):
            # If in position, check if trade is still open
            if in_position:
                if i >= pos_end:
                    # Timeout — exit at market
                    out = (closes[i] - pos_entry) * pos_dir
                    p = base_risk * (out / pos_sl_dist) - sp * 2
                    acc += p; pnl += p; nn += 1
                    if p > 0: w += 1
                    else: l += 1
                    if acc > peak: peak = acc
                    d2 = (peak - acc) / peak if peak > 0 else 0
                    if d2 > dd: dd = d2
                    trade_log.append({'entry_bar': pos_entry_bar, 'exit_bar': i,
                                      'dir': 'BUY' if pos_dir == 1 else 'SELL',
                                      'pnl': p, 'type': 'timeout'})
                    in_position = False
                    cooldown_until = i + cooldown
                continue
            
            # Not in position — look for signal
            if i < cooldown_until:
                continue
            
            bv = buy_votes[i]; sv = sell_votes[i]
            if atr_14[i] < 0.1:
                continue
            
            if bv >= min_votes:
                signal = 1; strength = bv / n_live
            elif sv >= min_votes:
                signal = -1; strength = sv / n_live
            else:
                continue
            
            # Dynamic sizing
            current_dd = (peak - acc) / peak if peak > 0 else 0
            if dynamic:
                if current_dd > 0.3: rd = base_risk * 0.5
                elif current_dd > 0.1: rd = base_risk * 0.75
                else: rd = min(base_risk * 1.5, 50)
                rd *= min(strength * 3, 1.5)
            else:
                rd = base_risk
            
            entry = closes[i]; d = signal
            sl_d = sl_m * atr_14[i]; tp_d = tp_m * atr_14[i]
            if d == 1: sl_p = entry - sl_d; tp_p = entry + tp_d
            else: sl_p = entry + sl_d; tp_p = entry - tp_d
            
            # Simulate trade
            out = 0; bp = entry; exit_bar = i
            for j in range(i + 1, min(i + max_hold + 1, n)):
                if d == 1:
                    if lows[j] <= sl_p:
                        out = -(entry - sl_p); exit_bar = j; break
                    bp = max(bp, highs[j])
                    nsl = bp - sl_d * 0.6
                    if nsl > sl_p: sl_p = nsl
                    if highs[j] >= tp_p:
                        out = (tp_p - entry); exit_bar = j; break
                else:
                    if highs[j] >= sl_p:
                        out = -(sl_p - entry); exit_bar = j; break
                    bp = min(bp, lows[j])
                    nsl = bp + sl_d * 0.6
                    if nsl < sl_p: sl_p = nsl
                    if lows[j] <= tp_p:
                        out = (entry - tp_p); exit_bar = j; break
            
            if out == 0:
                exit_bar = min(i + max_hold, n - 1)
                out = (closes[exit_bar] - entry) * d
            
            p = rd * (out / sl_d) - sp * 2
            acc += p; pnl += p; nn += 1
            if p > 0: w += 1
            else: l += 1
            if acc > peak: peak = acc
            d2 = (peak - acc) / peak if peak > 0 else 0
            if d2 > dd: dd = d2
            
            trade_log.append({
                'entry_bar': i, 'exit_bar': exit_bar,
                'dir': 'BUY' if d == 1 else 'SELL',
                'entry': entry, 'pnl': p, 'type': 'tp' if abs(out) >= tp_d - 0.01 else 'sl'
            })
            
            in_position = False
            cooldown_until = exit_bar + cooldown
            
            if acc <= 0: break
        
        # Analysis
        if trade_log:
            hold_bars = [t['exit_bar'] - t['entry_bar'] for t in trade_log]
            avg_hold = np.mean(hold_bars) if hold_bars else 0
            avg_hold_min = avg_hold * 5  # M5 bars = 5 min each
            trades_per_day = nn / (USE_ROWS / 288) if USE_ROWS > 0 else 0
            
            print(f"  Votes>={min_votes:2d} SL={sl_m} TP={tp_m}: "
                  f"WR={w/max(w+l,1):.1%} PnL=${pnl:,.0f} DD={dd:.1%} "
                  f"trades={nn} PF={w/max(l,1):.2f}")
            print(f"    → Avg hold: {avg_hold:.0f} bars ({avg_hold_min:.0f} min) | "
                  f"Frequency: {trades_per_day:.1f} trades/day | "
                  f"Avg ${pnl/nn:.2f}/trade")
        
        return {'n': nn, 'wr': w/max(w+l,1), 'pnl': pnl, 'acc': acc, 
                'dd': dd, 'w': w, 'l': l, 'trades_per_day': trades_per_day if trade_log else 0}
    
    # Test with cooldown=0 (enter immediately after close)
    print("── Cooldown=0 (immediate re-entry) ──")
    best_pnl = -999999; best = None
    for mv in [10, 15, 20, 25, 30]:
        for sl in [0.8, 1.0, 1.5]:
            for tp in [1.5, 2.0, 2.5, 3.0]:
                r = bt_realistic(mv, sl, tp, cooldown=0)
                if r['n'] > 10 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'mv': mv, 'sl': sl, 'tp': tp, 'cd': 0, **r}
    
    # Test with cooldown=3 (wait 15 min after close)
    print("\n── Cooldown=3 bars (15 min wait) ──")
    best_pnl2 = -999999; best2 = None
    for mv in [15, 20, 25]:
        for sl in [0.8, 1.0]:
            for tp in [2.0, 2.5, 3.0]:
                r = bt_realistic(mv, sl, tp, cooldown=3)
                if r['n'] > 10 and r['pnl'] > best_pnl2:
                    best_pnl2 = r['pnl']
                    best2 = {'mv': mv, 'sl': sl, 'tp': tp, 'cd': 3, **r}
    
    # Test with cooldown=6 (30 min wait)
    print("\n── Cooldown=6 bars (30 min wait) ──")
    best_pnl3 = -999999; best3 = None
    for mv in [15, 20, 25]:
        for sl in [0.8, 1.0]:
            for tp in [2.0, 2.5, 3.0]:
                r = bt_realistic(mv, sl, tp, cooldown=6)
                if r['n'] > 10 and r['pnl'] > best_pnl3:
                    best_pnl3 = r['pnl']
                    best3 = {'mv': mv, 'sl': sl, 'tp': tp, 'cd': 6, **r}
    
    # Best overall
    all_best = [b for b in [best, best2, best3] if b]
    if all_best:
        overall_best = max(all_best, key=lambda x: x['pnl'])
        print(f"\n═══ BEST OVERALL ═══")
        print(f"  Votes: {overall_best['mv']}/{n_live}")
        print(f"  SL: {overall_best['sl']} ATR, TP: {overall_best['tp']} ATR")
        print(f"  Cooldown: {overall_best['cd']} bars ({overall_best['cd']*5} min)")
        print(f"  Trades: {overall_best['n']:,}")
        print(f"  Win rate: {overall_best['wr']:.1%}")
        print(f"  P&L: ${overall_best['pnl']:,.2f}")
        print(f"  Account: ${overall_best['acc']:,.2f}")
        print(f"  Drawdown: {overall_best['dd']:.1%}")
        print(f"  Profit factor: {overall_best['w']/max(overall_best['l'],1):.2f}")
        print(f"  Trades/day: {overall_best['trades_per_day']:.1f}")
        
        years = USE_ROWS / (288 * 365)
        if overall_best['acc'] > 0:
            ann = (overall_best['acc'] / 1000) ** (1/years) - 1
            print(f"\n  Annual return: {ann:.1%}")
            for s in [100, 500, 1000]:
                for y in [1, 2, 3, 5]:
                    print(f"  ${s} → ${s*(1+ann)**y:,.0f} after {y}yr")
    
    print(f"\n═══ DONE ═══ ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
