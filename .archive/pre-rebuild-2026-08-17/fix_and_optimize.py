#!/usr/bin/env python3
"""
fix_and_optimize.py — Fix calibration + find optimal SL/TP using EXISTING models.
Doesn't retrain — just calibrates and optimizes.
"""
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import time, os, json, gc, pickle
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = f"{BASE}/models"

def main():
    t0 = time.time()
    print("═══ FIX & OPTIMIZE — CALIBRATION + SL/TP OPTIMIZATION ═══\n")
    
    # Load data
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    mmap_rows = 9999963
    fm = json.load(open(f"{BASE}/models/feature_map.json"))
    live_idx = fm['live_indices']
    n_live = len(live_idx)
    
    X = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(mmap_rows, nf))
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(meta['n_rows'],))
    prices = np.load(f"{BASE}/prices_tail.npy")
    n_prices = len(prices)
    
    y_data = y[-mmap_rows:]
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    
    print(f"Data: {mmap_rows:,} rows × {n_live} features")
    print(f"Prices: {n_prices:,} bars")
    
    # Load existing models
    seeds = [42, 7, 2026]
    models = []
    for s in seeds:
        m = lgb.Booster(model_file=f"{MODELS}/gold_lgb_model_s{s}.txt")
        models.append(m)
    print(f"Loaded {len(models)} models × {models[0].num_feature()} features")
    
    # Split: train (first 6M for calibration), test (last 4M for optimization)
    CALIB_SIZE = 6_000_000
    print(f"\nCalibration set: 0-{CALIB_SIZE:,}")
    print(f"Optimization set: {CALIB_SIZE:,}-{mmap_rows:,}")
    
    # Extract features for calibration set (chunked)
    print("\nExtracting calibration features...")
    X_calib = np.empty((CALIB_SIZE, n_live), dtype=np.float32)
    CHUNK = 500_000
    for cs in range(0, CALIB_SIZE, CHUNK):
        ce = min(cs + CHUNK, CALIB_SIZE)
        for j, fi in enumerate(live_idx):
            X_calib[cs:ce, j] = X[cs:ce, fi]
        if (cs // CHUNK) % 5 == 0:
            print(f"  {cs:,}/{CALIB_SIZE:,}", flush=True)
    print(f"  ✅ Done ({time.time()-t0:.0f}s)")
    
    # Get predictions on calibration set
    print("\nGenerating calibration predictions...")
    raw_calib = np.zeros(CALIB_SIZE, dtype=np.float64)
    for m in models:
        raw_calib += m.predict(X_calib) / len(models)
    del X_calib
    gc.collect()
    
    # Fit isotonic regression calibrator
    print("Fitting isotonic regression calibrator...")
    y_calib = y_data[:CALIB_SIZE]
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    ir.fit(raw_calib, y_calib)
    
    # Analyze calibration quality
    calib_probs = ir.transform(raw_calib)
    print("\n═══ CALIBRATION QUALITY ═══")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = calib_probs >= thr
        if mask.sum() > 0:
            pred_up = calib_probs[mask] > 0.5
            actual_up = y_calib[mask] == 1
            correct = (pred_up == actual_up).mean()
            print(f"  conf>={thr:.2f}: {mask.sum():>8,} trades, accuracy={correct:.3f}")
    
    # Save calibrator
    with open(f"{MODELS}/beast_calibrator.pkl", 'wb') as f:
        pickle.dump(ir, f)
    print(f"\n✅ Calibrator saved: {MODELS}/beast_calibrator.pkl")
    
    # NOW: Optimize SL/TP on the OPTIMIZATION set
    print("\n═══ OPTIMIZING SL/TP ═══")
    print("Extracting optimization features...")
    
    opt_size = mmap_rows - CALIB_SIZE
    X_opt = np.empty((opt_size, n_live), dtype=np.float32)
    for cs in range(0, opt_size, CHUNK):
        ce = min(cs + CHUNK, opt_size)
        for j, fi in enumerate(live_idx):
            X_opt[cs:ce, j] = X[CALIB_SIZE + cs:CALIB_SIZE + ce, fi]
    print(f"  ✅ Done")
    
    # Get predictions on optimization set
    print("Generating optimization predictions...")
    raw_opt = np.zeros(opt_size, dtype=np.float64)
    for m in models:
        raw_opt += m.predict(X_opt) / len(models)
    del X_opt
    gc.collect()
    
    # Calibrate
    cal_opt = ir.transform(raw_opt)
    
    # Backtest function
    def backtest(probs, start_bar, sl_m, tp_m, conf, trailing=True, risk_pct=0.02):
        acc = 1000.0
        peak = 1000.0
        wins = 0
        losses = 0
        total_pnl = 0
        max_dd = 0
        n_trades = 0
        spread = 0.30
        atr_period = 14
        
        for i in range(len(probs)):
            if probs[i] < conf:
                continue
            
            bar = start_bar + i
            if bar < 200 or bar >= n_prices - 37:
                continue
            
            trs = []
            for j in range(max(0, bar - atr_period), bar):
                tr = max(highs[j] - lows[j],
                         max(abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])))
                trs.append(tr)
            atr = np.mean(trs)
            if atr < 0.1:
                continue
            
            entry = closes[bar]
            direction = 1 if probs[i] > 0.5 else -1
            sl_dist = sl_m * atr
            tp_dist = tp_m * atr
            
            if direction == 1:
                sl_price = entry - sl_dist
                tp_price = entry + tp_dist
            else:
                sl_price = entry + sl_dist
                tp_price = entry - tp_dist
            
            outcome = 0
            best_price = entry
            for j in range(bar + 1, min(bar + 37, n_prices)):
                if direction == 1:
                    if lows[j] <= sl_price:
                        outcome = -sl_dist
                        break
                    best_price = max(best_price, highs[j])
                    if trailing:
                        nsl = best_price - sl_dist * 0.7
                        if nsl > sl_price:
                            sl_price = nsl
                    if highs[j] >= tp_price:
                        outcome = tp_dist
                        break
                else:
                    if highs[j] >= sl_price:
                        outcome = -sl_dist
                        break
                    best_price = min(best_price, lows[j])
                    if trailing:
                        nsl = best_price + sl_dist * 0.7
                        if nsl < sl_price:
                            sl_price = nsl
                    if lows[j] <= tp_price:
                        outcome = tp_dist
                        break
            
            if outcome == 0:
                continue
            
            risk = acc * risk_pct
            pnl = (outcome / sl_dist) * risk - spread * 2
            acc += pnl
            total_pnl += pnl
            n_trades += 1
            
            if pnl > 0: wins += 1
            else: losses += 1
            
            if acc > peak: peak = acc
            dd = (peak - acc) / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
        
        wr = wins / max(wins + losses, 1)
        return {'trades': n_trades, 'wr': wr, 'pnl': total_pnl, 
                'account': acc, 'dd': max_dd, 'wins': wins, 'losses': losses}
    
    # Grid search
    best = None
    best_pnl = -999999
    
    print("\nSearching optimal parameters...")
    for sl in [0.8, 1.0, 1.2, 1.5, 2.0]:
        for tp in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            for conf in [0.50, 0.55, 0.60, 0.65]:
                r = backtest(cal_opt, CALIB_SIZE, sl, tp, conf)
                if r['trades'] > 50 and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best = {'sl': sl, 'tp': tp, 'conf': conf, **r}
    
    if best:
        print(f"\n═══ OPTIMAL CONFIGURATION ═══")
        print(f"  SL: {best['sl']} ATR")
        print(f"  TP: {best['tp']} ATR")
        print(f"  Min confidence: {best['conf']}")
        print(f"  Trades: {best['trades']:,}")
        print(f"  Win rate: {best['wr']:.1%}")
        print(f"  Total P&L: ${best['pnl']:,.2f}")
        print(f"  Final account: ${best['account']:,.2f}")
        print(f"  Max drawdown: {best['dd']:.1%}")
        print(f"  Return: {(best['account']/1000-1)*100:.1f}%")
        print(f"  Profit factor: {best['wins']/max(best['losses'],1):.2f}")
        
        # Save optimal config
        config = {
            'version': 'beast_v1.0',
            'sl_atr': best['sl'],
            'tp_atr': best['tp'],
            'min_confidence': best['conf'],
            'trailing_stop': True,
            'trailing_pct': 0.7,
            'risk_per_trade': 0.02,
            'backtest_trades': best['trades'],
            'backtest_win_rate': best['wr'],
            'backtest_pnl': best['pnl'],
            'backtest_return': (best['account']/1000-1),
            'backtest_drawdown': best['dd'],
            'trained': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(f"{MODELS}/beast_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✅ Config saved: {MODELS}/beast_config.json")
    
    # Also show what happens with different risk levels
    print("\n═══ COMPOUND PROJECTIONS (with optimal params) ═══")
    if best:
        for start in [100, 500, 1000]:
            acc = start
            annual_mult = (best['account'] / 1000)
            years = (mmap_rows - CALIB_SIZE) / (288 * 365)
            annual_return = annual_mult ** (1/years) - 1
            print(f"\n  Starting: ${start}")
            for yr in [1, 2, 3, 5]:
                final = start * ((1 + annual_return) ** yr)
                print(f"    Year {yr}: ${final:,.0f}")
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)")

if __name__ == '__main__':
    main()
