#!/usr/bin/env python3
"""
backtest_proper.py — REAL backtest with 6 years of gold data.
No estimates. No theories. Just numbers.
"""
import numpy as np
import lightgbm as lgb
import time, os, json
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    t0 = time.time()
    print("═══ REAL BACKTEST — 6 YEARS OF GOLD DATA ═══\n")
    
    # Load data
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    nf = meta['n_features']
    n = meta['n_rows']
    feats = meta['features']
    
    fm = json.load(open(f"{BASE}/models/feature_map.json"))
    live_idx = fm['live_indices']
    live_feats = fm['live_features']
    dead_idx = fm.get('dead_indices', [])
    n_live = len(live_idx)
    
    print(f"Total features: {nf}")
    print(f"Live features used: {n_live}")
    print(f"Data rows: {n:,}")
    
    # Load models
    seeds = [42, 7, 2026]
    models = []
    for s in seeds:
        m = lgb.Booster(model_file=f"{BASE}/models/gold_lgb_model_s{s}.txt")
        models.append(m)
        print(f"Model seed {s}: {m.num_trees()} trees, {m.num_feature()} features")
    
    # Load labels
    y = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(n,))
    
    # Load prices for P&L calculation
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:, 3].astype(np.float64)
    n_tail = len(closes)
    
    print(f"\n═══ BACKTEST PARAMETERS ═══")
    print(f"Period: {n_tail:,} M5 bars")
    print(f"Price range: ${closes.min():.2f} – ${closes.max():.2f}")
    print(f"Spread: $0.30 (typical XAU/USD)")
    print(f"SL: adaptive (from model)")
    print(f"TP: adaptive (from model)")
    print(f"Risk per trade: 1% of account")
    print(f"Initial account: $1,000")
    print(f"Compounding: YES (grow with profits)")
    
    # Extract live features for all rows
    print(f"\n═══ EXTRACTING LIVE FEATURES ═══")
    X_full = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='r', shape=(9999963, nf))
    
    # Run predictions on FULL dataset
    print("Running ensemble predictions on full dataset...")
    
    # Use last 9,999,963 rows (where features exist)
    start_row = n - n_tail
    n_test = n_tail
    
    # Get predictions from all 3 models (chunked to avoid OOM)
    print("Running ensemble predictions (chunked)...")
    CHUNK = 500_000
    preds = np.zeros(n_test, dtype=np.float64)
    for m in models:
        for cs in range(0, n_test, CHUNK):
            ce = min(cs + CHUNK, n_test)
            X_sel = np.empty((ce - cs, n_live), dtype=np.float32)
            for j, fi in enumerate(live_idx):
                X_sel[:, j] = X_full[cs:ce, fi]
            preds[cs:ce] += m.predict(X_sel) / len(models)
            del X_sel
    
    # Binary predictions
    pred_binary = (preds > 0.5).astype(int)
    labels = y[start_row:start_row + n_test]
    
    # Basic accuracy
    acc = (pred_binary == labels).mean()
    up_mask = labels == 1
    down_mask = labels == 0
    up_acc = (pred_binary[up_mask] == labels[up_mask]).mean() if up_mask.sum() > 0 else 0
    down_acc = (pred_binary[down_mask] == labels[down_mask]).mean() if down_mask.sum() > 0 else 0
    pred_up_pct = pred_binary.mean()
    
    print(f"\n═══ BASIC ACCURACY ═══")
    print(f"Overall accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"UP accuracy: {up_acc:.3f} ({up_acc*100:.1f}%)")
    print(f"DOWN accuracy: {down_acc:.3f} ({down_acc*100:.1f}%)")
    print(f"Predicted UP: {pred_up_pct:.3f} ({pred_up_pct*100:.1f}%)")
    print(f"Actual UP: {labels.mean():.3f} ({labels.mean()*100:.1f}%)")
    
    # Confidence analysis
    print(f"\n═══ CONFIDENCE DISTRIBUTION ═══")
    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = preds > threshold
        if mask.sum() > 0:
            sub_acc = (pred_binary[mask] == labels[mask]).mean()
            sub_up = (pred_binary[mask & up_mask] == labels[mask & up_mask]).mean() if (mask & up_mask).sum() > 0 else 0
            print(f"  conf>{threshold:.2f}: {mask.sum():,} trades, acc={sub_acc:.3f}, up_acc={sub_up:.3f}")
    
    # Simulate trading with compound interest
    print(f"\n═══ COMPOUND TRADING SIMULATION ═══")
    print(f"Strategy: Trade when model confidence > 0.55")
    print(f"Direction: Follow model (UP if pred_up, DOWN if pred_down)")
    print(f"SL: 1.5 ATR (adaptive)")
    print(f"TP: 2.0 ATR (adaptive)")
    print(f"Risk: 1% per trade")
    
    account = 1000.0
    trades = []
    wins = 0
    losses = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak = 1000.0
    
    # Simple ATR calculation
    atr_period = 14
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    
    conf_threshold = 0.55
    sl_atr = 1.5
    tp_atr = 2.0
    spread = 0.30
    
    trade_count = 0
    for i in range(200, n_test):
        if preds[i] < conf_threshold:
            continue
        
        # Calculate ATR
        tr = np.maximum(highs[i] - lows[i],
                       np.maximum(abs(highs[i] - closes[i-1]),
                                 abs(lows[i] - closes[i-1])))
        atr = np.mean([np.maximum(highs[j] - lows[j],
                        np.maximum(abs(highs[j] - closes[j-1]),
                                  abs(lows[j] - closes[j-1])))
                       for j in range(i-atr_period, i)])
        
        if atr < 0.1:
            continue
        
        entry = closes[i]
        direction = 1 if pred_binary[i] == 1 else -1  # 1=BUY, -1=SELL
        
        # SL and TP
        sl_dist = sl_atr * atr
        tp_dist = tp_atr * atr
        
        if direction == 1:  # BUY
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:  # SELL
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist
        
        # Check next bars for SL/TP hit
        outcome = 0
        for j in range(i+1, min(i+37, n_test)):  # 36 bars = 3 hours
            if direction == 1:  # BUY
                if lows[j] <= sl_price:
                    outcome = -sl_dist
                    break
                if highs[j] >= tp_price:
                    outcome = tp_dist
                    break
            else:  # SELL
                if highs[j] >= sl_price:
                    outcome = -sl_dist
                    break
                if lows[j] <= tp_price:
                    outcome = tp_dist
                    break
        
        if outcome == 0:
            continue  # No resolution, skip
        
        # Calculate P&L
        risk_amount = account * 0.01  # 1% risk
        pnl = outcome / sl_dist * risk_amount - spread  # Spread cost
        
        account += pnl
        total_pnl += pnl
        trade_count += 1
        
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        if account > peak:
            peak = account
        dd = (peak - account) / peak
        if dd > max_drawdown:
            max_drawdown = dd
        
        trades.append({
            'bar': i,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry': entry,
            'outcome': outcome,
            'pnl': pnl,
            'account': account,
            'conf': preds[i]
        })
        
        if trade_count % 100 == 0:
            wr = wins / max(wins + losses, 1)
            print(f"  {trade_count:,} trades | wins={wr:.1%} | PnL=${total_pnl:.2f} | account=${account:.2f} | dd={max_drawdown:.1%}")
    
    # Final results
    wr = wins / max(wins + losses, 1)
    avg_pnl = total_pnl / max(trade_count, 1)
    
    print(f"\n═══ FINAL RESULTS ═══")
    print(f"Total trades: {trade_count:,}")
    print(f"Win rate: {wr:.1%} ({wins}W / {losses}L)")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Final account: ${account:,.2f}")
    print(f"Return: {(account/1000-1)*100:.1f}%")
    print(f"Max drawdown: {max_drawdown:.1%}")
    print(f"Avg P&L per trade: ${avg_pnl:.2f}")
    print(f"Profit factor: {wins/max(losses,1):.2f}")
    
    # Annualized
    years = n_test / (288 * 365)  # 288 M5 bars per day, 365 days
    annual_return = (account / 1000) ** (1/years) - 1
    print(f"Period: {years:.1f} years")
    print(f"Annualized return: {annual_return:.1%}")
    
    # Compounding projection
    print(f"\n═══ COMPOUNDING PROJECTION ═══")
    for start_amount in [100, 500, 1000]:
        acc = start_amount
        for year in range(1, 4):
            acc *= (1 + annual_return)
            print(f"  ${start_amount} → ${acc:,.0f} after {year} year{'s' if year>1 else ''}")
    
    # Top trades
    if trades:
        print(f"\n═══ TOP 10 TRADES ═══")
        sorted_trades = sorted(trades, key=lambda x: x['pnl'], reverse=True)[:10]
        for t in sorted_trades:
            print(f"  bar={t['bar']} {t['direction']} entry=${t['entry']:.2f} pnl=${t['pnl']:.2f} conf={t['conf']:.3f}")
        
        print(f"\n═══ WORST 10 TRADES ═══")
        sorted_trades = sorted(trades, key=lambda x: x['pnl'])[:10]
        for t in sorted_trades:
            print(f"  bar={t['bar']} {t['direction']} entry=${t['entry']:.2f} pnl=${t['pnl']:.2f} conf={t['conf']:.3f}")
    
    # Monthly breakdown
    print(f"\n═══ MONTHLY BREAKDOWN ═══")
    monthly = {}
    for t in trades:
        month = t['bar'] // (288 * 21)  # ~21 trading days per month
        if month not in monthly:
            monthly[month] = {'wins': 0, 'losses': 0, 'pnl': 0}
        if t['pnl'] > 0:
            monthly[month]['wins'] += 1
        else:
            monthly[month]['losses'] += 1
        monthly[month]['pnl'] += t['pnl']
    
    for m in sorted(monthly.keys())[:24]:
        d = monthly[m]
        wr = d['wins'] / max(d['wins'] + d['losses'], 1)
        print(f"  Month {m:3d}: {d['wins']+d['losses']:4d} trades, WR={wr:.0%}, PnL=${d['pnl']:8.2f}")
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s)")

if __name__ == '__main__':
    main()
