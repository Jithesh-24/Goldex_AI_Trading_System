#!/usr/bin/env python3
"""
multi_strategy_ensemble.py — 3-strategy ensemble for gold trading.
1. TREND: H4/D1 EMA crossover → M5 entry
2. MEAN REVERSION: Bollinger bands at key levels
3. VOLATILITY: Sell vol when high, buy vol when low

Each strategy runs independently. Combined via portfolio allocation.
No LightGBM prediction — pure strategy logic.
"""
import numpy as np
import json, os, time
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

class TrendStrategy:
    """H4/D1 trend → M5 entry. Follows the trend, doesn't predict."""
    
    def __init__(self, h4_ema_fast=20, h4_ema_slow=50, d1_ema=200):
        self.h4_ema_fast = h4_ema_fast
        self.h4_ema_slow = h4_ema_slow
        self.d1_ema = d1_ema
    
    def compute_htf(self, closes, timeframe='h4'):
        """Compute higher-timeframe EMAs from M5 data."""
        if timeframe == 'h4':
            # Resample M5 to H4 (48 bars per H4)
            period = 48
        elif timeframe == 'd1':
            period = 288  # 24*12 = 288 M5 bars per day
        else:
            period = 12  # H1
        
        # Downsample by taking every Nth close
        htf_closes = closes[::period]
        ema_fast = self._ema(htf_closes, self.h4_ema_fast) if timeframe == 'h4' else self._ema(htf_closes, self.d1_ema)
        ema_slow = self._ema(htf_closes, self.h4_ema_slow) if timeframe == 'h4' else None
        
        return htf_closes, ema_fast, ema_slow
    
    def _ema(self, data, period):
        """Exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def signal(self, closes, highs, lows, i):
        """
        Generate trend signal at bar i.
        Returns: 1 (BUY), -1 (SELL), 0 (no signal)
        """
        if i < 300:
            return 0
        
        # D1 trend (200 EMA)
        d1_closes = closes[::288]
        d1_ema200 = self._ema(d1_closes, self.d1_ema)
        d1_idx = i // 288
        if d1_idx >= len(d1_ema200):
            return 0
        d1_trend = 1 if closes[i] > d1_ema200[d1_idx] else -1
        
        # H4 trend (20/50 EMA crossover)
        h4_closes = closes[::48]
        h4_ema20 = self._ema(h4_closes, self.h4_ema_fast)
        h4_ema50 = self._ema(h4_closes, self.h4_ema_slow)
        h4_idx = i // 48
        if h4_idx >= len(h4_ema20):
            return 0
        h4_bullish = h4_ema20[h4_idx] > h4_ema50[h4_idx]
        
        # M5 entry: price pullback to H4 EMA
        h4_ema_val = h4_ema20[h4_idx]
        pullback = abs(closes[i] - h4_ema_val) / closes[i] < 0.003  # Within 0.3%
        
        if d1_trend == 1 and h4_bullish and pullback:
            return 1  # BUY
        elif d1_trend == -1 and not h4_bullish and pullback:
            return -1  # SELL
        
        return 0

class MeanReversionStrategy:
    """Bollinger bands at support/resistance → M5 entry."""
    
    def __init__(self, bb_period=20, bb_std=2.0, sr_lookback=100):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sr_lookback = sr_lookback
    
    def _ema(self, data, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def signal(self, closes, highs, lows, i):
        """
        Generate mean reversion signal at bar i.
        Returns: 1 (BUY), -1 (SELL), 0 (no signal)
        """
        if i < self.bb_period + self.sr_lookback:
            return 0
        
        # Bollinger Bands on H1 (12 M5 bars)
        h1_closes = closes[i-11:i+1]
        sma = np.mean(h1_closes)
        std = np.std(h1_closes)
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        
        # Support/Resistance from D1
        d1_highs = highs[i-287:i+1:288]  # Daily highs
        d1_lows = lows[i-287:i+1:288]    # Daily lows
        d1_closes_arr = closes[i-287:i+1:288]
        
        if len(d1_highs) < 2:
            return 0
        
        resistance = np.max(d1_highs[-5:])  # Last 5 days high
        support = np.min(d1_lows[-5:])      # Last 5 days low
        
        # Price at support + lower BB → BUY
        at_support = (closes[i] - support) / closes[i] < 0.002
        at_lower_bb = closes[i] < lower
        
        # Price at resistance + upper BB → SELL
        at_resistance = (resistance - closes[i]) / closes[i] < 0.002
        at_upper_bb = closes[i] > upper
        
        if at_support and at_lower_bb:
            return 1  # BUY
        elif at_resistance and at_upper_bb:
            return -1  # SELL
        
        return 0

class VolatilityStrategy:
    """Sell vol when high (mean reverts), buy vol when low (breakout)."""
    
    def __init__(self, vol_period=20, vol_threshold=1.5):
        self.vol_period = vol_period
        self.vol_threshold = vol_threshold
    
    def signal(self, closes, highs, lows, i):
        """
        Generate volatility signal at bar i.
        Returns: 1 (BUY breakout), -1 (SELL breakout), 0 (no signal)
        """
        if i < self.vol_period * 2:
            return 0
        
        # ATR
        trs = []
        for j in range(max(0, i - self.vol_period), i):
            tr = max(highs[j] - lows[j],
                     max(abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])))
            trs.append(tr)
        current_atr = np.mean(trs)
        
        # Historical ATR (longer period)
        hist_trs = []
        for j in range(max(0, i - self.vol_period * 4), i - self.vol_period):
            tr = max(highs[j] - lows[j],
                     max(abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])))
            hist_trs.append(tr)
        hist_atr = np.mean(hist_trs) if hist_trs else current_atr
        
        # Volatility ratio
        vol_ratio = current_atr / hist_atr if hist_atr > 0 else 1.0
        
        # High vol → mean reversion (sell)
        # Low vol → breakout (buy direction based on recent trend)
        if vol_ratio > self.vol_threshold:
            # Vol spike → expect mean reversion → SELL
            return -1
        elif vol_ratio < 0.5:
            # Vol contraction → expect breakout → BUY if uptrend
            recent_trend = np.mean(closes[i-20:i]) - np.mean(closes[i-40:i-20])
            if recent_trend > 0:
                return 1
            else:
                return -1
        
        return 0

class PortfolioManager:
    """Combines 3 strategies with dynamic allocation."""
    
    def __init__(self):
        self.strategies = {
            'trend': TrendStrategy(),
            'mean_rev': MeanReversionStrategy(),
            'vol': VolatilityStrategy(),
        }
        self.weights = {'trend': 0.40, 'mean_rev': 0.35, 'vol': 0.25}
        self.min_agreement = 2  # Need 2/3 strategies to agree
    
    def generate_signals(self, closes, highs, lows):
        """Generate ensemble signals for all bars."""
        n = len(closes)
        signals = np.zeros(n, dtype=np.int8)
        strengths = np.zeros(n, dtype=np.float64)
        
        for i in range(300, n):
            strat_signals = {}
            for name, strat in self.strategies.items():
                strat_signals[name] = strat.signal(closes, highs, lows, i)
            
            # Count agreement
            buy_count = sum(1 for s in strat_signals.values() if s == 1)
            sell_count = sum(1 for s in strat_signals.values() if s == -1)
            
            if buy_count >= self.min_agreement:
                signals[i] = 1
                strengths[i] = buy_count / 3.0
            elif sell_count >= self.min_agreement:
                signals[i] = -1
                strengths[i] = sell_count / 3.0
        
        return signals, strengths

def backtest_ensemble(closes, highs, lows, signals, strengths,
                      sl_mult=1.5, tp_mult=3.0, risk_pct=0.02, min_strength=0.5):
    """Backtest the ensemble with proper risk management."""
    n = len(closes)
    account = 1000.0
    peak = 1000.0
    trades = []
    wins = 0
    losses = 0
    total_pnl = 0
    max_dd = 0
    spread = 0.30
    
    # ATR for SL/TP
    atr_period = 14
    
    for i in range(300, n - 37):
        if signals[i] == 0:
            continue
        if strengths[i] < min_strength:
            continue
        
        # ATR
        trs = []
        for j in range(max(0, i - atr_period), i):
            tr = max(highs[j] - lows[j],
                     max(abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])))
            trs.append(tr)
        atr = np.mean(trs)
        if atr < 0.1:
            continue
        
        entry = closes[i]
        direction = signals[i]
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        
        if direction == 1:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist
        
        # Simulate
        outcome = 0
        best_price = entry
        for j in range(i + 1, min(i + 37, n)):
            if direction == 1:
                if lows[j] <= sl_price:
                    outcome = -(entry - sl_price)
                    break
                best_price = max(best_price, highs[j])
                nsl = best_price - sl_dist * 0.6
                if nsl > sl_price:
                    sl_price = nsl
                if highs[j] >= tp_price:
                    outcome = (tp_price - entry)
                    break
            else:
                if highs[j] >= sl_price:
                    outcome = -(sl_price - entry)
                    break
                best_price = min(best_price, lows[j])
                nsl = best_price + sl_dist * 0.6
                if nsl < sl_price:
                    sl_price = nsl
                if lows[j] <= tp_price:
                    outcome = (entry - tp_price)
                    break
        
        if outcome == 0:
            # Timeout exit
            if direction == 1:
                outcome = closes[min(i + 37, n - 1)] - entry
            else:
                outcome = entry - closes[min(i + 37, n - 1)]
        
        risk = account * risk_pct
        pnl = (outcome / sl_dist) * risk - spread * 2
        account += pnl
        total_pnl += pnl
        
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        if account > peak:
            peak = account
        dd = (peak - account) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        
        trades.append({
            'bar': i, 'direction': 'BUY' if direction == 1 else 'SELL',
            'entry': entry, 'pnl': pnl, 'account': account,
            'strength': strengths[i]
        })
        
        if account <= 0:
            break
    
    wr = wins / max(wins + losses, 1)
    return {
        'trades': len(trades), 'win_rate': wr, 'total_pnl': total_pnl,
        'final_account': account, 'max_drawdown': max_dd,
        'wins': wins, 'losses': losses,
        'profit_factor': wins / max(losses, 1),
        'return_pct': (account / 1000 - 1) * 100,
    }

def main():
    t0 = time.time()
    print("═══ MULTI-STRATEGY ENSEMBLE — GOLD ═══\n")
    
    # Load data
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    
    print(f"Data: {len(closes):,} M5 bars")
    print(f"Price range: ${closes.min():.2f} – ${closes.max():.2f}")
    
    # Generate signals
    print("\n═══ GENERATING SIGNALS ═══")
    pm = PortfolioManager()
    signals, strengths = pm.generate_signals(closes, highs, lows)
    
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    print(f"BUY signals: {buy_signals:,} ({buy_signals/len(closes)*100:.2f}%)")
    print(f"SELL signals: {sell_signals:,} ({sell_signals/len(closes)*100:.2f}%)")
    print(f"Avg strength: {strengths[strengths>0].mean():.3f}")
    
    # Backtest with different SL/TP
    print("\n═══ BACKTESTING ═══")
    best = None
    best_pnl = -999999
    
    for sl in [1.0, 1.5, 2.0]:
        for tp in [2.0, 3.0, 4.0, 5.0]:
            for ms in [0.5, 0.67]:
                r = backtest_ensemble(closes, highs, lows, signals, strengths,
                                     sl_mult=sl, tp_mult=tp, min_strength=ms)
                if r['trades'] > 20 and r['total_pnl'] > best_pnl:
                    best_pnl = r['total_pnl']
                    best = {'sl': sl, 'tp': tp, 'min_str': ms, **r}
                    print(f"  SL={sl} TP={tp} str>={ms}: trades={r['trades']}, "
                          f"WR={r['win_rate']:.1%}, PnL=${r['total_pnl']:.2f}, "
                          f"DD={r['max_drawdown']:.1%}")
    
    if best:
        print(f"\n═══ BEST RESULT ═══")
        print(f"  SL: {best['sl']} ATR")
        print(f"  TP: {best['tp']} ATR")
        print(f"  Min strength: {best['min_str']}")
        print(f"  Trades: {best['trades']:,}")
        print(f"  Win rate: {best['win_rate']:.1%}")
        print(f"  Total P&L: ${best['total_pnl']:,.2f}")
        print(f"  Final account: ${best['final_account']:,.2f}")
        print(f"  Max drawdown: {best['max_drawdown']:.1%}")
        print(f"  Return: {best['return_pct']:.1f}%")
        print(f"  Profit factor: {best['profit_factor']:.2f}")
        
        # Compounding projections
        print(f"\n═══ COMPOUNDING PROJECTIONS ═══")
        years = len(closes) / (288 * 365)
        annual_return = (best['final_account'] / 1000) ** (1/years) - 1
        print(f"  Annual return: {annual_return:.1%}")
        for start in [100, 500, 1000]:
            acc = start
            for yr in [1, 2, 3, 5]:
                acc_final = start * ((1 + annual_return) ** yr)
                print(f"  ${start} → ${acc_final:,.0f} after {yr} years")
    
    # Individual strategy performance
    print(f"\n═══ INDIVIDUAL STRATEGY PERFORMANCE ═══")
    for name, strat in pm.strategies.items():
        sig = np.zeros(len(closes), dtype=np.int8)
        for i in range(300, len(closes)):
            sig[i] = strat.signal(closes, highs, lows, i)
        
        buy_count = (sig == 1).sum()
        sell_count = (sig == -1).sum()
        
        if buy_count + sell_count > 0:
            r = backtest_ensemble(closes, highs, lows, sig, np.abs(sig).astype(float),
                                 sl_mult=1.5, tp_mult=3.0, min_strength=0.5)
            print(f"  {name:12s}: {buy_count:4d} buys, {sell_count:4d} sells, "
                  f"WR={r['win_rate']:.1%}, PnL=${r['total_pnl']:.2f}, "
                  f"trades={r['trades']}")
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s)")

if __name__ == '__main__':
    main()
