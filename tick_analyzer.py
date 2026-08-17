#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tick_analyzer.py — Real-time XM tick microstructure analysis

Captures ticks from xm_ticker.py's xm_tick_state.json and computes:
1. Order Flow Imbalance (OFI) — bid/ask volume ratio
2. VWAP deviation — price vs volume-weighted average
3. Realized volatility — high-frequency vol estimation
4. Spread dynamics — bid-ask spread changes
5. Momentum burst detection — rapid price movements
6. Mean reversion signals — fair value deviation
7. Trade toxicity (VPIN) — probability of informed trading
8. Session-based signals — London/NY overlap patterns
9. CVD (Cumulative Volume Delta) — net buying/selling pressure
10. Micro-price — volume-weighted mid price

Reads: /home/jith/.hermes/profiles/trading/cron/output/xm_tick_state.json
Writes: /home/jith/.hermes/profiles/trading/cron/output/tick_features.json
"""

import json
import os
import time
import math
from collections import deque
from datetime import datetime, timezone

OUT = "/home/jith/.hermes/profiles/trading/cron/output"
TICK_STATE = os.path.join(OUT, "xm_tick_state.json")
TICK_FEATURES = os.path.join(OUT, "tick_features.json")

# ── Configuration ──────────────────────────────────────────────
WINDOW_TICKS = 100        # Rolling window for microstructure features
WINDOW_SHORT = 20         # Short window for momentum
WINDOW_LONG = 500         # Long window for VWAP
SPREAD_ALERT_THRESHOLD = 2.0  # Spread z-score for alert
MOMENTUM_BURST_THRESHOLD = 0.0008  # 0.08% price move in short window
MEAN_REVERSION_THRESHOLD = 2.0  # Standard deviations from VWAP
VPIN_BUCKETS = 20         # VPIN estimation buckets
SESSION_OVERLAP_START = 7  # 07:00 UTC = 12:30 IST (London-NY overlap starts)
SESSION_OVERLAP_END = 16   # 16:00 UTC = 21:30 IST (overlap ends)


class TickAnalyzer:
    """Real-time tick microstructure analyzer."""
    
    def __init__(self):
        self.tick_buffer = deque(maxlen=WINDOW_LONG)
        self.bid_buffer = deque(maxlen=WINDOW_LONG)
        self.ask_buffer = deque(maxlen=WINDOW_LONG)
        self.spread_buffer = deque(maxlen=WINDOW_LONG)
        self.volume_buffer = deque(maxlen=WINDOW_LONG)
        self.cvd = 0.0  # Cumulative Volume Delta
        self.vpin_volume = deque(maxlen=WINDOW_LONG)
        self.vpin_buy_vol = 0.0
        self.vpin_sell_vol = 0.0
        self.last_vpin_compute = 0
        self.vpin_value = 0.5  # Neutral
        
    def add_tick(self, bid, ask, volume=1.0):
        """Add a new tick and compute all features."""
        now = time.time()
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pips = spread * 100  # Convert to pips (0.01 = 1 pip)
        
        self.tick_buffer.append((now, mid))
        self.bid_buffer.append(bid)
        self.ask_buffer.append(ask)
        self.spread_buffer.append(spread_pips)
        self.volume_buffer.append(volume)
        
        # Update CVD (simplified: compare to mid of previous tick)
        if len(self.tick_buffer) >= 2:
            prev_mid = self.tick_buffer[-2][1]
            if mid > prev_mid:
                self.cvd += volume  # Buying pressure
            elif mid < prev_mid:
                self.cvd -= volume  # Selling pressure
        
        # Compute all features
        features = self._compute_all(now, bid, ask, mid, spread_pips, volume)
        return features
    
    def _compute_all(self, now, bid, ask, mid, spread, volume):
        """Compute all microstructure features."""
        features = {}
        
        # 1. Basic features
        features["bid"] = round(bid, 2)
        features["ask"] = round(ask, 2)
        features["mid"] = round(mid, 2)
        features["spread_pips"] = round(spread, 1)
        features["cvd"] = round(self.cvd, 2)
        
        if len(self.tick_buffer) < 5:
            return features
        
        # 2. VWAP (Volume-Weighted Average Price)
        features["vwap"] = self._compute_vwap()
        features["vwap_dev"] = round((mid - features["vwap"]) / features["vwap"] * 10000, 2)  # basis points
        features["vwap_dev_z"] = self._compute_vwap_dev_z()
        
        # 3. Order Flow Imbalance (OFI)
        features["ofi"] = self._compute_ofi()
        features["ofi_signal"] = "BUY" if features["ofi"] > 0.6 else ("SELL" if features["ofi"] < 0.4 else "NEUTRAL")
        
        # 4. Realized Volatility
        features["rv_short"] = self._compute_realized_vol(WINDOW_SHORT)
        features["rv_long"] = self._compute_realized_vol(WINDOW_LONG)
        features["rv_ratio"] = round(features["rv_short"] / max(features["rv_long"], 1e-10), 3)
        
        # 5. Spread Dynamics
        features["spread_z"] = self._compute_spread_z()
        features["spread_alert"] = abs(features["spread_z"]) > SPREAD_ALERT_THRESHOLD
        
        # 6. Momentum Burst Detection
        features["momentum"] = self._compute_momentum()
        features["momentum_burst"] = abs(features["momentum"]) > MOMENTUM_BURST_THRESHOLD
        features["momentum_dir"] = "UP" if features["momentum"] > 0 else "DOWN"
        
        # 7. Mean Reversion Signal
        features["mean_rev_z"] = self._compute_mean_reversion_z()
        features["mean_rev_signal"] = "BUY" if features["mean_rev_z"] < -MEAN_REVERSION_THRESHOLD else (
            "SELL" if features["mean_rev_z"] > MEAN_REVERSION_THRESHOLD else "NEUTRAL"
        )
        
        # 8. Trade Toxicity (VPIN)
        self._update_vpin(volume, mid)
        features["vpin"] = self.vpin_value
        features["toxic_flow"] = self.vpin_value > 0.7  # High toxicity = informed trading
        
        # 9. Session-based features
        features["session"] = self._get_session()
        features["is_overlap"] = self._is_overlap()
        
        # 10. Micro-price (volume-weighted mid)
        features["micro_price"] = self._compute_micro_price()
        
        # 11. Bid-Ask Pressure (microstructure pressure)
        features["pressure"] = self._compute_pressure()
        
        # 12. Price acceleration (second derivative)
        features["acceleration"] = self._compute_acceleration()
        
        return features
    
    def _compute_vwap(self):
        """Volume-Weighted Average Price."""
        if len(self.tick_buffer) < 2:
            return self.tick_buffer[-1][1] if self.tick_buffer else 0
        
        # Simplified VWAP: use equal weight for recent ticks
        ticks = list(self.tick_buffer)[-min(WINDOW_LONG, len(self.tick_buffer)):]
        vwap = sum(t[1] for t in ticks) / len(ticks)
        return round(vwap, 2)
    
    def _compute_vwap_dev_z(self):
        """Z-score of price deviation from VWAP."""
        if len(self.tick_buffer) < 20:
            return 0
        
        vwap = self._compute_vwap()
        ticks = list(self.tick_buffer)[-20:]
        deviations = [(t[1] - vwap) for t in ticks]
        if not deviations:
            return 0
        
        mean_dev = sum(deviations) / len(deviations)
        std_dev = (sum((d - mean_dev) ** 2 for d in deviations) / len(deviations)) ** 0.5
        
        if std_dev < 1e-10:
            return 0
        
        return round((0 - mean_dev) / std_dev, 2)  # How far current is from mean
    
    def _compute_ofi(self):
        """Order Flow Imbalance: bid_volume / (bid_volume + ask_volume)."""
        if len(self.bid_buffer) < 10:
            return 0.5
        
        window = min(WINDOW_TICKS, len(self.bid_buffer))
        bids = list(self.bid_buffer)[-window:]
        asks = list(self.ask_buffer)[-window:]
        
        # Count bid-side vs ask-side ticks
        bid_ticks = sum(1 for i in range(1, len(bids)) if bids[i] >= bids[i-1])
        ask_ticks = sum(1 for i in range(1, len(asks)) if asks[i] <= asks[i-1])
        total = bid_ticks + ask_ticks
        
        if total == 0:
            return 0.5
        
        return bid_ticks / total
    
    def _compute_realized_vol(self, window):
        """Realized volatility over N ticks."""
        if len(self.tick_buffer) < window + 1:
            return 0
        
        ticks = list(self.tick_buffer)[-window-1:]
        returns = [(ticks[i][1] - ticks[i-1][1]) / ticks[i-1][1] for i in range(1, len(ticks))]
        
        if not returns:
            return 0
        
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        
        # Annualize (assuming 25ms ticks, ~288k ticks/day, 252 trading days)
        # But keep in per-tick terms for comparison
        return (var_r ** 0.5) * 10000  # In basis points
    
    def _compute_spread_z(self):
        """Z-score of current spread vs recent history."""
        if len(self.spread_buffer) < 20:
            return 0
        
        spreads = list(self.spread_buffer)
        current = spreads[-1]
        historical = spreads[:-1]
        
        mean = sum(historical) / len(historical)
        std = (sum((s - mean) ** 2 for s in historical) / len(historical)) ** 0.5
        
        if std < 1e-10:
            return 0
        
        return round((current - mean) / std, 2)
    
    def _compute_momentum(self):
        """Short-term momentum (price change over recent ticks)."""
        if len(self.tick_buffer) < WINDOW_SHORT + 1:
            return 0
        
        recent = self.tick_buffer[-1][1]
        past = self.tick_buffer[-WINDOW_SHORT][1]
        
        return (recent - past) / past
    
    def _compute_mean_reversion_z(self):
        """Z-score for mean reversion: how far is price from VWAP in std devs."""
        if len(self.tick_buffer) < 30:
            return 0
        
        vwap = self._compute_vwap()
        ticks = list(self.tick_buffer)[-30:]
        deviations = [(t[1] - vwap) for t in ticks]
        
        if not deviations:
            return 0
        
        std = (sum(d ** 2 for d in deviations) / len(deviations)) ** 0.5
        
        if std < 1e-10:
            return 0
        
        current_dev = self.tick_buffer[-1][1] - vwap
        return round(current_dev / std, 2)
    
    def _update_vpin(self, volume, price):
        """Volume-Synchronized Probability of Informed Trading (VPIN)."""
        self.vpin_volume.append(volume)
        
        # Classify volume as buy/sell using tick direction
        if len(self.tick_buffer) >= 2:
            if self.tick_buffer[-1][1] > self.tick_buffer[-2][1]:
                self.vpin_buy_vol += volume
            else:
                self.vpin_sell_vol += volume
        
        # Compute VPIN periodically
        now = time.time()
        if now - self.last_vpin_compute > 60:  # Every 60 seconds
            total_vol = self.vpin_buy_vol + self.vpin_sell_vol
            if total_vol > 0:
                self.vpin_value = abs(self.vpin_buy_vol - self.vpin_sell_vol) / total_vol
            self.vpin_buy_vol = 0
            self.vpin_sell_vol = 0
            self.last_vpin_compute = now
    
    def _get_session(self):
        """Current trading session."""
        hour = datetime.now(timezone.utc).hour
        if 7 <= hour < 16:
            return "LONDON"
        elif 12 <= hour < 21:
            return "NEW_YORK"
        elif 0 <= hour < 7:
            return "ASIA"
        else:
            return "OFF_HOURS"
    
    def _is_overlap(self):
        """London-NY overlap period."""
        hour = datetime.now(timezone.utc).hour
        return 12 <= hour < 16  # 12:00-16:00 UTC
    
    def _compute_micro_price(self):
        """Volume-weighted mid price (micro-price)."""
        if not self.bid_buffer or not self.ask_buffer:
            return 0
        
        bid = self.bid_buffer[-1]
        ask = self.ask_buffer[-1]
        
        # Simplified: equal weight (in reality, use order book depth)
        return round((bid + ask) / 2, 2)
    
    def _compute_pressure(self):
        """Microstructure pressure: bid vs ask dominance."""
        if len(self.bid_buffer) < 20:
            return 0
        
        window = 20
        bids = list(self.bid_buffer)[-window:]
        asks = list(self.ask_buffer)[-window:]
        
        # Count how many times bid moved up vs ask moved down
        bid_up = sum(1 for i in range(1, len(bids)) if bids[i] > bids[i-1])
        ask_down = sum(1 for i in range(1, len(asks)) if asks[i] < asks[i-1])
        total = window - 1
        
        if total == 0:
            return 0
        
        pressure = (bid_up - ask_down) / total
        return round(pressure, 3)
    
    def _compute_acceleration(self):
        """Price acceleration (second derivative of price)."""
        if len(self.tick_buffer) < 30:
            return 0
        
        # First derivative: momentum
        p1 = self.tick_buffer[-15][1]
        p2 = self.tick_buffer[-1][1]
        momentum1 = (p2 - p1) / p1
        
        # Second derivative: acceleration
        p0 = self.tick_buffer[-30][1]
        momentum0 = (p1 - p0) / p0
        
        acceleration = momentum1 - momentum0
        return round(acceleration * 10000, 4)  # In basis points


def read_tick_state():
    """Read latest tick state from xm_ticker.py."""
    try:
        with open(TICK_STATE, "r") as f:
            state = json.load(f)
        return state
    except:
        return None


def run_analyzer():
    """Main analyzer loop."""
    analyzer = TickAnalyzer()
    last_tick_time = 0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tick analyzer started")
    
    while True:
        try:
            state = read_tick_state()
            
            if state is None:
                time.sleep(0.1)
                continue
            
            # Extract bid/ask from xm_ticker state
            bid = state.get("bid", state.get("last", 0))
            ask = state.get("ask", state.get("last", 0))
            tick_time = state.get("time", 0)
            
            # Only process new ticks
            if tick_time <= last_tick_time:
                time.sleep(0.05)
                continue
            
            last_tick_time = tick_time
            
            # Compute features
            features = analyzer.add_tick(bid, ask, 1.0)
            features["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Write features
            with open(TICK_FEATURES, "w") as f:
                json.dump(features, f, indent=2)
            
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.05)  # 50ms polling


if __name__ == "__main__":
    run_analyzer()
