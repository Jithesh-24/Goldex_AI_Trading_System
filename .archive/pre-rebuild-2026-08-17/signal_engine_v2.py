#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_engine_v2.py — Enhanced signal generation with microstructure features

Combines:
1. LightGBM model predictions (from retrained model)
2. Microstructure features (from tick_analyzer.py)
3. Quantitative features (from advanced_features.py)
4. Regime detection (existing)
5. Event calendar (existing)
6. Macro context (existing)

Generates signals with:
- Direction (BUY/SELL)
- Confidence (calibrated probability)
- Entry/SL/TP levels (dynamic based on volatility)
- Signal rating (0-100)
- Reason codes (why this signal was generated)
"""

import json
import os
import time
from datetime import datetime, timezone

OUT = "/home/jith/.hermes/profiles/trading/cron/output"
TICK_FEATURES = os.path.join(OUT, "tick_features.json")
SIGNAL_OUTPUT = os.path.join(OUT, ".active_signal_ai_v2.json")

# ── Signal Generation Rules ────────────────────────────────────

def compute_signal_reasons(features):
    """Generate human-readable reasons for the signal."""
    reasons = []
    
    # Microstructure reasons
    if features.get("ofi", 0.5) > 0.6:
        reasons.append("OFI_BUY_PRESSURE")
    elif features.get("ofi", 0.5) < 0.4:
        reasons.append("OFI_SELL_PRESSURE")
    
    if features.get("vpin", 0.5) > 0.7:
        reasons.append("INFORMED_FLOW")
    
    if features.get("momentum_burst", False):
        reasons.append("MOMENTUM_BURST")
    
    if features.get("mean_rev_signal") == "BUY":
        reasons.append("MEAN_REVERSION_OVERSOLD")
    elif features.get("mean_rev_signal") == "SELL":
        reasons.append("MEAN_REVERSION_OVERBOUGHT")
    
    if features.get("spread_alert", False):
        reasons.append("WIDE_SPREAD")
    
    if features.get("is_overlap", False):
        reasons.append("LONDON_NY_OVERLAP")
    
    if features.get("pressure", 0) > 0.3:
        reasons.append("BID_DOMINANCE")
    elif features.get("pressure", 0) < -0.3:
        reasons.append("ASK_DOMINANCE")
    
    if features.get("acceleration", 0) > 0.0001:
        reasons.append("ACCELERATING_UP")
    elif features.get("acceleration", 0) < -0.0001:
        reasons.append("ACCELERATING_DOWN")
    
    return reasons


def compute_dynamic_sl_tp(features, direction, current_price):
    """Compute dynamic SL/TP based on volatility and microstructure."""
    # Use realized volatility for SL/TP sizing
    rv = features.get("rv_short", 0.001)
    if rv < 0.0001:
        rv = 0.001
    
    # Base ATR-like measure from recent price range
    # Use spread and volatility for SL/TP
    base_sl_pips = max(15, min(50, int(rv * 10000 * 2)))  # 15-50 pips
    base_tp_pips = base_sl_pips * 2  # 2:1 R:R minimum
    
    # Adjust for session
    if features.get("is_overlap", False):
        # Overlap = more volatile, wider SL
        base_sl_pips = int(base_sl_pips * 1.2)
        base_tp_pips = int(base_tp_pips * 1.2)
    
    # Adjust for spread
    spread = features.get("spread_pips", 10)
    if spread > 20:
        # Wide spread = institutional activity, wait for better entry
        base_sl_pips += int(spread * 0.5)
        base_tp_pips += int(spread * 0.5)
    
    if direction == "BUY":
        sl = current_price - base_sl_pips * 0.01
        tp = current_price + base_tp_pips * 0.01
    else:
        sl = current_price + base_sl_pips * 0.01
        tp = current_price - base_tp_pips * 0.01
    
    return round(sl, 2), round(tp, 2), base_sl_pips, base_tp_pips


def generate_signal(model_prediction, features, regime=None):
    """
    Generate a trading signal from model prediction and microstructure features.
    
    Args:
        model_prediction: dict with 'direction', 'probability', 'confidence'
        features: dict from tick_analyzer.py or advanced_features.py
        regime: optional current regime string
    
    Returns:
        dict with signal details
    """
    direction = model_prediction.get("direction", "NEUTRAL")
    prob = model_prediction.get("probability", 0.5)
    confidence = model_prediction.get("confidence", 0.5)
    
    current_price = features.get("mid", features.get("close", 0))
    if current_price == 0:
        return None
    
    # Compute signal reasons
    reasons = compute_signal_reasons(features)
    
    # Filter: don't trade against strong microstructure
    if direction == "BUY" and features.get("ofi", 0.5) < 0.3:
        reasons.append("FILTERED_SELL_PRESSURE")
        return None  # Don't buy into selling pressure
    
    if direction == "SELL" and features.get("ofi", 0.5) > 0.7:
        reasons.append("FILTERED_BUY_PRESSURE")
        return None  # Don't sell into buying pressure
    
    # Don't trade during toxic flow
    if features.get("vpin", 0.5) > 0.8:
        reasons.append("FILTERED_TOXIC_FLOW")
        return None
    
    # Don't trade with very wide spreads
    if features.get("spread_pips", 10) > 30:
        reasons.append("FILTERED_WIDE_SPREAD")
        return None
    
    # Compute dynamic SL/TP
    sl, tp, sl_pips, tp_pips = compute_dynamic_sl_tp(features, direction, current_price)
    
    # Signal rating (0-100)
    rating = compute_signal_rating(features, prob, confidence)
    
    # Build signal
    signal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "probability": round(prob, 4),
        "confidence": round(confidence, 4),
        "rating": rating,
        "entry": round(current_price, 2),
        "sl": sl,
        "tp": tp,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "rr_ratio": round(tp_pips / max(sl_pips, 1), 2),
        "reasons": reasons,
        "regime": regime,
        "microstructure": {
            "ofi": features.get("ofi", 0.5),
            "vpin": features.get("vpin", 0.5),
            "rv_short": features.get("rv_short", 0),
            "spread_pips": features.get("spread_pips", 10),
            "momentum": features.get("momentum", 0),
            "pressure": features.get("pressure", 0),
            "cvd": features.get("cvd", 0),
        },
        "session": features.get("session", "UNKNOWN"),
        "is_overlap": features.get("is_overlap", False),
    }
    
    return signal


def compute_signal_rating(features, prob, confidence):
    """
    Compute signal rating (0-100) based on multiple factors.
    
    Higher rating = stronger conviction signal.
    """
    rating = 0
    
    # Model probability (0-30 points)
    rating += int((prob - 0.5) * 60)  # 50% = 0, 60% = 6, 70% = 12
    
    # Confidence (0-20 points)
    rating += int(confidence * 20)
    
    # OFI alignment (0-15 points)
    ofi = features.get("ofi", 0.5)
    if prob > 0.5 and ofi > 0.6:
        rating += 15
    elif prob < 0.5 and ofi < 0.4:
        rating += 15
    elif 0.4 <= ofi <= 0.6:
        rating += 5  # Neutral OFI is okay
    
    # VPIN (0-10 points)
    vpin = features.get("vpin", 0.5)
    if vpin > 0.7:
        rating -= 10  # Toxic flow is bad
    elif vpin < 0.3:
        rating += 10  # Clean flow is good
    
    # Session bonus (0-10 points)
    if features.get("is_overlap", False):
        rating += 10
    elif features.get("session") == "LONDON":
        rating += 5
    elif features.get("session") == "NEW_YORK":
        rating += 5
    
    # Spread penalty (0-10 points)
    spread = features.get("spread_pips", 10)
    if spread < 10:
        rating += 10
    elif spread < 20:
        rating += 5
    elif spread > 30:
        rating -= 10
    
    # Momentum alignment (0-10 points)
    momentum = features.get("momentum", 0)
    if prob > 0.5 and momentum > 0:
        rating += 10
    elif prob < 0.5 and momentum < 0:
        rating += 10
    
    # Clamp to 0-100
    return max(0, min(100, rating))


def write_signal(signal):
    """Write signal to file."""
    if signal is None:
        # Remove old signal
        if os.path.exists(SIGNAL_OUTPUT):
            os.remove(SIGNAL_OUTPUT)
        return
    
    with open(SIGNAL_OUTPUT, "w") as f:
        json.dump(signal, f, indent=2)


if __name__ == "__main__":
    # Test with dummy data
    test_features = {
        "mid": 2400.50,
        "ofi": 0.65,
        "vpin": 0.3,
        "rv_short": 0.001,
        "spread_pips": 8,
        "momentum": 0.0001,
        "pressure": 0.4,
        "cvd": 150,
        "session": "LONDON",
        "is_overlap": False,
    }
    
    test_prediction = {
        "direction": "BUY",
        "probability": 0.65,
        "confidence": 0.7,
    }
    
    signal = generate_signal(test_prediction, test_features, regime="TRENDING_UP")
    print(json.dumps(signal, indent=2))
