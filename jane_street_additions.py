#!/usr/bin/env python3
"""
jane_street_additions.py — Jane Street ideology patches for ai_signal_engine.py
=============================================================================

Three critical additions for the v8.9 Jane Street upgrade:

1. COST-AWARE ENTRY GATING
   Don't fire a signal if the expected alpha < total transaction cost.
   Cost = spread + estimated slippage + commission.
   This prevents "barely positive EV" trades that get eaten by costs.

2. LIVE PnL TRACKING + DRAWDOWN CIRCUIT BREAKER
   Track running session PnL. If drawdown exceeds a threshold,
   pause trading for a cooldown period. Prevents tilt-bleeding.

3. FEATURE DRIFT DETECTION
   Monitor live feature distributions. If a feature drifts > 3σ from
   training mean, flag the signal as degraded confidence.

These are ADDITIVE patches — they don't change existing logic, only add gates.
"""
import json
import time
import os

# ═══════════════════════════════════════════════════════════════════
# 1. COST-AWARE ENTRY GATE
# ═══════════════════════════════════════════════════════════════════

# Gold XAUUSD typical costs (XM micro):
SPREAD_COST_USD = 0.30       # average spread on gold (0.30 = 3 pips)
SLIPPAGE_USD = 0.15          # estimated slippage in fast market
COMMISSION_USD = 0.00        # XM gold micro = 0 commission
TOTAL_COST_USD = SPREAD_COST_USD + SLIPPAGE_USD + COMMISSION_USD

def cost_aware_gate(exp_per_dollar_risked, sl_dist, atr):
    """
    Returns True if the trade's expected alpha exceeds total cost.
    
    exp_per_dollar_risked: from best_placement — expectancy per $ risked
    sl_dist: stop-loss distance in price units
    atr: current ATR
    
    The gate converts exp_per_dollar_risked to a dollar expectancy,
    then checks if it exceeds the total cost of the trade.
    """
    # Dollar expectancy = exp_per_dollar * risk_per_trade
    # risk_per_trade = sl_dist (in price units, roughly $ per lot micro)
    dollar_expectancy = exp_per_dollar_risked * sl_dist
    
    # Total cost for a round-trip (entry + exit spread + slippage)
    total_cost = TOTAL_COST_USD * 2  # both legs
    
    return dollar_expectancy > total_cost


# ═══════════════════════════════════════════════════════════════════
# 2. LIVE PnL TRACKING + DRAWDOWN CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════

PNL_STATE_FILE = "/home/jith/.hermes/profiles/trading/scripts/models/live_pnl_state.json"

# Circuit breaker thresholds:
MAX_DRAWDOWN_USD = -15.0     # pause if session PnL drops below -$15
COOLDOWN_SECONDS = 1800      # 30-minute cooldown after drawdown trip
MAX_TRADES_PER_HOUR = 4      # rate limit: max 4 trades per hour
MAX_DAILY_LOSSES = -25.0     # hard stop: halt rest of day if daily loss > $25

def load_pnl_state():
    """Load live PnL tracking state."""
    try:
        with open(PNL_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "session_pnl": 0.0,
            "trade_count": 0,
            "trade_times": [],
            "halt_until": 0,
            "daily_pnl": 0.0,
            "last_reset": "",
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
        }

def save_pnl_state(state):
    """Save live PnL tracking state."""
    os.makedirs(os.path.dirname(PNL_STATE_FILE), exist_ok=True)
    with open(PNL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def record_trade_result(state, pnl):
    """Record a completed trade's PnL."""
    state["session_pnl"] += pnl
    state["daily_pnl"] += pnl
    state["trade_count"] += 1
    state["trade_times"].append(time.time())
    # Keep only last 100 trade times
    state["trade_times"] = state["trade_times"][-100:]
    if pnl < 0:
        state["consecutive_losses"] += 1
        state["max_consecutive_losses"] = max(
            state["max_consecutive_losses"],
            state["consecutive_losses"]
        )
    else:
        state["consecutive_losses"] = 0
    save_pnl_state(state)

def daily_reset_if_needed(state):
    """Reset daily PnL tracking at midnight UTC."""
    import datetime
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    if state.get("last_reset") != today:
        state["daily_pnl"] = 0.0
        state["last_reset"] = today
        state["halt_until"] = 0
        state["trade_times"] = []
        save_pnl_state(state)
    return state

def circuit_breaker_check(state):
    """
    Returns (can_trade: bool, reason: str).
    Checks drawdown limits, cooldown periods, and rate limits.
    """
    now = time.time()
    
    # Check cooldown period
    if now < state.get("halt_until", 0):
        remaining = int(state["halt_until"] - now)
        return False, f"⏸ COOLDOWN: {remaining}s remaining (drawdown circuit breaker)"
    
    # Check session drawdown
    if state["session_pnl"] < MAX_DRAWDOWN_USD:
        state["halt_until"] = now + COOLDOWN_SECONDS
        save_pnl_state(state)
        return False, f"🚨 SESSION DRAWDOWN: ${state['session_pnl']:.2f} < ${MAX_DRAWDOWN_USD} — cooling {COOLDOWN_SECONDS}s"
    
    # Check daily loss limit
    if state["daily_pnl"] < MAX_DAILY_LOSSES:
        return False, f"🚨 DAILY LOSS LIMIT: ${state['daily_pnl']:.2f} < ${MAX_DAILY_LOSSES} — trading halted for rest of day"
    
    # Check rate limit (trades per hour)
    recent_trades = [t for t in state.get("trade_times", []) if now - t < 3600]
    if len(recent_trades) >= MAX_TRADES_PER_HOUR:
        return False, f"⏸ RATE LIMIT: {len(recent_trades)} trades in last hour (max {MAX_TRADES_PER_HOUR})"
    
    # Check consecutive losses (soft warning, not a hard gate)
    if state.get("consecutive_losses", 0) >= 3:
        return True, f"⚠️ WARNING: {state['consecutive_losses']} consecutive losses — proceed with caution"
    
    return True, "✅ All checks passed"


# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE DRIFT DETECTION (lightweight)
# ═══════════════════════════════════════════════════════════════════

DRIFT_STATS_FILE = "/home/jith/.hermes/profiles/trading/scripts/models/feature_drift_stats.json"

def load_drift_stats():
    """Load training-time feature statistics."""
    try:
        with open(DRIFT_STATS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def check_feature_drift(stats, fx, max_zscore=3.0):
    """
    Check if live features have drifted from training distribution.
    Returns (healthy: bool, drifted_features: list).
    
    stats: {feature_name: {mean: float, std: float}}
    fx: current feature dict
    max_zscore: threshold for flagging drift
    """
    if stats is None:
        return True, []
    
    drifted = []
    for feat_name, feat_stats in stats.items():
        if feat_name in fx and fx[feat_name] != 0.0:
            mean = feat_stats.get("mean", 0)
            std = feat_stats.get("std", 1)
            if std < 1e-8:
                continue
            zscore = abs((fx[feat_name] - mean) / std)
            if zscore > max_zscore:
                drifted.append((feat_name, fx[feat_name], mean, std, zscore))
    
    healthy = len(drifted) == 0
    return healthy, drifted


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION GUIDE
# ═══════════════════════════════════════════════════════════════════
#
# To patch ai_signal_engine.py, add these after the existing 3-layer filter:
#
# After line 1538 (if direction is not None and exp > 0):
#
#   # ── JANE STREET: COST-AWARE GATE ──
#   from jane_street_additions import cost_aware_gate
#   if not cost_aware_gate(exp, sl_dist, atr):
#       print(f"[{ts()}] 💰 cost gate: exp=${exp*sl_dist:.2f} < cost=${TOTAL_COST_USD*2:.2f}")
#       time.sleep(poll); continue
#
# After line 1538 (before signal rating):
#
#   # ── JANE STREET: CIRCUIT BREAKER ──
#   from jane_street_additions import load_pnl_state, circuit_breaker_check, daily_reset_if_needed
#   _pnl_state = daily_reset_if_needed(load_pnl_state())
#   _can_trade, _cb_reason = circuit_breaker_check(_pnl_state)
#   if not _can_trade:
#       print(f"[{ts()}] {_cb_reason}")
#       time.sleep(poll); continue
#
# After signal rating (line 1558):
#
#   # ── JANE STREET: FEATURE DRIFT CHECK ──
#   from jane_street_additions import load_drift_stats, check_feature_drift
#   _drift_stats = load_drift_stats()
#   _healthy, _drifted = check_feature_drift(_drift_stats, fx)
#   if not _healthy:
#       worst = max(_drifted, key=lambda x: x[4])
#       print(f"[{ts()}] 📊 drift: {worst[0]}={worst[1]:.4f} vs μ={worst[2]:.4f} (z={worst[4]:.1f})")
#       # Don't block, but reduce confidence
#       conf *= 0.8  # 20% confidence penalty for drift
#
# After trade execution:
#
#   # ── JANE STREET: RECORD TRADE RESULT ──
#   from jane_street_additions import record_trade_result
#   record_trade_result(_pnl_state, realized_pnl)
