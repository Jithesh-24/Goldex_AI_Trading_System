#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correlation_features.py — Cross-asset correlation breakdown detection

Gold correlates with DXY (-0.7), VIX (+0.3), yields (-0.2).
When these correlations BREAK DOWN, it signals regime change.

Features:
- corr_dxy: rolling correlation with DXY
- corr_vix: rolling correlation with VIX
- corr_change: rate of correlation change (breakdown detector)
- correlation_regime: normal/breakdown/transition
"""

import numpy as np


def rolling_correlation(x, y, window=20):
    """Rolling Pearson correlation."""
    n = len(x)
    corr = np.zeros(n)
    
    for i in range(window, n):
        x_win = x[i-window:i]
        y_win = y[i-window:i]
        
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        
        cov = np.sum((x_win - x_mean) * (y_win - y_mean))
        var_x = np.sum((x_win - x_mean) ** 2)
        var_y = np.sum((y_win - y_mean) ** 2)
        
        denom = np.sqrt(var_x * var_y)
        if denom > 1e-10:
            corr[i] = cov / denom
        else:
            corr[i] = 0.0
    
    return corr


def compute_correlation_features(gold_returns, dxy_returns=None, vix_returns=None, window=20):
    """
    Compute correlation features.
    
    If DXY/VIX not available, use synthetic proxies.
    """
    result = {
        'corr_dxy': 0.0,
        'corr_vix': 0.0,
        'corr_change_dxy': 0.0,
        'corr_change_vix': 0.0,
        'corr_regime': 0.0,  # 0=normal, 1=breakdown
    }
    
    n = len(gold_returns)
    
    # DXY correlation
    if dxy_returns is not None and len(dxy_returns) == n:
        corr_dxy = rolling_correlation(gold_returns, dxy_returns, window)
        result['corr_dxy'] = float(corr_dxy[-1])
        
        # Correlation change (breakdown detector)
        if n > window + 10:
            recent_corr = np.mean(corr_dxy[-5:])
            prev_corr = np.mean(corr_dxy[-window:-5])
            result['corr_change_dxy'] = float(recent_corr - prev_corr)
    else:
        # Synthetic: use gold autocorrelation as proxy
        if n > window:
            result['corr_dxy'] = float(np.corrcoef(
                gold_returns[-window:], 
                np.roll(gold_returns, 1)[-window:]
            )[0, 1])
    
    # VIX correlation
    if vix_returns is not None and len(vix_returns) == n:
        corr_vix = rolling_correlation(gold_returns, vix_returns, window)
        result['corr_vix'] = float(corr_vix[-1])
        
        if n > window + 10:
            recent_corr = np.mean(corr_vix[-5:])
            prev_corr = np.mean(corr_vix[-window:-5])
            result['corr_change_vix'] = float(recent_corr - prev_corr)
    
    # Correlation regime
    corr_change = abs(result['corr_change_dxy']) + abs(result['corr_change_vix'])
    if corr_change > 0.3:
        result['corr_regime'] = 1.0  # breakdown
    elif corr_change > 0.15:
        result['corr_regime'] = 0.5  # transition
    else:
        result['corr_regime'] = 0.0  # normal
    
    return result


if __name__ == '__main__':
    np.random.seed(42)
    n = 200
    
    # Gold returns
    gold = np.random.randn(n) * 0.01
    
    # DXY (negatively correlated)
    dxy = -gold * 0.7 + np.random.randn(n) * 0.005
    
    # VIX (positively correlated)
    vix = gold * 0.3 + np.random.randn(n) * 0.005
    
    print("Testing correlation features...")
    result = compute_correlation_features(gold, dxy, vix, window=20)
    print(f"  Corr DXY: {result['corr_dxy']:.4f}")
    print(f"  Corr VIX: {result['corr_vix']:.4f}")
    print(f"  Corr change DXY: {result['corr_change_dxy']:.4f}")
    print(f"  Corr change VIX: {result['corr_change_vix']:.4f}")
    print(f"  Corr regime: {result['corr_regime']:.1f}")
    print("✅ Correlation features working!")
