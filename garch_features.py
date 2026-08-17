#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
garch_features.py — GARCH volatility modeling

Models volatility clustering (the #1 stylized fact of financial markets).
Gold exhibits strong volatility clustering — high vol follows high vol.

EGARCH (Exponential GARCH) captures asymmetric effects:
- Bad news (price drops) increases volatility more than good news
- This is critical for gold trading

Features:
- garch_vol: conditional volatility from EGARCH(1,1)
- garch_forecast: 1-step ahead volatility forecast
- vol_regime: high/low volatility regime (from GARCH)
- vol_persistence: how long vol shocks last
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _egarch_loglik(params, returns, n):
    """EGARCH(1,1) log-likelihood for Numba."""
    omega, alpha, beta, gamma = params
    
    # Stationarity constraint
    if alpha + beta >= 1.0:
        return 1e10  # penalty
    
    log_h = np.log(np.var(returns) + 1e-10)  # initialize
    log_lik = 0.0
    
    for t in range(1, n):
        eps = returns[t-1] / np.exp(log_h/2 + 1e-10)
        log_h = (omega + 
                 alpha * (abs(eps) - np.sqrt(2/np.pi)) + 
                 gamma * eps + 
                 beta * log_h)
        
        # Bound log_h
        if log_h > 20:
            log_h = 20
        elif log_h < -20:
            log_h = -20
        
        h_t = np.exp(log_h)
        log_lik += -0.5 * (np.log(2*np.pi*h_t + 1e-10) + returns[t]**2 / (h_t + 1e-10))
    
    return -log_lik  # minimize negative log-lik


def fit_egarch(returns, method='grid'):
    """
    Fit EGARCH(1,1) model.
    
    Returns: omega, alpha, beta, gamma, conditional_vol
    """
    n = len(returns)
    if n < 100:
        return None
    
    # Grid search (robust, no gradient issues)
    best_params = None
    best_ll = 1e10
    
    # Parameter grids
    omegas = [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05]
    alphas = [0.05, 0.1, 0.15, 0.2]
    betas = [0.85, 0.9, 0.95]
    gammas = [-0.1, -0.05, 0.0, 0.05, 0.1]
    
    for omega in omegas:
        for alpha in alphas:
            for beta in betas:
                if alpha + beta >= 1.0:
                    continue
                for gamma in gammas:
                    params = np.array([omega, alpha, beta, gamma])
                    ll = _egarch_loglik(params, returns, n)
                    if ll < best_ll:
                        best_ll = ll
                        best_params = params
    
    if best_params is None:
        return None
    
    # Compute conditional volatility with best params
    omega, alpha, beta, gamma = best_params
    log_h = np.log(np.var(returns) + 1e-10)
    cond_vol = np.zeros(n)
    cond_vol[0] = np.exp(log_h)
    
    for t in range(1, n):
        eps = returns[t-1] / np.exp(log_h/2 + 1e-10)
        log_h = (omega + 
                 alpha * (abs(eps) - np.sqrt(2/np.pi)) + 
                 gamma * eps + 
                 beta * log_h)
        log_h = np.clip(log_h, -20, 20)
        cond_vol[t] = np.exp(log_h)
    
    return {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'persistence': alpha + beta,
        'cond_vol': cond_vol,
        'log_lik': -best_ll
    }


def compute_garch_features(returns, window=200):
    """
    Compute GARCH features for a return series.
    
    Returns dict with:
    - garch_vol: current conditional volatility
    - garch_forecast: 1-step ahead vol forecast
    - vol_regime: 0=low, 1=high (based on vol z-score)
    - vol_persistence: alpha + beta (how long shocks last)
    - vol_asymmetry: gamma (asymmetric news impact)
    - vol_shock: current innovation (surprise)
    """
    result = {
        'garch_vol': 0.0,
        'garch_forecast': 0.0,
        'vol_regime': 0.0,
        'vol_persistence': 0.0,
        'vol_asymmetry': 0.0,
        'vol_shock': 0.0
    }
    
    if len(returns) < window:
        return result
    
    # Fit EGARCH on recent returns
    fit = fit_egarch(returns[-window:])
    
    if fit is None:
        return result
    
    result['garch_vol'] = float(fit['cond_vol'][-1])
    result['vol_persistence'] = float(fit['persistence'])
    result['vol_asymmetry'] = float(fit['gamma'])
    
    # 1-step ahead forecast: h_{t+1} = omega + alpha*(|eps_t| - sqrt(2/pi)) + gamma*eps_t + beta*h_t
    eps = returns[-1] / (fit['cond_vol'][-1]**0.5 + 1e-10)
    log_h = np.log(fit['cond_vol'][-1] + 1e-10)
    log_h_next = (fit['omega'] + 
                  fit['alpha'] * (abs(eps) - np.sqrt(2/np.pi)) +
                  fit['gamma'] * eps +
                  fit['beta'] * log_h)
    result['garch_forecast'] = float(np.exp(np.clip(log_h_next, -20, 20)))
    
    # Vol regime (z-score of current vol)
    vol_mean = np.mean(fit['cond_vol'][-50:])
    vol_std = np.std(fit['cond_vol'][-50:]) + 1e-10
    vol_z = (fit['cond_vol'][-1] - vol_mean) / vol_std
    result['vol_regime'] = 1.0 if vol_z > 1.0 else 0.0  # high vol regime
    
    # Innovation (surprise)
    result['vol_shock'] = float(returns[-1] / (fit['cond_vol'][-1]**0.5 + 1e-10))
    
    return result


if __name__ == '__main__':
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.01
    
    # Add volatility clustering
    for i in range(1, n):
        returns[i] *= (1 + 0.5 * abs(returns[i-1]) * 100)
    
    print("Testing EGARCH...")
    result = compute_garch_features(returns, window=200)
    print(f"  GARCH vol: {result['garch_vol']:.6f}")
    print(f"  GARCH forecast: {result['garch_forecast']:.6f}")
    print(f"  Vol regime: {result['vol_regime']:.1f}")
    print(f"  Persistence: {result['vol_persistence']:.4f}")
    print(f"  Asymmetry: {result['vol_asymmetry']:.4f}")
    print(f"  Shock: {result['vol_shock']:.4f}")
    print("✅ EGARCH working!")
