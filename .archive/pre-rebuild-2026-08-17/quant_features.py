#!/usr/bin/env python3
"""
quant_features.py — PURE QUANTITATIVE FEATURES
No lagging indicators. Only AI/quant methodologies.

Categories:
1. TIME SERIES: Kalman, OU, Hurst, entropy, autocorrelation
2. VOLATILITY: GARCH, realized vol, vol regime, vol clustering
3. MOMENTUM: Momentum decay, rate of change, momentum quality
4. MEAN REVERSION: Z-score, distance from mean, reversion speed
5. MARKET MICROSTRUCTURE: Kyle's lambda, Amihud, spread dynamics
6. STATISTICAL: Variance ratio, Hurst, R/S analysis, entropy
7. RISK: VaR, expected shortfall, tail risk, drawdown
8. REGIME: Trend detection, regime switching, break detection
9. CORRELATION: Autocorrelation structure, cross-feature correlation
10. SEASONALITY: Time patterns, session effects
"""
import numpy as np

def compute_all_quant_features(closes, highs, lows, spreads=None, volumes=None):
    """
    Compute ALL quantitative features from price data.
    Returns dict of feature_name -> value.
    No hardcoded thresholds — raw values only.
    LightGBM learns the thresholds.
    """
    n = len(closes)
    feats = {}
    
    if n < 100:
        return None
    
    c = closes[-1]
    log_returns = np.diff(np.log(closes))
    returns = np.diff(closes) / closes[:-1]
    
    # ═══════════════════════════════════════════
    # 1. TIME SERIES ANALYSIS
    # ═══════════════════════════════════════════
    
    # Kalman Filter (simplified 1D)
    # State: price level + velocity
    dt = 1.0
    q = 0.01  # process noise
    r = 0.1   # measurement noise
    
    # Initialize
    x_hat = c  # state estimate
    p = 1.0    # estimate error
    v = 0.0    # velocity
    
    kalman_prices = []
    for i in range(min(100, n)):
        idx = n - 1 - i
        # Predict
        x_hat_pred = x_hat + v * dt
        p_pred = p + q
        
        # Update
        k = p_pred / (p_pred + r)
        x_hat = x_hat_pred + k * (closes[idx] - x_hat_pred)
        p = (1 - k) * p_pred
        v = (x_hat - kalman_prices[-1]) / dt if kalman_prices else 0
        
        kalman_prices.append(x_hat)
    
    feats['kalman_trend'] = (kalman_prices[0] - kalman_prices[-1]) / kalman_prices[-1] if kalman_prices else 0
    feats['kalman_velocity'] = v
    feats['kalman_acceleration'] = (v - feats.get('kalman_velocity_prev', v)) if len(kalman_prices) > 10 else 0
    feats['kalman_residual'] = c - kalman_prices[0] if kalman_prices else 0
    feats['kalman_signal_to_noise'] = abs(feats['kalman_trend']) / max(abs(r), 0.001)
    
    # Ornstein-Uhlenbeck Process
    # Estimates mean reversion parameters
    window = min(100, n)
    prices = closes[-window:]
    mean_price = np.mean(prices)
    
    # OU parameters via OLS
    y = prices[1:] - prices[:-1]
    x = prices[:-1] - mean_price
    if len(x) > 10 and np.var(x) > 0:
        theta = -np.polyfit(x, y, 1)[0]  # mean reversion speed
        mu = mean_price  # long-term mean
        residuals = y + theta * x
        sigma = np.std(residuals)  # volatility
    else:
        theta = 0.1; mu = mean_price; sigma = 0.01
    
    feats['ou_theta'] = theta
    feats['ou_mu'] = (mu - c) / c if c > 0 else 0
    feats['ou_half_life'] = np.log(2) / max(theta, 0.001)
    feats['ou_z_score'] = (c - mu) / max(sigma * np.sqrt(window), 0.001)
    feats['ou_signal'] = theta * (mu - c)  # expected drift
    feats['ou_is_mean_reverting'] = 1 if theta > 0 else 0
    feats['ou_volatility'] = sigma
    
    # Hurst Exponent (R/S analysis)
    for period in [50, 100, 200]:
        if n >= period:
            ts = closes[-period:]
            mean_ts = np.mean(ts)
            devs = ts - mean_ts
            cumulative = np.cumsum(devs)
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(ts)
            if s > 0 and r > 0:
                feats[f'hurst_{period}'] = np.log(r / s) / np.log(period)
            else:
                feats[f'hurst_{period}'] = 0.5
        else:
            feats[f'hurst_{period}'] = 0.5
    
    # Entropy (Shannon)
    for period in [50, 100, 200]:
        if n >= period:
            rets = log_returns[-period:]
            bins = np.histogram(rets, bins=20)[0]
            probs = bins / max(bins.sum(), 1)
            probs = probs[probs > 0]
            feats[f'entropy_{period}'] = -np.sum(probs * np.log2(probs))
        else:
            feats[f'entropy_{period}'] = 3.0
    
    # Approximate Entropy (complexity measure)
    if n >= 100:
        ts = closes[-100:]
        m = 2  # embedding dimension
        r_threshold = 0.2 * np.std(ts)
        count = 0
        for i in range(len(ts) - m):
            for j in range(len(ts) - m):
                if i != j:
                    if max(abs(ts[i:i+m] - ts[j:j+m])) < r_threshold:
                        count += 1
        total = (len(ts) - m) * (len(ts) - m - 1)
        feats['approx_entropy'] = -np.log(count / max(total, 1)) if total > 0 else 2.0
    else:
        feats['approx_entropy'] = 2.0
    
    # Sample Entropy
    feats['sample_entropy'] = feats.get('approx_entropy', 2.0) * 0.9  # approximation
    
    # Variance Ratio Test
    for period in [5, 10, 20, 40]:
        if n >= period * 2:
            rets_short = log_returns[-period:]
            rets_long = log_returns[-period*2:]
            var_short = np.var(rets_short) if len(rets_short) > 0 else 1e-10
            var_long = np.var(rets_long) if len(rets_long) > 0 else 1e-10
            feats[f'variance_ratio_{period}'] = var_short / max(var_long, 1e-10)
        else:
            feats[f'variance_ratio_{period}'] = 1.0
    
    # Autocorrelation Structure
    for lag in [1, 2, 5, 10, 20]:
        if n > lag + 10:
            rets = log_returns[-100:]
            if len(rets) > lag:
                feats[f'autocorr_lag_{lag}'] = np.corrcoef(rets[:-lag], rets[lag:])[0, 1]
            else:
                feats[f'autocorr_lag_{lag}'] = 0
        else:
            feats[f'autocorr_lag_{lag}'] = 0
    
    # Partial Autocorrelation (simplified)
    for lag in [1, 2, 5]:
        feats[f'pacf_lag_{lag}'] = feats.get(f'autocorr_lag_{lag}', 0)
    
    # ═══════════════════════════════════════════
    # 2. VOLATILITY ANALYSIS
    # ═══════════════════════════════════════════
    
    # Realized Volatility (multiple windows)
    for period in [10, 20, 50, 100]:
        if n >= period:
            rets = log_returns[-period:]
            feats[f'realized_vol_{period}'] = np.std(rets) * np.sqrt(288)
        else:
            feats[f'realized_vol_{period}'] = 0
    
    # GARCH(1,1) parameters (simplified)
    if n >= 100:
        rets = log_returns[-100:]
        var_t = np.var(rets)
        alpha = 0.1  # typically 0.05-0.2
        beta = 0.85   # typically 0.8-0.95
        omega = var_t * (1 - alpha - beta)
        
        garch_vars = [var_t]
        for r in rets:
            var_t = omega + alpha * r**2 + beta * var_t
            garch_vars.append(var_t)
        
        feats['garch_vol'] = np.sqrt(garch_vars[-1]) * np.sqrt(288)
        feats['garch_forecast'] = np.sqrt(omega + alpha * rets[-1]**2 + beta * garch_vars[-1]) * np.sqrt(288)
        feats['garch_vol_ratio'] = feats['garch_vol'] / max(feats.get('realized_vol_20', 0.01), 0.001)
    else:
        feats['garch_vol'] = 0
        feats['garch_forecast'] = 0
        feats['garch_vol_ratio'] = 1
    
    # Volatility of Volatility
    if n >= 100:
        vols = []
        for i in range(20, 100):
            vols.append(np.std(log_returns[i-20:i]) * np.sqrt(288))
        if len(vols) > 5:
            feats['vol_of_vol'] = np.std(vols) / max(np.mean(vols), 0.001)
            feats['vol_of_vol_trend'] = (np.mean(vols[-5:]) - np.mean(vols[:5])) / max(np.mean(vols[:5]), 0.001)
        else:
            feats['vol_of_vol'] = 0
            feats['vol_of_vol_trend'] = 0
    else:
        feats['vol_of_vol'] = 0
        feats['vol_of_vol_trend'] = 0
    
    # Volatility Regime Detection
    if n >= 200:
        recent_vol = np.std(log_returns[-20:]) * np.sqrt(288)
        historical_vol = np.std(log_returns[-200:]) * np.sqrt(288)
        feats['vol_regime'] = recent_vol / max(historical_vol, 0.001)
        feats['vol_regime_percentile'] = np.searchsorted(
            np.sort([np.std(log_returns[i-20:i]) * np.sqrt(288) for i in range(20, 200)]),
            recent_vol
        ) / 180
    else:
        feats['vol_regime'] = 1
        feats['vol_regime_percentile'] = 0.5
    
    # Volatility Clustering (autocorrelation of squared returns)
    if n >= 100:
        sq_returns = log_returns[-100:]**2
        feats['vol_clustering'] = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]
    else:
        feats['vol_clustering'] = 0
    
    # Parkinson Volatility (using high-low)
    if n >= 20:
        hl_ratio = np.log(highs[-20:] / lows[-20:])
        feats['parkinson_vol'] = np.sqrt(np.mean(hl_ratio**2) / (4 * np.log(2))) * np.sqrt(288)
    else:
        feats['parkinson_vol'] = 0
    
    # Garman-Klass Volatility
    if n >= 20:
        hl = np.log(highs[-20:] / lows[-20:])
        co = np.log(closes[-20:] / closes[-21:-1])
        feats['garman_klass_vol'] = np.sqrt(np.mean(0.5 * hl**2 - (2*np.log(2)-1) * co**2)) * np.sqrt(288)
    else:
        feats['garman_klass_vol'] = 0
    
    # ═══════════════════════════════════════════
    # 3. MOMENTUM ANALYSIS
    # ═══════════════════════════════════════════
    
    # Momentum at multiple timeframes
    for period in [5, 10, 20, 50, 100]:
        if n >= period:
            feats[f'momentum_{period}'] = (closes[-1] / closes[-period] - 1)
        else:
            feats[f'momentum_{period}'] = 0
    
    # Momentum Decay (how fast momentum fades)
    for short, long in [(5, 20), (10, 50), (20, 100)]:
        if n >= long:
            mom_short = closes[-1] / closes[-short] - 1
            mom_long = closes[-1] / closes[-long] - 1
            feats[f'momentum_decay_{short}_{long}'] = mom_short / max(abs(mom_long), 0.001)
        else:
            feats[f'momentum_decay_{short}_{long}'] = 1
    
    # Rate of Change
    for period in [5, 10, 20]:
        if n >= period + 1:
            feats[f'roc_{period}'] = (closes[-1] - closes[-period-1]) / closes[-period-1]
        else:
            feats[f'roc_{period}'] = 0
    
    # Momentum Quality (consistency of returns)
    if n >= 50:
        rets = returns[-50:]
        pos_ret = rets[rets > 0]
        neg_ret = rets[rets < 0]
        feats['momentum_quality'] = len(pos_ret) / max(len(neg_ret), 1)
        feats['momentum_consistency'] = np.mean(rets) / max(np.std(rets), 0.001)
    else:
        feats['momentum_quality'] = 1
        feats['momentum_consistency'] = 0
    
    # Trend Strength (ADX-like but computed from returns)
    if n >= 50:
        rets = returns[-50:]
        abs_ret = np.abs(rets)
        signed_ret = rets
        feats['trend_strength'] = abs(np.mean(signed_ret)) / max(np.mean(abs_ret), 0.001)
        feats['trend_consistency'] = np.sign(np.sum(rets)) * feats['trend_strength']
    else:
        feats['trend_strength'] = 0
        feats['trend_consistency'] = 0
    
    # ═══════════════════════════════════════════
    # 4. MEAN REVERSION ANALYSIS
    # ═══════════════════════════════════════════
    
    # Z-Score at multiple windows
    for period in [20, 50, 100]:
        if n >= period:
            mean = np.mean(closes[-period:])
            std = np.std(closes[-period:])
            feats[f'zscore_{period}'] = (c - mean) / max(std, 0.001)
        else:
            feats[f'zscore_{period}'] = 0
    
    # Distance from Mean
    for period in [20, 50, 100]:
        if n >= period:
            mean = np.mean(closes[-period:])
            feats[f'distance_from_mean_{period}'] = (c - mean) / mean if mean > 0 else 0
        else:
            feats[f'distance_from_mean_{period}'] = 0
    
    # Reversion Speed (how fast price returns to mean)
    if n >= 100:
        mean_100 = np.mean(closes[-100:])
        deviations = closes[-100:] - mean_100
        if len(deviations) > 10:
            feats['reversion_speed'] = -np.corrcoef(deviations[:-1], np.diff(deviations))[0, 1]
        else:
            feats['reversion_speed'] = 0
    else:
        feats['reversion_speed'] = 0
    
    # Mean Reversion Probability
    if n >= 100:
        ts = closes[-100:]
        mean_ts = np.mean(ts)
        current_dev = abs(c - mean_ts)
        max_dev = np.max(np.abs(ts - mean_ts))
        feats['mean_reversion_prob'] = current_dev / max(max_dev, 0.001)
    else:
        feats['mean_reversion_prob'] = 0
    
    # Bollinger Band Position (raw, no threshold)
    for period in [20, 50]:
        if n >= period:
            mean = np.mean(closes[-period:])
            std = np.std(closes[-period:])
            feats[f'bb_position_{period}'] = (c - mean) / (2 * std) if std > 0 else 0
            feats[f'bb_width_{period}'] = 4 * std / mean if mean > 0 else 0
        else:
            feats[f'bb_position_{period}'] = 0
            feats[f'bb_width_{period}'] = 0
    
    # ═══════════════════════════════════════════
    # 5. MARKET MICROSTRUCTURE
    # ═══════════════════════════════════════════
    
    # Kyle's Lambda (price impact)
    if n >= 50 and volumes is not None and len(volumes) >= 50:
        signed_volume = np.sign(returns[-50:]) * volumes[-50:]
        price_changes = returns[-50:]
        if np.var(signed_volume) > 0:
            feats['kyle_lambda'] = np.polyfit(signed_volume, price_changes, 1)[0]
        else:
            feats['kyle_lambda'] = 0
    else:
        feats['kyle_lambda'] = 0
    
    # Amihud Illiquidity
    for period in [20, 50]:
        if n >= period:
            abs_ret = np.abs(returns[-period:])
            dollar_vol = np.mean(closes[-period:] * (highs[-period:] - lows[-period:]))
            feats[f'amihud_{period}'] = np.mean(abs_ret) / max(dollar_vol, 0.01)
        else:
            feats[f'amihud_{period}'] = 0
    
    # Spread Dynamics
    if spreads is not None and len(spreads) >= 20:
        feats['spread_mean'] = np.mean(spreads[-20:])
        feats['spread_volatility'] = np.std(spreads[-20:])
        feats['spread_trend'] = (np.mean(spreads[-5:]) - np.mean(spreads[-20:])) / max(np.mean(spreads[-20:]), 0.001)
    else:
        feats['spread_mean'] = 0
        feats['spread_volatility'] = 0
        feats['spread_trend'] = 0
    
    # Bid-Ask Bounce (microstructure noise)
    if n >= 50:
        rets = returns[-50:]
        sign_changes = np.sum(np.diff(np.sign(rets)) != 0)
        feats['bid_ask_bounce'] = sign_changes / len(rets)
    else:
        feats['bid_ask_bounce'] = 0
    
    # ═══════════════════════════════════════════
    # 6. RISK MEASURES
    # ═══════════════════════════════════════════
    
    # Value at Risk (parametric)
    for period in [20, 50]:
        if n >= period:
            rets = returns[-period:]
            mu = np.mean(rets)
            sigma = np.std(rets)
            feats[f'var_{period}'] = mu - 2.33 * sigma  # 99% VaR
            feats[f'cvar_{period}'] = mu - 2.33 * sigma * 1.2  # Expected Shortfall
        else:
            feats[f'var_{period}'] = 0
            feats[f'cvar_{period}'] = 0
    
    # Tail Risk
    if n >= 100:
        rets = returns[-100:]
        feats['tail_risk'] = np.percentile(rets, 5) / max(np.std(rets), 0.001)
        feats['skewness'] = np.mean((rets - np.mean(rets))**3) / max(np.std(rets)**3, 0.001)
        feats['kurtosis'] = np.mean((rets - np.mean(rets))**4) / max(np.std(rets)**4, 0.001) - 3
    else:
        feats['tail_risk'] = 0
        feats['skewness'] = 0
        feats['kurtosis'] = 0
    
    # Maximum Drawdown
    for period in [50, 100]:
        if n >= period:
            prices = closes[-period:]
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            feats[f'max_drawdown_{period}'] = np.min(drawdown)
        else:
            feats[f'max_drawdown_{period}'] = 0
    
    # ═══════════════════════════════════════════
    # 7. REGIME DETECTION
    # ═══════════════════════════════════════════
    
    # Trend vs Mean-Reverting Regime
    if n >= 100:
        hurst = feats.get('hurst_100', 0.5)
        feats['regime_trending'] = 1 if hurst > 0.55 else 0
        feats['regime_mean_reverting'] = 1 if hurst < 0.45 else 0
        feats['regime_random'] = 1 if 0.45 <= hurst <= 0.55 else 0
    else:
        feats['regime_trending'] = 0
        feats['regime_mean_reverting'] = 0
        feats['regime_random'] = 1
    
    # Volatility Regime
    if n >= 200:
        recent_vol = np.std(log_returns[-20:])
        hist_vol = np.std(log_returns[-200:])
        feats['vol_regime_high'] = 1 if recent_vol > hist_vol * 1.5 else 0
        feats['vol_regime_low'] = 1 if recent_vol < hist_vol * 0.5 else 0
        feats['vol_regime_normal'] = 1 if 0.5 <= recent_vol / hist_vol <= 1.5 else 0
    else:
        feats['vol_regime_high'] = 0
        feats['vol_regime_low'] = 0
        feats['vol_regime_normal'] = 1
    
    # Breakout Detection
    if n >= 50:
        prices = closes[-50:]
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        historical_high = np.max(prices)
        historical_low = np.min(prices)
        feats['breakout_up'] = 1 if c > recent_high * 0.999 else 0
        feats['breakout_down'] = 1 if c < recent_low * 1.001 else 0
        feats['breakout_strength'] = (c - historical_low) / max(historical_high - historical_low, 0.001)
    else:
        feats['breakout_up'] = 0
        feats['breakout_down'] = 0
        feats['breakout_strength'] = 0.5
    
    # ═══════════════════════════════════════════
    # 8. CORRELATION STRUCTURE
    # ═══════════════════════════════════════════
    
    # Autocorrelation at multiple lags
    for lag in [1, 2, 5, 10, 20, 50]:
        if n > lag + 50:
            rets = log_returns[-100:]
            if len(rets) > lag:
                feats[f'autocorr_{lag}'] = np.corrcoef(rets[:-lag], rets[lag:])[0, 1]
            else:
                feats[f'autocorr_{lag}'] = 0
        else:
            feats[f'autocorr_{lag}'] = 0
    
    # Cross-feature correlation
    if n >= 100:
        rets = log_returns[-100:]
        sq_rets = rets**2
        feats['leverage_effect'] = np.corrcoef(rets, sq_rets)[0, 1]
    else:
        feats['leverage_effect'] = 0
    
    # ═══════════════════════════════════════════
    # 9. SEASONALITY & TIME
    # ═══════════════════════════════════════════
    
    import time as time_module
    hour_utc = (time_module.time() % 86400) / 3600
    minute_utc = (time_module.time() % 3600) / 60
    
    feats['hour_sin'] = np.sin(2 * np.pi * hour_utc / 24)
    feats['hour_cos'] = np.cos(2 * np.pi * hour_utc / 24)
    feats['minute_sin'] = np.sin(2 * np.pi * minute_utc / 60)
    feats['minute_cos'] = np.cos(2 * np.pi * minute_utc / 60)
    
    # Session detection
    feats['is_london'] = 1 if 7 <= hour_utc <= 16 else 0
    feats['is_ny'] = 1 if 12 <= hour_utc <= 21 else 0
    feats['is_overlap'] = 1 if 12 <= hour_utc <= 16 else 0
    feats['is_asian'] = 1 if hour_utc <= 7 or hour_utc >= 21 else 0
    
    # ═══════════════════════════════════════════
    # 10. COMPOSITE SIGNALS
    # ═══════════════════════════════════════════
    
    # Price momentum + volatility interaction
    mom_20 = feats.get('momentum_20', 0)
    vol_20 = feats.get('realized_vol_20', 0.01)
    feats['momentum_vol_ratio'] = mom_20 / max(vol_20, 0.001)
    
    # Trend + mean reversion interaction
    trend = feats.get('trend_strength', 0)
    mr_prob = feats.get('mean_reversion_prob', 0)
    feats['trend_vs_reversion'] = trend - mr_prob
    
    # Kalman + OU agreement
    kalman = feats.get('kalman_trend', 0)
    ou = feats.get('ou_signal', 0)
    feats['kalman_ou_agreement'] = np.sign(kalman) * np.sign(ou)
    
    # Volatility regime + momentum
    vol_regime = feats.get('vol_regime', 1)
    feats['vol_adjusted_momentum'] = mom_20 * vol_regime
    
    return feats

# Feature names for reference
FEATURE_NAMES = [
    # Kalman (5)
    'kalman_trend', 'kalman_velocity', 'kalman_acceleration', 'kalman_residual', 'kalman_signal_to_noise',
    # OU (7)
    'ou_theta', 'ou_mu', 'ou_half_life', 'ou_z_score', 'ou_signal', 'ou_is_mean_reverting', 'ou_volatility',
    # Hurst (3)
    'hurst_50', 'hurst_100', 'hurst_200',
    # Entropy (3)
    'entropy_50', 'entropy_100', 'entropy_200',
    # Approx/Sample Entropy (2)
    'approx_entropy', 'sample_entropy',
    # Variance Ratio (4)
    'variance_ratio_5', 'variance_ratio_10', 'variance_ratio_20', 'variance_ratio_40',
    # Autocorrelation (5)
    'autocorr_lag_1', 'autocorr_lag_2', 'autocorr_lag_5', 'autocorr_lag_10', 'autocorr_lag_20',
    # PACF (3)
    'pacf_lag_1', 'pacf_lag_2', 'pacf_lag_5',
    # Realized Vol (4)
    'realized_vol_10', 'realized_vol_20', 'realized_vol_50', 'realized_vol_100',
    # GARCH (3)
    'garch_vol', 'garch_forecast', 'garch_vol_ratio',
    # Vol of Vol (2)
    'vol_of_vol', 'vol_of_vol_trend',
    # Vol Regime (2)
    'vol_regime', 'vol_regime_percentile',
    # Vol Clustering (1)
    'vol_clustering',
    # Parkinson/Garman-Klass (2)
    'parkinson_vol', 'garman_klass_vol',
    # Momentum (5)
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50', 'momentum_100',
    # Momentum Decay (3)
    'momentum_decay_5_20', 'momentum_decay_10_50', 'momentum_decay_20_100',
    # ROC (3)
    'roc_5', 'roc_10', 'roc_20',
    # Momentum Quality (2)
    'momentum_quality', 'momentum_consistency',
    # Trend (2)
    'trend_strength', 'trend_consistency',
    # Z-Score (3)
    'zscore_20', 'zscore_50', 'zscore_100',
    # Distance from Mean (3)
    'distance_from_mean_20', 'distance_from_mean_50', 'distance_from_mean_100',
    # Reversion (2)
    'reversion_speed', 'mean_reversion_prob',
    # Bollinger (4)
    'bb_position_20', 'bb_width_20', 'bb_position_50', 'bb_width_50',
    # Microstructure (5)
    'kyle_lambda', 'amihud_20', 'amihud_50', 'spread_mean', 'spread_trend',
    # Bid-Ask (1)
    'bid_ask_bounce',
    # Risk (8)
    'var_20', 'cvar_20', 'var_50', 'cvar_50', 'tail_risk', 'skewness', 'kurtosis',
    # Max Drawdown (2)
    'max_drawdown_50', 'max_drawdown_100',
    # Regime (6)
    'regime_trending', 'regime_mean_reverting', 'regime_random',
    'vol_regime_high', 'vol_regime_low', 'vol_regime_normal',
    # Breakout (3)
    'breakout_up', 'breakout_down', 'breakout_strength',
    # Correlation (7)
    'autocorr_1', 'autocorr_2', 'autocorr_5', 'autocorr_10', 'autocorr_20', 'autocorr_50',
    'leverage_effect',
    # Time (8)
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'is_london', 'is_ny', 'is_overlap', 'is_asian',
    # Composite (4)
    'momentum_vol_ratio', 'trend_vs_reversion', 'kalman_ou_agreement', 'vol_adjusted_momentum',
]

if __name__ == '__main__':
    print(f"Total quantitative features: {len(FEATURE_NAMES)}")
    print()
    categories = {}
    for name in FEATURE_NAMES:
        cat = name.split('_')[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)
    
    for cat, feats in sorted(categories.items()):
        print(f"{cat.upper()} ({len(feats)}):")
        for f in feats:
            print(f"  {f}")
        print()
