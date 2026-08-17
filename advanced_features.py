#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_features.py — FAST vectorized gold quantitative features

Uses numpy vectorized operations for speed on large datasets.
All features use only past data (no lookahead bias).

Features:
1. hurst — mean reversion vs trending (rescaled range)
2. amihud — price impact per unit volume
3. kyle_lambda — informed trading coefficient
4. vpin — informed trading probability
5. return_autocorr — return persistence
6. variance_ratio — random walk test
7. momentum_half_life — momentum decay speed
8. entropy — market randomness
9. support_dist — distance to recent support
10. resistance_dist — distance to recent resistance
"""

import numpy as np
import pandas as pd
from numba import njit, prange


@njit(cache=True)
def _rolling_hurst(prices, max_lag=15):
    """Vectorized Hurst via rescaled range (simplified)."""
    n = len(prices)
    out = np.full(n, 0.5)
    for i in range(max_lag + 2, n):
        best_h = 0.5
        rs_sum = 0.0
        cnt = 0
        for lag in range(2, min(max_lag + 1, (i // 4) + 1)):
            seg_len = lag
            n_segs = (i + 1) // seg_len
            if n_segs < 1:
                continue
            for s in range(min(n_segs, 5)):  # max 5 segments for speed
                start = i - (s + 1) * seg_len + 1
                if start < 0:
                    break
                mean_val = 0.0
                for j in range(start, start + seg_len):
                    mean_val += prices[j]
                mean_val /= seg_len

                cumdev = 0.0
                r_max = 0.0
                r_min = 0.0
                var_val = 0.0
                for j in range(start, start + seg_len):
                    cumdev += prices[j] - mean_val
                    if cumdev > r_max:
                        r_max = cumdev
                    if cumdev < r_min:
                        r_min = cumdev
                    dev = prices[j] - mean_val
                    var_val += dev * dev

                R = r_max - r_min
                S2 = var_val / seg_len
                if S2 > 1e-20:
                    rs_sum += np.log(R / np.sqrt(S2))
                    cnt += 1

        if cnt > 1:
            x_mean = 0.0
            for lag in range(2, min(max_lag + 1, (i // 4) + 1)):
                x_mean += np.log(lag)
            x_mean /= cnt
            y_sum = 0.0
            x2_sum = 0.0
            idx = 0
            for lag in range(2, min(max_lag + 1, (i // 4) + 1)):
                y = rs_sum / cnt  # approximate
                x = np.log(lag) - x_mean
                y_sum += x * y
                x2_sum += x * x
                idx += 1
            if x2_sum > 1e-20:
                best_h = max(0.0, min(1.0, y_sum / x2_sum))
        out[i] = best_h
    return out


@njit(cache=True)
def _rolling_amihud(returns, volumes, window=20):
    """Rolling Amihud illiquidity."""
    n = len(returns)
    out = np.zeros(n)
    for i in range(window, n):
        s = 0.0
        for j in range(i - window, i):
            v = volumes[j]
            if v > 1e-10:
                s += abs(returns[j]) / v
        out[i] = s / window
    return out


@njit(cache=True)
def _rolling_kyle(returns, volumes, window=20):
    """Rolling Kyle's lambda (price impact)."""
    n = len(returns)
    out = np.zeros(n)
    for i in range(window, n):
        # signed volume
        sv_sum = 0.0
        sv2_sum = 0.0
        rsv_sum = 0.0
        r_sum = 0.0
        for j in range(i - window, i):
            sv = volumes[j] if returns[j] >= 0 else -volumes[j]
            sv_sum += sv
            sv2_sum += sv * sv
            rsv_sum += returns[j] * sv
            r_sum += returns[j]
        sv_mean = sv_sum / window
        r_mean = r_sum / window
        cov = rsv_sum / window - r_mean * sv_mean
        var = sv2_sum / window - sv_mean * sv_mean
        if var > 1e-20:
            out[i] = cov / var
    return out


@njit(cache=True)
def _rolling_vpin(volumes, returns, n_buckets=10, window=100):
    """Rolling VPIN (simplified)."""
    n = len(volumes)
    out = np.zeros(n)
    for i in range(window, n):
        bucket_size = window // n_buckets
        total_vpin = 0.0
        for b in range(n_buckets):
            start = i - window + b * bucket_size
            end = start + bucket_size
            buy_vol = 0.0
            sell_vol = 0.0
            for j in range(start, end):
                if returns[j] >= 0:
                    buy_vol += volumes[j]
                else:
                    sell_vol += volumes[j]
            total = buy_vol + sell_vol
            if total > 1e-10:
                total_vpin += abs(buy_vol - sell_vol) / total
        out[i] = total_vpin / n_buckets
    return out


@njit(cache=True)
def _rolling_autocorr(returns, lag=5, window=20):
    """Rolling autocorrelation at given lag."""
    n = len(returns)
    out = np.zeros(n)
    for i in range(window + lag, n):
        x_mean = 0.0
        y_mean = 0.0
        for j in range(window):
            x_mean += returns[i - window - lag + j]
            y_mean += returns[i - window + j]
        x_mean /= window
        y_mean /= window
        cov = 0.0
        var_x = 0.0
        var_y = 0.0
        for j in range(window):
            x = returns[i - window - lag + j] - x_mean
            y = returns[i - window + j] - y_mean
            cov += x * y
            var_x += x * x
            var_y += y * y
        denom = np.sqrt(var_x * var_y)
        if denom > 1e-20:
            out[i] = cov / denom
    return out


@njit(cache=True)
def _rolling_var_ratio(returns, period=10, window=50):
    """Rolling variance ratio test."""
    n = len(returns)
    out = np.ones(n)
    for i in range(window, n):
        # variance of 1-period returns
        var1 = 0.0
        for j in range(i - window, i):
            var1 += returns[j] * returns[j]
        var1 /= window
        if var1 < 1e-20:
            continue
        # variance of period returns
        n_periods = window // period
        if n_periods < 2:
            continue
        mean_p = 0.0
        for k in range(n_periods):
            s = 0.0
            for j in range(period):
                s += returns[i - window + k * period + j]
            mean_p += s
        mean_p /= n_periods
        var_p = 0.0
        for k in range(n_periods):
            s = 0.0
            for j in range(period):
                s += returns[i - window + k * period + j]
            var_p += (s - mean_p) * (s - mean_p)
        var_p /= n_periods
        out[i] = var_p / (period * var1)
    return out


@njit(cache=True)
def _rolling_entropy(returns, n_bins=8, window=50):
    """Rolling Shannon entropy."""
    n = len(returns)
    out = np.full(n, 2.0)  # max entropy for 8 bins = 3.0
    for i in range(window, n):
        # find min/max in window
        rmin = returns[i - window]
        rmax = returns[i - window]
        for j in range(i - window + 1, i):
            if returns[j] < rmin:
                rmin = returns[j]
            if returns[j] > rmax:
                rmax = returns[j]
        rng = rmax - rmin
        if rng < 1e-20:
            out[i] = 0.0
            continue
        # count bins
        counts = np.zeros(n_bins)
        for j in range(i - window, i):
            b = int((returns[j] - rmin) / rng * (n_bins - 0.01))
            if b < 0:
                b = 0
            if b >= n_bins:
                b = n_bins - 1
            counts[b] += 1.0
        # entropy
        ent = 0.0
        for b in range(n_bins):
            p = counts[b] / window
            if p > 1e-10:
                ent -= p * np.log2(p)
        out[i] = ent
    return out


@njit(cache=True)
def _rolling_momentum_half_life(returns, window=20):
    """Approximate momentum half-life from first zero-crossing of ACF."""
    n = len(returns)
    out = np.full(n, window / 2.0)
    for i in range(window, n):
        # compute ACF at lags 1..window/2
        x_mean = 0.0
        for j in range(i - window, i):
            x_mean += returns[j]
        x_mean /= window
        var_x = 0.0
        for j in range(i - window, i):
            d = returns[j] - x_mean
            var_x += d * d
        if var_x < 1e-20:
            continue
        prev_ac = 0.0
        half_life = window / 2.0
        max_lag = min(window // 2, 15)
        for lag in range(1, max_lag + 1):
            cov = 0.0
            cnt = 0
            for j in range(i - window + lag, i):
                cov += (returns[j - lag] - x_mean) * (returns[j] - x_mean)
                cnt += 1
            if cnt > 0:
                ac = cov / (var_x * cnt / window)
            else:
                ac = 0.0
            if lag == 1 and abs(ac) < 0.01:
                half_life = 1.0
                break
            if prev_ac > 0 and ac < prev_ac * 0.5:
                half_life = float(lag)
                break
            prev_ac = abs(ac)
        out[i] = half_life
    return out


def compute_advanced_features(df):
    """Compute all advanced features for a DataFrame."""
    result = df.copy()

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df.get('volume', pd.Series(np.ones(len(df)))).values.astype(np.float64)

    returns = np.diff(close, prepend=close[0]) / np.maximum(np.abs(np.roll(close, 1)), 1e-10)
    returns[0] = 0.0

    print("  Computing Hurst...", end=" ", flush=True)
    result['hurst'] = _rolling_hurst(close)

    print("Amihud...", end=" ", flush=True)
    result['amihud'] = _rolling_amihud(returns, volume)

    print("Kyle...", end=" ", flush=True)
    result['kyle_lambda'] = _rolling_kyle(returns, volume)

    print("VPIN...", end=" ", flush=True)
    result['vpin'] = _rolling_vpin(volume, returns)

    print("Autocorr...", end=" ", flush=True)
    result['return_autocorr'] = _rolling_autocorr(returns)

    print("VarRatio...", end=" ", flush=True)
    result['variance_ratio'] = _rolling_var_ratio(returns)

    print("HalfLife...", end=" ", flush=True)
    result['momentum_half_life'] = _rolling_momentum_half_life(returns)

    print("Entropy...", end=" ", flush=True)
    result['entropy'] = _rolling_entropy(returns)

    # Support/Resistance distance (fast)
    print("S/R...", end=" ", flush=True)
    sr_window = 50
    n = len(df)
    sup_dist = np.zeros(n)
    res_dist = np.zeros(n)
    for i in range(sr_window, n):
        recent_high = np.max(high[i - sr_window:i + 1])
        recent_low = np.min(low[i - sr_window:i + 1])
        rng = recent_high - recent_low
        if rng > 1e-10:
            sup_dist[i] = (close[i] - recent_low) / rng
            res_dist[i] = (recent_high - close[i]) / rng
        else:
            sup_dist[i] = 0.5
            res_dist[i] = 0.5
    result['support_dist'] = sup_dist
    result['resistance_dist'] = res_dist

    print("DONE")
    return result


ADVANCED_FEATURE_NAMES = [
    'hurst', 'amihud', 'kyle_lambda', 'vpin',
    'return_autocorr', 'variance_ratio', 'momentum_half_life', 'entropy',
    'support_dist', 'resistance_dist'
]

if __name__ == '__main__':
    np.random.seed(42)
    n = 2000
    price = 2000 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        'open': price - np.random.rand(n) * 0.3,
        'high': price + np.random.rand(n) * 0.5,
        'low': price - np.random.rand(n) * 0.5,
        'close': price,
        'volume': np.random.randint(100, 1000, n).astype(float),
        'session': np.random.choice(['LONDON', 'NY', 'ASIAN'], n),
        'time': pd.date_range('2024-01-01', periods=n, freq='5min')
    })
    result = compute_advanced_features(df)
    for col in ADVANCED_FEATURE_NAMES:
        print(f"  {col}: mean={result[col].mean():.4f}, std={result[col].std():.4f}")
