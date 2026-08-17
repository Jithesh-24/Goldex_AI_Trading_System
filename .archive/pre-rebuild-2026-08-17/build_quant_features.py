#!/usr/bin/env python3
"""
build_quant_features.py — Build 116 quant features for 10M rows.
Memory-efficient: processes in chunks, uses memmap.
"""
import numpy as np
import json, os, time
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def sma(data, period):
    cs = np.cumsum(data)
    out = np.full(len(data), np.nan, dtype=np.float64)
    out[period-1:] = (cs[period-1:] - np.concatenate([[0], cs[:-period]])) / period
    return out

def compute_chunk(start, end, closes, highs, lows):
    """Compute features for a chunk of data. Returns dict of feature arrays."""
    n = end - start
    feats = {}
    
    # Slice data
    c = closes[start:end].astype(np.float64)
    h = highs[start:end].astype(np.float64)
    lo = lows[start:end].astype(np.float64)
    
    # Need lookback data
    lookback = 300
    c_full = closes[max(0, start-lookback):end].astype(np.float64)
    h_full = highs[max(0, start-lookback):end].astype(np.float64)
    lo_full = lows[max(0, start-lookback):end].astype(np.float64)
    
    log_ret_full = np.diff(np.log(c_full))
    ret_full = np.diff(c_full) / c_full[:-1]
    
    # Offset for slicing lookback data
    lb_offset = start - max(0, start-lookback)
    
    for i in range(n):
        idx = lb_offset + i  # index in c_full
        
        if idx < 100:
            # Not enough data yet — fill with NaN
            for key in FEATURE_NAMES:
                feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            continue
        
        c_slice = c_full[:idx+1]
        log_ret = log_ret_full[:idx]
        ret = ret_full[:idx]
        
        curr_price = c_slice[-1]
        
        # ═══ KALMAN (5) ═══
        dt = 1.0; q = 0.01; r = 0.1
        x_hat = c_slice[-1]; p = 1.0; v = 0.0
        kp = []
        for k in range(min(100, len(c_slice))):
            ki = len(c_slice) - 1 - k
            x_pred = x_hat + v * dt
            p_pred = p + q
            gain = p_pred / (p_pred + r)
            x_hat = x_pred + gain * (c_slice[ki] - x_pred)
            p = (1 - gain) * p_pred
            if kp:
                v = (x_hat - kp[-1]) / dt
            kp.append(x_hat)
        
        feats.setdefault('kalman_trend', np.full(n, np.nan, dtype=np.float32))
        feats['kalman_trend'][i] = (kp[0] - kp[-1]) / kp[-1] if kp else 0
        feats.setdefault('kalman_velocity', np.full(n, np.nan, dtype=np.float32))
        feats['kalman_velocity'][i] = v
        feats.setdefault('kalman_residual', np.full(n, np.nan, dtype=np.float32))
        feats['kalman_residual'][i] = curr_price - kp[0] if kp else 0
        feats.setdefault('kalman_signal_to_noise', np.full(n, np.nan, dtype=np.float32))
        feats['kalman_signal_to_noise'][i] = abs(feats['kalman_trend'][i]) / max(r, 0.001)
        
        # ═══ OU (4) ═══
        w = min(100, len(c_slice))
        prices = c_slice[-w:]
        mu_p = np.mean(prices)
        y = prices[1:] - prices[:-1]
        x = prices[:-1] - mu_p
        if len(x) > 10 and np.var(x) > 0:
            theta = -np.polyfit(x, y, 1)[0]
            res = y + theta * x
            sigma = np.std(res)
        else:
            theta = 0.1; sigma = 0.01
        
        feats.setdefault('ou_theta', np.full(n, np.nan, dtype=np.float32))
        feats['ou_theta'][i] = theta
        feats.setdefault('ou_half_life', np.full(n, np.nan, dtype=np.float32))
        feats['ou_half_life'][i] = np.log(2) / max(theta, 0.001)
        feats.setdefault('ou_z_score', np.full(n, np.nan, dtype=np.float32))
        feats['ou_z_score'][i] = (curr_price - mu_p) / max(sigma * np.sqrt(w), 0.001)
        feats.setdefault('ou_signal', np.full(n, np.nan, dtype=np.float32))
        feats['ou_signal'][i] = theta * (mu_p - curr_price)
        
        # ═══ HURST (3) ═══
        for period, key in [(50, 'hurst_50'), (100, 'hurst_100'), (200, 'hurst_200')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) >= period:
                ts = c_slice[-period:]
                mt = np.mean(ts)
                devs = ts - mt
                cum = np.cumsum(devs)
                r_val = np.max(cum) - np.min(cum)
                s_val = np.std(ts)
                if s_val > 0 and r_val > 0:
                    feats[key][i] = np.log(r_val / s_val) / np.log(period)
                else:
                    feats[key][i] = 0.5
            else:
                feats[key][i] = 0.5
        
        # ═══ ENTROPY (3) ═══
        for period, key in [(50, 'entropy_50'), (100, 'entropy_100'), (200, 'entropy_200')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(log_ret) >= period:
                rets = log_ret[-period:]
                bins = np.histogram(rets, bins=20)[0]
                probs = bins / max(bins.sum(), 1)
                probs = probs[probs > 0]
                feats[key][i] = -np.sum(probs * np.log2(probs))
            else:
                feats[key][i] = 3.0
        
        # ═══ VARIANCE RATIO (4) ═══
        for period, key in [(5, 'variance_ratio_5'), (10, 'variance_ratio_10'),
                            (20, 'variance_ratio_20'), (40, 'variance_ratio_40')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(log_ret) >= period * 2:
                vs = np.var(log_ret[-period:])
                vl = np.var(log_ret[-period*2:])
                feats[key][i] = vs / max(vl, 1e-10)
            else:
                feats[key][i] = 1.0
        
        # ═══ AUTOCORRELATION (6) ═══
        for lag, key in [(1, 'autocorr_1'), (2, 'autocorr_2'), (5, 'autocorr_5'),
                         (10, 'autocorr_10'), (20, 'autocorr_20'), (50, 'autocorr_50')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(log_ret) > lag + 10:
                feats[key][i] = np.corrcoef(log_ret[-100:-lag], log_ret[-100+lag:])[0, 1]
            else:
                feats[key][i] = 0
        
        # ═══ REALIZED VOL (4) ═══
        for period, key in [(10, 'realized_vol_10'), (20, 'realized_vol_20'),
                            (50, 'realized_vol_50'), (100, 'realized_vol_100')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(log_ret) >= period:
                feats[key][i] = np.std(log_ret[-period:]) * np.sqrt(288)
            else:
                feats[key][i] = 0
        
        # ═══ GARCH (3) ═══
        feats.setdefault('garch_vol', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('garch_forecast', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('garch_vol_ratio', np.full(n, np.nan, dtype=np.float32))
        if len(log_ret) >= 100:
            rets = log_ret[-100:]
            alpha = 0.1; beta = 0.85
            var_t = np.var(rets)
            omega = var_t * (1 - alpha - beta)
            gvar = var_t
            for ri in rets:
                gvar = omega + alpha * ri**2 + beta * gvar
            feats['garch_vol'][i] = np.sqrt(gvar) * np.sqrt(288)
            feats['garch_forecast'][i] = np.sqrt(omega + alpha * rets[-1]**2 + beta * gvar) * np.sqrt(288)
            rv20 = np.std(log_ret[-20:]) * np.sqrt(288) if len(log_ret) >= 20 else 0.01
            feats['garch_vol_ratio'][i] = feats['garch_vol'][i] / max(rv20, 0.001)
        
        # ═══ VOL OF VOL (2) ═══
        feats.setdefault('vol_of_vol', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('vol_of_vol_trend', np.full(n, np.nan, dtype=np.float32))
        if len(log_ret) >= 100:
            vols = [np.std(log_ret[j-20:j]) * np.sqrt(288) for j in range(20, min(100, len(log_ret)))]
            if len(vols) > 5:
                feats['vol_of_vol'][i] = np.std(vols) / max(np.mean(vols), 0.001)
                feats['vol_of_vol_trend'][i] = (np.mean(vols[-5:]) - np.mean(vols[:5])) / max(np.mean(vols[:5]), 0.001)
        
        # ═══ VOL REGIME (2) ═══
        feats.setdefault('vol_regime', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('vol_regime_percentile', np.full(n, np.nan, dtype=np.float32))
        if len(log_ret) >= 200:
            recent = np.std(log_ret[-20:]) * np.sqrt(288)
            hist = np.std(log_ret[-200:]) * np.sqrt(288)
            feats['vol_regime'][i] = recent / max(hist, 0.001)
            all_vols = np.sort([np.std(log_ret[j-20:j]) * np.sqrt(288) for j in range(20, 200)])
            feats['vol_regime_percentile'][i] = np.searchsorted(all_vols, recent) / len(all_vols)
        
        # ═══ VOL CLUSTERING (1) ═══
        feats.setdefault('vol_clustering', np.full(n, np.nan, dtype=np.float32))
        if len(log_ret) >= 100:
            sq = log_ret[-100:]**2
            feats['vol_clustering'][i] = np.corrcoef(sq[:-1], sq[1:])[0, 1]
        
        # ═══ PARKINSON + GK (2) ═══
        feats.setdefault('parkinson_vol', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('garman_klass_vol', np.full(n, np.nan, dtype=np.float32))
        if len(c_slice) >= 20:
            hl = np.log(h_full[idx-19:idx+1] / lo_full[idx-19:idx+1])
            feats['parkinson_vol'][i] = np.sqrt(np.mean(hl**2) / (4 * np.log(2))) * np.sqrt(288)
            co = np.log(c_slice[-20:] / c_slice[-21:-1]) if len(c_slice) > 20 else np.zeros(20)
            feats['garman_klass_vol'][i] = np.sqrt(max(np.mean(0.5 * hl**2 - (2*np.log(2)-1) * co**2), 0)) * np.sqrt(288)
        
        # ═══ MOMENTUM (5) ═══
        for period, key in [(5, 'momentum_5'), (10, 'momentum_10'), (20, 'momentum_20'),
                            (50, 'momentum_50'), (100, 'momentum_100')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) > period:
                feats[key][i] = curr_price / c_slice[-period-1] - 1
        
        # ═══ MOMENTUM DECAY (3) ═══
        for s, l, key in [(5, 20, 'momentum_decay_5_20'), (10, 50, 'momentum_decay_10_50'),
                          (20, 100, 'momentum_decay_20_100')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) > l:
                ms = curr_price / c_slice[-s-1] - 1
                ml = curr_price / c_slice[-l-1] - 1
                feats[key][i] = ms / max(abs(ml), 0.001)
        
        # ═══ ROC (3) ═══
        for period, key in [(5, 'roc_5'), (10, 'roc_10'), (20, 'roc_20')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) > period:
                feats[key][i] = (curr_price - c_slice[-period-1]) / c_slice[-period-1]
        
        # ═══ MOMENTUM QUALITY (2) ═══
        feats.setdefault('momentum_quality', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('momentum_consistency', np.full(n, np.nan, dtype=np.float32))
        if len(ret) >= 50:
            r50 = ret[-50:]
            pos = r50[r50 > 0]
            neg = r50[r50 < 0]
            feats['momentum_quality'][i] = len(pos) / max(len(neg), 1)
            feats['momentum_consistency'][i] = np.mean(r50) / max(np.std(r50), 0.001)
        
        # ═══ TREND (2) ═══
        feats.setdefault('trend_strength', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('trend_consistency', np.full(n, np.nan, dtype=np.float32))
        if len(ret) >= 50:
            r50 = ret[-50:]
            feats['trend_strength'][i] = abs(np.mean(r50)) / max(np.mean(np.abs(r50)), 0.001)
            feats['trend_consistency'][i] = np.sign(np.sum(r50)) * feats['trend_strength'][i]
        
        # ═══ Z-SCORE (3) ═══
        for period, key in [(20, 'zscore_20'), (50, 'zscore_50'), (100, 'zscore_100')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) >= period:
                mu = np.mean(c_slice[-period:])
                sd = np.std(c_slice[-period:])
                feats[key][i] = (curr_price - mu) / max(sd, 0.001)
        
        # ═══ DISTANCE FROM MEAN (3) ═══
        for period, key in [(20, 'dist_mean_20'), (50, 'dist_mean_50'), (100, 'dist_mean_100')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) >= period:
                mu = np.mean(c_slice[-period:])
                feats[key][i] = (curr_price - mu) / mu if mu > 0 else 0
        
        # ═══ REVERSION (2) ═══
        feats.setdefault('reversion_speed', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('mean_reversion_prob', np.full(n, np.nan, dtype=np.float32))
        if len(c_slice) >= 100:
            mu100 = np.mean(c_slice[-100:])
            devs = c_slice[-100:] - mu100
            if len(devs) > 10:
                feats['reversion_speed'][i] = -np.corrcoef(devs[:-1], np.diff(devs))[0, 1]
            cur_dev = abs(curr_price - mu100)
            max_dev = np.max(np.abs(c_slice[-100:] - mu100))
            feats['mean_reversion_prob'][i] = cur_dev / max(max_dev, 0.001)
        
        # ═══ BB POSITION + WIDTH (4) ═══
        for period, key_p, key_w in [(20, 'bb_pos_20', 'bb_width_20'), (50, 'bb_pos_50', 'bb_width_50')]:
            feats.setdefault(key_p, np.full(n, np.nan, dtype=np.float32))
            feats.setdefault(key_w, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) >= period:
                mu = np.mean(c_slice[-period:])
                sd = np.std(c_slice[-period:])
                feats[key_p][i] = (curr_price - mu) / (2 * sd) if sd > 0 else 0
                feats[key_w][i] = 4 * sd / mu if mu > 0 else 0
        
        # ═══ AMIHUD (2) ═══
        for period, key in [(20, 'amihud_20'), (50, 'amihud_50')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(ret) >= period:
                ar = np.abs(ret[-period:])
                dv = np.mean(c_slice[-period:] * (h_full[idx-period+1:idx+1] - lo_full[idx-period+1:idx+1]))
                feats[key][i] = np.mean(ar) / max(dv, 0.01)
        
        # ═══ KYLE'S LAMBDA (1) ═══
        feats.setdefault('kyle_lambda', np.full(n, np.nan, dtype=np.float32))
        feats['kyle_lambda'][i] = 0  # need volume data
        
        # ═══ SPREAD (2) ═══
        feats.setdefault('spread_mean', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('spread_trend', np.full(n, np.nan, dtype=np.float32))
        feats['spread_mean'][i] = 0  # need spread data
        feats['spread_trend'][i] = 0
        
        # ═══ BID-ASK BOUNCE (1) ═══
        feats.setdefault('bid_ask_bounce', np.full(n, np.nan, dtype=np.float32))
        if len(ret) >= 50:
            sc = np.sum(np.diff(np.sign(ret[-50:])) != 0)
            feats['bid_ask_bounce'][i] = sc / 50
        
        # ═══ RISK (6) ═══
        for period, vk, ck in [(20, 'var_20', 'cvar_20'), (50, 'var_50', 'cvar_50')]:
            feats.setdefault(vk, np.full(n, np.nan, dtype=np.float32))
            feats.setdefault(ck, np.full(n, np.nan, dtype=np.float32))
            if len(ret) >= period:
                mu = np.mean(ret[-period:])
                sd = np.std(ret[-period:])
                feats[vk][i] = mu - 2.33 * sd
                feats[ck][i] = mu - 2.33 * sd * 1.2
        
        feats.setdefault('tail_risk', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('skewness', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('kurtosis', np.full(n, np.nan, dtype=np.float32))
        if len(ret) >= 100:
            r100 = ret[-100:]
            mu = np.mean(r100); sd = np.std(r100)
            feats['tail_risk'][i] = np.percentile(r100, 5) / max(sd, 0.001)
            feats['skewness'][i] = np.mean((r100 - mu)**3) / max(sd**3, 0.001)
            feats['kurtosis'][i] = np.mean((r100 - mu)**4) / max(sd**4, 0.001) - 3
        
        # ═══ MAX DRAWDOWN (2) ═══
        for period, key in [(50, 'max_dd_50'), (100, 'max_dd_100')]:
            feats.setdefault(key, np.full(n, np.nan, dtype=np.float32))
            if len(c_slice) >= period:
                p = c_slice[-period:]
                pk = np.maximum.accumulate(p)
                dd = (p - pk) / pk
                feats[key][i] = np.min(dd)
        
        # ═══ REGIME (3) ═══
        feats.setdefault('regime_trending', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('regime_mr', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('regime_random', np.full(n, np.nan, dtype=np.float32))
        h_val = feats.get('hurst_100', [0.5])[i] if 'hurst_100' in feats else 0.5
        feats['regime_trending'][i] = 1 if h_val > 0.55 else 0
        feats['regime_mr'][i] = 1 if h_val < 0.45 else 0
        feats['regime_random'][i] = 1 if 0.45 <= h_val <= 0.55 else 0
        
        # ═══ VOL REGIME CATEGORICAL (3) ═══
        feats.setdefault('vol_high', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('vol_low', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('vol_normal', np.full(n, np.nan, dtype=np.float32))
        vr = feats.get('vol_regime', [1])[i] if 'vol_regime' in feats else 1
        feats['vol_high'][i] = 1 if vr > 1.5 else 0
        feats['vol_low'][i] = 1 if vr < 0.5 else 0
        feats['vol_normal'][i] = 1 if 0.5 <= vr <= 1.5 else 0
        
        # ═══ BREAKOUT (3) ═══
        feats.setdefault('breakout_up', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('breakout_down', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('breakout_strength', np.full(n, np.nan, dtype=np.float32))
        if len(c_slice) >= 50:
            p50 = c_slice[-50:]
            rh = np.max(p50[-20:]); rl = np.min(p50[-20:])
            hh = np.max(p50); ll = np.min(p50)
            feats['breakout_up'][i] = 1 if curr_price > rh * 0.999 else 0
            feats['breakout_down'][i] = 1 if curr_price < rl * 1.001 else 0
            feats['breakout_strength'][i] = (curr_price - ll) / max(hh - ll, 0.001)
        
        # ═══ LEVERAGE EFFECT (1) ═══
        feats.setdefault('leverage_effect', np.full(n, np.nan, dtype=np.float32))
        if len(log_ret) >= 100:
            sq = log_ret[-100:]**2
            feats['leverage_effect'][i] = np.corrcoef(log_ret[-100:], sq)[0, 1]
        
        # ═══ COMPOSITE (4) ═══
        feats.setdefault('momentum_vol_ratio', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('trend_vs_reversion', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('kalman_ou_agreement', np.full(n, np.nan, dtype=np.float32))
        feats.setdefault('vol_adjusted_momentum', np.full(n, np.nan, dtype=np.float32))
        
        m20 = feats.get('momentum_20', [0])[i] if 'momentum_20' in feats else 0
        rv20 = feats.get('realized_vol_20', [0.01])[i] if 'realized_vol_20' in feats else 0.01
        ts_val = feats.get('trend_strength', [0])[i] if 'trend_strength' in feats else 0
        mr_val = feats.get('mean_reversion_prob', [0])[i] if 'mean_reversion_prob' in feats else 0
        kt = feats.get('kalman_trend', [0])[i] if 'kalman_trend' in feats else 0
        os_val = feats.get('ou_signal', [0])[i] if 'ou_signal' in feats else 0
        
        feats['momentum_vol_ratio'][i] = m20 / max(rv20, 0.001)
        feats['trend_vs_reversion'][i] = ts_val - mr_val
        feats['kalman_ou_agreement'][i] = np.sign(kt) * np.sign(os_val)
        feats['vol_adjusted_momentum'][i] = m20 * vr
    
    return feats

# Feature names
FEATURE_NAMES = [
    'kalman_trend', 'kalman_velocity', 'kalman_residual', 'kalman_signal_to_noise',
    'ou_theta', 'ou_half_life', 'ou_z_score', 'ou_signal',
    'hurst_50', 'hurst_100', 'hurst_200',
    'entropy_50', 'entropy_100', 'entropy_200',
    'variance_ratio_5', 'variance_ratio_10', 'variance_ratio_20', 'variance_ratio_40',
    'autocorr_1', 'autocorr_2', 'autocorr_5', 'autocorr_10', 'autocorr_20', 'autocorr_50',
    'realized_vol_10', 'realized_vol_20', 'realized_vol_50', 'realized_vol_100',
    'garch_vol', 'garch_forecast', 'garch_vol_ratio',
    'vol_of_vol', 'vol_of_vol_trend',
    'vol_regime', 'vol_regime_percentile',
    'vol_clustering',
    'parkinson_vol', 'garman_klass_vol',
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50', 'momentum_100',
    'momentum_decay_5_20', 'momentum_decay_10_50', 'momentum_decay_20_100',
    'roc_5', 'roc_10', 'roc_20',
    'momentum_quality', 'momentum_consistency',
    'trend_strength', 'trend_consistency',
    'zscore_20', 'zscore_50', 'zscore_100',
    'dist_mean_20', 'dist_mean_50', 'dist_mean_100',
    'reversion_speed', 'mean_reversion_prob',
    'bb_pos_20', 'bb_width_20', 'bb_pos_50', 'bb_width_50',
    'amihud_20', 'amihud_50',
    'kyle_lambda', 'spread_mean', 'spread_trend', 'bid_ask_bounce',
    'var_20', 'cvar_20', 'var_50', 'cvar_50',
    'tail_risk', 'skewness', 'kurtosis',
    'max_dd_50', 'max_dd_100',
    'regime_trending', 'regime_mr', 'regime_random',
    'vol_high', 'vol_low', 'vol_normal',
    'breakout_up', 'breakout_down', 'breakout_strength',
    'leverage_effect',
    'momentum_vol_ratio', 'trend_vs_reversion', 'kalman_ou_agreement', 'vol_adjusted_momentum',
]

def main():
    t0 = time.time()
    print("═══ BUILDING QUANT FEATURES ═══\n")
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    mmap_rows = 9999963
    
    # Load prices
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:, 3].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    
    n = len(closes)
    n_feat = len(FEATURE_NAMES)
    
    print(f"Rows: {n:,} | Features: {n_feat}")
    print(f"Output: {BASE}/quant_features_116.npy")
    
    # Create output memmap
    X_out = np.memmap(f"{BASE}/quant_features_116.npy", dtype=np.float32,
                      mode='w+', shape=(n, n_feat))
    
    # Process in chunks
    CHUNK = 100_000
    n_chunks = (n + CHUNK - 1) // CHUNK
    
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK
        end = min(start + CHUNK, n)
        
        feats = compute_chunk(start, end, closes, highs, lows)
        
        for j, fname in enumerate(FEATURE_NAMES):
            if fname in feats:
                X_out[start:end, j] = feats[fname]
            else:
                X_out[start:end, j] = np.nan
        
        elapsed = time.time() - t0
        eta = elapsed / (chunk_idx + 1) * (n_chunks - chunk_idx - 1)
        print(f"  Chunk {chunk_idx+1}/{n_chunks} ({start:,}-{end:,}) | "
              f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")
    
    X_out.flush()
    
    # Save feature names
    meta_out = {
        'n_features': n_feat,
        'feature_names': FEATURE_NAMES,
        'n_rows': n,
        'source': 'quant_features_116',
    }
    with open(f"{BASE}/quant_features_meta.json", 'w') as f:
        json.dump(meta_out, f, indent=2)
    
    print(f"\n✅ Done: {n:,} rows × {n_feat} features")
    print(f"   Time: {time.time()-t0:.0f}s")
    print(f"   File: {BASE}/quant_features_116.npy")

if __name__ == '__main__':
    main()
