#!/usr/bin/env python3
"""
full_pipeline.py — COMPLETE AI PIPELINE: features → train → validate → save.
Runs in one shot. No per-row Python loops for slow functions.
Uses windowed approximations for Hurst/Entropy/Autocorr.
"""
import numpy as np
import lightgbm as lgb
import json, os, time, sys
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

# ═══ FAST MATH FUNCTIONS ═══

def fast_rolling_mean(x, w):
    cs = np.cumsum(x); out = np.full_like(x, np.nan)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out

def fast_rolling_std(x, w):
    m = fast_rolling_mean(x, w); m2 = fast_rolling_mean(x**2, w)
    return np.sqrt(np.maximum(m2 - m**2, 0))

def fast_rolling_max(x, w):
    n = len(x); out = np.full(n, np.nan)
    for i in range(w-1, n, max(1, w//4)):
        out[i] = np.max(x[max(0,i-w+1):i+1])
    # forward fill
    idx = np.where(~np.isnan(out))[0]
    if len(idx) > 0:
        out[:idx[0]] = out[idx[0]]
        for k in range(len(idx)-1):
            out[idx[k]:idx[k+1]] = out[idx[k]]
        out[idx[-1]:] = out[idx[-1]]
    return out

def fast_rolling_min(x, w):
    n = len(x); out = np.full(n, np.nan)
    for i in range(w-1, n, max(1, w//4)):
        out[i] = np.min(x[max(0,i-w+1):i+1])
    idx = np.where(~np.isnan(out))[0]
    if len(idx) > 0:
        out[:idx[0]] = out[idx[0]]
        for k in range(len(idx)-1):
            out[idx[k]:idx[k+1]] = out[idx[k]]
        out[idx[-1]:] = out[idx[-1]]
    return out

def fast_autocorr(x, lag, window=200):
    """Vectorized rolling autocorr using correlation of x and shifted x."""
    n = len(x)
    out = np.full(n, np.nan)
    # Process in blocks
    step = window
    for i in range(lag + window, n, step):
        end = min(i + step, n)
        seg = x[i-lag-window:end]
        if len(seg) > lag + 10:
            for j in range(i, min(end, n)):
                s = x[max(0,j-window):j]
                if len(s) > lag + 2:
                    out[j] = np.corrcoef(s[:-lag], s[lag:])[0, 1]
    return out

def fast_autocorr_vectorized(x, lag, window=500):
    """Approximate rolling autocorr using fixed windows sampled every 100 bars."""
    n = len(x)
    out = np.full(n, np.nan)
    sample_step = 100
    for i in range(lag + window, n, sample_step):
        s = x[i-window:i]
        if len(s) > lag + 5:
            out[i] = np.corrcoef(s[:-lag], s[lag:])[0, 1]
    # Interpolate gaps
    valid = ~np.isnan(out)
    if np.sum(valid) > 2:
        xp = np.where(valid)[0]
        fp = out[valid]
        all_x = np.arange(n)
        out = np.interp(all_x, xp, fp)
    return out

def fast_hurst(x, period, sample_step=500):
    """Approximate Hurst: sample every sample_step bars."""
    n = len(x)
    out = np.full(n, 0.5)
    for i in range(period, n, sample_step):
        ts = x[i-period:i]
        mt = np.mean(ts)
        devs = ts - mt
        cum = np.cumsum(devs)
        r_val = np.max(cum) - np.min(cum)
        s_val = np.std(ts)
        if s_val > 0 and r_val > 0:
            out[i] = np.log(r_val / s_val) / np.log(period)
    # Interpolate
    valid = ~np.isnan(out) & (out != 0.5)
    if np.sum(valid) > 2:
        xp = np.where(valid)[0]
        fp = out[valid]
        out = np.interp(np.arange(n), xp, fp)
    return out

def fast_entropy(x, period, sample_step=500):
    """Approximate entropy: sample every sample_step bars."""
    n = len(x)
    out = np.full(n, 3.0)
    for i in range(period, n, sample_step):
        r = x[i-period:i]
        bins = np.histogram(r, bins=20)[0]
        probs = bins / max(bins.sum(), 1)
        probs = probs[probs > 0]
        out[i] = -np.sum(probs * np.log2(probs))
    # Interpolate
    valid = ~np.isnan(out) & (out != 3.0)
    if np.sum(valid) > 2:
        xp = np.where(valid)[0]
        fp = out[valid]
        out = np.interp(np.arange(n), xp, fp)
    return out

def ema(x, p):
    alpha = 2.0 / (p + 1)
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out

# ═══ MAIN PIPELINE ═══

def main():
    t0 = time.time()
    print("═══ FULL AI PIPELINE: FEATURES → TRAIN → VALIDATE ═══\n")
    
    prices = np.load(f"{BASE}/prices_tail.npy")
    opens = prices[:, 0].astype(np.float64)
    highs = prices[:, 1].astype(np.float64)
    lows = prices[:, 2].astype(np.float64)
    closes = prices[:, 3].astype(np.float64)
    n = len(closes)
    print(f"Data: {n:,} bars")
    
    log_ret = np.diff(np.log(closes))
    log_ret = np.concatenate([[0], log_ret])
    ret = np.diff(closes) / closes[:-1]
    ret = np.concatenate([[0], ret])
    
    # Labels
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    y_full = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(meta['n_rows'],))
    y = y_full[:n].astype(np.int8)
    
    feats = {}
    def add(name, vals):
        feats[name] = vals.astype(np.float32)
    
    # ═══ MOMENTUM ═══
    for p in [5,10,20,50,100]:
        add(f'mom_{p}', closes / np.roll(closes, p) - 1)
    for p in [5,10,20]:
        add(f'roc_{p}', closes / np.roll(closes, p+1) - 1)
    add('mom_decay_5_20', feats['mom_5'] / np.maximum(np.abs(feats['mom_20']), 0.001))
    add('mom_decay_10_50', feats['mom_10'] / np.maximum(np.abs(feats['mom_50']), 0.001))
    add('mom_decay_20_100', feats['mom_20'] / np.maximum(np.abs(feats['mom_100']), 0.001))
    print(f"  Momentum ({time.time()-t0:.0f}s)")
    
    # ═══ VOLATILITY ═══
    for p in [10,20,50,100]:
        add(f'rvol_{p}', fast_rolling_std(log_ret, p) * np.sqrt(288))
    rv20 = feats['rvol_20']
    rv100 = feats['rvol_100']
    
    # GARCH
    alpha_g, beta_g = 0.1, 0.85
    var0 = np.var(log_ret[1:100]) if n > 100 else 0.001
    omega = var0 * (1 - alpha_g - beta_g)
    gv = np.zeros(n); gv[0] = var0
    for i in range(1, n):
        gv[i] = omega + alpha_g * log_ret[i]**2 + beta_g * gv[i-1]
    add('garch_vol', np.sqrt(gv) * np.sqrt(288))
    add('garch_ratio', np.sqrt(gv) * np.sqrt(288) / np.maximum(rv20, 0.001))
    add('garch_forecast', np.roll(np.sqrt(gv) * np.sqrt(288), -1))
    
    # Vol of vol
    add('vol_of_vol', fast_rolling_std(rv20, 50) / np.maximum(fast_rolling_mean(rv20, 50), 0.001))
    add('vol_of_vol_trend', (fast_rolling_mean(rv20, 5) - fast_rolling_mean(rv20, 50)) / np.maximum(fast_rolling_mean(rv20, 50), 0.001))
    
    # Vol regime
    add('vol_regime', rv20 / np.maximum(rv100, 0.001))
    add('vol_regime_pctile', fast_rolling_mean((rv20 > rv100).astype(float), 200))
    add('vol_clustering', fast_autocorr_vectorized(log_ret**2, 1))
    
    # Parkinson + GK
    hl = np.log(np.maximum(highs, 1) / np.maximum(lows, 1))
    co = log_ret
    add('parkinson_vol', np.sqrt(fast_rolling_mean(hl**2, 20) / (4*np.log(2))) * np.sqrt(288))
    add('gk_vol', np.sqrt(np.maximum(fast_rolling_mean(0.5*hl**2 - (2*np.log(2)-1)*co**2, 20), 0)) * np.sqrt(288))
    
    # Vol categorical
    add('vol_high', (feats['vol_regime'] > 1.5).astype(float))
    add('vol_low', (feats['vol_regime'] < 0.5).astype(float))
    add('vol_normal', ((feats['vol_regime'] >= 0.5) & (feats['vol_regime'] <= 1.5)).astype(float))
    print(f"  Volatility ({time.time()-t0:.0f}s)")
    
    # ═══ Z-SCORE + BB ═══
    for p in [20,50,100]:
        m = fast_rolling_mean(closes, p); s = fast_rolling_std(closes, p)
        add(f'zscore_{p}', (closes - m) / np.maximum(s, 0.001))
        add(f'dist_mean_{p}', (closes - m) / np.maximum(m, 0.001))
    for p in [20,50]:
        m = fast_rolling_mean(closes, p); s = fast_rolling_std(closes, p)
        add(f'bb_pos_{p}', (closes - m) / (2*np.maximum(s, 0.001)))
        add(f'bb_width_{p}', 4*s / np.maximum(m, 0.001))
    print(f"  Z-score/BB ({time.time()-t0:.0f}s)")
    
    # ═══ HURST (sampled) ═══
    for p in [50,100,200]:
        add(f'hurst_{p}', fast_hurst(closes, p, sample_step=200))
    print(f"  Hurst ({time.time()-t0:.0f}s)")
    
    # ═══ ENTROPY (sampled) ═══
    for p in [50,100,200]:
        add(f'entropy_{p}', fast_entropy(log_ret, p, sample_step=200))
    print(f"  Entropy ({time.time()-t0:.0f}s)")
    
    # ═══ VARIANCE RATIO ═══
    for p in [5,10,20,40]:
        vs = fast_rolling_std(log_ret, p)**2
        vs2 = fast_rolling_std(log_ret, p*2)**2
        add(f'var_ratio_{p}', vs / np.maximum(vs2, 1e-10))
    
    # ═══ AUTOCORRELATION (sampled) ═══
    for lag in [1,2,5,10,20,50]:
        add(f'autocorr_{lag}', fast_autocorr_vectorized(log_ret, lag))
    print(f"  Autocorr ({time.time()-t0:.0f}s)")
    
    # ═══ OU ═══
    ou_dev = closes - fast_rolling_mean(closes, 100)
    ou_ac = fast_autocorr_vectorized(ou_dev, 1)
    add('ou_theta', -ou_ac)
    add('ou_half_life', np.log(2) / np.maximum(-ou_ac, 0.001))
    add('ou_z_score', ou_dev / np.maximum(fast_rolling_std(closes, 100), 0.001))
    add('ou_signal', -ou_ac * (fast_rolling_mean(closes, 100) - closes))
    
    # ═══ KALMAN ═══
    kf = ema(closes, 5); ks = ema(closes, 50)
    add('kalman_trend', (kf - ks) / np.maximum(ks, 0.001))
    add('kalman_velocity', kf - np.roll(kf, 1))
    add('kalman_residual', closes - kf)
    add('kalman_sn', np.abs(kf - ks) / np.maximum(fast_rolling_std(closes, 50), 0.001))
    print(f"  OU/Kalman ({time.time()-t0:.0f}s)")
    
    # ═══ MOMENTUM QUALITY ═══
    pos_r = fast_rolling_mean((ret > 0).astype(float), 50)
    add('mom_quality', pos_r / np.maximum(1 - pos_r, 0.01))
    add('mom_consistency', fast_rolling_mean(ret, 50) / np.maximum(fast_rolling_std(ret, 50), 0.001))
    add('trend_strength', np.abs(fast_rolling_mean(ret, 50)) / np.maximum(fast_rolling_mean(np.abs(ret), 50), 0.001))
    add('trend_consistency', np.sign(fast_rolling_mean(ret, 50)) * feats['trend_strength'])
    
    # ═══ REVERSION ═══
    add('rev_speed', -fast_autocorr_vectorized(ou_dev, 1))
    add('mr_prob', np.abs(ou_dev) / np.maximum(fast_rolling_max(np.abs(ou_dev), 100), 0.001))
    
    # ═══ RISK ═══
    for p, vk, ck in [(20,'var_20','cvar_20'),(50,'var_50','cvar_50')]:
        mu = fast_rolling_mean(ret, p); sd = fast_rolling_std(ret, p)
        add(vk, mu - 2.33*sd); add(ck, mu - 2.33*sd*1.2)
    add('skewness', fast_rolling_mean((ret - fast_rolling_mean(ret,100))**3, 100) / np.maximum(fast_rolling_std(ret,100)**3, 0.001))
    add('kurtosis', fast_rolling_mean((ret - fast_rolling_mean(ret,100))**4, 100) / np.maximum(fast_rolling_std(ret,100)**4, 0.001) - 3)
    peak = np.maximum.accumulate(closes)
    add('max_dd', (closes - peak) / np.maximum(peak, 0.001))
    
    # ═══ REGIME ═══
    h100 = feats['hurst_100']
    add('regime_trend', (h100 > 0.55).astype(float))
    add('regime_mr', (h100 < 0.45).astype(float))
    add('regime_random', ((h100 >= 0.45) & (h100 <= 0.55)).astype(float))
    
    # ═══ BREAKOUT ═══
    rh20 = fast_rolling_max(highs, 20); rl20 = fast_rolling_min(lows, 20)
    add('breakout_up', (closes >= rh20*0.999).astype(float))
    add('breakout_down', (closes <= rl20*1.001).astype(float))
    rh50 = fast_rolling_max(highs, 50); rl50 = fast_rolling_min(lows, 50)
    add('breakout_str', (closes - rl50) / np.maximum(rh50 - rl50, 0.001))
    
    # ═══ COMPOSITE ═══
    add('mom_vol_ratio', feats['mom_20'] / np.maximum(rv20, 0.001))
    add('trend_vs_mr', feats['trend_strength'] - feats['mr_prob'])
    add('kal_ou_agree', np.sign(feats['kalman_trend']) * np.sign(feats['ou_signal']))
    add('vol_adj_mom', feats['mom_20'] * feats['vol_regime'])
    print(f"  All features done ({time.time()-t0:.0f}s)")
    
    # ═══ WRITE FEATURES ═══
    names = sorted(feats.keys())
    n_feat = len(names)
    print(f"\n  Total: {n_feat} features")
    
    X = np.memmap(f"{BASE}/quant_features_116.npy", dtype=np.float32, mode='w+', shape=(n, n_feat))
    for j, name in enumerate(names):
        X[:, j] = feats[name]
    X.flush()
    
    meta_out = {'n_features': n_feat, 'feature_names': names, 'n_rows': n}
    with open(f"{BASE}/quant_features_meta.json", 'w') as f:
        json.dump(meta_out, f, indent=2)
    
    print(f"  Features saved: {os.path.getsize(f'{BASE}/quant_features_116.npy')/1e9:.1f}GB")
    
    # Free memory — delete features dict, prices
    del feats; del prices; del opens; del highs; del lows; del closes
    del log_ret; del ret
    import gc; gc.collect()
    print(f"  Memory freed for training")
    
    # ═══ TRAIN LIGHTGBM ═══
    print("\n═══ TRAINING LIGHTGBM ═══")
    
    # Replace NaN in-place (no copy!)
    # Process in chunks to avoid memory spike
    CHUNK = 500_000
    for ci in range(0, n, CHUNK):
        end = min(ci + CHUNK, n)
        chunk = X[ci:end]
        mask = np.isnan(chunk)
        if mask.any():
            chunk[mask] = 0.0
    del mask
    gc.collect()
    
    # Split: 70% train, 30% test (use memmap slices, no copy)
    split = int(n * 0.7)
    y_tr, y_te = y[:split], y[split:]
    
    fname_list = [f"f{i}" for i in range(n_feat)]
    params = {
        'objective': 'binary', 'metric': 'auc',
        'num_leaves': 31, 'max_depth': 8,
        'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'min_child_samples': 100, 'verbose': -1,
    }
    
    # Train 3 seeds
    for seed in [42, 7, 2026]:
        print(f"\n  Training seed {seed}...")
        dtrain = lgb.Dataset(X[:split], label=y_tr, feature_name=names)
        dval = lgb.Dataset(X[split:], label=y_te, feature_name=names, reference=dtrain)
        
        model = lgb.train(params, dtrain, num_boost_round=500,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        
        preds = model.predict(X[split:])
        pred_bin = (preds > 0.5).astype(int)
        acc = (pred_bin == y_te).mean()
        up_acc = pred_bin[y_te == 1].mean() if (y_te == 1).sum() > 0 else 0
        dn_acc = pred_bin[y_te == 0].mean() if (y_te == 0).sum() > 0 else 0
        
        print(f"    Accuracy: {acc:.4f}")
        print(f"    UP recall: {up_acc:.4f}")
        print(f"    DOWN recall: {dn_acc:.4f}")
        print(f"    Trees: {model.num_trees()}")
        print(f"    Pred UP: {pred_bin.mean():.3f} (actual: {y_te.mean():.3f})")
        
        # Feature importance
        imp = model.feature_importance(importance_type='gain')
        top5 = np.argsort(imp)[-5:][::-1]
        print(f"    Top features:")
        for idx in top5:
            print(f"      {names[idx]:25s} = {imp[idx]:.0f}")
        
        model.save_model(f"{BASE}/models/quant_lgb_s{seed}.txt")
    
    # Save ensemble.json
    ensemble = {
        'models': [f'quant_lgb_s42', f'quant_lgb_s7', f'quant_lgb_s2026'],
        'base_tf': 'm5',
        'n_features': n_feat,
        'feature_names': names,
        'source': 'quant_116',
    }
    with open(f"{BASE}/models/quant_ensemble.json", 'w') as f:
        json.dump(ensemble, f, indent=2)
    
    total_time = time.time() - t0
    print(f"\n═══ PIPELINE COMPLETE ═══")
    print(f"  Features: {n_feat}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Models: quant_lgb_s42.txt, quant_lgb_s7.txt, quant_lgb_s2026.txt")
    print(f"  Ensemble: quant_ensemble.json")
    
    # Write completion marker
    with open(f"{BASE}/pipeline_complete.json", 'w') as f:
        json.dump({'status': 'done', 'features': n_feat, 'time': total_time, 
                   'timestamp': time.time()}, f)

if __name__ == '__main__':
    main()
