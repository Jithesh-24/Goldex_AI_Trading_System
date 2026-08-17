#!/usr/bin/env python3
"""
full_pipeline_v2.py — OOM-SAFE: writes features to memmap immediately.
No accumulation in dict. Each feature written then freed.
"""
import numpy as np
import lightgbm as lgb
import json, os, time, gc
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

def fast_rolling_mean(x, w):
    cs = np.cumsum(x); out = np.full_like(x, np.nan)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out

def fast_rolling_std(x, w):
    m = fast_rolling_mean(x, w); m2 = fast_rolling_mean(x**2, w)
    return np.sqrt(np.maximum(m2 - m**2, 0))

def fast_rolling_max(x, w):
    n = len(x); out = np.full(n, np.nan)
    step = max(1, w//4)
    for i in range(w-1, n, step):
        out[i] = np.max(x[max(0,i-w+1):i+1])
    idx = np.where(~np.isnan(out))[0]
    if len(idx) > 0:
        out[:idx[0]] = out[idx[0]]
        for k in range(len(idx)-1): out[idx[k]:idx[k+1]] = out[idx[k]]
        out[idx[-1]:] = out[idx[-1]]
    return out

def fast_rolling_min(x, w):
    n = len(x); out = np.full(n, np.nan)
    step = max(1, w//4)
    for i in range(w-1, n, step):
        out[i] = np.min(x[max(0,i-w+1):i+1])
    idx = np.where(~np.isnan(out))[0]
    if len(idx) > 0:
        out[:idx[0]] = out[idx[0]]
        for k in range(len(idx)-1): out[idx[k]:idx[k+1]] = out[idx[k]]
        out[idx[-1]:] = out[idx[-1]]
    return out

def fast_autocorr_vec(x, lag):
    n = len(x); out = np.full(n, np.nan)
    step = 100
    for i in range(lag+500, n, step):
        s = x[max(0,i-500):i]
        if len(s) > lag + 5:
            out[i] = np.corrcoef(s[:-lag], s[lag:])[0, 1]
    valid = ~np.isnan(out)
    if np.sum(valid) > 2:
        out = np.interp(np.arange(n), np.where(valid)[0], out[valid])
    return out

def fast_hurst(x, period, step=500):
    n = len(x); out = np.full(n, 0.5)
    for i in range(period, n, step):
        ts = x[i-period:i]; mt = np.mean(ts); devs = ts - mt
        cum = np.cumsum(devs); r = np.max(cum)-np.min(cum); s = np.std(ts)
        if s > 0 and r > 0: out[i] = np.log(r/s)/np.log(period)
    valid = (out != 0.5)
    if np.sum(valid) > 2: out = np.interp(np.arange(n), np.where(valid)[0], out[valid])
    return out

def fast_entropy(x, period, step=500):
    n = len(x); out = np.full(n, 3.0)
    for i in range(period, n, step):
        r = x[i-period:i]; bins = np.histogram(r, bins=20)[0]
        probs = bins/max(bins.sum(),1); probs = probs[probs>0]
        out[i] = -np.sum(probs*np.log2(probs))
    valid = (out != 3.0)
    if np.sum(valid) > 2: out = np.interp(np.arange(n), np.where(valid)[0], out[valid])
    return out

def ema(x, p):
    alpha = 2.0/(p+1); out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)): out[i] = alpha*x[i] + (1-alpha)*out[i-1]
    return out

def main():
    t0 = time.time()
    print("═══ FULL AI PIPELINE V2 (OOM-SAFE) ═══\n")
    
    prices = np.load(f"{BASE}/prices_tail.npy")
    opens = prices[:,0].astype(np.float64)
    highs = prices[:,1].astype(np.float64)
    lows = prices[:,2].astype(np.float64)
    closes = prices[:,3].astype(np.float64)
    n = len(closes)
    del prices
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    y_full = np.memmap(f"{BASE}/train_data_y.npy", dtype=np.int8, mode='r', shape=(meta['n_rows'],))
    y = y_full[:n].copy()  # small: 10M int8 = 10MB
    
    log_ret = np.concatenate([[0], np.diff(np.log(closes))])
    ret = np.concatenate([[0], np.diff(closes)/closes[:-1]])
    
    print(f"Data: {n:,} bars\n")
    
    # Pre-compute shared intermediates (keep small set)
    rv20 = fast_rolling_std(log_ret, 20) * np.sqrt(288)
    rv100 = fast_rolling_std(log_ret, 100) * np.sqrt(288)
    mom20 = closes/np.roll(closes, 20) - 1
    
    # Feature names and index
    NAMES = [
        'mom_5','mom_10','mom_20','mom_50','mom_100',
        'roc_5','roc_10','roc_20',
        'mom_decay_5_20','mom_decay_10_50','mom_decay_20_100',
        'rvol_10','rvol_20','rvol_50','rvol_100',
        'garch_vol','garch_ratio','garch_forecast',
        'vol_of_vol','vol_of_vol_trend',
        'vol_regime','vol_regime_pctile','vol_clustering',
        'parkinson_vol','gk_vol',
        'vol_high','vol_low','vol_normal',
        'zscore_20','zscore_50','zscore_100',
        'dist_mean_20','dist_mean_50','dist_mean_100',
        'bb_pos_20','bb_width_20','bb_pos_50','bb_width_50',
        'hurst_50','hurst_100','hurst_200',
        'entropy_50','entropy_100','entropy_200',
        'var_ratio_5','var_ratio_10','var_ratio_20','var_ratio_40',
        'autocorr_1','autocorr_2','autocorr_5','autocorr_10','autocorr_20','autocorr_50',
        'ou_theta','ou_half_life','ou_z_score','ou_signal',
        'kalman_trend','kalman_velocity','kalman_residual','kalman_sn',
        'mom_quality','mom_consistency','trend_strength','trend_consistency',
        'rev_speed','mr_prob',
        'var_20','cvar_20','var_50','cvar_50',
        'skewness','kurtosis','max_dd',
        'regime_trend','regime_mr','regime_random',
        'breakout_up','breakout_down','breakout_str',
        'mom_vol_ratio','trend_vs_mr','kal_ou_agree','vol_adj_mom',
    ]
    n_feat = len(NAMES)
    print(f"Features: {n_feat}")
    
    X = np.memmap(f"{BASE}/quant_features_116.npy", dtype=np.float32, mode='w+', shape=(n, n_feat))
    
    def write_feat(idx, data):
        X[:, idx] = data.astype(np.float32)
    
    # ═══ WRITE FEATURES ONE BY ONE ═══
    # MOMENTUM (11)
    for i, p in enumerate([5,10,20,50,100]):
        write_feat(i, closes/np.roll(closes,p) - 1)
    for i, p in enumerate([5,10,20]):
        write_feat(5+i, closes/np.roll(closes,p+1) - 1)
    write_feat(8, X[:,0]/np.maximum(np.abs(X[:,2]), 0.001))
    write_feat(9, X[:,1]/np.maximum(np.abs(X[:,3]), 0.001))
    write_feat(10, X[:,2]/np.maximum(np.abs(X[:,4]), 0.001))
    print(f"  [{time.time()-t0:.0f}s] Momentum")
    
    # VOLATILITY (14)
    for i, p in enumerate([10,20,50,100]):
        write_feat(11+i, fast_rolling_std(log_ret, p)*np.sqrt(288))
    
    alpha_g, beta_g = 0.1, 0.85
    var0 = np.var(log_ret[1:100])
    omega = var0*(1-alpha_g-beta_g)
    gv = np.zeros(n); gv[0]=var0
    for i in range(1,n): gv[i] = omega + alpha_g*log_ret[i]**2 + beta_g*gv[i-1]
    write_feat(15, np.sqrt(gv)*np.sqrt(288))
    write_feat(16, np.sqrt(gv)*np.sqrt(288)/np.maximum(rv20, 0.001))
    write_feat(17, np.roll(np.sqrt(gv)*np.sqrt(288), -1))
    
    rv20_v = X[:,12]  # rvol_20
    rv100_v = X[:,13]  # rvol_100
    write_feat(18, fast_rolling_std(rv20_v, 50)/np.maximum(fast_rolling_mean(rv20_v, 50), 0.001))
    write_feat(19, (fast_rolling_mean(rv20_v, 5)-fast_rolling_mean(rv20_v, 50))/np.maximum(fast_rolling_mean(rv20_v, 50), 0.001))
    write_feat(20, rv20_v/np.maximum(rv100_v, 0.001))
    write_feat(21, fast_rolling_mean((rv20_v > rv100_v).astype(float), 200))
    write_feat(22, fast_autocorr_vec(log_ret**2, 1))
    
    hl = np.log(np.maximum(highs,1)/np.maximum(lows,1))
    write_feat(23, np.sqrt(fast_rolling_mean(hl**2, 20)/(4*np.log(2)))*np.sqrt(288))
    write_feat(24, np.sqrt(np.maximum(fast_rolling_mean(0.5*hl**2-(2*np.log(2)-1)*log_ret**2, 20), 0))*np.sqrt(288))
    
    vr = X[:,20]
    write_feat(25, (vr > 1.5).astype(float))
    write_feat(26, (vr < 0.5).astype(float))
    write_feat(27, ((vr >= 0.5)&(vr <= 1.5)).astype(float))
    del hl; gc.collect()
    print(f"  [{time.time()-t0:.0f}s] Volatility")
    
    # Z-SCORE + BB (10)
    for i, p in enumerate([20,50,100]):
        m = fast_rolling_mean(closes, p); s = fast_rolling_std(closes, p)
        write_feat(28+i, (closes-m)/np.maximum(s, 0.001))
        write_feat(31+i, (closes-m)/np.maximum(m, 0.001))
    for i, p in enumerate([20,50]):
        m = fast_rolling_mean(closes, p); s = fast_rolling_std(closes, p)
        write_feat(34+i, (closes-m)/(2*np.maximum(s, 0.001)))
        write_feat(36+i, 4*s/np.maximum(m, 0.001))
    print(f"  [{time.time()-t0:.0f}s] Z-score/BB")
    
    # HURST (3)
    for i, p in enumerate([50,100,200]):
        write_feat(38+i, fast_hurst(closes, p, 200))
    print(f"  [{time.time()-t0:.0f}s] Hurst")
    
    # ENTROPY (3)
    for i, p in enumerate([50,100,200]):
        write_feat(41+i, fast_entropy(log_ret, p, 200))
    print(f"  [{time.time()-t0:.0f}s] Entropy")
    
    # VARIANCE RATIO (4)
    for i, p in enumerate([5,10,20,40]):
        vs = fast_rolling_std(log_ret, p)**2; vs2 = fast_rolling_std(log_ret, p*2)**2
        write_feat(44+i, vs/np.maximum(vs2, 1e-10))
    
    # AUTOCORRELATION (6)
    for i, lag in enumerate([1,2,5,10,20,50]):
        write_feat(48+i, fast_autocorr_vec(log_ret, lag))
    print(f"  [{time.time()-t0:.0f}s] Autocorr")
    
    # OU (4)
    ou_dev = closes - fast_rolling_mean(closes, 100)
    ou_ac = fast_autocorr_vec(ou_dev, 1)
    write_feat(54, -ou_ac)
    write_feat(55, np.log(2)/np.maximum(-ou_ac, 0.001))
    write_feat(56, ou_dev/np.maximum(fast_rolling_std(closes, 100), 0.001))
    write_feat(57, -ou_ac*(fast_rolling_mean(closes, 100)-closes))
    
    # KALMAN (4)
    kf = ema(closes, 5); ks = ema(closes, 50)
    write_feat(58, (kf-ks)/np.maximum(ks, 0.001))
    write_feat(59, kf - np.roll(kf, 1))
    write_feat(60, closes - kf)
    write_feat(61, np.abs(kf-ks)/np.maximum(fast_rolling_std(closes, 50), 0.001))
    del kf, ks, ou_dev, ou_ac; gc.collect()
    print(f"  [{time.time()-t0:.0f}s] OU/Kalman")
    
    # MOMENTUM QUALITY (4)
    pos_r = fast_rolling_mean((ret > 0).astype(float), 50)
    write_feat(62, pos_r/np.maximum(1-pos_r, 0.01))
    write_feat(63, fast_rolling_mean(ret, 50)/np.maximum(fast_rolling_std(ret, 50), 0.001))
    write_feat(64, np.abs(fast_rolling_mean(ret, 50))/np.maximum(fast_rolling_mean(np.abs(ret), 50), 0.001))
    write_feat(65, np.sign(fast_rolling_mean(ret, 50))*X[:, 64])
    
    # REVERSION (2)
    write_feat(66, -fast_autocorr_vec(closes - fast_rolling_mean(closes, 100), 1))
    dev = closes - fast_rolling_mean(closes, 100)
    write_feat(67, np.abs(dev)/np.maximum(fast_rolling_max(np.abs(dev), 100), 0.001))
    del dev; gc.collect()
    
    # RISK (5)
    for i, (p, vk, ck) in enumerate([(20,68,69),(50,70,71)]):
        mu = fast_rolling_mean(ret, p); sd = fast_rolling_std(ret, p)
        write_feat(vk, mu - 2.33*sd); write_feat(ck, mu - 2.33*sd*1.2)
    write_feat(72, fast_rolling_mean((ret-fast_rolling_mean(ret,100))**3, 100)/np.maximum(fast_rolling_std(ret,100)**3, 0.001))
    write_feat(73, fast_rolling_mean((ret-fast_rolling_mean(ret,100))**4, 100)/np.maximum(fast_rolling_std(ret,100)**4, 0.001) - 3)
    peak = np.maximum.accumulate(closes)
    write_feat(74, (closes-peak)/np.maximum(peak, 0.001))
    del peak; gc.collect()
    
    # REGIME (3)
    h100 = X[:,39]  # hurst_100
    write_feat(75, (h100 > 0.55).astype(float))
    write_feat(76, (h100 < 0.45).astype(float))
    write_feat(77, ((h100 >= 0.45)&(h100 <= 0.55)).astype(float))
    
    # BREAKOUT (3)
    rh20 = fast_rolling_max(highs, 20); rl20 = fast_rolling_min(lows, 20)
    write_feat(78, (closes >= rh20*0.999).astype(float))
    write_feat(79, (closes <= rl20*1.001).astype(float))
    rh50 = fast_rolling_max(highs, 50); rl50 = fast_rolling_min(lows, 50)
    write_feat(80, (closes-rl50)/np.maximum(rh50-rl50, 0.001))
    
    # COMPOSITE (4)
    write_feat(81, mom20/np.maximum(rv20, 0.001))
    write_feat(82, X[:,64] - X[:,67])
    write_feat(83, np.sign(X[:,58])*np.sign(X[:,57]))
    write_feat(84, mom20*vr)
    
    del closes, opens, highs, lows, log_ret, ret, mom20, rv20, rv100
    gc.collect()
    print(f"  [{time.time()-t0:.0f}s] ALL {n_feat} features written")
    
    X.flush()
    
    # Save meta
    meta_out = {'n_features': n_feat, 'feature_names': NAMES, 'n_rows': n}
    with open(f"{BASE}/quant_features_meta.json", 'w') as f:
        json.dump(meta_out, f, indent=2)
    print(f"  File: {os.path.getsize(f'{BASE}/quant_features_116.npy')/1e9:.1f}GB")
    
    # ═══ TRAIN LIGHTGBM ═══
    print(f"\n═══ TRAINING ({time.time()-t0:.0f}s so far) ═══")
    
    # NaN fill in chunks
    for ci in range(0, n, 500_000):
        end = min(ci+500_000, n)
        chunk = X[ci:end]
        m = np.isnan(chunk)
        if m.any(): chunk[m] = 0.0
    gc.collect()
    
    split = int(n * 0.7)
    # Use last 2M rows for training to fit in RAM (7.5GB total)
    train_start = max(0, split - 2_000_000)
    X_tr = X[train_start:split]; X_te = X[split:]
    y_tr = y[train_start:split]; y_te = y[split:]
    print(f"  Training rows: {len(y_tr):,} | Test rows: {len(y_te):,}")
    
    params = {
        'objective':'binary', 'metric':'auc',
        'num_leaves':31, 'max_depth':8, 'learning_rate':0.05,
        'feature_fraction':0.8, 'bagging_fraction':0.8, 'bagging_freq':5,
        'reg_alpha':0.1, 'reg_lambda':1.0, 'min_child_samples':100, 'verbose':-1,
    }
    
    for seed in [42, 7, 2026]:
        print(f"\n  Seed {seed}...")
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=NAMES)
        dval = lgb.Dataset(X_te, label=y_te, feature_name=NAMES, reference=dtrain)
        
        model = lgb.train(params, dtrain, num_boost_round=500,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        
        preds = model.predict(X_te)
        pred_bin = (preds > 0.5).astype(int)
        acc = (pred_bin == y_te).mean()
        
        print(f"    Acc: {acc:.4f} | Trees: {model.num_trees()} | Pred UP: {pred_bin.mean():.3f}")
        
        imp = model.feature_importance(importance_type='gain')
        top5 = np.argsort(imp)[-5:][::-1]
        for idx in top5:
            print(f"      {NAMES[idx]:25s} gain={imp[idx]:.0f}")
        
        model.save_model(f"{BASE}/models/quant_lgb_s{seed}.txt")
    
    ensemble = {
        'models': [f'quant_lgb_s42', 'quant_lgb_s7', 'quant_lgb_s2026'],
        'base_tf': 'm5', 'n_features': n_feat,
        'feature_names': NAMES, 'source': 'quant_116',
    }
    with open(f"{BASE}/models/quant_ensemble.json", 'w') as f:
        json.dump(ensemble, f, indent=2)
    
    total = time.time() - t0
    print(f"\n═══ DONE: {total:.0f}s ({total/60:.1f}min) ═══")
    print(f"  {n_feat} features × {n:,} rows → 3 LightGBM models")
    
    with open(f"{BASE}/pipeline_complete.json", 'w') as f:
        json.dump({'status':'done','features':n_feat,'time':total,'ts':time.time()}, f)

if __name__ == '__main__':
    main()
