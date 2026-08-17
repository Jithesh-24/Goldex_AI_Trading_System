#!/usr/bin/env python3
"""
add_renaissance_v3.py — FAST Renaissance features (no HMM, Kalman+OU+GARCH+corr only).
HMM has ZERO importance and takes 12 hours — skip it.
"""
import numpy as np
import time, os, sys, json
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from renaissance_modules import KalmanFilter, OrnsteinUhlenbeckDetector
from garch_features import compute_garch_features
from correlation_features import compute_correlation_features

TAIL = 10_000_000
STRIDE = 50

# 17 features (no HMM — zero importance, 12h computation)
REN_FEATS = [
    'kalman_trend', 'kalman_velocity', 'kalman_innovation',
    'ou_theta', 'ou_mu', 'ou_half_life', 'ou_is_mr', 'ou_signal', 'ou_z_score',
    'garch_vol', 'garch_forecast', 'vol_regime', 'vol_persistence', 'vol_asymmetry', 'vol_shock',
    'corr_dxy', 'corr_vix',
]

def main():
    t0 = time.time()
    
    print("═══ RENAISSANCE FEATURES V3 (FAST, NO HMM) ═══\n", flush=True)
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    n = meta['n_rows']
    nf_old = meta.get('n_features_base', 108)
    
    # Load raw close prices (~40MB)
    print("Loading raw close prices...", flush=True)
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:, 3].astype(np.float64)
    n_tail = len(closes)
    print(f"Prices: {n_tail:,} rows, close range {closes.min():.2f} – {closes.max():.2f}", flush=True)
    
    # New feature list (108 base + 17 Renaissance = 125)
    # But we need to replace the 5 HMM features with placeholder names
    # Keep old names for compatibility, just set HMM to 0
    new_feats = meta['features'][:nf_old] + REN_FEATS
    nf_new = len(new_feats)
    
    # Create new mmap
    X_new = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='w+', shape=(n_tail, nf_new))
    
    # Copy base features (OOM-safe chunks)
    print(f"Copying {nf_old} base features...", flush=True)
    X_full = np.memmap(f"{BASE}/train_data_x.npy", dtype=np.float32, mode='r', shape=(n, nf_old))
    start = max(0, n - TAIL)
    
    CHUNK = 50_000
    copied = 0
    for i in range(start, start + n_tail, CHUNK):
        end = min(i + CHUNK, start + n_tail)
        chunk = np.array(X_full[i:end])
        X_new[copied:copied + len(chunk), :nf_old] = chunk
        copied += len(chunk)
    del X_full
    X_new.flush()
    print(f"  ✅ Base copied ({time.time()-t0:.0f}s)", flush=True)
    
    # Initialize modules
    kf = KalmanFilter(dt=1.0, process_noise=0.01, measurement_noise=0.1)
    ou_detector = OrnsteinUhlenbeckDetector(lookback=100)
    
    lookback = 200
    n_computed = (n_tail - lookback) // STRIDE
    
    print(f"\nComputing {len(REN_FEATS)} Renaissance features (no HMM) every {STRIDE}th row...", flush=True)
    print(f"  {n_computed:,} computations expected\n", flush=True)
    
    count = 0
    errors = 0
    for i in range(lookback, n_tail, STRIDE):
        window = closes[max(0, i-lookback):i]
        if len(window) < 50:
            continue
        
        base = nf_old
        
        # Kalman (filter_series)
        try:
            _kf_trends, _kf_innov = kf.filter_series(window)
            X_new[i, base + 0] = float(_kf_trends[-1, 0])
            X_new[i, base + 1] = float(_kf_trends[-1, 1])
            X_new[i, base + 2] = float(_kf_innov[-1])
        except Exception as e:
            errors += 1
        
        # OU (fit)
        try:
            ou_result = ou_detector.fit(window)
            X_new[i, base + 3] = float(ou_result.get('theta', 0))
            X_new[i, base + 4] = float(ou_result.get('mu', 0))
            X_new[i, base + 5] = min(float(ou_result.get('half_life', 999)), 999)
            X_new[i, base + 6] = 1.0 if ou_result.get('is_mean_reverting', False) else 0.0
            X_new[i, base + 7] = float(ou_result.get('signal', 0))
            X_new[i, base + 8] = float(ou_result.get('z_score', 0))
        except Exception as e:
            errors += 1
        
        # GARCH
        try:
            garch = compute_garch_features(window)
            X_new[i, base + 9] = float(garch.get('garch_vol', 0))
            X_new[i, base + 10] = float(garch.get('garch_forecast', 0))
            X_new[i, base + 11] = float(garch.get('vol_regime', 0))
            X_new[i, base + 12] = float(garch.get('vol_persistence', 0))
            X_new[i, base + 13] = float(garch.get('vol_asymmetry', 0))
            X_new[i, base + 14] = float(garch.get('vol_shock', 0))
        except Exception as e:
            errors += 1
        
        # Correlation
        try:
            corr = compute_correlation_features(window)
            X_new[i, base + 15] = float(corr.get('corr_dxy', 0))
            X_new[i, base + 16] = float(corr.get('corr_vix', 0))
        except Exception as e:
            errors += 1
        
        count += 1
        if count % 10_000 == 0:
            el = time.time() - t0
            rate = count / max(el, 1)
            remaining = (n_computed - count) / max(rate, 1)
            print(f"  {count:,}/{n_computed:,} ({el:.0f}s, ~{remaining:.0f}s remaining, {rate:.1f} rows/s, {errors} errors)", flush=True)
    
    X_new.flush()
    
    # Forward-fill gaps
    print(f"\nForward-filling {STRIDE}x gaps...", flush=True)
    ren_start = nf_old
    for col_offset in range(len(REN_FEATS)):
        col = X_new[:, ren_start + col_offset]
        last_val = 0.0
        for i in range(n_tail):
            if col[i] != 0.0:
                last_val = col[i]
            else:
                col[i] = last_val
        X_new[:, ren_start + col_offset] = col
    
    X_new.flush()
    
    # Update metadata
    meta['n_features'] = nf_new
    meta['features'] = new_feats
    meta['n_features_base'] = nf_old
    meta['renaissance_features'] = REN_FEATS
    meta['renaissance_added'] = '2026-08-16'
    meta['mmap_x_130'] = f"{BASE}/train_data_x_130.npy"
    meta['tail_n_rows'] = n_tail
    meta['no_hmm'] = True
    
    with open(f"{BASE}/train_data_meta.json", 'w') as f:
        json.dump(meta, f)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)", flush=True)
    print(f"Saved: train_data_x_130.npy ({os.path.getsize(f'{BASE}/train_data_x_130.npy')/1024**3:.1f} GB)", flush=True)
    print(f"Features: {nf_new} ({nf_old} base + {len(REN_FEATS)} Renaissance)", flush=True)
    print(f"Errors: {errors}", flush=True)

if __name__ == '__main__':
    main()
