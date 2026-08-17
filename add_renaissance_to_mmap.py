#!/usr/bin/env python3
"""
add_renaissance_to_mmap.py — Add 22 Renaissance features using RAW close prices.
Computes every 10th row, forward-fills gaps (Renaissance features change slowly).
"""
import numpy as np
import time, os, sys, json
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from renaissance_modules import HMMRegimeDetector, KalmanFilter, OrnsteinUhlenbeckDetector
from garch_features import compute_garch_features
from correlation_features import compute_correlation_features

TAIL = 10_000_000
STRIDE = 50  # compute every 50th row (Renaissance features change slowly)

REN_FEATS = [
    'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'hmm_prob_3',
    'kalman_trend', 'kalman_velocity', 'kalman_innovation',
    'ou_theta', 'ou_mu', 'ou_half_life', 'ou_is_mr', 'ou_signal', 'ou_z_score',
    'garch_vol', 'garch_forecast', 'vol_regime', 'vol_persistence', 'vol_asymmetry', 'vol_shock',
    'corr_dxy', 'corr_vix',
]

def main():
    t0 = time.time()
    
    print("═══ ADDING 22 RENAISSANCE FEATURES (STRIDE=10) ═══\n", flush=True)
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    n = meta['n_rows']
    old_feats = meta['features']
    nf_old = meta.get('n_features_base', len(old_feats))  # 108 base features
    old_feats = old_feats[:nf_old]  # only base features, not the zero Renaissance ones
    
    # Load raw close prices
    print("Loading raw close prices...", flush=True)
    prices = np.load(f"{BASE}/prices_tail.npy")
    closes = prices[:, 3].astype(np.float64)
    n_tail = len(closes)
    print(f"Prices: {n_tail:,} rows, close range {closes.min():.2f} – {closes.max():.2f}", flush=True)
    
    # Load base features (last 10M rows)
    start = max(0, n - TAIL)
    print(f"Loading mmap: rows {start:,}–{n:,} ({n_tail:,} rows × {nf_old} features)", flush=True)
    X_full = np.memmap(f"{BASE}/train_data_x.npy", dtype=np.float32, mode='r', shape=(n, nf_old))
    X = np.array(X_full[start:start + n_tail])
    del X_full
    
    nf_new = nf_old + len(REN_FEATS)
    new_feats = old_feats + REN_FEATS
    
    # Create new mmap
    X_new = np.memmap(f"{BASE}/train_data_x_130.npy", dtype=np.float32, mode='w+', shape=(n_tail, nf_new))
    X_new[:, :nf_old] = X
    X_new.flush()
    del X
    
    # Initialize modules
    hmm = HMMRegimeDetector(n_states=4, n_iter=10)
    kf = KalmanFilter(dt=1.0, process_noise=0.01, measurement_noise=0.1)
    ou_detector = OrnsteinUhlenbeckDetector(lookback=100)
    
    lookback = 200
    n_computed = (n_tail - lookback) // STRIDE
    
    print(f"\nComputing Renaissance features every {STRIDE}th row ({n_computed:,} computations)...", flush=True)
    
    count = 0
    for i in range(lookback, n_tail, STRIDE):
        window = closes[max(0, i-lookback):i]
        if len(window) < 50:
            continue
        
        base = nf_old
        
        # HMM
        try:
            hmm.fit(window[-150:])
            regime, probs = hmm.get_current_regime(window, window=min(50, len(window)))
            X_new[i, base + 0] = float(regime)
            for p in range(4):
                X_new[i, base + 1 + p] = float(probs[p]) if p < len(probs) else 0.25
        except:
            X_new[i, base + 0] = 2.0
            for p in range(4):
                X_new[i, base + 1 + p] = 0.25
        
        # Kalman
        try:
            _kf_trends, _kf_innov = kf.filter_series(window)
            X_new[i, base + 5] = float(_kf_trends[-1, 0])
            X_new[i, base + 6] = float(_kf_trends[-1, 1])
            X_new[i, base + 7] = float(_kf_innov[-1])
        except:
            pass
        
        # OU
        try:
            ou_result = ou_detector.fit(window)
            X_new[i, base + 8] = float(ou_result.get('theta', 0))
            X_new[i, base + 9] = float(ou_result.get('mu', 0))
            X_new[i, base + 10] = min(float(ou_result.get('half_life', 999)), 999)
            X_new[i, base + 11] = 1.0 if ou_result.get('is_mean_reverting', False) else 0.0
            X_new[i, base + 12] = float(ou_result.get('signal', 0))
            X_new[i, base + 13] = float(ou_result.get('z_score', 0))
        except:
            pass
        
        # GARCH
        try:
            garch = compute_garch_features(window)
            X_new[i, base + 14] = float(garch.get('garch_vol', 0))
            X_new[i, base + 15] = float(garch.get('garch_forecast', 0))
            X_new[i, base + 16] = float(garch.get('vol_regime', 0))
            X_new[i, base + 17] = float(garch.get('vol_persistence', 0))
            X_new[i, base + 18] = float(garch.get('vol_asymmetry', 0))
            X_new[i, base + 19] = float(garch.get('vol_shock', 0))
        except:
            pass
        
        # Correlation
        try:
            corr = compute_correlation_features(window)
            X_new[i, base + 20] = float(corr.get('corr_dxy', 0))
            X_new[i, base + 21] = float(corr.get('corr_vix', 0))
        except:
            pass
        
        count += 1
        if count % 10_000 == 0:
            el = time.time() - t0
            rate = count / max(el, 1)
            remaining = (n_computed - count) / max(rate, 1)
            print(f"  {count:,}/{n_computed:,} ({el:.0f}s, ~{remaining:.0f}s remaining, {rate:.1f} rows/s)", flush=True)
    
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
    meta['renaissance_features'] = REN_FEATS
    meta['renaissance_added'] = '2026-08-15'
    meta['mmap_x_130'] = f"{BASE}/train_data_x_130.npy"
    meta['tail_n_rows'] = n_tail
    
    with open(f"{BASE}/train_data_meta.json", 'w') as f:
        json.dump(meta, f)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)", flush=True)
    print(f"Saved: train_data_x_130.npy ({os.path.getsize(f'{BASE}/train_data_x_130.npy')/1024**3:.1f} GB)", flush=True)
    print(f"Features: {nf_new} ({nf_old} base + {len(REN_FEATS)} Renaissance)", flush=True)

if __name__ == '__main__':
    main()
