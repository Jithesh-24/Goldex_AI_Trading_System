#!/usr/bin/env python3
"""
retrain_with_renaissance.py — Adds Renaissance features to matrix, then retrains.

Called automatically after base retrain completes.
Computes HMM, Kalman, OU features for each row and adds to matrix.
"""
import pandas as pd
import numpy as np
import time
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "gold_features_m5_full.csv")
OUTPUT = os.path.join(BASE, "gold_features_m5_renaissance.csv")

sys.path.insert(0, BASE)


def compute_row_renaissance(prices, returns, idx, lookback=200):
    """Compute Renaissance features for a single row from its price history."""
    result = {}
    
    if len(prices) < lookback:
        # Not enough history — fill with defaults
        result['hmm_regime'] = 2.0  # ranging
        for i in range(4):
            result[f'hmm_prob_{i}'] = 0.25
        result['kalman_trend'] = prices[-1] if len(prices) > 0 else 0
        result['kalman_velocity'] = 0.0
        result['kalman_innovation'] = 0.0
        result['ou_theta'] = 0.0
        result['ou_mu'] = np.mean(prices) if len(prices) > 0 else 0
        result['ou_half_life'] = 999.0
        result['ou_is_mr'] = 0.0
        result['ou_signal'] = 0.0
        result['ou_z_score'] = 0.0
        # GARCH defaults
        result['garch_vol'] = 0.0
        result['garch_forecast'] = 0.0
        result['vol_regime'] = 0.0
        result['vol_persistence'] = 0.0
        result['vol_asymmetry'] = 0.0
        result['vol_shock'] = 0.0
        # Correlation defaults
        result['corr_dxy'] = 0.0
        result['corr_change_dxy'] = 0.0
        result['corr_regime'] = 0.0
        return result
    
    # Use last lookback prices
    ph = prices[-lookback:]
    ret = returns[-lookback:]
    
    # HMM Regime Detection
    try:
        from renaissance_modules import HMMRegimeDetector
        hmm = HMMRegimeDetector(n_states=4, n_iter=8)
        hmm.fit(ret[-150:])
        regime, probs = hmm.get_current_regime(ret, window=50)
        result['hmm_regime'] = float(regime)
        for i in range(4):
            result[f'hmm_prob_{i}'] = float(probs[i])
    except Exception:
        result['hmm_regime'] = 2.0
        for i in range(4):
            result[f'hmm_prob_{i}'] = 0.25
    
    # Kalman Filter
    try:
        from renaissance_modules import KalmanFilter
        kf = KalmanFilter(dt=1.0, process_noise=0.01, measurement_noise=0.1)
        trends, innovations = kf.filter_series(ph[-100:])
        result['kalman_trend'] = float(trends[-1, 0])
        result['kalman_velocity'] = float(trends[-1, 1])
        result['kalman_innovation'] = float(innovations[-1])
    except Exception:
        result['kalman_trend'] = ph[-1]
        result['kalman_velocity'] = 0.0
        result['kalman_innovation'] = 0.0
    
    # Ornstein-Uhlenbeck
    try:
        from renaissance_modules import OrnsteinUhlenbeckDetector
        ou = OrnsteinUhlenbeckDetector(lookback=100)
        ou_result = ou.fit(ph)
        result['ou_theta'] = float(ou_result['theta'])
        result['ou_mu'] = float(ou_result['mu'])
        result['ou_half_life'] = min(float(ou_result['half_life']), 999.0)
        result['ou_is_mr'] = 1.0 if ou_result['is_mean_reverting'] else 0.0
        result['ou_signal'] = float(ou_result['signal'])
        result['ou_z_score'] = float(ou_result.get('z_score', 0))
    except Exception:
        result['ou_theta'] = 0.0
        result['ou_mu'] = ph[-1]
        result['ou_half_life'] = 999.0
        result['ou_is_mr'] = 0.0
        result['ou_signal'] = 0.0
        result['ou_z_score'] = 0.0
    
    # GARCH Volatility
    try:
        from garch_features import compute_garch_features
        _garch = compute_garch_features(ret[-200:], window=200)
        result['garch_vol'] = _garch['garch_vol']
        result['garch_forecast'] = _garch['garch_forecast']
        result['vol_regime'] = _garch['vol_regime']
        result['vol_persistence'] = _garch['vol_persistence']
        result['vol_asymmetry'] = _garch['vol_asymmetry']
        result['vol_shock'] = _garch['vol_shock']
    except Exception:
        pass
    
    # Correlation Breakdown
    try:
        from correlation_features import compute_correlation_features
        _corr = compute_correlation_features(ret[-200:], window=20)
        result['corr_dxy'] = _corr['corr_dxy']
        result['corr_change_dxy'] = _corr['corr_change_dxy']
        result['corr_regime'] = _corr['corr_regime']
    except Exception:
        pass
    
    return result


def main():
    print("═══ RENAISSANCE FEATURE ADDITION ═══")
    print(f"Input: {INPUT}")
    print(f"Output: {OUTPUT}")
    
    # Read column names
    t0 = time.time()
    header = pd.read_csv(INPUT, nrows=0)
    cols = list(header.columns)
    print(f"Source columns: {len(cols)}")
    
    # Renaissance feature names
    ren_cols = [
        'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'hmm_prob_3',
        'kalman_trend', 'kalman_velocity', 'kalman_innovation',
        'ou_theta', 'ou_mu', 'ou_half_life', 'ou_is_mr', 'ou_signal', 'ou_z_score',
        'garch_vol', 'garch_forecast', 'vol_regime', 'vol_persistence', 'vol_asymmetry', 'vol_shock',
        'corr_dxy', 'corr_change_dxy', 'corr_regime'
    ]
    
    # Check for close/price columns
    has_close = 'close' in cols
    has_ret_1 = 'ret_1' in cols
    
    if not has_close:
        print("ERROR: 'close' column not found in matrix!")
        sys.exit(1)
    
    print(f"Adding {len(ren_cols)} Renaissance features")
    print(f"Total new columns: {len(cols) + len(ren_cols)}")
    
    # Process in chunks
    CHUNK = 200_000
    first_chunk = True
    total_rows = 0
    
    for chunk_idx, chunk in enumerate(pd.read_csv(INPUT, chunksize=CHUNK)):
        t_chunk = time.time()
        
        # Compute returns for OU
        close = chunk['close'].values
        returns = np.diff(close, prepend=close[0]) / np.maximum(np.abs(np.roll(close, 1)), 1e-10)
        returns[0] = 0.0
        
        # Initialize Renaissance feature columns
        ren_data = {col: np.zeros(len(chunk)) for col in ren_cols}
        
        # Compute for each row (with rolling window)
        LOOKBACK = 200
        for i in range(len(chunk)):
            # Build price history from current chunk + overlap
            start = max(0, i - LOOKBACK)
            ph = close[start:i+1]
            ret = returns[start:i+1]
            
            ren = compute_row_renaissance(ph, ret, i, lookback=LOOKBACK)
            for col in ren_cols:
                ren_data[col][i] = ren.get(col, 0.0)
        
        # Add Renaissance columns to chunk
        for col in ren_cols:
            chunk[col] = ren_data[col]
        
        # Write to output
        if first_chunk:
            chunk.to_csv(OUTPUT, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(OUTPUT, index=False, mode='a', header=False)
        
        total_rows += len(chunk)
        elapsed = time.time() - t0
        rate = total_rows / elapsed if elapsed > 0 else 0
        eta = (32_509_156 - total_rows) / rate if rate > 0 else 0
        
        print(f"  [{total_rows:>10,} / 32,509,156] "
              f"({total_rows/32_509_156*100:.1f}%) "
              f"ETA: {eta/60:.0f}min "
              f"({time.time()-t_chunk:.0f}s/chunk)")
    
    total_time = time.time() - t0
    print(f"\nDONE: {total_rows:,} rows, {len(cols) + len(ren_cols)} columns ({total_time:.0f}s)")
    print(f"Output: {OUTPUT}")


if __name__ == '__main__':
    main()
