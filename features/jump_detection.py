"""Family C -- jump/change detection. Moved from research/features_v3.py
lines 521-553. Deviation from verbatim: cusum_k is now an explicit
parameter (was `from learning.train import CUSUM_K`) -- features/ must
never import learning/ (test_boundary.py)."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def compute_jump_detection(shared: SharedInputs, cusum_k: float) -> dict:
    ret1, sign1, ewma_vol, c = shared.ret1, shared.sign1, shared.ewma_vol, shared.c
    f = {}
    threshold = np.clip(cusum_k * np.nan_to_num(ewma_vol, nan=np.nanmedian(ewma_vol)) * c, 1e-6, None)
    cusum_pos = np.zeros(len(c)); cusum_neg = np.zeros(len(c))
    sp, sn = 0.0, 0.0
    for i in range(1, len(c)):
        diff = c[i] - c[i - 1]
        sp = max(0.0, sp + diff); sn = min(0.0, sn + diff)
        if sp > threshold[i] or sn < -threshold[i]:
            sp, sn = 0.0, 0.0
        cusum_pos[i], cusum_neg[i] = sp, sn
    f["cusum_distance_to_threshold"] = np.maximum(cusum_pos, -cusum_neg) / np.clip(threshold, 1e-9, None)
    local_vol_price = ewma_vol * c
    is_jump = np.abs(ret1 * c) > 3 * np.clip(local_vol_price, 1e-9, None)
    is_jump_s = pd.Series(is_jump.astype(np.float64))
    f["jump_intensity_60"] = is_jump_s.rolling(60).sum().to_numpy()
    jump_mag = pd.Series(np.where(is_jump, np.abs(ret1), np.nan))
    f["jump_magnitude_mean_60"] = jump_mag.rolling(60, min_periods=1).mean().to_numpy()
    jump_dir = pd.Series(np.where(is_jump, sign1, np.nan))
    f["jump_direction_bias_60"] = jump_dir.rolling(60, min_periods=1).mean().to_numpy()
    changepoint = (cusum_pos == 0) & (cusum_neg == 0)
    changepoint[0] = False
    bars_since = np.full(len(c), np.nan)
    last_cp = -1
    for i in range(len(c)):
        if changepoint[i]:
            last_cp = i
        bars_since[i] = i - last_cp if last_cp >= 0 else np.nan
    f["bars_since_last_changepoint"] = bars_since
    f["changepoint_intensity_240"] = pd.Series(changepoint.astype(np.float64)).rolling(240).sum().to_numpy()
    shock = np.abs(ret1) / np.clip(ewma_vol, 1e-9, None)
    f["vol_shock_zscore"] = (pd.Series(shock) - pd.Series(shock).rolling(60).mean()).to_numpy()
    # exposed for family I (regime_state.py, Task 13), which reuses bars_since
    f["_bars_since_last_changepoint_internal"] = bars_since
    return f
