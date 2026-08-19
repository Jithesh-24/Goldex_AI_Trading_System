"""Family B -- volatility dynamics candidates (beyond the baseline
EWMA/GK/RS/YZ/bipower/jump in features/volatility.py, which stays
untouched). Moved from research/features_v3.py lines 501-519."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def compute_volatility_dynamics(shared: SharedInputs) -> dict:
    ret1_s = pd.Series(shared.ret1)
    ewma_vol = shared.ewma_vol
    times = shared.times
    base_feat = shared.base_feat
    f = {}
    f["realized_variance_20"] = (ret1_s ** 2).rolling(20).sum().to_numpy()
    ret1_up = pd.Series(np.where(shared.ret1 > 0, shared.ret1, 0.0))
    ret1_down = pd.Series(np.where(shared.ret1 < 0, shared.ret1, 0.0))
    f["realized_semivar_upside_20"] = (ret1_up ** 2).rolling(20).sum().to_numpy()
    f["realized_semivar_downside_20"] = (ret1_down ** 2).rolling(20).sum().to_numpy()
    log_hl = np.log(shared.h / shared.l)
    f["parkinson_vol_60"] = np.sqrt(pd.Series(log_hl ** 2).rolling(60).mean().to_numpy() / (4 * np.log(2)))
    f["vol_acceleration_30"] = ewma_vol - pd.Series(ewma_vol).shift(30).to_numpy()
    f["vol_of_vol_60"] = pd.Series(ewma_vol).rolling(60).std().to_numpy()
    ev_daily = pd.Series(ewma_vol, index=times).resample("1D").last()
    vol_pctile_252 = ev_daily.rolling(252, min_periods=60).rank(pct=True).shift(1)
    f["vol_percentile_252"] = vol_pctile_252.reindex(times, method="ffill").to_numpy()
    ev_roll_mean_60 = pd.Series(ewma_vol).rolling(60).mean()
    ev_roll_std_60 = pd.Series(ewma_vol).rolling(60).std()
    f["vol_zscore_60"] = ((pd.Series(ewma_vol) - ev_roll_mean_60) / ev_roll_std_60).to_numpy()
    gk20 = base_feat["gk_vol_20"].to_numpy()
    gk240 = base_feat["gk_vol_240"].to_numpy()
    f["vol_compression_ratio"] = np.where(gk240 > 1e-12, gk20 / gk240, np.nan)
    return f
