"""Family H -- historical microstructure, honestly scoped: tick_volume +
spread only (no tick stream, no order book in the 6.7yr CSV -- see spec
section 2/Task 1's real historical_coverage.py measurement for exact
tick_volume degradation date and spread's near-constant-98.9%-of-history
fact). Moved from research/features_v3.py lines 643-653."""
import pandas as pd

from features._shared import SharedInputs


def compute_microstructure_history(shared: SharedInputs) -> dict:
    tv_s = pd.Series(shared.tick_vol)
    sp_s = pd.Series(shared.spread)
    times = shared.times
    f = {}
    f["tick_volume_zscore_60"] = ((tv_s - tv_s.rolling(60).mean()) / tv_s.rolling(60).std()).to_numpy()
    f["tick_volume_accel_20"] = (tv_s.rolling(20).mean() - tv_s.rolling(20).mean().shift(20)).to_numpy()
    f["spread_change_1"] = sp_s.diff().to_numpy()
    sp_daily = sp_s.copy(); sp_daily.index = times
    spread_pctile = sp_daily.resample("1D").last().rolling(252, min_periods=60).rank(pct=True).shift(1)
    f["spread_percentile_252"] = spread_pctile.reindex(times, method="ffill").to_numpy()
    f["spread_volatility_60"] = sp_s.rolling(60).std().to_numpy()
    f["tick_volume_spread_ratio"] = (tv_s / (sp_s + 1.0)).to_numpy()
    return f
