"""Family G -- time/session encoding (UTC; MT5 server-time offset not
modeled -- relative encodings only). Moved from research/features_v3.py
lines 612-641. NOTE: session_asian/london/ny below are generic UTC hour
bands, a DIFFERENT and independent concept from
market/state_engine.py's is_market_closed() (the empirically-derived XM
GOLD.i# open/closed schedule) -- this family encodes WHICH session,
is_market_closed answers WHETHER the market is open at all. Not merged."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def compute_temporal(shared: SharedInputs) -> dict:
    times, ewma_vol, tick_vol, base_feat = shared.times, shared.ewma_vol, shared.tick_vol, shared.base_feat
    hour = times.hour.to_numpy(dtype=np.float64)
    minute = times.minute.to_numpy(dtype=np.float64)
    dow = times.dayofweek.to_numpy(dtype=np.float64)
    f = {}
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    f["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["session_asian"] = ((hour >= 0) & (hour < 8)).astype(np.float64)
    f["session_london"] = ((hour >= 8) & (hour < 16)).astype(np.float64)
    f["session_ny"] = ((hour >= 13) & (hour < 21)).astype(np.float64)
    f["session_london_ny_overlap"] = ((hour >= 13) & (hour < 16)).astype(np.float64)
    session_id = np.select(
        [f["session_london_ny_overlap"] > 0, f["session_london"] > 0, f["session_ny"] > 0, f["session_asian"] > 0],
        [3, 1, 2, 0], default=0)
    session_change = pd.Series(session_id).diff().abs() > 0
    f["session_transition_flag"] = session_change.astype(np.float64).to_numpy()
    hour_key = pd.Series(hour, index=times)
    ev_by_hour_ref = pd.Series(ewma_vol, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["vol_conditional_on_session"] = (pd.Series(ewma_vol) - ev_by_hour_ref.to_numpy()).to_numpy()
    ret5 = base_feat["ret_5"].to_numpy()
    ret5_by_hour_ref = pd.Series(ret5, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["ret_conditional_on_session"] = (pd.Series(ret5) - ret5_by_hour_ref.to_numpy()).to_numpy()
    tv_by_hour_ref = pd.Series(tick_vol, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["activity_conditional_on_session"] = (pd.Series(tick_vol) - tv_by_hour_ref.to_numpy()).to_numpy()
    return f
