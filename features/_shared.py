"""Common precomputed arrays every family-Batch compute_<family>()
function needs -- built once per call (replay: once per DataFrame; live:
once per bounded window) instead of every family recomputing ret1/sign1/
etc independently. Mirrors research/features_v3.py's former
build_candidate_features() setup block (lines 455-471), unchanged math."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SharedInputs:
    df: pd.DataFrame
    base_feat: pd.DataFrame
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    log_c: np.ndarray
    ret1: np.ndarray
    sign1: np.ndarray
    times: pd.DatetimeIndex
    ewma_vol: np.ndarray
    kalman_resid: np.ndarray
    hurst_120: np.ndarray
    tick_vol: np.ndarray
    spread: np.ndarray


def build_shared_inputs(df: pd.DataFrame, base_feat: pd.DataFrame) -> SharedInputs:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    log_c = np.log(c)
    ret1 = np.diff(log_c, prepend=log_c[0])
    ret1[0] = np.nan
    sign1 = np.nan_to_num(np.sign(ret1), nan=0.0)
    times = pd.to_datetime(df["time"].to_numpy())
    ewma_vol = base_feat["ewma_vol"].to_numpy(dtype=np.float64)
    kalman_resid = base_feat["kalman_residual_z"].to_numpy(dtype=np.float64)
    hurst_120 = base_feat["hurst_120"].to_numpy(dtype=np.float64)
    tick_vol = df["tick_volume"].to_numpy(dtype=np.float64) if "tick_volume" in df.columns else np.full(len(df), np.nan)
    spread = df["spread"].to_numpy(dtype=np.float64) if "spread" in df.columns else np.full(len(df), np.nan)
    return SharedInputs(df=df, base_feat=base_feat, o=o, h=h, l=l, c=c, log_c=log_c,
                         ret1=ret1, sign1=sign1, times=times, ewma_vol=ewma_vol,
                         kalman_resid=kalman_resid, hurst_120=hurst_120,
                         tick_vol=tick_vol, spread=spread)
