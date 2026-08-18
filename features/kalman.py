"""
1D local-level Kalman filter, hand-rolled (no filterpy dependency needed for
something this small). State = latent "true" price level, observation =
close. Adaptive alternative to a moving average: the gain adjusts itself
bar-by-bar based on the filter's own uncertainty instead of a fixed lookback
window, so it doesn't lag a fixed-N-bar MA does on a regime change.

Causal by construction: state at bar i uses only observations <= i.
"""
import numba
import numpy as np


@numba.njit(cache=True)
def kalman_local_level(price: np.ndarray, q: float, r: float):
    """q = process (state) variance, r = observation (measurement) variance.
    Higher q/r ratio -> filter trusts new observations more -> tracks faster
    but noisier; lower -> smoother but laggier. Returns (level, velocity,
    residual) arrays, same length as price. `residual` = price - level,
    a mean-reverting-ish signal (distance from the filtered trend).
    """
    n = len(price)
    level = np.empty(n, dtype=np.float64)
    velocity = np.empty(n, dtype=np.float64)
    residual = np.empty(n, dtype=np.float64)

    # 2-state (level, velocity) constant-velocity model
    x0, x1 = price[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    level[0] = x0
    velocity[0] = x1
    residual[0] = 0.0

    for i in range(1, n):
        # predict
        x0_pred = x0 + x1
        x1_pred = x1
        p00_pred = p00 + p01 + p10 + p11 + q
        p01_pred = p01 + p11
        p10_pred = p10 + p11
        p11_pred = p11 + q

        # update (observe price[i])
        y = price[i] - x0_pred
        s = p00_pred + r
        k0 = p00_pred / s
        k1 = p10_pred / s

        x0 = x0_pred + k0 * y
        x1 = x1_pred + k1 * y

        p00 = (1 - k0) * p00_pred
        p01 = (1 - k0) * p01_pred
        p10 = p10_pred - k1 * p00_pred
        p11 = p11_pred - k1 * p01_pred

        level[i] = x0
        velocity[i] = x1
        residual[i] = y

    return level, velocity, residual
