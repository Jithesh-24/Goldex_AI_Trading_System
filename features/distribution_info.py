"""Family D -- distribution / information theory. Moved from
research/features_v3.py lines 132-265 (kernels) and 555-563 (assembly),
math unchanged."""
import numba
import numpy as np
import pandas as pd

from features._shared import SharedInputs


@numba.njit(cache=True)
def shannon_entropy_returns(ret, window, nbins):
    n = len(ret)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = ret[i - window:i]
        lo = seg.min()
        hi = seg.max()
        rng = hi - lo
        if rng < 1e-15:
            out[i] = 0.0
            continue
        counts = np.zeros(nbins, dtype=np.float64)
        for j in range(window):
            b = int((seg[j] - lo) / rng * nbins)
            if b >= nbins:
                b = nbins - 1
            counts[b] += 1
        h = 0.0
        for c in counts:
            if c > 0:
                p = c / window
                h -= p * np.log2(p)
        out[i] = h
    return out


@numba.njit(cache=True)
def permutation_entropy(ret, window, order):
    """Bandt-Pompe permutation entropy, order=3 (6 possible orderings of 3
    consecutive values) -- ordinal-pattern complexity, distinct from the
    value-distribution entropy above (this is invariant to monotonic
    rescaling, sensitive only to up/down ORDER patterns)."""
    n = len(ret)
    out = np.full(n, np.nan)
    n_patterns = 1
    for k in range(2, order + 1):
        n_patterns *= k  # order! possible orderings
    for i in range(window, n):
        seg = ret[i - window:i]
        counts = np.zeros(n_patterns, dtype=np.float64)
        n_windows = window - order + 1
        for j in range(n_windows):
            a, b, c = seg[j], seg[j + 1], seg[j + 2]
            # rank the 3 values into one of 6 patterns (order=3 fixed here)
            if a < b:
                if b < c:
                    p = 0
                elif a < c:
                    p = 1
                else:
                    p = 2
            else:
                if a < c:
                    p = 3
                elif b < c:
                    p = 4
                else:
                    p = 5
            counts[p] += 1
        h = 0.0
        for cnt in counts:
            if cnt > 0:
                p = cnt / n_windows
                h -= p * np.log2(p)
        out[i] = h / np.log2(n_patterns)  # normalized to [0,1]
    return out


@numba.njit(cache=True)
def sample_entropy(ret, window, m, r_mult):
    """Sample entropy, deliberately small window (20, not 60+) -- SampEn is
    an O(window^2) estimator and unstable on short samples; a 60-bar window
    would be both slow at 2.4M-row scale and a noisy estimate, so this uses
    the smaller window the estimator is actually reliable at (matches the
    library's own 'avoid unstable estimators when sample size is
    inadequate' instruction)."""
    n = len(ret)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = ret[i - window:i]
        r = r_mult * seg.std()
        if r < 1e-15:
            out[i] = 0.0
            continue
        nm = window - m
        nm1 = window - m - 1
        count_m = 0
        count_m1 = 0
        for a in range(nm):
            for b in range(a + 1, nm):
                match_m = True
                for k in range(m):
                    if abs(seg[a + k] - seg[b + k]) > r:
                        match_m = False
                        break
                if match_m:
                    count_m += 1
                    if a < nm1 and b < nm1 and abs(seg[a + m] - seg[b + m]) <= r:
                        count_m1 += 1
        if count_m == 0 or count_m1 == 0:
            out[i] = np.nan
        else:
            out[i] = -np.log(count_m1 / count_m)
    return out


@numba.njit(cache=True)
def mi_proxy_sign_lag(sign, lag, window):
    """Cheap causal mutual-information proxy: plug-in MI estimate between
    sign(ret_1) and its own value `lag` bars earlier, over the trailing
    window, via a 3x3 (up/down/flat x up/down/flat) contingency table --
    avoids an expensive kernel-density MI estimator while still answering
    'is there information in the lagged sign beyond linear correlation'."""
    n = len(sign)
    out = np.full(n, np.nan)
    for i in range(window + lag, n):
        joint = np.zeros((3, 3))
        for j in range(i - window, i):
            a = sign[j]
            b = sign[j - lag]
            ai = 0 if a > 0 else (1 if a < 0 else 2)
            bi = 0 if b > 0 else (1 if b < 0 else 2)
            joint[ai, bi] += 1.0
        joint /= window
        pa = joint.sum(axis=1)
        pb = joint.sum(axis=0)
        mi = 0.0
        for x in range(3):
            for y in range(3):
                if joint[x, y] > 1e-12 and pa[x] > 1e-12 and pb[y] > 1e-12:
                    mi += joint[x, y] * np.log2(joint[x, y] / (pa[x] * pb[y]))
        out[i] = mi
    return out


def compute_distribution_info(shared: SharedInputs) -> dict:
    ret1, sign1 = shared.ret1, shared.sign1
    ret1_s = pd.Series(ret1)
    f = {}
    ret_std_60 = ret1_s.rolling(60).std()
    f["tail_probability_60"] = (np.abs(ret1_s) > 2 * ret_std_60).rolling(60).mean().to_numpy()
    f["shannon_entropy_returns_60"] = shannon_entropy_returns(ret1, 60, 8)
    f["permutation_entropy_60"] = permutation_entropy(ret1, 60, 3)
    f["sample_entropy_20"] = sample_entropy(ret1, 20, 2, 0.2)
    r2 = ret1_s ** 2
    f["return_concentration_60"] = ((r2 ** 2).rolling(60).sum() / (r2.rolling(60).sum() ** 2)).to_numpy()
    f["mi_proxy_sign_lag5_240"] = mi_proxy_sign_lag(sign1, 5, 240)
    return f
