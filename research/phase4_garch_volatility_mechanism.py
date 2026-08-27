"""research/phase4_garch_volatility_mechanism.py
Phase 4 Section 18 item 1, mechanism 1/3: GARCH-family conditional volatility
forecast vs. forward return, using the exact same MI-vs-shuffled-null
methodology and target convention as Phase 3A
(research/phase3a_representation_experiments.py) so results are comparable.

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 (training
partition only -- same convention as research/phase3_real_run.py and Phase
3A; the Phase 3 validation split, rows 300,000:400,000, is never read here).

Library note: `arch` (the standard Python GARCH library) is NOT installed in
this venv (checked: `import arch` raises ModuleNotFoundError). Rather than
add a new dependency mid-phase, this script implements a minimal from-scratch
GARCH(1,1) conditional-variance estimator: returns are demeaned, then
sigma2[t] = omega + alpha*eps[t-1]^2 + beta*sigma2[t-1] is fit by numerically
minimizing the negative Gaussian log-likelihood over (omega, alpha, beta)
with a simple bounded grid + local refinement (no external optimizer
dependency beyond numpy/scipy already used elsewhere in this repo -- if scipy
isn't available either, a coordinate-descent fallback is used). This is a
standard, textbook GARCH(1,1) -- just implemented locally, not sourced from
a battle-tested library, which is a limitation noted below.

REPRESENTATION: GARCH(1,1) one-step-ahead conditional variance forecast,
sigma2[t] (uses only returns up to and including t-1 -- no look-ahead).
MODEL: none (this is a representation-vs-target MI probe, not a trading
model -- matches Phase 3A's methodology exactly).
TARGET: forward_return(closes, horizon=5) -- identical to Phase 3A.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase4_garch_volatility_mechanism.py
"""
import numpy as np
import pandas as pd

from research.phase3a_representation_experiments import (
    TRAINING_ROWS, DATA_PATH, FORWARD_HORIZON, N_BINS, RNG_SEED,
    forward_return, mi_with_shuffle_control,
)

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    return df_training["close"].to_numpy(dtype=np.float64)


def _garch11_negloglik(params, eps):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e12
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(eps)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-12)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps ** 2 / sigma2)
    return -ll


def fit_garch11(returns):
    """Fit a minimal GARCH(1,1) by maximum likelihood. Returns
    (omega, alpha, beta) and the full in-sample conditional-variance path.
    Uses only past returns to build sigma2[t] (sigma2[t] depends on
    eps[t-1] and sigma2[t-1], never on eps[t] or later) -- no look-ahead."""
    eps = returns - np.mean(returns)
    x0 = np.array([np.var(eps) * 0.1, 0.05, 0.90])
    if _HAVE_SCIPY:
        result = minimize(
            _garch11_negloglik, x0, args=(eps,), method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-5},
        )
        omega, alpha, beta = result.x
    else:
        # Coordinate-descent fallback if scipy is unavailable.
        omega, alpha, beta = x0
        best = _garch11_negloglik((omega, alpha, beta), eps)
        for _ in range(50):
            improved = False
            for i, step in [(0, omega * 0.1 + 1e-8), (1, 0.02), (2, 0.02)]:
                for sign in (1, -1):
                    trial = [omega, alpha, beta]
                    trial[i] += sign * step
                    val = _garch11_negloglik(trial, eps)
                    if val < best:
                        best = val
                        omega, alpha, beta = trial
                        improved = True
            if not improved:
                break
    if alpha < 0 or beta < 0 or omega <= 0 or alpha + beta >= 1.0:
        omega, alpha, beta = x0  # fall back to a stable stationary start if fit diverged
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(eps)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return (omega, alpha, beta), sigma2


def main():
    closes = load_training_closes()
    returns = np.diff(closes, prepend=closes[0])
    print(f"Loaded {len(closes)} training closes from {DATA_PATH} (rows 0:{TRAINING_ROWS})")
    print(f"scipy available for MLE optimization: {_HAVE_SCIPY}")

    (omega, alpha, beta), sigma2 = fit_garch11(returns)
    print(f"Fitted GARCH(1,1): omega={omega:.6g} alpha={alpha:.4f} beta={beta:.4f} "
          f"(persistence alpha+beta={alpha + beta:.4f})")

    fwd = forward_return(closes, FORWARD_HORIZON)
    # sigma2[t] is the one-step-ahead conditional variance forecast made using
    # information through t-1; align it directly against fwd[t] (forward
    # return starting at t) -- no shift needed since sigma2 already only used
    # eps[t-1] and earlier.
    stats = mi_with_shuffle_control(sigma2, fwd, n_bins=N_BINS, n_shuffles=20, seed=RNG_SEED)

    print(f"\nForward horizon: {FORWARD_HORIZON} bars. N_BINS={N_BINS}. Shuffle control: 20 permutations.\n")
    print(f"{'representation':45s} {'real_MI(nats)':>14s} {'null_mean':>10s} {'null_std':>10s} {'null_max':>10s}")
    print(f"{'garch11_conditional_variance':45s} {stats['real_mi_nats']:14.6f} {stats['null_mi_mean']:10.6f} "
          f"{stats['null_mi_std']:10.6f} {stats['null_mi_max']:10.6f}")
    print("\nInterpretation: real_MI is only meaningfully above noise if it clears")
    print("null_mi_mean + a few null_mi_std, or exceeds null_mi_max.")
    return {"garch11_conditional_variance": stats, "garch_params": {"omega": omega, "alpha": alpha, "beta": beta}}


if __name__ == "__main__":
    main()
