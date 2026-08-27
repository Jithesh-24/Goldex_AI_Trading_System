"""research/phase4_distributional_mechanism.py
Phase 4 Section 18 item 1, mechanism 3/3: rolling skew/kurtosis and a simple
jump-detection flag vs. forward return, using the exact same
MI-vs-shuffled-null methodology and target convention as Phase 3A
(research/phase3a_representation_experiments.py).

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 (training
partition only; the Phase 3 validation split, rows 300,000:400,000, is
never read here).

REPRESENTATION (all computed from bar-to-bar returns using only a trailing
window ending at t-1, so representation[t] never uses return[t] or later --
no look-ahead):
  1. rolling_skew: skewness of the trailing WINDOW returns
  2. rolling_kurtosis: excess kurtosis of the trailing WINDOW returns
  3. jump_flag: 1.0 if |return[t-1]| exceeds JUMP_Z_THRESHOLD times the
     trailing realized volatility (excluding the jump bar itself), else 0.0
MODEL: none (representation-vs-target MI probe, matching Phase 3A).
TARGET: forward_return(closes, horizon=5) -- identical to Phase 3A.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase4_distributional_mechanism.py
"""
import numpy as np
import pandas as pd

from research.phase3a_representation_experiments import (
    TRAINING_ROWS, DATA_PATH, FORWARD_HORIZON, N_BINS, RNG_SEED,
    forward_return, mi_with_shuffle_control,
)

WINDOW = 30
JUMP_Z_THRESHOLD = 3.0


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    return df_training["close"].to_numpy(dtype=np.float64)


def _rolling_moment(returns, window, order, center=True):
    """Rolling standardized moment (skew for order=3, excess kurtosis for
    order=4) of `returns` over a trailing window ending strictly before the
    current index -- rolling_stat[t] is computed from returns[t-window:t],
    never including returns[t]."""
    n = len(returns)
    out = np.full(n, np.nan)
    for t in range(window, n):
        w = returns[t - window:t]
        mean = np.mean(w)
        std = np.std(w)
        if std <= 0:
            out[t] = 0.0
            continue
        m = np.mean(((w - mean) / std) ** order)
        out[t] = m - 3.0 if order == 4 else m  # excess kurtosis subtracts 3
    return out


def jump_detection_flag(returns, window=WINDOW, z_threshold=JUMP_Z_THRESHOLD):
    """1.0 if |returns[t-1]| exceeds z_threshold * trailing realized vol
    (computed from returns[t-1-window:t-1], excluding the candidate jump bar
    itself), else 0.0. Uses only returns strictly before t -- flag[t] never
    depends on returns[t] or later."""
    n = len(returns)
    out = np.full(n, np.nan)
    for t in range(window + 1, n):
        prior_window = returns[t - 1 - window:t - 1]
        vol = np.std(prior_window)
        if vol <= 0:
            out[t] = 0.0
            continue
        out[t] = 1.0 if abs(returns[t - 1]) > z_threshold * vol else 0.0
    return out


def main():
    closes = load_training_closes()
    returns = np.diff(closes, prepend=closes[0])
    print(f"Loaded {len(closes)} training closes from {DATA_PATH} (rows 0:{TRAINING_ROWS})")

    fwd = forward_return(closes, FORWARD_HORIZON)

    representations = {
        "rolling_skew": _rolling_moment(returns, WINDOW, order=3),
        "rolling_excess_kurtosis": _rolling_moment(returns, WINDOW, order=4),
        "jump_detection_flag": jump_detection_flag(returns, WINDOW, JUMP_Z_THRESHOLD),
    }

    n_jumps = int(np.nansum(representations["jump_detection_flag"]))
    print(f"Jump flags fired: {n_jumps} / {len(closes)} bars "
          f"(window={WINDOW}, z_threshold={JUMP_Z_THRESHOLD})")

    print(f"\nForward horizon: {FORWARD_HORIZON} bars. N_BINS={N_BINS}. Shuffle control: 20 permutations.\n")
    print(f"{'representation':45s} {'real_MI(nats)':>14s} {'null_mean':>10s} {'null_std':>10s} {'null_max':>10s}")
    results = {}
    for name, repr_series in representations.items():
        stats = mi_with_shuffle_control(repr_series, fwd, n_bins=N_BINS, n_shuffles=20, seed=RNG_SEED)
        results[name] = stats
        print(f"{name:45s} {stats['real_mi_nats']:14.6f} {stats['null_mi_mean']:10.6f} "
              f"{stats['null_mi_std']:10.6f} {stats['null_mi_max']:10.6f}")

    print("\nInterpretation: real_MI is only meaningfully above noise if it clears")
    print("null_mi_mean + a few null_mi_std, or exceeds null_mi_max.")
    return results


if __name__ == "__main__":
    main()
