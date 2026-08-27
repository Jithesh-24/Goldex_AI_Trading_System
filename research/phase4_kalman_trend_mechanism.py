"""research/phase4_kalman_trend_mechanism.py
Phase 4 Section 18 item 1, mechanism 2/3: a Kalman-filtered price
level/trend estimate vs. forward return, using the exact same
MI-vs-shuffled-null methodology and target convention as Phase 3A
(research/phase3a_representation_experiments.py).

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 (training
partition only; the Phase 3 validation split, rows 300,000:400,000, is
never read here).

REPRESENTATION: a standard local-level-with-trend (constant-velocity)
Kalman filter, state = [level, velocity], observation = close price. Filtered
state at time t uses only observations through t (a proper forward filter,
no smoother/backward pass) -- no look-ahead. Two derived scalars are tested:
  1. the filtered trend/velocity estimate v[t]
  2. the innovation (observation minus one-step-ahead prediction), a
     "surprise" signal
MODEL: none (representation-vs-target MI probe, matching Phase 3A).
TARGET: forward_return(closes, horizon=5) -- identical to Phase 3A.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase4_kalman_trend_mechanism.py
"""
import numpy as np
import pandas as pd

from research.phase3a_representation_experiments import (
    TRAINING_ROWS, DATA_PATH, FORWARD_HORIZON, N_BINS, RNG_SEED,
    forward_return, mi_with_shuffle_control,
)


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    return df_training["close"].to_numpy(dtype=np.float64)


def kalman_level_trend_filter(closes, process_var_level=1e-4, process_var_velocity=1e-6, obs_var=1.0):
    """Constant-velocity Kalman filter over price closes.
    State x = [level, velocity]. Transition: level += velocity each step.
    Returns (filtered_level, filtered_velocity, innovation), all built using
    only observations up to and including the current index (a forward
    filter -- x[t] is the posterior after observing close[t], derived only
    from close[0..t])."""
    n = len(closes)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[process_var_level, 0.0], [0.0, process_var_velocity]])
    R = np.array([[obs_var]])

    x = np.array([closes[0], 0.0])
    P = np.eye(2) * 1.0

    levels = np.empty(n)
    velocities = np.empty(n)
    innovations = np.empty(n)
    levels[0], velocities[0], innovations[0] = x[0], x[1], 0.0

    for t in range(1, n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # Update
        z = closes[t]
        y = z - (H @ x_pred)[0]  # innovation, uses only close[t] and the prior state
        S = (H @ P_pred @ H.T)[0, 0] + R[0, 0]
        K = (P_pred @ H.T) / S
        x = x_pred + (K.flatten() * y)
        P = (np.eye(2) - K @ H) @ P_pred

        levels[t] = x[0]
        velocities[t] = x[1]
        innovations[t] = y

    return levels, velocities, innovations


def main():
    closes = load_training_closes()
    print(f"Loaded {len(closes)} training closes from {DATA_PATH} (rows 0:{TRAINING_ROWS})")

    # obs_var scaled to typical squared 1-bar move so the filter isn't
    # dominated by an arbitrary unit choice.
    typical_move_var = float(np.var(np.diff(closes)))
    levels, velocities, innovations = kalman_level_trend_filter(
        closes, process_var_level=typical_move_var * 0.1,
        process_var_velocity=typical_move_var * 1e-4, obs_var=typical_move_var,
    )

    fwd = forward_return(closes, FORWARD_HORIZON)

    representations = {
        "kalman_filtered_velocity": velocities,
        "kalman_innovation": innovations,
    }

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
