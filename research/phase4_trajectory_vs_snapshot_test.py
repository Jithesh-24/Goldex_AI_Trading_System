"""research/phase4_trajectory_vs_snapshot_test.py
Phase 4 Section 18 item 2: does a short sequence window of market-state
vectors (a full trajectory) carry more information about eventual trade
outcome than the single decision-time snapshot Phase 3A tested?

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 (training
partition only). A local, fixed, rule-based decider (below) generates real
decisions/trajectories on this data via simulator.replay.run_replay -- this
is NOT a new roster candidate (it lives only in this script, is not added to
candidates/ or any tournament roster, and is not wired into
research/phase2_tournament.py or research/phase3_tournament.py). It exists
only to produce real DECIDE/MANAGE/POSITION_CLOSED records with
observation_features and decision_id set, so the trajectory-assembly module
(research/phase4_trajectory_assembly.py) has something real to assemble.

REPRESENTATION:
  (1) single-snapshot: the DECIDE record's observation_features vector alone.
  (2) full-trajectory: the same vector, plus a summary (mean, std, last) of
      the observation_features vectors from every MANAGE record recorded
      while the position was open, plus the number of MANAGE steps -- this
      is the concatenated/summarized full sequence, causal (it only uses
      information seen up to the close of the trade, same information the
      terminal label itself is derived from -- there is no forecasting
      claim here, this tests information CONTENT of the sequence vs the
      snapshot about an outcome that has, by construction, already happened
      by the time both representations are fully observed).
MODEL: sklearn LogisticRegression(), default/fixed hyperparameters, no
tuning -- same "fixed simple model" discipline as Phase 3A's shallow-tree
probe.
TARGET: sign of realized_pnl (1 if positive, 0 otherwise) for each closed
trade trajectory.
TRAIN/VALIDATION SPLIT: chronological, 80% earliest trajectories / 20% latest
trajectories, by decide_timestamp -- carved entirely within the training
partition (rows 0:300,000). The Phase 3 validation split (rows
300,000:400,000) is never touched.
NULL CONTROL: for each representation, labels are shuffled on the training
split, the same fixed model is retrained on the shuffled labels, and
evaluated on the (real) test split, repeated 20 times -- the real model's
test accuracy is judged against this null distribution, not against 50%
alone (class balance may not be exactly 50/50).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase4_trajectory_vs_snapshot_test.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.replay import run_replay
from research.phase4_trajectory_assembly import assemble_trajectories

TRAINING_ROWS = 300_000
DATA_PATH = "data/gold_seed_merged_full6yr.csv"
Z_LOOKBACK = 20
Z_THRESHOLD = 1.5
TRAIN_FRACTION = 0.8
N_NULL_SHUFFLES = 20
RNG_SEED = 42


class _FixedRuleZScoreDecider:
    """Local, fixed, rule-based decider for generating real trajectories only
    -- deliberately the same mechanism family as
    candidates/statistical_null.py's MomentumMeanReversionCandidate (a
    transparent z-score mean-reversion rule), reimplemented locally rather
    than importing that candidate directly so this script never needs to
    touch or wire into candidates/ or any tournament roster. Exposes
    observation_features via the last_decision_features convention
    simulator/replay.py already reads (Phase 3A)."""

    def __init__(self, lookback=Z_LOOKBACK, z_threshold=Z_THRESHOLD):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self._closes = []
        self.last_decision_features = None

    def _features(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > self.lookback:
                self._closes.pop(0)
        if len(self._closes) < self.lookback:
            return None
        arr = np.asarray(self._closes)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std <= 0:
            return None
        z = (market_state.mid - mean) / std
        return {"z": z, "rolling_mean": mean, "rolling_std": std}

    def decide(self, market_state, account):
        feats = self._features(market_state)
        self.last_decision_features = feats
        if feats is None:
            return ("NO_TRADE", None, None)
        z = feats["z"]
        if z <= -self.z_threshold:
            return ("LONG", None, None)
        if z >= self.z_threshold:
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        feats = self._features(market_state)
        self.last_decision_features = feats
        if feats is None:
            return "HOLD"
        if abs(feats["z"]) <= self.z_threshold / 2.0:
            return "EXIT"
        return "HOLD"


def load_training_df():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    return df.iloc[:TRAINING_ROWS].reset_index(drop=True)


def snapshot_vector(features):
    if features is None:
        return None
    return np.array([features["z"], features["rolling_mean"], features["rolling_std"]], dtype=np.float64)


def trajectory_vector(trajectory):
    snap = snapshot_vector(trajectory.decide_observation_features)
    if snap is None:
        return None
    manage_vecs = [snapshot_vector(f) for f in trajectory.manage_observation_sequence]
    manage_vecs = [v for v in manage_vecs if v is not None]
    if manage_vecs:
        stacked = np.stack(manage_vecs)
        summary = np.concatenate([stacked.mean(axis=0), stacked.std(axis=0), stacked[-1]])
    else:
        # No MANAGE steps before close -- pad with zeros rather than dropping
        # the trajectory, so both representations are evaluated on the same
        # set of trades.
        summary = np.zeros(9)
    return np.concatenate([snap, summary, [len(trajectory.manage_observation_sequence)]])


def build_dataset(trajectories):
    trajectories = sorted(trajectories, key=lambda t: t.decide_timestamp)
    snap_rows, traj_rows, labels, timestamps = [], [], [], []
    for t in trajectories:
        if t.decide_observation_features is None or t.realized_pnl is None:
            continue
        snap_v = snapshot_vector(t.decide_observation_features)
        traj_v = trajectory_vector(t)
        if snap_v is None or traj_v is None:
            continue
        snap_rows.append(snap_v)
        traj_rows.append(traj_v)
        labels.append(1 if t.realized_pnl > 0 else 0)
        timestamps.append(t.decide_timestamp)
    return np.array(snap_rows), np.array(traj_rows), np.array(labels), timestamps


def chronological_split(n, train_fraction=TRAIN_FRACTION):
    split = int(n * train_fraction)
    return split


def evaluate_with_null(X, y, split, n_shuffles=N_NULL_SHUFFLES, seed=RNG_SEED):
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    if len(np.unique(y_train)) < 2 or len(X_test) == 0:
        return None

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    real_acc = float(model.score(X_test, y_test))

    rng = np.random.default_rng(seed)
    null_accs = []
    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y_train)
        if len(np.unique(y_shuffled)) < 2:
            continue
        null_model = LogisticRegression(max_iter=1000)
        null_model.fit(X_train, y_shuffled)
        null_accs.append(float(null_model.score(X_test, y_test)))

    return {
        "real_test_accuracy": real_acc,
        "null_mean_accuracy": float(np.mean(null_accs)) if null_accs else None,
        "null_std_accuracy": float(np.std(null_accs)) if null_accs else None,
        "n_test": len(X_test), "n_train": len(X_train),
        "positive_fraction_test": float(np.mean(y_test)),
    }


def main():
    df = load_training_df()
    print(f"Loaded {len(df)} training rows from {DATA_PATH} (rows 0:{TRAINING_ROWS})")

    decider = _FixedRuleZScoreDecider()
    config = SimulatedExecutionConfig()
    recorder = run_replay(df, decider.decide, decider.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    trajectories = assemble_trajectories(recorder.all_records())
    print(f"Assembled {len(trajectories)} real trade trajectories from the fixed z-score rule decider")

    snap_X, traj_X, y, timestamps = build_dataset(trajectories)
    n = len(y)
    print(f"Usable trajectories with complete features: {n} "
          f"(positive-outcome fraction: {float(np.mean(y)) if n else float('nan'):.4f})")
    if n < 40:
        print("Too few real trajectories for a meaningful chronological train/test split -- reporting as-is.")

    split = chronological_split(n)
    print(f"\nChronological split: train=[0:{split}] test=[{split}:{n}] (80/20, earliest-first)\n")

    snap_result = evaluate_with_null(snap_X, y, split)
    traj_result = evaluate_with_null(traj_X, y, split)

    print(f"{'representation':20s} {'test_acc':>10s} {'null_mean':>10s} {'null_std':>10s} {'n_train':>8s} {'n_test':>7s}")
    for name, result in [("single_snapshot", snap_result), ("full_trajectory", traj_result)]:
        if result is None:
            print(f"{name:20s} -- insufficient data for a valid split (e.g. single-class training labels)")
            continue
        print(f"{name:20s} {result['real_test_accuracy']:10.4f} {result['null_mean_accuracy']:10.4f} "
              f"{result['null_std_accuracy']:10.4f} {result['n_train']:8d} {result['n_test']:7d}")

    print("\nInterpretation: a representation's real test accuracy is only meaningfully")
    print("above null if it clears null_mean + a few null_std. Compare snapshot vs.")
    print("trajectory results directly -- neither being above null is a valid, complete")
    print("negative outcome (see Section 18/28 of the Phase 4 design doc).")
    return {"single_snapshot": snap_result, "full_trajectory": traj_result, "n_trajectories": n}


if __name__ == "__main__":
    main()
