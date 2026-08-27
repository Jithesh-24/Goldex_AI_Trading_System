"""research/phase4_mechanism_oos_check.py
Phase 4 Stage 1 follow-up: applies Phase 3A's Section D out-of-sample
treatment (research/phase3a_raw_path_geometry_probe.py) to the three new
representation families validated by Stage 1's marginal-MI probes --
GARCH(1,1) conditional variance, Kalman filtered velocity/innovation, and
rolling skew/excess kurtosis -- to test whether their large raw MI (Stage 1)
survives a real chronological train/test predictive check, or evaporates the
same way Phase 3A's momentum/path representations did.

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 (training
partition only; the Phase 3 validation split, rows 300,000:400,000, is never
read here). This matches all Phase 3A/Phase 4 Stage 1 scripts.

MODEL: DecisionTreeRegressor(max_depth=4, random_state=42) -- identical,
fixed configuration to research/phase3a_raw_path_geometry_probe.py. Chosen
before looking at any result here; not tuned against the test split.

TRAIN/TEST SPLIT: internal chronological split within the training
partition -- rows 0:240,000 train, rows 240,000:300,000 test. Same split
point as Phase 3A's Section D script. The real Phase 3 validation split
(rows 300,000:400,000) is never touched.

TARGET: forward_return(closes, horizon=5) -- identical convention to Phase
3A and all three Stage 1 mechanism scripts.

Representations tested (reusing each Stage 1 script's fitting/filtering
function directly, not reimplemented):
  1. GARCH(1,1) conditional variance (sigma2) alone
  2. Kalman filtered velocity + innovation (2 features)
  3. Rolling skew + excess kurtosis (2 features)
  4. All five features combined (sigma2, velocity, innovation, skew, kurtosis)

Each is evaluated against a shuffled-label-null control using an
identically-configured fresh model, exactly as in Phase 3A's Section D.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase4_mechanism_oos_check.py
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from research.phase3a_representation_experiments import (
    TRAINING_ROWS, DATA_PATH, FORWARD_HORIZON, forward_return,
)
from research.phase4_garch_volatility_mechanism import fit_garch11
from research.phase4_kalman_trend_mechanism import kalman_level_trend_filter
from research.phase4_distributional_mechanism import _rolling_moment, WINDOW

INTERNAL_SPLIT_ROW = 240_000  # internal train/test split WITHIN the training partition
RANDOM_STATE = 42
MAX_DEPTH = 4


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    return df.iloc[:TRAINING_ROWS]["close"].to_numpy(dtype=np.float64)


def evaluate(name, X, y, split_row):
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    row_idx = np.where(valid)[0]
    train_mask = row_idx < split_row
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    model = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    dir_acc = float(np.mean(np.sign(pred) == np.sign(y_test)))

    rng = np.random.default_rng(RANDOM_STATE)
    y_train_shuffled = rng.permutation(y_train)
    null_model = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    null_model.fit(X_train, y_train_shuffled)
    null_pred = null_model.predict(X_test)
    null_r2 = r2_score(y_test, null_pred)
    null_dir_acc = float(np.mean(np.sign(null_pred) == np.sign(y_test)))

    print(f"{name:45s} n_train={len(y_train):>7d} n_test={len(y_test):>6d} "
          f"R2={r2:9.5f} dir_acc={dir_acc:.4f}   |   null_R2={null_r2:9.5f} null_dir_acc={null_dir_acc:.4f}")
    return {"r2": r2, "dir_acc": dir_acc, "null_r2": null_r2, "null_dir_acc": null_dir_acc}


def main():
    closes = load_training_closes()
    returns = np.diff(closes, prepend=closes[0])
    y = forward_return(closes, FORWARD_HORIZON)

    print(f"Loaded {len(closes)} training closes from {DATA_PATH} (rows 0:{TRAINING_ROWS})")

    # Reuse Stage 1's fitting/filtering functions directly -- not reimplemented.
    (omega, alpha, beta), sigma2 = fit_garch11(returns)
    print(f"Fitted GARCH(1,1): omega={omega:.6g} alpha={alpha:.4f} beta={beta:.4f} "
          f"(persistence alpha+beta={alpha + beta:.4f})")

    typical_move_var = float(np.var(np.diff(closes)))
    _, velocity, innovation = kalman_level_trend_filter(
        closes, process_var_level=typical_move_var * 0.1,
        process_var_velocity=typical_move_var * 1e-4, obs_var=typical_move_var,
    )

    skew = _rolling_moment(returns, WINDOW, order=3)
    kurt = _rolling_moment(returns, WINDOW, order=4)

    garch_repr = sigma2.reshape(-1, 1)
    kalman_repr = np.column_stack([velocity, innovation])
    dist_repr = np.column_stack([skew, kurt])
    combined_repr = np.column_stack([sigma2, velocity, innovation, skew, kurt])

    print(f"\nModel: DecisionTreeRegressor(max_depth={MAX_DEPTH}, random_state={RANDOM_STATE}), fixed before results.")
    print(f"Internal split row (within training partition): {INTERNAL_SPLIT_ROW} "
          f"(train rows 0:{INTERNAL_SPLIT_ROW}, test rows {INTERNAL_SPLIT_ROW}:{TRAINING_ROWS})")
    print(f"Forward horizon: {FORWARD_HORIZON} bars.\n")

    results = {}
    results["garch_sigma2"] = evaluate("GARCH(1,1) conditional variance (1 feat)", garch_repr, y, INTERNAL_SPLIT_ROW)
    results["kalman_velocity_innovation"] = evaluate("Kalman velocity + innovation (2 feat)", kalman_repr, y, INTERNAL_SPLIT_ROW)
    results["skew_kurtosis"] = evaluate("Rolling skew + excess kurtosis (2 feat)", dist_repr, y, INTERNAL_SPLIT_ROW)
    results["combined_all5"] = evaluate("Combined: all 5 features", combined_repr, y, INTERNAL_SPLIT_ROW)

    return results


if __name__ == "__main__":
    main()
