"""research/phase3a_raw_path_geometry_probe.py
Phase 3A Section D: raw path-geometry probe. Compares three representations
for predicting short-horizon forward return using a single FIXED,
non-tuned model chosen before looking at any result:
DecisionTreeRegressor(max_depth=4, random_state=42).

Representations compared:
  A. Phase-3 engineered scalars (momentum scalar, multi-scale vol ratio,
     vol-regime transition) -- 3 features.
  B. Raw short price-path window: last PATH_WINDOW normalized returns.
  C. A + B combined.
  Plus a shuffled-label control (fit on shuffled training labels, evaluate
  against the real, unshuffled test labels) to show what "no information"
  looks like for this exact model/split.

Uses ONLY the real training partition (first 300k rows). An internal,
temporally later slice of that partition is carved out as the test set so
the actual Phase 3 validation split (rows 300000:400000) is never touched.
Model choice, hyperparameters, and the internal split point are fixed
before this script is run against results -- not iterated afterward.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase3a_raw_path_geometry_probe.py
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

TRAINING_ROWS = 300_000
DATA_PATH = "data/gold_seed_merged_full6yr.csv"

FORWARD_HORIZON = 5
MOMENTUM_LOOKBACK = 10
PATH_WINDOW = 15
VOL_WINDOWS = (10, 30, 100)

INTERNAL_SPLIT_ROW = 240_000  # internal train/test split WITHIN the training partition
RANDOM_STATE = 42
MAX_DEPTH = 4


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    return df.iloc[:TRAINING_ROWS]["close"].to_numpy(dtype=np.float64)


def build_phase3_scalars(closes):
    returns = np.diff(closes, prepend=closes[0])
    n = len(closes)
    mom = np.full(n, np.nan)
    mom[MOMENTUM_LOOKBACK:] = closes[MOMENTUM_LOOKBACK:] - closes[:-MOMENTUM_LOOKBACK]

    vols = {}
    for w in VOL_WINDOWS:
        vol = np.full(n, np.nan)
        for i in range(w, n):
            vol[i] = np.std(returns[i - w:i])
        vols[w] = vol
    short_w, long_w = min(VOL_WINDOWS), max(VOL_WINDOWS)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = vols[short_w] / vols[long_w]
    vol_ratio[~np.isfinite(vol_ratio)] = np.nan

    finite_short = vols[short_w][np.isfinite(vols[short_w])]
    edges = np.unique(np.quantile(finite_short, np.linspace(0, 1, 4)))
    regime = np.full(n, np.nan)
    valid = np.isfinite(vols[short_w])
    regime[valid] = np.clip(np.digitize(vols[short_w][valid], edges[1:-1]), 0, len(edges) - 2)
    transition = np.full(n, np.nan)
    transition[1:] = regime[1:] - regime[:-1]

    return np.column_stack([mom, vol_ratio, transition])


def build_raw_path_features(closes, window=PATH_WINDOW):
    returns = np.diff(closes, prepend=closes[0])
    n = len(closes)
    feats = np.full((n, window), np.nan)
    for i in range(window, n):
        w = returns[i - window:i]
        std = np.std(w)
        feats[i, :] = w / std if std > 0 else w
    return feats


def forward_return(closes, horizon=FORWARD_HORIZON):
    fwd = np.full(len(closes), np.nan)
    fwd[:-horizon] = closes[horizon:] - closes[:-horizon]
    return fwd


def evaluate(name, X, y, split_row):
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

    print(f"{name:35s} n_train={len(y_train):>7d} n_test={len(y_test):>6d} "
          f"R2={r2:9.5f} dir_acc={dir_acc:.4f}   |   null_R2={null_r2:9.5f} null_dir_acc={null_dir_acc:.4f}")
    return {"r2": r2, "dir_acc": dir_acc, "null_r2": null_r2, "null_dir_acc": null_dir_acc}


def main():
    closes = load_training_closes()
    y = forward_return(closes, FORWARD_HORIZON)
    scalars = build_phase3_scalars(closes)
    raw_path = build_raw_path_features(closes, PATH_WINDOW)
    combined = np.column_stack([scalars, raw_path])

    print(f"Model: DecisionTreeRegressor(max_depth={MAX_DEPTH}, random_state={RANDOM_STATE}), fixed before results.")
    print(f"Internal split row (within training partition): {INTERNAL_SPLIT_ROW} "
          f"(train rows 0:{INTERNAL_SPLIT_ROW}, test rows {INTERNAL_SPLIT_ROW}:{TRAINING_ROWS})")
    print(f"Forward horizon: {FORWARD_HORIZON} bars.\n")

    evaluate("A: Phase-3 engineered scalars (3 feat)", scalars, y, INTERNAL_SPLIT_ROW)
    evaluate(f"B: raw price-path window ({PATH_WINDOW} feat)", raw_path, y, INTERNAL_SPLIT_ROW)
    evaluate("C: combined (A+B)", combined, y, INTERNAL_SPLIT_ROW)


if __name__ == "__main__":
    main()
