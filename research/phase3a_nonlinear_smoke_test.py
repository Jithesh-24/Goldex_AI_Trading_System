"""research/phase3a_nonlinear_smoke_test.py
Phase 3A Section E: a single modest nonlinear model smoke test -- can more
model capacity extract anything from the richer (combined engineered +
raw-path) representation that the fixed linear/tree probes in Sections B/D
could not? This is a probe, not a candidate: it is not wired into
research/phase2_tournament.py's verdict machinery, not added to the
candidate roster, and is never run against the real Phase 3 validation
split (rows 300000:400000) -- it reuses the exact same internal,
temporally-later slice of the training partition as Section D.

Model: sklearn.ensemble.HistGradientBoostingRegressor (already available in
this venv, no new dependency), fixed hyperparameters decided before looking
at any result: max_depth=4, max_iter=100, learning_rate=0.05,
random_state=42. Compared against a shuffled-label control and against
Section D's fixed-tree R2/direction-accuracy numbers as a baseline
reference point.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase3a_nonlinear_smoke_test
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from research.phase3a_raw_path_geometry_probe import (
    load_training_closes, build_phase3_scalars, build_raw_path_features, forward_return,
    TRAINING_ROWS, FORWARD_HORIZON, PATH_WINDOW, INTERNAL_SPLIT_ROW,
)

RANDOM_STATE = 42
MAX_DEPTH = 4
MAX_ITER = 100
LEARNING_RATE = 0.05


def evaluate(name, X, y, split_row):
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    row_idx = np.where(valid)[0]
    train_mask = row_idx < split_row
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    model = HistGradientBoostingRegressor(
        max_depth=MAX_DEPTH, max_iter=MAX_ITER, learning_rate=LEARNING_RATE, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    dir_acc = float(np.mean(np.sign(pred) == np.sign(y_test)))

    rng = np.random.default_rng(RANDOM_STATE)
    y_train_shuffled = rng.permutation(y_train)
    null_model = HistGradientBoostingRegressor(
        max_depth=MAX_DEPTH, max_iter=MAX_ITER, learning_rate=LEARNING_RATE, random_state=RANDOM_STATE
    )
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

    print(f"Model: HistGradientBoostingRegressor(max_depth={MAX_DEPTH}, max_iter={MAX_ITER}, "
          f"learning_rate={LEARNING_RATE}, random_state={RANDOM_STATE}), fixed before results.")
    print(f"Internal split row (within training partition, same as Section D): {INTERNAL_SPLIT_ROW}")
    print(f"Forward horizon: {FORWARD_HORIZON} bars.\n")
    print("Reference (Section D, fixed decision tree, max_depth=4): combined R2=-0.00538, dir_acc=0.4838\n")

    evaluate("Combined repr, HGB nonlinear", combined, y, INTERNAL_SPLIT_ROW)


if __name__ == "__main__":
    main()
