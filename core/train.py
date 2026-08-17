"""
Full training pipeline: raw M1 -> Tier1/Tier2 features -> CUSUM event
sampling -> triple-barrier primary direction labels -> purged walk-forward
OOF primary training -> meta-labeling (precision filter) -> final models.

Two-stage architecture (de Prado ch. 3):
  PRIMARY  : recall-leaning direction classifier. 3-class (short/flat/long),
             answers "is there a directional opportunity here at all".
  META     : precision classifier, trained ONLY on rows where the primary's
             own out-of-fold prediction fired (non-flat), target = did the
             barrier in that assumed direction actually pay off (win) before
             the adverse side. Answers "of the opportunities primary found,
             which are precise enough to act on" — this is the filter that
             kills "confident but wrong" signals from the old system.

Live signal = primary fires a side AND meta P(win) exceeds a threshold.
TP/SL prices are read straight off the same vol-scaled barrier widths used
to build the labels, so what gets sent to Telegram is exactly what the
model was scored against, not a separate hand-tuned distance.

Run: python3 -m core.train [--rows N] [--out-dir models]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from core.data import load_raw_m1
from core.features import build_features
from core.labeling import TripleBarrierConfig, cusum_filter, triple_barrier_labels
from core.cv import PurgedWalkForwardCV

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TB_CFG = TripleBarrierConfig(pt_mult=1.5, sl_mult=1.0, max_holding=90, min_vol=1e-6)
CUSUM_K = 2.5
N_SPLITS = 6
EMBARGO_BARS = TB_CFG.max_holding * 2
META_PROB_THRESHOLD = 0.55  # only used at inference; kept here as the documented default


def assemble_dataset(rows: int = None):
    """Returns (feat_df, close/high/low arrays, event t0_idx). `rows` caps to
    the most recent N raw M1 bars for a fast dry run."""
    df = load_raw_m1()
    if rows:
        df = df.tail(rows).reset_index(drop=True)
    feat = build_features(df)

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    vol = feat["ewma_vol"].to_numpy(dtype=np.float64)
    vol_filled = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[np.isfinite(vol)]))
    threshold = np.clip(CUSUM_K * vol_filled * close, 1e-6, None)
    event_mask = cusum_filter(close, threshold)

    feature_cols = [c for c in feat.columns if c != "time"]
    warmup_ok = feat[feature_cols].notna().all(axis=1).to_numpy()
    horizon_ok = np.arange(len(df)) < (len(df) - TB_CFG.max_holding - 1)
    valid = event_mask & warmup_ok & horizon_ok
    t0_idx = np.where(valid)[0]

    print(f"assemble_dataset: {len(df):,} bars -> {len(t0_idx):,} CUSUM events "
          f"({len(t0_idx) / len(df) * 100:.1f}% of bars), {len(feature_cols)} features")
    return feat, close, high, low, vol, t0_idx, feature_cols


def train_primary_oof(feat, close, high, low, vol, t0_idx, feature_cols):
    """Purged walk-forward OOF primary predictions, needed to build honest
    meta-labels (using the primary's actual generalization behavior, not its
    in-sample fit, which would make the meta stage trivially overfit)."""
    labels = triple_barrier_labels(close, high, low, t0_idx, vol, TB_CFG, side=None)
    y = labels["label"].to_numpy()  # -1/0/1
    y_cat = y + 1  # catboost wants 0..2
    t1 = labels["t1"].to_numpy()

    X = feat.loc[t0_idx, feature_cols].reset_index(drop=True)
    y_cat = pd.Series(y_cat).reset_index(drop=True)
    t0 = pd.Series(t0_idx).reset_index(drop=True)
    t1 = pd.Series(t1).reset_index(drop=True)

    cv = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_bars=EMBARGO_BARS)
    oof_pred = np.full(len(X), -1, dtype=np.int64)
    oof_proba = np.full((len(X), 3), np.nan, dtype=np.float64)
    fold_metrics = []

    for fold, (train_pos, test_pos) in enumerate(cv.split(t0.to_numpy(), t1.to_numpy())):
        model = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, loss_function="MultiClass",
            class_weights=[1.0, 1.0, 1.0], random_seed=42, verbose=False,
            thread_count=-1,
        )
        model.fit(X.iloc[train_pos], y_cat.iloc[train_pos])
        proba = model.predict_proba(X.iloc[test_pos])
        pred = proba.argmax(axis=1)
        oof_pred[test_pos] = pred
        oof_proba[test_pos] = proba

        acc = (pred == y_cat.iloc[test_pos].to_numpy()).mean()
        fired = pred != 1  # not-flat
        fired_acc = (pred[fired] == y_cat.iloc[test_pos].to_numpy()[fired]).mean() if fired.sum() else float("nan")
        fold_metrics.append({"fold": fold, "n_train": len(train_pos), "n_test": len(test_pos),
                              "acc": acc, "fire_rate": fired.mean(), "fired_acc": fired_acc})
        print(f"  fold {fold}: train={len(train_pos):,} test={len(test_pos):,} "
              f"acc={acc:.3f} fire_rate={fired.mean():.3f} fired_acc={fired_acc:.3f}")

    has_oof = oof_pred >= 0
    return X, y_cat, t0, t1, oof_pred, oof_proba, has_oof, fold_metrics


def build_meta_labels(close, high, low, vol, t0_idx_arr, oof_pred, has_oof):
    """side = primary's OOF direction (+1/-1), only for rows it actually fired
    on (non-flat). Meta target = did that side's TP hit before its SL."""
    fired = has_oof & (oof_pred != 1)
    side = np.where(oof_pred[fired] == 2, 1.0, -1.0)  # class 2=up(+1), class 0=down(-1)
    t0_sub = t0_idx_arr[fired]
    meta_labels = triple_barrier_labels(close, high, low, t0_sub, vol, TB_CFG, side=side)
    return fired, side, meta_labels


def train_final_models(X, y_cat, feature_cols, fired, side, meta_labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    primary_final = CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.05, loss_function="MultiClass",
        random_seed=42, verbose=False, thread_count=-1,
    )
    primary_final.fit(X, y_cat)
    primary_final.save_model(os.path.join(out_dir, "primary.cbm"))

    X_meta = X.loc[fired].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = meta_labels["label"].to_numpy()

    meta_final = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05, loss_function="Logloss",
        random_seed=42, verbose=False, thread_count=-1,
    )
    meta_final.fit(X_meta, y_meta)
    meta_final.save_model(os.path.join(out_dir, "meta.cbm"))

    meta_cols = feature_cols + ["assumed_side"]
    with open(os.path.join(out_dir, "feature_cols.json"), "w") as f:
        json.dump({"primary": feature_cols, "meta": meta_cols,
                   "tb_cfg": TB_CFG.__dict__, "cusum_k": CUSUM_K,
                   "meta_prob_threshold": META_PROB_THRESHOLD}, f, indent=2)

    print(f"saved primary.cbm, meta.cbm, feature_cols.json -> {out_dir}")
    print(f"meta training set: {len(X_meta):,} rows, win rate {y_meta.mean():.3f} "
          f"(baseline before precision filter)")
    return primary_final, meta_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None, help="cap to most recent N raw bars (dry run)")
    ap.add_argument("--out-dir", type=str, default=os.path.join(BASE, "models"))
    args = ap.parse_args()

    t_start = time.time()
    feat, close, high, low, vol, t0_idx, feature_cols = assemble_dataset(rows=args.rows)

    print("\n== primary OOF training (purged walk-forward) ==")
    X, y_cat, t0, t1, oof_pred, oof_proba, has_oof, fold_metrics = train_primary_oof(
        feat, close, high, low, vol, t0_idx, feature_cols)

    print("\n== building meta-labels from primary OOF ==")
    fired, side, meta_labels = build_meta_labels(close, high, low, vol, t0_idx, oof_pred, has_oof)
    print(f"primary fired on {fired.sum():,}/{has_oof.sum():,} OOF rows "
          f"({fired.sum() / max(has_oof.sum(), 1) * 100:.1f}%)")

    print("\n== final model fit (full data) ==")
    train_final_models(X, y_cat, feature_cols, fired, side, meta_labels, args.out_dir)

    summary = {
        "n_bars": int(len(feat)), "n_events": int(len(t0_idx)),
        "fold_metrics": fold_metrics,
        "n_fired_oof": int(fired.sum()), "n_oof": int(has_oof.sum()),
        "meta_win_rate_baseline": float(meta_labels["label"].mean()),
        "elapsed_sec": time.time() - t_start,
    }
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE in {summary['elapsed_sec']:.1f}s. Summary -> {args.out_dir}/train_summary.json")


if __name__ == "__main__":
    main()
