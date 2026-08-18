"""
Full training pipeline: raw M1 -> Tier1/Tier2 features -> CUSUM event
sampling -> triple-barrier direction labels -> purged walk-forward OOF
primary training -> meta-labeling (precision filter) -> final models.

Two-stage architecture:
  EVENT GATE : CUSUM filter -- only bars with a statistically significant
               cumulative move become candidate rows at all. This is the
               "is there an opportunity" recall gate (the old 3-class
               primary's flat/no-flat role, moved here after finding the
               vertical-timeout ("truly nothing happened") outcome is only
               ~3-5% of CUSUM events -- CUSUM itself already screens out
               almost all the "nothing happened" bars, so a 3rd flat class
               downstream just adds a near-empty, hard-to-learn class).
  PRIMARY    : binary direction classifier (up vs down) on CUSUM events.
  META       : precision classifier, trained on the primary's own
               out-of-fold predictions (never in-sample fit, which would
               make the meta stage trivially overfit), target = did the
               barrier in that assumed direction pay off (win) before the
               adverse side. Answers "of primary's calls, which are precise
               enough to act on" -- the filter that kills "confident but
               wrong" signals.

Live signal = primary's side AND meta P(win) exceeds a threshold.
TP/SL prices are read straight off the same vol-scaled barrier widths used
to build the labels, so what gets sent to Telegram is exactly what the
model was scored against, not a separate hand-tuned distance.

Calibration notes (found via diagnostic checks before landing on these
defaults, not guesses):
  - Barrier width MUST be horizon-scaled (vol * sqrt(max_holding) * scale),
    not raw per-bar vol -- raw per-bar vol made barriers so tight they were
    touched by 1-bar noise almost immediately (mean holding ~2.7 bars
    against a 90-bar horizon).
  - Direction label barriers MUST be symmetric (pt==sl) -- asymmetric
    widths bias "which side touched first" toward whichever barrier is
    narrower, independent of real market direction (caught a spurious
    58.8%/38.1% down/up split this way that vanished under symmetric
    widths).
  - The real edge here is short-horizon mean-reversion after a CUSUM spike:
    a trivial "bet against the last 5min move" rule beats "bet with it" in
    every year 2020-2026 (52.0% -> 50.1%, decaying -- real but thin and
    getting arbitraged away over time). Edge is stronger at shorter holding
    horizons (15 bars: 51.4% vs 180 bars: 50.7%); max_holding=45 is a
    balance between edge strength and a holding period a human can actually
    react to and manage (~10-15min average).
  - GBDT with only continuous return features UNDERPERFORMED that trivial
    rule (histogram binning can't reproduce an exact zero-threshold split
    as precisely as sign() does) -- fixed by adding explicit sign_ret_*
    features (see core/features.py) and switching to early-stopped binary
    Logloss instead of a 3-class softmax over a near-empty flat class.

Run: python3 -m core.train [--rows N] [--out-dir models]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from core.data import load_raw_m1
from core.features import build_features
from core.labeling import TripleBarrierConfig, cusum_filter, triple_barrier_labels
from core.cv import PurgedWalkForwardCV

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TB_CFG_DIR = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
TB_CFG_TRADE = TripleBarrierConfig(pt_mult=1.5, sl_mult=1.0, max_holding=45, min_vol=1e-6)
HORIZON_VOL_SCALE = 0.45
CUSUM_K = 2.5
N_SPLITS = 6
EMBARGO_BARS = TB_CFG_DIR.max_holding * 2
META_PROB_THRESHOLD = 0.60  # only used at inference; kept here as the documented default --
# calibrated from core/backtest.py's sequential (one-trade-at-a-time) OOF sweep: 0.60 gives
# ~5 signals/day at 58.3% win rate / net +0.21R, the best fit for manual one-at-a-time execution.

CATBOOST_KW = dict(depth=4, iterations=2000, learning_rate=0.02, l2_leaf_reg=15,
                    loss_function="Logloss", random_seed=42, verbose=False,
                    thread_count=-1, early_stopping_rounds=100)
VAL_FRACTION = 0.15  # chronological tail slice of each fold's TRAIN set, for early stopping


def _fit_with_early_stopping(X, y, train_pos):
    """Carve the tail VAL_FRACTION of `train_pos` (already chronological)
    as an early-stopping validation set -- still causal, no test-fold data
    involved, just prevents the fit from running past where it stops
    generalizing on this particular training window."""
    cut = int(len(train_pos) * (1 - VAL_FRACTION))
    tr, va = train_pos[:cut], train_pos[cut:]
    model = CatBoostClassifier(**CATBOOST_KW)
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
    return model


def assemble_dataset(rows: int = None, exclude: frozenset = frozenset()):
    """Returns (feat_df, close/high/low arrays, horizon-scaled vol, event
    t0_idx, feature_cols). `rows` caps to the most recent N raw M1 bars for
    a fast dry run. `exclude` drops named columns from the PREDICTIVE
    feature_cols list (e.g. {"spread", "tick_volume"} -- see Phase 2 data-
    semantics fix) -- the columns are still computed in `feat` (spread stays
    available for the execution-cost layer in core/signal.py and the
    backtest), they are just never shown to the model. Default excludes
    nothing, preserving exact prior behavior for existing callers."""
    df = load_raw_m1()
    if rows:
        df = df.tail(rows).reset_index(drop=True)
    feat = build_features(df)

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    vol = feat["ewma_vol"].to_numpy(dtype=np.float64)  # per-bar (1-minute) std of log returns
    vol_filled = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[np.isfinite(vol)]))
    threshold = np.clip(CUSUM_K * vol_filled * close, 1e-6, None)
    event_mask = cusum_filter(close, threshold)

    vol_tb = vol_filled * np.sqrt(TB_CFG_DIR.max_holding) * HORIZON_VOL_SCALE

    feature_cols = [c for c in feat.columns if c != "time" and c not in exclude]
    warmup_ok = feat[feature_cols].notna().all(axis=1).to_numpy()
    horizon_ok = np.arange(len(df)) < (len(df) - TB_CFG_DIR.max_holding - 1)
    valid = event_mask & warmup_ok & horizon_ok
    t0_idx = np.where(valid)[0]

    print(f"assemble_dataset: {len(df):,} bars -> {len(t0_idx):,} CUSUM events "
          f"({len(t0_idx) / len(df) * 100:.1f}% of bars), {len(feature_cols)} features")
    return feat, close, high, low, vol_tb, t0_idx, feature_cols


def label_events(close, high, low, vol, t0_idx, feature_cols, feat):
    """Symmetric direction labels, vertical-timeout (flat) events dropped --
    CUSUM already screens for "something happened"; keeping a near-empty
    3rd class just makes the classifier harder to fit for no benefit."""
    labels = triple_barrier_labels(close, high, low, t0_idx, vol, TB_CFG_DIR, side=None)
    y = labels["label"].to_numpy()
    t1 = labels["t1"].to_numpy()
    nz = y != 0
    t0_nz = t0_idx[nz]
    y_bin = pd.Series((y[nz] == 1).astype(np.int64)).reset_index(drop=True)  # 1=up, 0=down
    t0 = pd.Series(t0_nz).reset_index(drop=True)
    t1_nz = pd.Series(t1[nz]).reset_index(drop=True)
    X = feat.loc[t0_nz, feature_cols].reset_index(drop=True)
    print(f"label_events: {len(t0_idx):,} events -> {len(t0_nz):,} directional "
          f"({(~nz).mean() * 100:.1f}% dropped as vertical-timeout flat)")
    return X, y_bin, t0, t1_nz, t0_nz


def train_primary_oof(X, y_bin, t0, t1):
    """Purged walk-forward OOF primary predictions."""
    cv = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_bars=EMBARGO_BARS)
    oof_pred = np.full(len(X), -1, dtype=np.int64)
    oof_proba = np.full(len(X), np.nan, dtype=np.float64)
    fold_metrics = []

    for fold, (train_pos, test_pos) in enumerate(cv.split(t0.to_numpy(), t1.to_numpy())):
        model = _fit_with_early_stopping(X, y_bin, train_pos)
        proba = model.predict_proba(X.iloc[test_pos])[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        oof_pred[test_pos] = pred
        oof_proba[test_pos] = proba

        acc = (pred == y_bin.iloc[test_pos].to_numpy()).mean()
        fold_metrics.append({"fold": fold, "n_train": len(train_pos), "n_test": len(test_pos),
                              "acc": acc, "best_iter": model.get_best_iteration()})
        print(f"  fold {fold}: train={len(train_pos):,} test={len(test_pos):,} "
              f"acc={acc:.4f} best_iter={model.get_best_iteration()}")

    has_oof = oof_pred >= 0
    return oof_pred, oof_proba, has_oof, fold_metrics


def build_meta_labels(close, high, low, vol, t0_nz, oof_pred, has_oof):
    """side = primary's OOF direction (+1/-1). Meta target = did that side's
    TP hit before its SL, using the asymmetric TRADE config (real reward:risk
    structure, not the symmetric one used to determine direction)."""
    side = np.where(oof_pred[has_oof] == 1, 1.0, -1.0)
    t0_sub = t0_nz[has_oof]
    meta_labels = triple_barrier_labels(close, high, low, t0_sub, vol, TB_CFG_TRADE, side=side)
    return has_oof, side, meta_labels


FEATURE_SCHEMA_VERSION = "v2-2026-08-18"  # bumped from the implicit unversioned original 28-col
# schema when spread/tick_volume were dropped from the predictive matrix (Phase 2 data-semantics
# fix) -- see feature_cols.json's "excluded_features"/"schema_version" fields for what produced
# any given model artifact.


def train_final_models(X, y_bin, t0, has_oof, side, meta_labels, feature_cols, out_dir,
                        excluded_features=frozenset(), dataset_meta=None):
    os.makedirs(out_dir, exist_ok=True)
    all_pos = np.arange(len(X))

    primary_final = _fit_with_early_stopping(X, y_bin, all_pos)
    primary_final.save_model(os.path.join(out_dir, "primary.cbm"))

    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())

    meta_final = _fit_with_early_stopping(X_meta, y_meta, np.arange(len(X_meta)))
    meta_final.save_model(os.path.join(out_dir, "meta.cbm"))

    meta_cols = feature_cols + ["assumed_side"]
    with open(os.path.join(out_dir, "feature_cols.json"), "w") as f:
        json.dump({"primary": feature_cols, "meta": meta_cols,
                   "tb_cfg_dir": TB_CFG_DIR.__dict__, "tb_cfg_trade": TB_CFG_TRADE.__dict__,
                   # signal.py's TP/SL formula uses tb_cfg_trade's pt_mult/sl_mult against
                   # vol that's ALREADY *sqrt(max_holding)*HORIZON_VOL_SCALE -- live engine
                   # must apply the same scaling to raw per-bar vol before calling score().
                   "horizon_vol_scale": HORIZON_VOL_SCALE, "max_holding": TB_CFG_TRADE.max_holding,
                   "cusum_k": CUSUM_K, "meta_prob_threshold": META_PROB_THRESHOLD,
                   # -- versioning (Phase 2 step 10): "what exact model+features produced this
                   # signal" should always be answerable from this one file.
                   "schema_version": FEATURE_SCHEMA_VERSION,
                   "excluded_features": sorted(excluded_features),
                   "catboost_kw": CATBOOST_KW, "val_fraction": VAL_FRACTION,
                   "n_splits": N_SPLITS, "embargo_bars": EMBARGO_BARS,
                   "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "dataset": dataset_meta or {}}, f, indent=2)

    print(f"saved primary.cbm, meta.cbm, feature_cols.json -> {out_dir}")
    print(f"meta training set: {len(X_meta):,} rows, win rate {y_meta.mean():.3f} "
          f"(baseline before precision filter)")
    return primary_final, meta_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None, help="cap to most recent N raw bars (dry run)")
    ap.add_argument("--out-dir", type=str, default=os.path.join(BASE, "models"))
    ap.add_argument("--exclude-features", type=str, default="",
                     help="comma-separated feature names to drop from the PREDICTIVE matrix "
                          "(e.g. spread,tick_volume) -- Phase 2 data-semantics fix. Empty by "
                          "default so retrain_daily.py's scheduled nightly call is UNCHANGED "
                          "unless explicitly opted in; pass this flag by hand to build a "
                          "validated v2 artifact without touching the automatic promotion path.")
    args = ap.parse_args()
    exclude = frozenset(c.strip() for c in args.exclude_features.split(",") if c.strip())

    t_start = time.time()
    feat, close, high, low, vol, t0_idx, feature_cols = assemble_dataset(rows=args.rows, exclude=exclude)
    X, y_bin, t0, t1, t0_nz = label_events(close, high, low, vol, t0_idx, feature_cols, feat)
    dataset_meta = {"n_bars": int(len(feat)), "excluded_features": sorted(exclude),
                     "date_range": [str(pd.to_datetime(feat["time"].to_numpy())[0]),
                                    str(pd.to_datetime(feat["time"].to_numpy())[-1])]}

    print("\n== primary OOF training (purged walk-forward, binary direction) ==")
    oof_pred, oof_proba, has_oof, fold_metrics = train_primary_oof(X, y_bin, t0, t1)
    mean_oof_acc = np.mean([f["acc"] for f in fold_metrics])
    print(f"mean OOF accuracy: {mean_oof_acc:.4f}")

    print("\n== building meta-labels from primary OOF ==")
    has_oof, side, meta_labels = build_meta_labels(close, high, low, vol, t0_nz, oof_pred, has_oof)
    print(f"meta training set size: {has_oof.sum():,}")

    print("\n== final model fit (full data) ==")
    train_final_models(X, y_bin, t0, has_oof, side, meta_labels, feature_cols, args.out_dir,
                        excluded_features=exclude, dataset_meta=dataset_meta)

    summary = {
        "n_bars": int(len(feat)), "n_events": int(len(t0_idx)), "n_directional": int(len(X)),
        "fold_metrics": fold_metrics, "mean_oof_acc": float(mean_oof_acc),
        "n_meta_train": int(has_oof.sum()),
        "meta_win_rate_baseline": float(meta_labels["label"].mean()),
        "schema_version": FEATURE_SCHEMA_VERSION, "excluded_features": sorted(exclude),
        "elapsed_sec": time.time() - t_start,
    }
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE in {summary['elapsed_sec']:.1f}s. Summary -> {args.out_dir}/train_summary.json")


if __name__ == "__main__":
    main()
