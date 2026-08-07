"""FINAL EDITION (v7 2026-08-02) — Dual-model LightGBM trainer.

Architecture:
  MODEL A "placement" (how-to-place): 24 rows/bar (2 dir × 4 sl × 3 tp),
     predicts P(win | market state, HTF context, session, events, order
     flow, placement geometry, direction). The engine sweeps the grid and
     fires the max-expectancy placement — the model decides SL/TP.
  MODEL B "direction" (when-to-trade): 1 row/bar, predicts P(up-move |
     market state) — the "don't catch falling knives" teacher. At inference
     the engine multiplies: final Exp = Exp_buy × P(up) vs Exp_sell ×
     (1−P(up)). No gates — a learned probability multiplier.

ENSEMBLE: 3 seeds per model, averaged probabilities (variance reduction,
smoother calibration). Recency weighting: rows weighted by exp(−age/τ)
so recent market behavior counts more — the model ADAPTS to the current
regime without any hardcoded regime logic.

Walk-forward stays strict: train on past only, test on future only.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, os, sys, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import fit_calibration, save_calibration, apply_calibration

BASE = "/home/jith/.hermes/profiles/trading/scripts"
FEAT_CSV = os.environ.get("FEAT_CSV", f"{BASE}/gold_features.csv")
MODEL_DIR = f"{BASE}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_EXCLUDE = {"time", "target", "fwd_return", "mfe_atr", "mfa_atr"}
# v8: mfe/mfa are forward-looking (measured at resolution) — placement-prior
# calibration inputs, NEVER model features (lookahead leak otherwise).
SEEDS = [42, 7, 2026]          # ensemble seeds
RECENCY_TAU_DAYS = 120.0       # exp(−age_days/120) sample weight
ROWS_PER_BAR = 48              # 2 dir × 6 sl × 4 tp (must match features.py)

def lgb_params(seed):
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbose": -1,
        "num_threads": 4,   # 8-logical-CPU i5-10210U; OOM fixed via Dataset.subset (was 1)
        "seed": seed,
    }

def load_data():
    """float32 direct-read, raw-price cols excluded at read time (memory-safe
    on the 2GB gateway cgroup). X is a VIEW; df freed before LightGBM."""
    from features import RAW_PRICE_COLS
    all_cols = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    feats = [c for c in all_cols if c not in FEATURE_EXCLUDE and c not in RAW_PRICE_COLS]
    read_cols = feats + ["target", "direction", "time"]
    dtype_map = {c: np.float32 for c in read_cols if c != "time"}
    df = pd.read_csv(FEAT_CSV, usecols=read_cols, dtype=dtype_map)
    df["time"] = pd.to_datetime(df["time"])
    return df, feats

def walk_forward_splits(n, n_chunks=6):
    """Time-ordered rolling splits, aligned to BAR boundaries (multiple of
    ROWS_PER_BAR) so the same bar's rows never straddle train/test."""
    n_bars = n // ROWS_PER_BAR
    chunk = n_bars // n_chunks
    splits = []
    for i in range(2, n_chunks):
        train_bars = i * chunk
        test_bars = min(chunk, n_bars - train_bars)
        if test_bars <= 0:
            break
        tr_end = train_bars * ROWS_PER_BAR
        te_end = (train_bars + test_bars) * ROWS_PER_BAR
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    return splits

def recency_weights(times, tau_days=RECENCY_TAU_DAYS):
    """exp(−age/τ) — recent bars weighted higher. τ=120d: a bar 2 weeks old
    weighs 0.89×, 6 months old 0.22×, 3 years old ~0.0001×. The model ADAPTS
    to the current regime; ancient history still teaches faintly."""
    ts = times.astype("datetime64[s]").astype(np.int64)
    now = ts.max()
    age_days = (now - ts) / 86400.0
    w = np.exp(-age_days / tau_days)
    w = w / w.mean()  # normalize so total weight ≈ n (LR stays sane)
    return w.astype(np.float32)

def train_one(dtr, Xte, seed):
    """Train on a subset Dataset (shares parent's bin mapper — zero copy).
    Predict on the raw test slice (small, transient — LightGBM 4.7 refuses
    Dataset prediction and subset get_data() fails after free_raw_data)."""
    p = lgb_params(seed)
    model = lgb.train(p, dtr, num_boost_round=600)
    prob = model.predict(Xte)
    return model, prob

def evaluate(probs, y, label=""):
    probs = np.array(probs); y = np.array(y)
    pred = (probs >= 0.5).astype(int)
    acc = (pred == y).mean()
    up_mask = pred == 1
    dn_mask = pred == 0
    p_up = y[up_mask].mean() if up_mask.any() else float("nan")
    p_dn = (1 - y[dn_mask]).mean() if dn_mask.any() else float("nan")
    print(f"  {label}: acc={acc:.3f} | P(up correct)={p_up:.3f} | P(down correct)={p_dn:.3f} | n={len(y)}")
    return {"acc": acc, "p_up": p_up, "p_dn": p_dn, "n": len(y)}

def main():
    t0 = time.time()
    df, feats = load_data()
    times = df["time"].values
    print(f"FINAL EDITION v7 — {len(df):,} rows | {len(feats)} features | "
          f"{pd.Timestamp(times[0])} -> {pd.Timestamp(times[-1])}")
    print(f"Target balance: {df['target'].value_counts().to_dict()}")

    X = df[feats].values          # float32 VIEW (df already float32)
    y = df["target"].values.astype(np.int8)
    n = len(df)
    w = recency_weights(times)
    del df
    import gc; gc.collect()
    print(f"Recency weights: τ={RECENCY_TAU_DAYS}d, w_max/w_min = {w.max():.1f}/{w.min():.6f}")

    splits = walk_forward_splits(n)
    print(f"Walk-forward: {len(splits)} test windows\n")

    # ── ENSEMBLE WALK-FORWARD: 3 seeds, average OOF probs ──
    # Test-mask is identical for all seeds (same splits) — compute once.
    test_mask = np.zeros(n, dtype=bool)
    for _, te_idx in splits:
        test_mask[te_idx] = True
    oof_sum = np.zeros(n)
    oof_y = np.zeros(n, dtype=int)
    seed_models = {s: [] for s in SEEDS}
    # v7.3 OOM FIX: bin the FULL dataset ONCE, then use Dataset.subset() for
    # each walk-forward window. subset() shares the bin mapper — NO per-window
    # float32 copies (X[tr_idx] was 2.5GB/window on the doubled 6.35M-row
    # matrix → 7.7GB peak on a 7.6GB machine → OOM kill at 17:44 on 08-02).
    # Binning once: ~0.7GB binned + 2.7GB parent X (kept for predict slices).
    d_full = lgb.Dataset(X, label=y, weight=w, free_raw_data=True,
                         params={"max_bin": 255, "num_threads": 4})
    import gc; gc.collect()
    for s in SEEDS:
        print(f"── seed {s} ──")
        oof_s = np.zeros(n)
        for k, (tr_idx, te_idx) in enumerate(splits):
            dtr = d_full.subset(tr_idx)
            Xte = np.ascontiguousarray(X[te_idx])
            model, prob = train_one(dtr, Xte, s)
            seed_models[s].append(model)
            oof_s[te_idx] = prob
            oof_y[te_idx] = y[te_idx]
            del dtr, Xte
        oof_sum += oof_s
        t1, t2 = pd.Timestamp(times[tr_idx[0]]), pd.Timestamp(times[te_idx[-1]])
        print(f"  seed {s} windows done ({t1.date()} -> {t2.date()})")

    oof_probs = oof_sum / len(SEEDS)
    all_probs = oof_probs[test_mask]
    all_y = oof_y[test_mask]

    print("\n=== AGGREGATE OUT-OF-SAMPLE (3-seed ensemble) ===")
    res = evaluate(all_probs, all_y, "ALL TEST WINDOWS")

    # ── CALIBRATION on ensemble OOF ──
    knots = fit_calibration(all_probs, all_y)
    cal_check = apply_calibration(np.array([0.3, 0.5, 0.6, 0.8]), knots)
    print(f"\nCalibration fitted on {knots['n']} OOF preds ({len(knots['knots_p'])} knots)")
    print(f"  raw 0.30→{cal_check[0]:.2f}  0.50→{cal_check[1]:.2f}  0.60→{cal_check[2]:.2f}  0.80→{cal_check[3]:.2f}")
    # v7.3f: the select_p / select_p_buy / select_p_sell confidence-floor
    # gate was removed from calibrate.py (v7.1 already stopped reading it in
    # the live engine — harness, not harden). No longer computed or persisted.
    save_calibration(knots, f"{MODEL_DIR}/calibration.json")

    # ── OOF cache for backtest ──
    np.save(f"{MODEL_DIR}/oof_probs.npy", oof_probs.astype(np.float32))
    np.save(f"{MODEL_DIR}/oof_targets.npy", oof_y.astype(np.int8))
    print(f"OOF saved: {MODEL_DIR}/oof_probs.npy ({n} rows)")

    # ── FINAL MODELS (all data) + atomic swap, one per seed ──
    for i, s in enumerate(SEEDS):
        final = lgb.train(lgb_params(s), lgb.Dataset(X, label=y, weight=w, free_raw_data=True),
                          num_boost_round=600)
        name = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        tmp = name + ".tmp"
        final.save_model(tmp)
        os.replace(tmp, name)
        print(f"Model saved: {name} (atomic swap)")
        del final; gc.collect()

    with open(f"{MODEL_DIR}/features.json", "w") as f:
        json.dump(feats, f)
    with open(f"{MODEL_DIR}/ensemble.json", "w") as f:
        json.dump({"type": "placement", "seeds": SEEDS,
                   "models": [f"gold_lgb_model_s{s}.txt" for s in SEEDS],
                   "recency_tau_days": RECENCY_TAU_DAYS,
                   "base_tf": os.environ.get("PRIOR_BAR_SECS", "180") == "300" and "m5" or "m1"},
                  f, indent=2)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in res.items()}, f, indent=2)
    print(f"\nEnsemble config: {MODEL_DIR}/ensemble.json")
    print(f"✅ FINAL EDITION placement model done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
