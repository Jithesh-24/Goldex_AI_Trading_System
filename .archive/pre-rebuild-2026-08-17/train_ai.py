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
import json, os, sys, time, gc
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import fit_calibration, save_calibration, apply_calibration

BASE = "/home/jith/.hermes/profiles/trading/scripts"
FEAT_CSV = os.environ.get("FEAT_CSV", f"{BASE}/gold_features.csv")
MODEL_DIR = f"{BASE}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_EXCLUDE = {"time", "target", "fwd_return", "mfe_atr", "mfa_atr"}
# v8.9c: REMOVED macro features (DXY, yields — 1-day lag from Yahoo)
# Pure gold system: 122 gold-intrinsic features only
MACRO_FEATURES = {"dxy_z", "dxy_5d_chg", "tnx_level", "tnx_5d_chg",
                  "gc_5d_chg", "gld_5d_chg", "eur_5d_chg"}
FEATURE_EXCLUDE |= MACRO_FEATURES
# v8.7 M5-only mandate: HTF columns stay in the matrix but are never read
from features import HTF_FEATURES
FEATURE_EXCLUDE |= HTF_FEATURES
# v8: mfe/mfa are forward-looking (measured at resolution) — placement-prior
# calibration inputs, NEVER model features (lookahead leak otherwise).
SEEDS = [42, 7, 2026]  # 3 seeds (i5 HW limit: 10 seeds = 90h on 32.5M rows)
RECENCY_TAU_DAYS = 120.0       # exp(−age_days/120) sample weight
ROWS_PER_BAR = 84              # 2 dir × 6 sl × 7 tp (must match features.py)

def lgb_params(seed):
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbose": -1,
        "num_threads": 8,   # 8-logical-CPU i5-10210U
        "seed": seed,
    }

def load_data():
    """v9.0 MMAP FIX (2026-08-14): Pre-converted numpy mmap files load in
    seconds instead of 32 hours. csv_to_mmap.py converts CSV → mmap once.
    Falls back to streaming CSV if mmap not found."""
    from features import RAW_PRICE_COLS
    
    META = f"{BASE}/train_data_meta.json"
    MMAP_X = f"{BASE}/train_data_x.npy"
    MMAP_Y = f"{BASE}/train_data_y.npy"
    MMAP_T = f"{BASE}/train_data_t.npy"
    
    # Try mmap first (seconds)
    if os.path.exists(MMAP_X) and os.path.exists(MMAP_Y) and os.path.exists(MMAP_T) and os.path.exists(META):
        import json
        with open(META) as f:
            meta = json.load(f)
        n_rows = meta['n_rows']
        feats = meta['features']
        print(f"  Loading pre-converted mmap ({n_rows:,} rows × {len(feats)} features)...", flush=True)
        t0 = time.time()
        X = np.memmap(MMAP_X, dtype=np.float32, mode='r', shape=(n_rows, len(feats)))
        y = np.memmap(MMAP_Y, dtype=np.int8, mode='r', shape=(n_rows,))
        times = np.memmap(MMAP_T, dtype='datetime64[s]', mode='r', shape=(n_rows,))
        # Force first page load
        _ = X[0:1]; _ = y[0]; _ = times[0]
        print(f"  ✅ Mmap loaded in {time.time()-t0:.1f}s (vs 32h CSV)", flush=True)
        # Copy times to RAM for walk-forward (small: 32.5M × 8 bytes = 260MB)
        times = np.array(times)
        mm_x = MMAP_X
        mm_y = MMAP_Y
        return X, y, times, feats, mm_x, mm_y
    
    # Fallback: streaming CSV (old way, slow)
    print("  ⚠️ No mmap found — falling back to CSV (will take ~32 hours!)", flush=True)
    print("  Run csv_to_mmap.py to create fast-loading mmap files.", flush=True)

def walk_forward_splits(n, n_chunks=3):
    """Time-ordered rolling splits, aligned to BAR boundaries (multiple of
    ROWS_PER_BAR) so the same bar's rows never straddle train/test."""
    n_bars = n // ROWS_PER_BAR
    chunk = n_bars // n_chunks
    splits = []
    for i in range(1, n_chunks):
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
    model = lgb.train(p, dtr, num_boost_round=200)
    prob = model.predict(Xte)
    return model, prob

def predict_chunked(model, X, te_idx, chunk=1_000_000):
    """v8.9 OOM FIX #2 (2026-08-11): predict on the memmap in chunks instead
    of materializing np.ascontiguousarray(X[te_idx]) — a 32.5M-row test
    window was a 2.1GB float32 copy that, on top of the 3.1GB binned
    dataset, peaked 6G+ on a 7.5Gi box → OOM-killed the v8.8 chain at
    19:02 (6G mem + 5.5G swap). Chunks of 1M rows are ~384MB each,
    transient, and free immediately. Same exact predictions, no copy."""
    out = np.zeros(len(te_idx), dtype=np.float64)
    for i in range(0, len(te_idx), chunk):
        sl = te_idx[i:i + chunk]
        out[i:i + len(sl)] = model.predict(np.ascontiguousarray(X[sl]))
    return out

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
    X, y, times, feats, mm_x, mm_y = load_data()
    print(f"FINAL EDITION v7 — {len(X):,} rows | {len(feats)} features | "
          f"{pd.Timestamp(times[0])} -> {pd.Timestamp(times[-1])}")
    print(f"Target balance: {np.bincount(y.astype(np.int64)).tolist()}", flush=True)

    n = len(X)
    w = recency_weights(times)

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
    # v8.9 OOM FIX #3 (2026-08-11, third kill at 21:03): NEVER build d_full.
    # The old design binned ALL 32.5M rows once (3.1GB) and kept it alive
    # while every subset() added another ~2.3GB → 6.3G + 5.4G swap on a
    # 7.5Gi box → OOM-killed. Instead: bin EACH window directly from a
    # contiguous memmap slice VIEW (X[a:b] on np.memmap = zero-copy view;
    # LightGBM reads pages through the OS page cache, not into RSS), train,
    # predict chunked, then del + gc — peak ≈ largest window's binned set
    # (~2.6GB) + transients ≈ 3.5GB total. max_bin 63 halves histogram
    # memory vs 255. Splits are contiguous by construction (walk_forward).
    for s in SEEDS:
        print(f"── seed {s} ──")
        oof_s = np.zeros(n)
        for k, (tr_idx, te_idx) in enumerate(splits):
            tr_end = tr_idx[-1] + 1
            te_end = te_idx[-1] + 1
            dtr = lgb.Dataset(X[0:tr_end], label=y[0:tr_end], weight=w[0:tr_end],
                              free_raw_data=True,
                              params={"max_bin": 31, "num_threads": 4})
            p = lgb_params(s)
            model = lgb.train(p, dtr, num_boost_round=200)
            prob = predict_chunked(model, X, te_idx)
            seed_models[s].append(model)
            oof_s[te_idx] = prob
            oof_y[te_idx] = y[te_idx]
            del dtr, model
            gc.collect()
        oof_sum += oof_s
        t1, t2 = pd.Timestamp(times[tr_idx[0]]), pd.Timestamp(times[te_idx[-1]])
        print(f"  seed {s} windows done ({t1.date()} -> {t2.date()})")

    # v8.9 OOM FIX #2b: nothing to free here anymore — d_full no longer exists.

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
    # v9.0 OOM FIX: max_bin 31 (was 63) to avoid OOM on 7.5GB RAM.
    # 32.5M rows × 108 features × 4 bytes = 13GB mmap (disk-backed).
    # max_bin 31 → ~1.5GB binned dataset per model. del + gc between seeds.
    for i, s in enumerate(SEEDS):
        print(f"\n── Final model: seed {s} ({i+1}/{len(SEEDS)}) ──", flush=True)
        final = lgb.train(lgb_params(s),
                          lgb.Dataset(X, label=y, weight=w, free_raw_data=True,
                                      params={"max_bin": 31, "num_threads": 4}),
                          num_boost_round=200)
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

    # v8.9 OOM FIX: drop the disk-backed memmaps now that models are written
    try:
        del X, y, w, times
        gc.collect()
        os.remove(mm_x)
        os.remove(mm_y)
        print(f"memmap cleaned: {mm_x}, {mm_y}")
    except Exception as e:
        print(f"memmap cleanup warn: {e}")

if __name__ == "__main__":
    main()
