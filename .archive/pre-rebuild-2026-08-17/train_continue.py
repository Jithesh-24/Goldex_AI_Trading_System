"""WARM-START CONTINUATION trainer (v7.5, 2026-08-06).

User directive: KEEP the models trained on all XAUUSD backtest history (they
"know the past" — every regime) and let them KEEP LEARNING with live data.
No cold rebuild from scratch.

Strategy (Claude plan: warm-start / continuation):
  - Load each deployed seed model's weights (the history-informed model).
  - Continue boosting on a RECENT WINDOW of the feature matrix (last 180d,
    incl. appended live outcomes) so the model ADAPTS to the current regime
    while never losing its historical foundation. Full-matrix training would
    double-buffer past 7.5GB RAM (OOM 137), and warm-start means history is
    already in the base weights.
  - Store the continued models (same names the engine loads) with atomic swap.
  - OOF probs are computed in a SEPARATE streaming pass over the FULL matrix
    (chunked predict, ~40MB peak per chunk) so downstream calibration stays
    index-aligned with gold_features.csv rows.
"""
import numpy as np
import os
import pandas as pd
import lightgbm as lgb
import json, os, sys, time, gc, subprocess as _sp
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import fit_calibration

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MODEL_DIR = f"{BASE}/models"
FEAT_CSV = os.environ.get("FEAT_CSV", f"{BASE}/gold_features_m5.csv")
FEATURE_EXCLUDE = {"time", "target", "fwd_return", "mfe_atr", "mfa_atr"}
# v8.7 M5-only mandate: HTF columns stay in the matrix but are never read
from features import HTF_FEATURES
FEATURE_EXCLUDE |= HTF_FEATURES
# v8: mfe/mfa are forward-looking (measured at resolution) — placement-prior
# calibration inputs, NEVER model features (lookahead leak otherwise).
SEEDS = [42, 7, 2026]
RECENCY_TAU_DAYS = 120.0
ROWS_PER_BAR = 84              # 2 dir × 6 sl × 7 tp (must match features.py)
CONTINUE_ROUNDS = 200      # extra boosting rounds on top of the base model
WINDOW_DAYS = 180          # recent training window (warm-start adapt)
CHUNK = 500_000

def lgb_params(seed):
    return {"objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 50, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
            "verbose": -1, "num_threads": 4, "seed": seed}
# NOTE: i5-10210U = 4 physical cores / 8 logical. LightGBM with num_threads=8
# spins on hyperthreaded logical CPUs -> 24x SLOWER (17s@1T, 8.6s@4T, 209s@8T).
# num_threads=4 (physical cores) is the empirical optimum on this machine.

def recency_weights(times_sec, tau_days=RECENCY_TAU_DAYS):
    # times_sec: int64 epoch-seconds
    age = (times_sec.max() - times_sec) / 86400.0
    w = np.exp(-age / tau_days); w = w / w.mean()
    return w.astype(np.float32)

def row_count():
    n = int(_sp.run(["bash", "-c", f"wc -l < {FEAT_CSV}"],
                    capture_output=True, text=True).stdout.strip())
    return n - 1

def load_window(feats, cutoff_sec):
    """Stream matrix, keep only rows with time >= cutoff_sec (recent window)."""
    read_cols = feats + ["target", "time"]
    dtype = {c: np.float32 for c in read_cols if c != "time"}
    parts_x, parts_y, parts_t = [], [], []
    for chunk in pd.read_csv(FEAT_CSV, usecols=read_cols, dtype=dtype,
                             chunksize=CHUNK, low_memory=False):
        t = pd.to_datetime(chunk["time"]).astype("datetime64[s]").astype("int64").values
        m = t >= cutoff_sec
        if m.any():
            parts_x.append(chunk.loc[m, feats].values)
            parts_y.append(chunk.loc[m, "target"].values.astype(np.int8))
            parts_t.append(t[m])
        del chunk
    if not parts_x:
        return None
    X = np.vstack(parts_x).astype(np.float32)
    y = np.concatenate(parts_y).astype(np.int8)
    times = np.concatenate(parts_t)
    del parts_x, parts_y, parts_t; gc.collect()
    return X, y, times

def main():
    t0 = time.time()
    from features import RAW_PRICE_COLS
    all_cols = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    feats = [c for c in all_cols if c not in FEATURE_EXCLUDE and c not in RAW_PRICE_COLS]

    # ── PASS 1: load recent window + warm-start continuation ──
    # Need the max time first: locate the time column position in the header
    # (M1 matrix: col 98; M5 matrix: col 104 — NEVER hardcode, scan it).
    hdr = _sp.run(["bash", "-c", f"head -1 {FEAT_CSV}"],
                  capture_output=True, text=True, timeout=60).stdout.strip()
    _col_time = [i for i, c in enumerate(hdr.split(","), 1) if c.strip() == "time"]
    if not _col_time:
        print("no time column in header; abort"); return 1
    _tpos = _col_time[0]
    raw_max = _sp.run(
        ["bash", "-c", f"awk -F, 'NR>1 && ${_tpos}>max {{max=${_tpos}}} END {{print max}}' {FEAT_CSV}"],
        capture_output=True, text=True, timeout=600).stdout.strip()
    last_ts = int(pd.Timestamp(raw_max).timestamp())
    cutoff = int(last_ts) - WINDOW_DAYS * 86400
    loaded = load_window(feats, cutoff)
    if loaded is None:
        print("no rows in window; abort"); return 1
    X, y, times = loaded
    print(f"CONTINUE — window {len(X):,} rows | {len(feats)} feats | "
          f"{datetime.utcfromtimestamp(times[0]).date()} -> "
          f"{datetime.utcfromtimestamp(times[-1]).date()} (last {WINDOW_DAYS}d)")
    w = recency_weights(times)
    del times; gc.collect()

    dset = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    for s in SEEDS:
        base = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        _can_warm = False
        if os.path.exists(base):
            try:
                _b = lgb.Booster(model_file=base)
                _can_warm = _b.num_feature() == len(feats)
                if not _can_warm:
                    print(f"  seed {s}: base has {_b.num_feature()} feats vs "
                          f"matrix {len(feats)} — cold start (v8 M5 switch)")
            except Exception:
                _can_warm = False
        if _can_warm:
            model = lgb.train(lgb_params(s), dset,
                              num_boost_round=CONTINUE_ROUNDS,
                              init_model=base)
            print(f"  seed {s}: warm-start from base +{CONTINUE_ROUNDS} rounds")
        else:
            model = lgb.train(lgb_params(s), dset,
                              num_boost_round=600)
            print(f"  seed {s}: no base → cold 600 rounds")
        name = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        tmp = name + ".tmp"; model.save_model(tmp); os.replace(tmp, name)
        del model; gc.collect()
    del X, y, w, dset; gc.collect()

    # ── PASS 2: full-length aligned OOF via streaming predict ──
    # fit_calibration_by_rr.py consumes oof_probs.npy/oof_targets.npy index-
    # aligned with gold_features.csv rows (direction, rr_buy, rr_sell).
    n = row_count()
    oof = np.zeros(n, dtype=np.float32)
    oof_y = np.zeros(n, dtype=np.int8)
    boosters = [lgb.Booster(model_file=f"{MODEL_DIR}/gold_lgb_model_s{s}.txt")
                for s in SEEDS]
    for b in boosters:
        b.params = dict(b.params, num_threads=4)
    read_cols = feats + ["target", "time"]
    dtype = {c: np.float32 for c in read_cols if c != "time"}
    start = 0
    for chunk in pd.read_csv(FEAT_CSV, usecols=read_cols, dtype=dtype,
                             chunksize=CHUNK, low_memory=False):
        end = start + len(chunk)
        Xc = chunk[feats].values.astype(np.float32)
        p = np.zeros(len(chunk), dtype=np.float32)
        for b in boosters:
            p += b.predict(Xc) / len(SEEDS)
        oof[start:end] = p
        oof_y[start:end] = chunk["target"].values
        start = end
        del chunk, Xc, p; gc.collect()
    del boosters; gc.collect()
    acc = ((oof >= 0.5).astype(int) == oof_y).mean()
    print(f"  OOF aligned len={len(oof):,} | acc={acc:.3f}")
    np.save(f"{MODEL_DIR}/oof_probs.npy", oof)
    np.save(f"{MODEL_DIR}/oof_targets.npy", oof_y)
    try:
        knots = fit_calibration(oof, oof_y)
        with open(f"{MODEL_DIR}/calibration.json", "w") as f:
            json.dump(knots, f)
        print(f"  calibration refit on {len(oof):,} OOF")
    except Exception as e:
        print("  calibration: skipped", e)
    with open(f"{MODEL_DIR}/ensemble.json", "w") as f:
        json.dump({"type": "placement", "seeds": SEEDS,
                   "models": [f"gold_lgb_model_s{s}.txt" for s in SEEDS],
                   "recency_tau_days": RECENCY_TAU_DAYS, "mode": "warm-start-continue",
                   "base_tf": os.environ.get("PRIOR_BAR_SECS", "180") == "300" and "m5" or "m1"},
                  f, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] ✅ continuation complete in {time.time()-t0:.0f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
