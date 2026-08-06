"""WARM-START CONTINUATION trainer (v7.4, 2026-08-03).

User directive: KEEP the models trained on all XAUUSD backtest history (they
"know the past" — every regime) and let them KEEP LEARNING with live data.
No cold rebuild from scratch.

Strategy (Claude plan: warm-start / continuation):
  - Load each deployed seed model's weights (the history-informed model).
  - Continue boosting on the current live feature matrix (gold_features.csv,
    includes appended live outcomes) so the model ADAPTS to the recent regime
    while never losing its historical foundation.
  - Store the continued models (deep... same names the engine loads) with
    atomic swap. Also re-emit OOF probs for backtest/calibration consistency.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import fit_calibration

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MODEL_DIR = f"{BASE}/models"
FEAT_CSV = f"{BASE}/gold_features.csv"
FEATURE_EXCLUDE = {"time", "target", "fwd_return"}
SEEDS = [42, 7, 2026]
RECENCY_TAU_DAYS = 120.0
ROWS_PER_BAR = 48
CONTINUE_ROUNDS = 200      # extra boosting rounds on top of the base model

def lgb_params(seed):
    return {"objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 50, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
            "verbose": -1, "num_threads": 8, "seed": seed}

def recency_weights(times_sec, tau_days=RECENCY_TAU_DAYS):
    # times_sec: int64 epoch-seconds (already converted during streaming load)
    age = (times_sec.max() - times_sec) / 86400.0
    w = np.exp(-age / tau_days); w = w / w.mean()
    return w.astype(np.float32)

def load_matrix_streaming(feats):
    """Chunked loader: preallocated float32 X / int8 y / int64 epoch-sec times.
    Peak ~3.5GB for 8.2M rows instead of read_csv double-buffer OOM."""
    from features import RAW_PRICE_COLS
    import subprocess as _sp
    n = int(_sp.run(["bash", "-c", f"wc -l < {FEAT_CSV}"],
                    capture_output=True, text=True).stdout.strip()) - 1
    if n < 1000:
        print("too few rows; abort"); return None
    X = np.empty((n, len(feats)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    times = np.empty(n, dtype=np.int64)  # epoch seconds
    read_cols = feats + ["target", "time"]
    dtype = {c: np.float32 for c in read_cols if c != "time"}
    start = 0
    for chunk in pd.read_csv(FEAT_CSV, usecols=read_cols, dtype=dtype,
                             chunksize=500_000, low_memory=False):
        end = start + len(chunk)
        X[start:end] = chunk[feats].values
        y[start:end] = chunk["target"].values
        times[start:end] = (pd.to_datetime(chunk["time"]).astype("datetime64[s]").astype("int64")).values
        start = end
        del chunk
    return X, y, times

def main():
    t0 = time.time()
    from features import RAW_PRICE_COLS
    all_cols = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    feats = [c for c in all_cols if c not in FEATURE_EXCLUDE and c not in RAW_PRICE_COLS]
    loaded = load_matrix_streaming(feats)
    if loaded is None:
        return 1
    X, y, times = loaded
    print(f"CONTINUE — {len(X):,} rows | {len(feats)} feats | "
          f"{datetime.utcfromtimestamp(times[0]).date()} -> "
          f"{datetime.utcfromtimestamp(times[-1]).date()}")
    w = recency_weights(times)
    del times; import gc; gc.collect()

    # ── warm-start continuation from existing history model per seed ──
    for s in SEEDS:
        base = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        if os.path.exists(base):
            model = lgb.train(lgb_params(s),
                              lgb.Dataset(X, label=y, weight=w, free_raw_data=True),
                              num_boost_round=CONTINUE_ROUNDS,
                              init_model=base)
            print(f"  seed {s}: warm-start from base +{CONTINUE_ROUNDS} rounds")
        else:
            model = lgb.train(lgb_params(s),
                              lgb.Dataset(X, label=y, weight=w, free_raw_data=True),
                              num_boost_round=600)
            print(f"  seed {s}: no base → cold 600 rounds")
        name = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        tmp = name + ".tmp"; model.save_model(tmp); os.replace(tmp, name)
        del model; gc.collect()

    # ── full-length aligned OOF for downstream calibration ──
    # fit_calibration_by_rr.py consumes oof_probs.npy/oof_targets.npy index-
    # aligned with gold_features.csv rows (direction, rr_buy, rr_sell), so they
    # must be length-n walk-forward probs from the CONTINUED ensemble — NOT a
    # 0.5 placeholder (that would flatten calibration to garbage).
    oof = np.zeros(len(X)); oof_cnt = np.zeros(len(X))
    n_bars = len(X) // ROWS_PER_BAR
    chunk = max(n_bars // 5, 1)
    for i in range(2, 5):
        tr_end = i * chunk * ROWS_PER_BAR
        te_end = min((i + 1) * chunk * ROWS_PER_BAR, len(X))
        if te_end <= tr_end:
            break
        te_idx = np.arange(tr_end, te_end, dtype=int)
        for s in SEEDS:
            b = lgb.Booster(model_file=f"{MODEL_DIR}/gold_lgb_model_s{s}.txt")
            oof[te_idx] += b.predict(X[te_idx]) / len(SEEDS)
        oof_cnt[te_idx] = 1
    miss = oof_cnt == 0
    if miss.any():
        for s in SEEDS:
            b = lgb.Booster(model_file=f"{MODEL_DIR}/gold_lgb_model_s{s}.txt")
            oof[miss] += b.predict(X[miss]) / len(SEEDS)
    oof_y = y.astype(np.int8)
    acc = ((oof >= 0.5).astype(int) == oof_y).mean()
    print(f"  OOF aligned len={len(oof):,} | acc={acc:.3f}")
    np.save(f"{MODEL_DIR}/oof_probs.npy", oof.astype(np.float32))
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
                   "recency_tau_days": RECENCY_TAU_DAYS, "mode": "warm-start-continue"},
                  f, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] ✅ continuation complete in {time.time()-t0:.0f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())