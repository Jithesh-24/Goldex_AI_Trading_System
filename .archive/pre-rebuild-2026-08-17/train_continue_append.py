"""train_continue_append.py — v8.4b (2026-08-08)

WARM-START CONTINUATION on the NEWLY APPENDED rows ONLY.

User mandate (verbatim intent):
  "no full retrain only the missing data should be retrained and appended
   so the model will be aware of all kind of markets... not only 6 years
   of rally... in those data so many regimes happened"

Strategy:
  - Deployed models already KNOW the 6yr history (weights frozen base).
  - Continue boosting ONLY on the appended missing-period rows, warm-
    starting from each deployed seed (init_model), +CONTINUE_ROUNDS per
    period. Chained across periods: 2021 -> 2024H2 -> 2025H2 so every
    specialist sees every new regime in order (memory-bounded: one
    period's rows at a time).
  - Regime specialists: bucket new rows by regime_bin, continue each
    regime's specialist on ITS new rows.
  - Global ensemble: continue each seed on all new rows of each period.
  - Atomic swap (.tmp -> os.replace) + touch configs so the engine
    hot-reloads (it watches config mtime).
"""
import numpy as np
import os
import pandas as pd
import lightgbm as lgb
import json, sys, time, gc
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MODEL_DIR = f"{BASE}/models"
FEAT_CSV = f"{BASE}/gold_features_m5.csv"
FEATURE_EXCLUDE = {"time", "target", "fwd_return", "mfe_atr", "mfa_atr"}
SEEDS = [42, 7, 2026]
CONTINUE_ROUNDS = int(os.environ.get("CONTINUE_ROUNDS", "200"))
CHUNK = 500_000
REGIME_KEYS = F.REGIME_KEYS
REGIME_NAMES = F.REGIME_NAMES
MIN_ROWS = 5_000

# ── NEW ROWS = the appended layers (missing periods), oldest first ──
# Original matrix covered: 2019-12..2020-08, 2022-03..09, 2023-01..04,
# 2024-01..05, 2025-01..05, 2026-06..08. Everything ELSE is new.
NEW_ROW_RANGES = [
    ("2020H2", "2020-09-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022Q1", "2022-01-01", "2022-03-01"),
    ("2022Q4", "2022-10-01", "2023-01-01"),
    ("2023H2", "2023-05-01", "2024-01-01"),
    ("2024H2", "2024-06-01", "2025-01-01"),
    ("2025H2", "2025-06-01", "2026-01-01"),
    ("2026H1", "2026-01-01", "2026-06-01"),
]

def lgb_params(seed):
    return {"objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 50, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
            "verbose": -1, "num_threads": 4, "seed": seed}

def recency_weights(times_sec, tau_days=120.0):
    age = (times_sec.max() - times_sec) / 86400.0
    w = np.exp(-age / tau_days); w = w / w.mean()
    return w.astype(np.float32)

def regime_of_frame(df):
    """Vectorized regime assignment — mirrors F.regime_bin() EXACTLY.

    Pure pandas/NumPy version of the per-row thresholds in features.regime_bin
    (trend first, then vol overlays, then range bins). MUST stay in lockstep
    with the engine's per-row version — the trainer and engine must agree on
    which regime a bar belongs to.
    """
    te = df["trend_ema"].astype(float).values
    ts = df["trend_slope"].astype(float).values
    bb = df["bb_pctile"].astype(float).values
    ap = df["atr_pctile"].astype(float).values
    vs = df["vol_spike"].astype(float).values
    ns = df["news_candle"].astype(float).values
    rsi = df["rsi_14"].astype(float).values
    volr = np.abs(df["m1_d1_vol_ratio"].astype(float).values)

    out = np.full(len(df), -1, dtype=np.int8)
    # trend bins first — indices MUST match F.REGIME_NAMES order:
    # 0 STRONG_UP, 1 UP, 2 DOWN, 3 STRONG_DOWN, 4 RANGE_TIGHT,
    # 5 RANGE_WIDE, 6 HIGH_VOL, 7 QUIET_LOW_VOL
    m = (te > 1.2) & (ts * te > 0)
    out[m] = 0  # STRONG_UP
    m = (te > 0.4) & (out == -1)
    out[m] = 1  # UP
    m = (te < -1.2) & (ts * te > 0)
    out[m] = 3  # STRONG_DOWN
    m = (te < -0.4) & (out == -1)
    out[m] = 2  # DOWN
    # vol overlays for non-trend bars
    m = (out == -1) & ((vs > 2.0) | (ns > 0.4) | (ap > 0.85))
    out[m] = 6  # HIGH_VOL
    m = (out == -1) & (ap < 0.15) & (ns < 0.2)
    out[m] = 7  # QUIET_LOW_VOL
    # range bins
    m = (out == -1) & (bb < 0.35)
    out[m] = 4  # RANGE_TIGHT
    m = (out == -1) & (rsi > 60) & (volr > 1.2)
    out[m] = 1  # UP
    m = (out == -1) & (rsi < 40) & (volr > 1.2)
    out[m] = 2  # DOWN
    out[out == -1] = 5  # RANGE_WIDE
    return out

# indices must match F.REGIME_NAMES order
REGIME_IDX = {n: i for i, n in enumerate(F.REGIME_NAMES)}
assert list(REGIME_IDX) == F.REGIME_NAMES, "regime order mismatch"

def load_period(feats, a, b):
    """Stream ONE period's rows (bounded memory). Returns (X, y, times, regs)."""
    hdr = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    read_cols = feats + ["target", "time"] + list(REGIME_KEYS)
    dt = {c: np.float32 for c in read_cols if c != "time"}
    parts_x, parts_y, parts_t, parts_r = [], [], [], []
    for chunk in pd.read_csv(FEAT_CSV, usecols=read_cols, dtype=dt,
                             chunksize=CHUNK, low_memory=False):
        t = pd.to_datetime(chunk["time"], utc=True).dt.tz_localize(None)
        m = (t >= a) & (t < b)
        # target NaN = no forward bars (matrix tail) — cannot train, drop
        m &= chunk["target"].notna().values
        if m.any():
            parts_x.append(chunk.loc[m, feats].values)
            parts_y.append(chunk.loc[m, "target"].values.astype(np.int8))
            parts_t.append(t[m].values.astype("datetime64[s]").astype("int64"))
            parts_r.append(regime_of_frame(chunk.loc[m, REGIME_KEYS]))
        del chunk; gc.collect()
    if not parts_x:
        return None
    X = np.vstack(parts_x).astype(np.float32)
    y = np.concatenate(parts_y).astype(np.int8)
    times = np.concatenate(parts_t)
    regs = np.concatenate(parts_r)
    del parts_x, parts_y, parts_t, parts_r; gc.collect()
    return X, y, times, regs

def continue_model(model_path, X, y, w, tag):
    """Warm-start one model path with +CONTINUE_ROUNDS on (X, y, w)."""
    for s in SEEDS:
        base = model_path.replace("{seed}", str(s))
        if os.path.exists(base):
            model = lgb.train(lgb_params(s), lgb.Dataset(X, label=y, weight=w),
                              num_boost_round=CONTINUE_ROUNDS, init_model=base)
            how = f"+{CONTINUE_ROUNDS}"
        else:
            model = lgb.train(lgb_params(s), lgb.Dataset(X, label=y, weight=w),
                              num_boost_round=600)
            how = "cold600"
        tmp = base + ".tmp"; model.save_model(tmp); os.replace(tmp, base)
        del model; gc.collect()
    print(f"  {tag}: warm-start {how} on {len(X):,} rows", flush=True)

def main():
    t0 = time.time()
    hdr = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    feats = [c for c in hdr if c not in FEATURE_EXCLUDE and c not in F.RAW_PRICE_COLS]
    print(f"═══ WARM-START CONTINUATION (new rows only) ═══\nfeats: {len(feats)}", flush=True)

    # PASS 1: per-period regime specialists (chained across periods)
    for label, a, b in NEW_ROW_RANGES:
        loaded = load_period(feats, a, b)
        if loaded is None:
            print(f"  {label}: no rows in range — skip", flush=True)
            continue
        X, y, times, regs = loaded
        w = recency_weights(times)
        print(f"  {label}: {len(X):,} rows | {datetime.utcfromtimestamp(times.min()).date()} "
              f"-> {datetime.utcfromtimestamp(times.max()).date()} | WR {y.mean():.3f}", flush=True)
        # per-regime buckets (regs is int8 index array — compare against indices!)
        for ridx, r in enumerate(REGIME_NAMES):
            m = regs == ridx
            nm = int(m.sum())
            if nm < MIN_ROWS:
                if nm:
                    print(f"    {r}: {nm:,} — below {MIN_ROWS}, skip", flush=True)
                continue
            path = f"{MODEL_DIR}/spec_{r.lower()}_s{{seed}}.txt"
            continue_model(path, X[m], y[m], w[m], f"{r} ({label})")
        del X, y, times, regs, w; gc.collect()

    # PASS 2: global ensemble on all new rows, chained across periods
    for label, a, b in NEW_ROW_RANGES:
        loaded = load_period(feats, a, b)
        if loaded is None:
            continue
        X, y, times, regs = loaded
        w = recency_weights(times)
        path = f"{MODEL_DIR}/gold_lgb_model_s{{seed}}.txt"
        continue_model(path, X, y, w, f"global ({label})")
        del X, y, times, regs, w; gc.collect()

    # touch configs so engine hot-reloads next tick
    for p in ("regime_specialists.json", "ensemble.json"):
        fp = f"{MODEL_DIR}/{p}"
        if os.path.exists(fp):
            os.utime(fp, None)
    print("  touched configs — engine hot-reload next tick", flush=True)
    print(f"✅ continuation done in {time.time()-t0:.0f}s", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
