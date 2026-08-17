"""train_global_m5.py — REBUILD the M5 global placement ensemble (v8.4b fix).

WHY: the EOD loop (retrain_loop.py → train_continue.py on the M1 matrix
gold_features.csv) overwrote the deployed M5 global models with M1-lineage
models (102 Column_* features). The engine's TF guard refused the poisoned
config and kept last-valid M5 in memory, but the .txt files on disk are
destroyed (gitignored, no backup). PASS 2 of train_continue_append.py then
crashed: 108 features in data vs 102 in init model.

THIS SCRIPT: cold-rebuilds the global 3-seed ensemble at M5 (base_tf=m5,
108 features) on the FULL 18.5M-row matrix, memory-safe:
  - Pass A: stream the matrix ONCE in time order, write per-bucket .npy
    snapshots (X, y, times) to tmp — peak RAM = one bucket (~1.4GB).
  - Pass B: bucket 0 cold-train (600 rounds), buckets 1..N warm-continue
    (+200 rounds) — chaining through ALL 6yr in time order, so the final
    models have seen every bar of the continuum (recency-weighted).
Writes gold_lgb_model_s{42,7,2026}.txt atomically + ensemble.json
(base_tf=m5) + features.json. Engine hot-reloads on ensemble.json mtime.

Run: python train_global_m5.py   (~90 min, num_threads=4, bounded RAM)
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

FEAT_CSV = f"{BASE}/gold_features_m5.csv"
MODEL_DIR = f"{BASE}/models"
TMP = "/home/jith/.hermes/profiles/trading/tmp/global_rebuild"
SEEDS = [42, 7, 2026]
CHUNK = 500_000
CONTINUE_ROUNDS = 200
COLD_ROUNDS = 600

# 6 time-ordered buckets covering 2019-12-01 → 2026-08-06 (2442 days / 6)
BUCKETS = [
    ("2019-12-01", "2021-01-12"),
    ("2021-01-12", "2022-02-23"),
    ("2022-02-23", "2023-04-06"),
    ("2023-04-06", "2024-05-18"),
    ("2024-05-18", "2025-06-29"),
    ("2025-06-29", "2026-08-07"),
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

def main():
    t0 = time.time()
    os.makedirs(TMP, exist_ok=True)
    hdr = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    # Use the EXACT live feature list (features.json = what the engine feeds
    # predict). Never recompute from column exclusions — a mismatch would
    # silently re-poison the model.
    with open(f"{MODEL_DIR}/features.json") as f:
        feats = json.load(f)
    missing = [c for c in feats if c not in hdr]
    if missing:
        print(f"❌ features.json lists {len(missing)} cols missing from matrix: "
              f"{missing[:8]}", flush=True)
        return 1
    print(f"═══ GLOBAL M5 REBUILD ═══\nfeats: {len(feats)} (exact live list)",
          flush=True)

    # ── Pass A: bucket the matrix (stream once, time-ordered) ──
    # Memory-safe: assign each chunk's rows to their buckets, flush a bucket
    # as soon as the stream has passed its upper bound. At most 2 buckets
    # pending at once (chunks span ≤1 boundary). Never drops straddle rows.
    b_lo = [np.datetime64(a) for a, _ in BUCKETS]
    b_hi = [np.datetime64(b) for _, b in BUCKETS]
    read_cols = feats + ["target", "time"]
    dt = {c: np.float32 for c in read_cols if c != "time"}
    n_rows = 0
    pending = {}  # bi -> [x_chunks], [y_chunks], [t_chunks]
    def flush_bucket(bi):
        if bi not in pending or not pending[bi][0]:
            return
        xp, yp, tp = pending.pop(bi)
        X = np.vstack(xp); y = np.concatenate(yp); times = np.concatenate(tp)
        np.save(f"{TMP}/b{bi}_X.npy", X); np.save(f"{TMP}/b{bi}_y.npy", y)
        np.save(f"{TMP}/b{bi}_t.npy", times)
        print(f"  bucket {bi} flushed: {len(X):,} rows | "
              f"{pd.Timestamp(times.min(), unit='s').date()} -> "
              f"{pd.Timestamp(times.max(), unit='s').date()} | WR {y.mean():.3f}",
              flush=True)
        del X, y, times; gc.collect()
    for ci, chunk in enumerate(pd.read_csv(FEAT_CSV, usecols=read_cols,
                                           dtype=dt, chunksize=CHUNK,
                                           low_memory=False)):
        t = pd.to_datetime(chunk["time"], utc=True).dt.tz_localize(None)
        t64 = t.values.astype("datetime64[s]")
        ok = chunk["target"].notna().values
        Xc = chunk[feats].values.astype(np.float32)
        yc = chunk["target"].values.astype(np.int8)
        # assign rows to ALL buckets they fall in (straddle-safe)
        for bi in range(len(BUCKETS)):
            m = (t64 >= b_lo[bi]) & (t64 < b_hi[bi]) & ok
            if m.any():
                if bi not in pending:
                    pending[bi] = ([], [], [])
                pending[bi][0].append(Xc[m]); pending[bi][1].append(yc[m])
                pending[bi][2].append(t64[m].astype("int64"))
        # flush buckets whose window has fully passed (stream is time-ordered)
        if len(t64):
            tmin = t64.min()
            for bi in sorted(pending):
                if b_hi[bi] <= tmin:
                    flush_bucket(bi)
        n_rows += len(chunk)
        del chunk, t, t64, Xc, yc; gc.collect()
    for bi in sorted(pending):
        flush_bucket(bi)
    print(f"Pass A done: {n_rows:,} total rows scanned ({time.time()-t0:.0f}s)",
          flush=True)

    # ── Pass B: chain cold → warm across buckets, per seed ──
    for s in SEEDS:
        out = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        for bi in range(len(BUCKETS)):
            X = np.load(f"{TMP}/b{bi}_X.npy"); y = np.load(f"{TMP}/b{bi}_y.npy")
            times = np.load(f"{TMP}/b{bi}_t.npy")
            w = recency_weights(times)
            dset = lgb.Dataset(X, label=y, weight=w, free_raw_data=True)
            if bi == 0 or not os.path.exists(out):
                model = lgb.train(lgb_params(s), dset, num_boost_round=COLD_ROUNDS)
                how = f"cold{COLD_ROUNDS}"
            else:
                model = lgb.train(lgb_params(s), dset, num_boost_round=CONTINUE_ROUNDS,
                                  init_model=out)
                how = f"+{CONTINUE_ROUNDS}"
            tmp = out + ".tmp"; model.save_model(tmp); os.replace(tmp, out)
            print(f"  seed {s} bucket {bi}: {how} on {len(X):,} rows", flush=True)
            del X, y, times, w, dset, model; gc.collect()

    # ── stamp configs ──
    with open(f"{MODEL_DIR}/features.json", "w") as f:
        json.dump(feats, f)
    with open(f"{MODEL_DIR}/ensemble.json", "w") as f:
        json.dump({"type": "placement", "seeds": SEEDS,
                   "models": [f"gold_lgb_model_s{s}.txt" for s in SEEDS],
                   "recency_tau_days": 120.0, "mode": "warm-start-continue",
                   "base_tf": "m5"}, f, indent=2)
    os.utime(f"{MODEL_DIR}/ensemble.json", None)
    print(f"✅ GLOBAL M5 REBUILD COMPLETE ({time.time()-t0:.0f}s) — "
          f"ensemble.json stamped base_tf=m5, engine hot-reloads next tick",
          flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
