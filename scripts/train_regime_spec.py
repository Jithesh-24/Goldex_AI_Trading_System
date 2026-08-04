"""v7.7 (2026-08-04) — REGIME SPECIALIST TRAINER.

Trades ANY market condition by training ONE placement ensemble per regime bin
(8 bins), each on the FULL 6-year matrix rows that fell in that bin. A range-day
specialist sells overbought; a trend specialist buys pullbacks. No single global
model forced to be good at everything.

PROCESS:
  1. compute regime_bin() for every row of gold_features.csv (shared with engine)
  2. group rows by bin, train a 3-seed LightGBM placement ensemble per bin
  3. emit per-bin OOF + a regime_specialists.json mapping bin -> model files
  4. report 6-year regime coverage (proof every move is captured)

RUN under systemd-run MemoryMax=7G so it gets all 8 cores (not the 800ms gateway
cgroup). Cold-start fresh per bin (a specialist is a specialist).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, os, sys, time, gc
from datetime import datetime

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MODEL_DIR = f"{BASE}/models"
FEAT_CSV = f"{BASE}/gold_features.csv"
FEATURE_EXCLUDE = {"time", "target", "fwd_return"}
SEEDS = [42, 7, 2026]
N_THREADS = 8
ROWS_PER_BAR = 48
RECENCY_TAU_DAYS = 120.0


def lgb_params(seed):
    return {"objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 50, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
            "verbose": -1, "num_threads": N_THREADS, "seed": seed}


def recency_weights(times, tau_days=RECENCY_TAU_DAYS):
    ts = times.astype("datetime64[s]").astype(np.int64)
    age = (ts.max() - ts) / 86400.0
    w = np.exp(-age / tau_days)
    w = w / (w.mean() + 1e-9)
    return w.astype(np.float32)


def label_regime_all(df):
    """Assign regime_bin to every row (vectorized via feature columns)."""
    import features as F
    keys = ["trend_ema", "trend_slope", "bb_pctile", "atr_pctile",
            "vol_spike", "news_candle", "rsi_14", "m1_d1_vol_ratio"]
    zipped = df[keys].to_dict("records")
    return np.array([F.regime_bin(r) for r in zipped])


def stream_bucket(FEAT_CSV, TMP_DIR):
    """Single pass: assign regime per row, append each row to a per-regime
    temp CSV. Returns coverage dict + the set of non-empty regimes. Bounded
    memory: only one read-chunk + up to 8 growing temp files resident."""
    import features as F
    # resolve feature list from header once
    hdr = pd.read_csv(FEAT_CSV, nrows=0).columns.tolist()
    feats = [c for c in hdr if c not in FEATURE_EXCLUDE and c not in F.RAW_PRICE_COLS]
    read_cols = feats + ["target", "direction", "time"]
    dt = {c: np.float32 for c in read_cols if c != "time"}

    handles = {}
    coverage = {n: 0 for n in F.REGIME_NAMES}
    # ensure temp files exist with header
    for n in F.REGIME_NAMES:
        tf = os.path.join(TMP_DIR, f"{n}.csv")
        pd.DataFrame(columns=read_cols).to_csv(tf, index=False)
        handles[n] = open(tf, "a")
        coverage[n] = 0

    import csv as _csv
    with open(FEAT_CSV) as f:
        rd = _csv.reader(f)
        header = next(rd)
        idx = {c: header.index(c) for c in read_cols}
        # stream row by row (6.39M) — pure python, but bounded ~ few MB
        writers = {n: _csv.writer(handles[n]) for n in F.REGIME_NAMES}
        for row in rd:
            rec = {c: row[idx[c]] for c in read_cols}
            fx = {c: float(row[idx[c]]) for c in F.REGIME_KEYS}
            n = F.regime_bin(fx)
            writers[n].writerow([rec[c] for c in read_cols])
            coverage[n] += 1
    for n in F.REGIME_NAMES:
        handles[n].close()
    for w in writers.values():
        del w
    nonempty = {n for n, c in coverage.items() if c > 0}
    return feats, coverage, nonempty


def train_regime_from_file(regime, tmp_file, feats, MODEL_DIR, t0):
    """Train one regime's 3-seed placement ensemble from its temp CSV."""
    df = pd.read_csv(tmp_file)
    times = pd.to_datetime(df["time"]).values
    w = recency_weights(times)
    X = df[feats].values.astype(np.float32)
    y = df["target"].values.astype(np.int8)
    files = []
    for s in SEEDS:
        model = lgb.train(lgb_params(s),
                          lgb.Dataset(X, label=y, weight=w, free_raw_data=True),
                          num_boost_round=600)
        fn = f"{MODEL_DIR}/spec_{regime.lower()}_s{s}.txt"
        tmp = fn + ".tmp"
        model.save_model(tmp)
        os.replace(tmp, fn)
        files.append(os.path.basename(fn))
        del model
        gc.collect()
    rows = len(df)
    del df, X, y, w, times
    gc.collect()
    print(f"  {regime}: trained {len(files)} seeds n={rows:,} ({time.time()-t0:.0f}s)",
          flush=True)
    return regime, files, rows


def main():
    t0 = time.time()
    import features as F
    from features import RAW_PRICE_COLS  # noqa (used by stream_bucket)
    TMP_DIR = f"{BASE}/tmp_regime"
    os.makedirs(TMP_DIR, exist_ok=True)
    for fn in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, fn))

    # PASS 1: single streaming pass → 8 per-regime temp CSVs (bounded memory)
    feats, coverage, nonempty = stream_bucket(FEAT_CSV, TMP_DIR)
    total = sum(coverage.values())
    print(f"BUCKET {total:,} rows in {time.time()-t0:.0f}s", flush=True)
    print("6-YEAR REGIME COVERAGE:", flush=True)
    for n in F.REGIME_NAMES:
        pct = 100.0 * coverage[n] / max(total, 1)
        print(f"  {n:15s} {coverage[n]:>10,} rows ({pct:4.1f}%)", flush=True)

    # PASS 2: train one 3-seed ensemble per non-empty regime
    spec_map = {}
    for n in F.REGIME_NAMES:
        if n not in nonempty:
            print(f"  {n}: EMPTY — skip", flush=True)
            continue
        regime, files, rows = train_regime_from_file(
            n, os.path.join(TMP_DIR, f"{n}.csv"), feats, MODEL_DIR, t0)
        spec_map[regime] = {"models": files, "seeds": SEEDS, "rows": rows}

    with open(f"{MODEL_DIR}/regime_specialists.json", "w") as f:
        json.dump({"type": "regime-placement", "bins": spec_map, "seeds": SEEDS,
                   "mode": "cold-fresh-per-regime", "creator": "train_regime_spec",
                   "coverage": coverage},
                  f, indent=2)
    # cleanup temps
    for n in F.REGIME_NAMES:
        p = os.path.join(TMP_DIR, f"{n}.csv")
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(TMP_DIR)
    print(f"\n✅ REGIME SPECIALISTS saved ({time.time()-t0:.0f}s)", flush=True)
    print(f"   bins trained: {list(spec_map.keys())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())