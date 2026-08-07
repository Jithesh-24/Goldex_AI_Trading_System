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
import os
import pandas as pd
import lightgbm as lgb
import json, os, sys, time, gc
from datetime import datetime

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MODEL_DIR = f"{BASE}/models"
FEAT_CSV = os.environ.get("FEAT_CSV", f"{BASE}/gold_features.csv")
FEATURE_EXCLUDE = {"time", "target", "fwd_return"}
SEEDS = [42, 7, 2026]
# 2026-08-06 FIX: i5-10210U = 4 physical cores / 8 logical. num_threads=8
# spins on hyperthreaded logical CPUs -> 24x SLOWER (17s@1T, 8.6s@4T, 209s@8T).
N_THREADS = 4
ROWS_PER_BAR = 48
RECENCY_TAU_DAYS = 120.0
CAL_SPLIT = 0.9   # calibration OOF: train on first 90% (by time), hold out 10%
CAL_ROUNDS = 300  # calibration pass early-stops; deployment keeps 600


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
    read_cols = feats + ["target", "direction", "rr_buy", "rr_sell", "time"]
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


def rr_bucket_spec(rr):
    """Map an effective RR to the nearest grid ratio."""
    from features import TP_RATIOS
    return min(TP_RATIOS, key=lambda t: abs(rr - t))


def _fit_spec_calibration(regime, oof, oofy, drr, dirs, min_support=5000):
    """Fit per-dir × per-RR calibration for ONE regime from its own OOF.

    Mirrors fit_calibration_by_rr.py but on the SPECIALIST's honest OOF,
    so the deployed probability scale matches the model that produced it.

    v7.7c FIX (2026-08-07): the old version derived the direction mask from
    `drr` (the RR VALUE, always 1.3-3.0) — `drr > 0.5` matched EVERY row, so
    all rows were fitted into BUY curves and SELL calibration was NEVER
    written. At signal time SELL fell back to raw (overconfident) model
    probability while BUY was calibrated → the placement sweep systematically
    favored SELL (bot shorted every uptrend, 6 SL / 1 TP in 40 min). Now the
    true direction mask is passed in explicitly and saved alongside.
    """
    from features import TP_RATIOS
    from fit_calibration_by_rr import pava_bins
    import numpy as _np
    out = {}
    dirs = _np.asarray(dirs, dtype=bool)
    dirs_buy = dirs                 # TRUE = BUY row (direction feature > 0.5)
    dirs_sell = ~dirs               # FALSE = SELL row
    _np.save(f"{MODEL_DIR}/dirmask_spec_{regime.lower()}.npy", dirs)
    for di, dname in enumerate(["BUY", "SELL"]):
        d_mask = dirs_buy if dname == "BUY" else dirs_sell
        for ri, t in enumerate(TP_RATIOS):
            key = f"{dname}_{t}"
            m = d_mask & (_np.array([rr_bucket_spec(r) for r in drr]) == t)
            if m.sum() < min_support:
                continue
            ps, ys = pava_bins(oof[m], oofy[m])
            out[key] = {"knots_p": ps.tolist(), "knots_y": ys.tolist(),
                        "n": int(m.sum())}
    if out:
        path = f"{MODEL_DIR}/calibration_by_drr_spec_{regime.lower()}.json"
        with open(path + ".tmp", "w") as f:
            json.dump(out, f, indent=2)
        os.replace(path + ".tmp", path)
    return out


def train_regime_from_file(regime, tmp_file, feats, MODEL_DIR, t0):
    """Train one regime's 3-seed placement ensemble from its temp CSV.

    v7.7b (2026-08-06): ALSO emit a per-regime OOF + per-dir×RR calibration.
    Previously specialists were calibrated with the BASE model's curves
    (calibration_by_drr.json) — wrong model, wrong probability scale.
    Now: train a calibration pass on the first 90% of the regime's rows
    (by time), predict the held-out 10% → honest specialist OOF → fit
    calibration_by_drr_spec_<regime>.json. The engine picks the specialist
    curve when it routes to this regime (fallback: base curves).
    """
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

    # ── v7.7b per-regime OOF + calibration ──
    # Time-ordered 90/10 split; train calibration seeds on the EARLY 90%,
    # predict the RECENT 10% (the regime the engine will actually face).
    try:
        order = np.argsort(times.astype("datetime64[s]").astype(np.int64))
        cut = int(len(order) * CAL_SPLIT)
        tr = order[:cut]
        va = order[cut:]
        if len(tr) > 20000 and len(va) > 5000:
            Xva = X[va]
            yva = y[va]
            oof = np.zeros(len(va), dtype=np.float32)
            for s in SEEDS:
                cal_params = dict(lgb_params(s))
                cal_params.pop("early_stopping_rounds", None)
                md = lgb.train(
                    cal_params,
                    lgb.Dataset(X[tr], label=y[tr], weight=w[tr],
                                free_raw_data=True),
                    num_boost_round=CAL_ROUNDS,
                    valid_sets=[lgb.Dataset(Xva, label=yva, weight=w[va])],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
                oof += md.predict(Xva, num_iteration=md.best_iteration or CAL_ROUNDS)
                del md
                gc.collect()
            oof /= len(SEEDS)
            np.save(f"{MODEL_DIR}/oof_spec_{regime.lower()}.npy", oof)
            np.save(f"{MODEL_DIR}/oofy_spec_{regime.lower()}.npy", yva)
            # per-row direction + effective RR → fit per-dir×RR curves
            drr = np.where(df["direction"].values[va] > 0.5,
                           df["rr_buy"].values[va], df["rr_sell"].values[va])
            np.save(f"{MODEL_DIR}/drr_spec_{regime.lower()}.npy", drr)
            dirs_va = df["direction"].values[va] > 0.5  # v7.7c: TRUE=BUY mask
            _fit_spec_calibration(regime, oof, yva, drr, dirs_va)
            print(f"  {regime}: OOF n={len(va):,} | WR {yva.mean():.1%} | "
                  f"calibration saved", flush=True)
    except Exception as e:
        print(f"  {regime}: ⚠ OOF/calibration skipped: {e}", flush=True)

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

    # PASS 2: train one 3-seed ensemble per non-empty regime.
    # v7.7c (2026-08-06): RESUMABLE. If all 3 seed files for a regime exist
    # AND are newer than the bucket start, skip re-training (gateway
    # shutdowns killed this run twice; completed regimes must not be wasted).
    # The per-regime OOF calibration is also skipped if already saved.
    spec_map = {}
    for n in F.REGIME_NAMES:
        if n not in nonempty:
            print(f"  {n}: EMPTY — skip", flush=True)
            continue
        key = n.lower()
        seed_files = [f"{MODEL_DIR}/spec_{key}_s{s}.txt" for s in SEEDS]
        cal_file = f"{MODEL_DIR}/calibration_by_drr_spec_{key}.json"
        try:
            # Fresh = written within the last 24h (today's retrain), NOT older
            # than the old Aug 4 deployment. Using t_bucket_done would reject
            # seeds written in a PREVIOUS (killed) run of today's retrain.
            all_fresh = all(
                os.path.exists(p) and os.path.getmtime(p) >= time.time() - 86400
                for p in seed_files)
            cal_ok = os.path.exists(cal_file)
        except Exception:
            all_fresh = False
            cal_ok = False
        if all_fresh and cal_ok:
            print(f"  {n}: SKIP (already trained with fixes, calibration present)",
                  flush=True)
            spec_map[n] = {"models": [os.path.basename(p) for p in seed_files],
                           "seeds": SEEDS, "rows": coverage[n]}
            continue
        regime, files, rows = train_regime_from_file(
            n, os.path.join(TMP_DIR, f"{n}.csv"), feats, MODEL_DIR, t0)
        spec_map[regime] = {"models": files, "seeds": SEEDS, "rows": rows}

    with open(f"{MODEL_DIR}/regime_specialists.json", "w") as f:
        json.dump({"type": "regime-placement", "bins": spec_map, "seeds": SEEDS,
                   "mode": "cold-fresh-per-regime", "creator": "train_regime_spec",
                   "coverage": coverage,
                   "base_tf": os.environ.get("PRIOR_BAR_SECS", "180") == "300" and "m5" or "m1"},
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