"""FINAL EDITION (v7 2026-08-02) — DIRECTION MODEL ("when to trade").

The placement model answers "given I take THIS trade, does it win?".
The direction model answers "is the market ABOUT to move up?" — a pure
directional prior the engine multiplies into expectancy:

    final_exp(BUY)  = Exp_placement(BUY)  × P(up)
    final_exp(SELL) = Exp_placement(SELL) × (1 − P(up))

No gates, no thresholds — a learned probability multiplier. This is what
teaches "don't catch falling knives": in a downtrend P(up) is low, so BUY
expectancy gets crushed and SELL wins the comparison naturally.

Data: 1 row per bar (no placement geometry), same market features, target
= did close rise ≥ spread over the next 60 bars (trade-realistic horizon,
matches MAX_TARGET_BARS). 3-seed ensemble, atomic save to
models/direction_s{s}.txt + models/direction_features.json.

Memory-safe: built by sampling 1 row per bar from the existing placement
matrix (direction==1 rows carry the same market features + direction=1),
target re-derived from fwd prices is NOT in the matrix — instead we reuse
the placement target for direction=BUY as a first-order proxy? NO — the
placement target depends on SL/TP geometry. The correct directional label
needs raw closes. We rebuild the label from the SEED (gold_seed_multi.csv)
the same way features were built: per-period _feature_block + horizon-60
directional target on close prices.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, os, sys, time, gc
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MODEL_DIR = f"{BASE}/models"
SEEDS = [42, 7, 2026]
RECENCY_TAU_DAYS = 120.0
HORIZON_BARS = 60           # "did it move up within the next hour"
MIN_BARS_PER_PERIOD = 300

def lgb_params(seed):
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbose": -1,
        "num_threads": 4,   # 8-logical-CPU machine; direction data is small (1 row/bar)
        "seed": seed,
    }

def directional_label(df, horizon=HORIZON_BARS):
    """target = 1 if the +band is TOUCHED FIRST (before the -band) within
    `horizon` bars; 0 if the -band touches first. Mirrors real SL/TP
    resolution: the level hit first decides the trade.

    v7.3b (2026-08-02): REPLACED the close-vs-close $0.20 label. That label
    was ~48% up-rate noise — the model learned a coin flip (acc 0.5008) and
    the direction prior could not suppress wrong-side trades (TREND-UP WR
    19.1%, PF 0.71). A FIRST-TOUCH label with a vol-scaled band (~1.5 ATR)
    keeps every bar (no ambiguous drops) and discriminates: in an uptrend
    the +band touches first far more often, giving the model a REAL signal.
    Threshold scales with per-bar ATR so it works across 2020-2026 regimes.
    """
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    n = len(df)
    if "atr_14" in df.columns:
        atr = df["atr_14"].values
    else:
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(cl, 1)), np.abs(lo - np.roll(cl, 1))))
        atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    atr = np.nan_to_num(atr, nan=1.0)
    atr[atr < 0.05] = 0.05
    tgt = np.zeros(n)
    band = 1.5  # ATR multiples; wide enough that most bars resolve first-touch
    for i in range(n - 1):
        j_end = min(i + 1 + horizon, n)
        up = hi[i+1:j_end] >= cl[i] + band * atr[i]
        dn = lo[i+1:j_end] <= cl[i] - band * atr[i]
        if up.any() and not dn.any():
            tgt[i] = 1.0
        elif dn.any() and not up.any():
            tgt[i] = 0.0
        elif up.any() and dn.any():
            # both touched — the FIRST touch wins (real resolution)
            tgt[i] = 1.0 if np.argmax(up) <= np.argmax(dn) else 0.0
        else:
            tgt[i] = 0.5  # neither touched — dropped later
    df["target"] = tgt
    return df

def build_direction_matrix(seed_csv):
    """1 row/bar directional dataset: market features only (no geometry)."""
    df = pd.read_csv(seed_csv)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    t = df["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(df)]
    periods = [df.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)]
    periods = [p for p in periods if len(p) >= MIN_BARS_PER_PERIOD]
    blocks = []
    for p in periods:
        fdf = F._feature_block(p)
        fdf = directional_label(fdf)
        fdf = fdf.dropna().reset_index(drop=True)
        # v7.3b: drop ambiguous first-touch rows (both/neither band touched
        # within horizon) — they carry no directional signal and would
        # truncate to 0 via int8 and poison the binary classifier.
        fdf = fdf[fdf["target"] != 0.5].reset_index(drop=True)
        market_cols = [c for c in fdf.columns
                       if c not in ("time", "target", "fwd_return")
                       and c not in F.RAW_PRICE_COLS]
        out = fdf[market_cols + ["target", "time"]].copy()
        blocks.append(out)
    final = pd.concat(blocks, ignore_index=True)
    final = final.sort_values("time").reset_index(drop=True)
    return final, [c for c in final.columns if c not in ("time", "target")]

def recency_weights(times, tau_days=RECENCY_TAU_DAYS):
    ts = times.astype("datetime64[s]").astype(np.int64)
    age_days = (ts.max() - ts) / 86400.0
    w = np.exp(-age_days / tau_days)
    return (w / w.mean()).astype(np.float32)

def evaluate(probs, y, label=""):
    probs = np.array(probs); y = np.array(y)
    pred = (probs >= 0.5).astype(int)
    acc = (pred == y).mean()
    up_mask = pred == 1
    p_up = y[up_mask].mean() if up_mask.any() else float("nan")
    p_dn = (1 - y[~up_mask]).mean() if (~up_mask).any() else float("nan")
    print(f"  {label}: acc={acc:.3f} | P(up correct)={p_up:.3f} | P(down correct)={p_dn:.3f} | n={len(y)}")
    return {"acc": acc, "p_up": p_up, "p_dn": p_dn, "n": len(y)}

def main():
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] ═══ DIRECTION MODEL (v7) ═══")
    multi = f"{BASE}/gold_seed_multi.csv"
    seed_csv = multi if os.path.exists(multi) else f"{BASE}/gold_seed.csv"
    matrix, feats = build_direction_matrix(seed_csv)
    print(f"Direction matrix: {len(matrix):,} rows (1/bar) | {len(feats)} feats | "
          f"{matrix['time'].iloc[0]} -> {matrix['time'].iloc[-1]}")
    print(f"Target balance: {matrix['target'].value_counts().to_dict()}")

    times = pd.to_datetime(matrix["time"]).values
    X = matrix[feats].values.astype(np.float32)
    y = matrix["target"].values.astype(np.int8)
    w = recency_weights(times)
    del matrix
    gc.collect()

    # walk-forward on BAR rows (no placement grouping)
    n = len(y)
    n_chunks = 6
    chunk = n // n_chunks
    splits = []
    for i in range(2, n_chunks):
        tr_end = i * chunk
        te_end = min((i + 1) * chunk, n)
        if te_end <= tr_end:
            break
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))

    test_mask = np.zeros(n, dtype=bool)
    for _, te_idx in splits:
        test_mask[te_idx] = True

    oof_sum = np.zeros(n)
    oof_y = np.zeros(n, dtype=int)
    for s in SEEDS:
        print(f"── seed {s} ──")
        oof_s = np.zeros(n)
        for k, (tr_idx, te_idx) in enumerate(splits):
            dtr = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=w[tr_idx],
                              free_raw_data=True)
            model = lgb.train(lgb_params(s), dtr, num_boost_round=400)
            oof_s[te_idx] = model.predict(X[te_idx])
            oof_y[te_idx] = y[te_idx]
        oof_sum += oof_s
        print(f"  seed {s} done")
    oof_probs = oof_sum / len(SEEDS)
    print("\n=== DIRECTION OOS (3-seed ensemble) ===")
    res = evaluate(oof_probs[test_mask], oof_y[test_mask], "ALL TEST WINDOWS")

    # persist OOF for backtest cross-check. times[] lets the backtest align
    # these rows to the placement matrix's bars by timestamp (v7.3f) — the
    # direction matrix is built independently (from gold_seed_multi.csv, not
    # sliced from gold_features.csv) so row indices don't correspond 1:1.
    np.save(f"{MODEL_DIR}/dir_oof_probs.npy", oof_probs.astype(np.float32))
    np.save(f"{MODEL_DIR}/dir_oof_targets.npy", oof_y.astype(np.int8))
    np.save(f"{MODEL_DIR}/dir_oof_times.npy", times.astype("datetime64[s]").astype(np.int64))
    np.save(f"{MODEL_DIR}/dir_oof_mask.npy", test_mask)

    # final models (all data) + atomic swap
    for s in SEEDS:
        final = lgb.train(lgb_params(s),
                          lgb.Dataset(X, label=y, weight=w, free_raw_data=True),
                          num_boost_round=400)
        name = f"{MODEL_DIR}/direction_s{s}.txt"
        tmp = name + ".tmp"
        final.save_model(tmp)
        os.replace(tmp, name)
        print(f"Direction model saved: {name} (atomic swap)")
        del final; gc.collect()

    with open(f"{MODEL_DIR}/direction_features.json", "w") as f:
        json.dump(feats, f)
    with open(f"{MODEL_DIR}/direction_ensemble.json", "w") as f:
        json.dump({"type": "direction", "seeds": SEEDS,
                   "models": [f"direction_s{s}.txt" for s in SEEDS],
                   "horizon_bars": HORIZON_BARS}, f, indent=2)
    with open(f"{MODEL_DIR}/direction_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in res.items()}, f, indent=2)
    print(f"✅ DIRECTION model done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
