"""v7.6 (2026-08-03) — DIRECTION MODEL: HTF-regime PERSISTENCE label.

WHY: the v7 direction model predicates M1 direction at 60 bars (first-touch
±1.5 ATR band). Experiments (2026-08-03, 62k bars) proved raw M1 direction
is UNLEARNABLE at horizons 8–60 AND 4h forward-close (OOS 0.493–0.525,
coin flip). Root cause: raw price direction on gold is noise — but the
REGIME is not. H1-trend persistence (does the H1 trend still point the
same way in 4h?) measured OOS acc 0.892 on the 2020-2026 multi-era seed:
trends persist, and that persistence is the learnable signal.

FIX: predict regime persistence, not price. Label:
    target = 1  if h1_trend[t + 241] > 0   (H1 trend still RISING in 4h)
             0  if h1_trend[t + 241] <= 0  (H1 trend still FALLING in 4h)
The engine contract is unchanged: final_exp(BUY) = Exp(BUY) × P(up). In a
falling H1 regime (h1_trend < 0 today), P(up) → low, BUY expectancy is
crushed, SELL wins the comparison — the model LEARNS "don't buy the
falling knife" instead of having it hardcoded. NO gates, NO thresholds.

Label uses the SAME h1_trend the features expose (causal: resample + shift(1),
only completed H1 bars), so no lookahead: at time t the model sees the
current H1 trend and predicts whether it persists. 93-feature market vector
unchanged.

Uses the ARCHIVED multi-era seed (2020-2026, 264k bars, regime diversity).
Walk-forward OOS must beat 0.53 (coin-flip ceil) or the honest gate refuses
to swap — direction_prior() neutralizes sub-noise models anyway.

3-seed ensemble, recency-weighted, atomic swap to the SAME filenames the
engine hot-reloads.
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
ARCH = "/home/jith/.hermes/profiles/trading/archives/backtest-data-2026-08-03"
SEEDS = [42, 7, 2026]
RECENCY_TAU_DAYS = 180.0
HORIZON_BARS = 240        # 4h window — the trade's realistic holding horizon
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
        "num_threads": 8,
        "seed": seed,
    }

def build_matrix(seed_csv, horizon=HORIZON_BARS):
    df = pd.read_csv(seed_csv)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    t = df["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(df)]
    periods = [df.iloc[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    periods = [p for p in periods if len(p) >= MIN_BARS_PER_PERIOD]

    blocks = []
    for p in periods:
        fdf = p.copy()
        fdf = F._feature_block(fdf)
        fdf = fdf.dropna().reset_index(drop=True)
        n = len(fdf)
        # ── PRICE-DIRECTION label (v7.8, 2026-08-04) ──
        # The engine consumes p_up as P(price rises over the trade horizon),
        # so that is EXACTLY what the model must be trained to predict.
        # OLD target (v7.6) was H1-trend PERSISTENCE: h1_trend[t+241]>0 —
        # trivially learnable (acc 0.905) but NOT price direction → the
        # model 'sold in buy markets' (its p_up ~0 in any falling H1 regime
        # regardless of what price actually did next). The GATE only
        # neutralized that damage; it couldn't teach the edge. Now the label
        # IS forward price direction over the same horizon the engine holds
        # trades: price up → target 1, price down → target 0. A model that
        # learns this is genuinely market-aware: high p_up ⟺ buy market,
        # low p_up ⟺ sell market. Causal, no lookahead (uses close[t+h]).
        cl = fdf["close"].values.astype(np.float64)
        fw = np.full(n, np.nan)
        if n > horizon + 1:
            fw[:-horizon - 1] = cl[horizon + 1:] - cl[:-horizon - 1]
        tgt = np.full(n, np.nan)
        if n > horizon + 1:
            tgt[:-horizon - 1] = (fw[:-horizon - 1] > 0).astype(float)
        fdf["target"] = tgt
        # PRICE fwd_return over the SAME horizon — for the honest
        # price-direction gate (2026-08-04). NOT a feature (excluded below);
        # only used to measure whether the model's p_up actually predicts
        # forward PRICE direction (the thing the engine consumes p_up for).
        fdf["fwd_return"] = fw
        fdf = fdf[fdf["target"].notna()].reset_index(drop=True)
        market_cols = [c for c in fdf.columns
                       if c not in ("time", "target", "fwd_return")
                       and c not in F.RAW_PRICE_COLS]
        # keep fwd_return for the honest price-direction gate (not a feature)
        out = fdf[market_cols + ["target", "time", "fwd_return"]].copy()
        blocks.append(out)
    final = pd.concat(blocks, ignore_index=True)
    final = final.sort_values("time").reset_index(drop=True)
    feats = [c for c in final.columns if c not in ("time", "target", "fwd_return")]
    return final, feats

def recency_weights(times, tau_days=RECENCY_TAU_DAYS):
    ts = times.astype("datetime64[s]").astype(np.int64)
    age_days = (ts.max() - ts) / 86400.0
    w = np.exp(-age_days / tau_days)
    return (w / w.mean()).astype(np.float32)

def _price_dir_acc(p_up, fwd):
    """Does the model's P(up) actually predict fwd PRICE direction?
    (The engine uses p_up as P(price up) — this measures that honestly.)
    p_up >= 0.5 → 'price up'; compare to actual fwd_return sign."""
    p_up = np.array(p_up); fwd = np.array(fwd)
    ok = ~np.isnan(fwd)
    if ok.sum() == 0:
        return float("nan")
    pred = (p_up[ok] >= 0.5).astype(int)
    act = (fwd[ok] > 0).astype(int)
    return float((pred == act).mean())

def evaluate(probs, y, label=""):
    probs = np.array(probs); y = np.array(y)
    pred = (probs >= 0.5).astype(int)
    acc = (pred == y).mean()
    up_mask = pred == 1
    p_up = y[up_mask].mean() if up_mask.any() else float("nan")
    p_dn = (1 - y[~up_mask]).mean() if (~up_mask).any() else float("nan")
    print(f"  {label}: acc={acc:.3f} | P(up)={p_up:.3f} | P(dn)={p_dn:.3f} | n={len(y)}")
    return {"acc": acc, "p_up": p_up, "p_dn": p_dn, "n": len(y)}

def train_final(seed_csv, horizon):
    matrix, feats = build_matrix(seed_csv, horizon)
    times = pd.to_datetime(matrix["time"]).values
    X = matrix[feats].values.astype(np.float32)
    y = matrix["target"].values.astype(np.int8)
    w = recency_weights(times)
    fwd_all = matrix["fwd_return"].values.astype(np.float32)  # for price-direction gate (read before del)
    del matrix; gc.collect()

    # walk-forward OOS report (3-seed ensemble, aligned)
    n = len(y); n_chunks = 5; chunk = n // n_chunks
    splits = []
    for i in range(2, n_chunks):
        tr_end = i * chunk; te_end = min((i + 1) * chunk, n)
        if te_end <= tr_end: break
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    test_mask = np.zeros(n, dtype=bool)
    for _, te in splits: test_mask[te] = True
    oof_sum = np.zeros(n); oof_y = np.zeros(n, dtype=int)
    for s in SEEDS:
        oof_s = np.zeros(n)
        for tr, te in splits:
            dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], free_raw_data=True)
            m = lgb.train(lgb_params(s), dtr, num_boost_round=500)
            oof_s[te] = m.predict(X[te]); oof_y[te] = y[te]
        oof_sum += oof_s
    oof_probs = oof_sum / len(SEEDS)
    print("\n=== DIRECTION (HTF, v7.6) OOS (3-seed ensemble) ===")
    res = evaluate(oof_probs[test_mask], oof_y[test_mask], "ALL TEST WINDOWS")

    # ── HONEST GATE 2: PRICE-DIRECTION accuracy (2026-08-04) ──
    # The engine consumes p_up as P(price up over the trade horizon), but the
    # label only measures H1-trend PERSISTENCE — trivially learnable (0.905)
    # and saturating (p_up pinned ~0.00 in any falling H1 regime). A model
    # with high persistence acc but NO price-direction edge forces the engine
    # to sell into rallies / buy into dumps (observed 08-04: 3 SELLs in a
    # rising tape, all SL'd, while placement wanted BUY at the 4020 low).
    # So we ALSO measure: does p_up actually predict fwd price direction?
    # If acc_price <= majority baseline → no real edge → neutralize.
    fwd = fwd_all
    tm = test_mask
    acc_price = _price_dir_acc(oof_probs[tm], fwd[tm])
    base_up = float((fwd[tm] > 0).mean())
    base_maj = max(base_up, 1 - base_up)
    print(f"  PRICE-DIRECTION OOS: acc={acc_price:.3f} | base-up={base_up:.3f} | "
          f"majority-baseline={base_maj:.3f} | edge={acc_price - base_maj:+.3f}")
    res["acc_price"] = float(acc_price)
    res["base_up"] = base_up

    # HONEST GATE (not a hardening rule — an integrity check): if the new
    # label STILL cannot beat coin-flip noise, do NOT overwrite the deployed
    # model. direction_prior() would neutralize it anyway (acc <= 0.53 → 0.5),
    # so writing it would only confuse metrics. Keep the previous model.
    if res["acc"] <= 0.53:
        print(f"⚠️ OOS acc {res['acc']:.3f} <= 0.53 — NO demonstrable HTF-direction edge.")
        print("   Keeping existing deployed direction model. No swap performed.")
        return res["acc"]
    # GATE 2: persistence edge alone is NOT enough — the model must also beat
    # the majority baseline on ACTUAL price direction, else it is a saturating
    # persistence parrot that overrides the placement model's real call.
    if acc_price <= base_maj + 0.02:
        print(f"⚠️ OOS acc {res['acc']:.3f} BUT price-direction acc {acc_price:.3f} "
              f"<= majority baseline {base_maj:.3f} — model cannot beat 'always "
              f"bet the majority' on price direction. Keeping deployed model "
              f"(engine will neutralize it via acc_price). No swap performed.")
        return res["acc"]

    np.save(f"{MODEL_DIR}/dir_oof_probs.npy", oof_probs.astype(np.float32))
    np.save(f"{MODEL_DIR}/dir_oof_targets.npy", oof_y.astype(np.int8))
    np.save(f"{MODEL_DIR}/dir_oof_times.npy", times.astype("datetime64[s]").astype(np.int64))
    np.save(f"{MODEL_DIR}/dir_oof_mask.npy", test_mask)

    # final models
    for s in SEEDS:
        final = lgb.train(lgb_params(s),
                          lgb.Dataset(X, label=y, weight=w, free_raw_data=True),
                          num_boost_round=500)
        name = f"{MODEL_DIR}/direction_s{s}.txt"
        tmp = name + ".tmp"; final.save_model(tmp); os.replace(tmp, name)
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
    print(f"✅ v7.6 HTF direction model done in {time.time() - t0:.0f}s | OOS acc {res['acc']:.3f}")
    return res["acc"]


if __name__ == "__main__":
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] ═══ DIRECTION MODEL v7.6 (HTF trend target) ═══")
    multi = f"{ARCH}/gold_seed_multi.csv"
    seed_csv = multi if os.path.exists(multi) else f"{BASE}/gold_seed.csv"
    print(f"seed: {seed_csv} (exists={os.path.exists(seed_csv)})")
    print(f"\nTraining final models @ {HORIZON_BARS/60:.0f}h horizon...")
    train_final(seed_csv, HORIZON_BARS)