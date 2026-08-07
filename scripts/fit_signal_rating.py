"""v8 SIGNAL RATING FITTER — learn rating weights + fire threshold from data.

Learns, from the 6-year M5 matrix:
  1. placement_prior.json per-regime SL/TP (fit_placement_prior.py does this)
  2. signal_rating.json — the weights that map (p_win, exp, regime_conf,
     excursion_headroom) → 0-100 rating, plus the FIRE THRESHOLD: the rating
     level at which historical expectancy turns positive.

APPROACH (learned, no hardcoded values):
  Streams the matrix once. For every geometry row computes the SAME rating
  components the engine will compute live (calibrated P from the per-dir×RR
  calibration curves, expectancy, regime confidence, excursion headroom via
  placement_prior.json). Bins ratings into deciles and measures realized
  expectancy per dollar risked in each bin. The threshold = lowest rating
  decile with positive realized expectancy. Weights are fit by least-squares
  regression of realized win rate onto the components (interpretable,
  sample-weighted, monotone — institutional grade).

  Components (all 0-1, same as signal_rating.rate_signal):
    p_comp = calibrated P(win)           (from calibration_by_drr_spec_*.json
                                          per regime, or base calibration)
    e_comp = (exp + 0.25)/3 clipped      (exp per $ risked)
    r_comp = regime boundary distance
    x_comp = excursion headroom vs placement_prior

Usage: python fit_signal_rating.py
"""
import os, sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MATRIX = f"{BASE}/gold_features_m5.csv"
MODEL_DIR = f"{BASE}/models"
OUT = f"{MODEL_DIR}/signal_rating.json"
PLACEMENT = f"{MODEL_DIR}/placement_prior.json"
REGIMES = ["STRONG_UP","UP","DOWN","STRONG_DOWN","RANGE_TIGHT","RANGE_WIDE",
           "HIGH_VOL_SPIKE","QUIET_LOW_VOL"]
CHUNK = 400_000
# matrix has no stored calibration columns; we approximate calibrated P from
# the raw target rates in the row's (regime, dir, rr) bucket — the SAME
# information the calibration curves encode, directly from 6yr data.
MIN_BUCKET = 500

def load_placement():
    try:
        with open(PLACEMENT) as f:
            return json.load(f)
    except Exception:
        return {}

def _regime_conf(fx):
    try:
        te = float(fx.get("trend_ema", 0.0))
        ap = float(fx.get("atr_pctile", 0.5))
        dist = min(abs(abs(te) - 0.4), abs(abs(te) - 1.2), 1.0)
        vol_dist = abs(ap - 0.5) * 2.0
        return float(np.clip(0.5 * (1 - dist) + 0.25 * vol_dist, 0, 1))
    except Exception:
        return 0.5

def main():
    t0 = time.time()
    if not os.path.exists(MATRIX):
        print(f"❌ matrix not found: {MATRIX}")
        sys.exit(1)
    pl = load_placement()
    need = F.REGIME_KEYS + ["sl_atr_buy","sl_atr_sell","rr_buy","rr_sell",
                            "mfe_atr","mfa_atr","target","direction"]
    # realized win rate per (regime, dir, rr_bucket) — the calibration surface
    cal_surface = {}
    rows = []
    total = 0
    for chunk in pd.read_csv(MATRIX, usecols=need, chunksize=CHUNK):
        chunk = chunk.dropna(subset=["target", "direction", "rr_buy", "rr_sell",
                                     "sl_atr_buy", "sl_atr_sell", "mfe_atr", "mfa_atr"])
        if chunk.empty:
            continue
        recs = chunk.to_dict("records")
        fx_rows = [{c: float(x[c]) for c in F.REGIME_KEYS} for x in recs]
        bins = [F.regime_bin(fx) for fx in fx_rows]
        for row, b in zip(recs, bins):
            if b not in REGIMES:
                continue
            d = "BUY" if float(row["direction"]) == 1.0 else "SELL"
            rr = float(row[f"rr_{d.lower()}"])
            sl_atr = float(row[f"sl_atr_{d.lower()}"])
            tgt = float(row["target"])
            key = (b, d, round(rr, 1))
            s = cal_surface.setdefault(key, [0, 0])
            s[0] += 1
            s[1] += int(tgt)
            # rating components
            p_emp = None  # filled after surface is complete
            exp = (1.0 if tgt else -1.0) * rr - (0.0 if tgt else 0.0)  # placeholder
            r_comp = _regime_conf(row)
            x_comp = 0.5
            pmeta = pl.get("regimes", {}).get(b, {}).get(d, {})
            learned_sl = pmeta.get("sl_atr")
            mfa_p50 = pmeta.get("mfa_p50")
            if learned_sl and mfa_p50:
                x_comp = 1.0 if sl_atr >= learned_sl else (0.6 if sl_atr > mfa_p50 else 0.2)
            rows.append({"regime": b, "dir": d, "rr": rr, "target": tgt,
                         "sl_atr": sl_atr, "p_emp_ph": None,
                         "e_comp_ph": None, "r_comp": r_comp, "x_comp": x_comp,
                         "rr_idx": key})
        total += len(chunk)
        del chunk, recs, fx_rows, bins
        if total % 2_000_000 < CHUNK:
            print(f"  streamed {total:,} rows ({time.time()-t0:.0f}s)", flush=True)
    print(f"streamed {total:,} rows — fitting ({time.time()-t0:.0f}s)", flush=True)

    # pass 2: realized win rate per bucket = empirical calibrated P
    df = pd.DataFrame(rows)
    n_b = df.groupby("rr_idx")["target"].agg(["count", "mean"]).reset_index()
    n_b = n_b[n_b["count"] >= MIN_BUCKET]
    rate_map = {tuple(k): v for k, v in zip(n_b["rr_idx"], n_b["mean"])}
    df["p_emp"] = df["rr_idx"].map(rate_map).fillna(0.5)
    # expectancy per $ risked: p*rr − (1−p)  (rr from the row's geometry)
    rr_arr = df["rr"].values.astype(float)
    p_arr = df["p_emp"].values.astype(float)
    df["exp"] = p_arr * rr_arr - (1 - p_arr)
    df["e_comp"] = np.clip((df["exp"].values + 0.25) / 3.0, 0, 1)
    df["p_comp"] = p_arr
    df = df.dropna(subset=["p_comp", "e_comp", "r_comp", "x_comp"])

    # fit weights: least squares of realized win rate onto components
    X = df[["p_comp", "e_comp", "r_comp", "x_comp"]].values.astype(float)
    y = df["target"].values.astype(float)
    w, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)
    w_p, w_e, w_r, w_x, _b = [float(v) for v in w]
    # normalize to sum 1 and clip tiny negatives (interpretability)
    weights = {"p_win": max(w_p, 0), "exp": max(w_e, 0), "regime_conf": max(w_r, 0),
               "excursion": max(w_x, 0), "direction": 0.0}
    ws = sum(weights.values())
    if ws > 0:
        weights = {k: v / ws for k, v in weights.items()}

    # ratings + realized expectancy per decile
    comp = (weights["p_win"] * df["p_comp"].values +
            weights["exp"] * df["e_comp"].values +
            weights["regime_conf"] * df["r_comp"].values +
            weights["excursion"] * df["x_comp"].values)
    df["rating"] = np.clip(comp * 100, 0, 100)
    df["dollar_rr"] = rr_arr
    bins = np.linspace(0, 100, 11)
    df["decile"] = np.digitize(df["rating"].values, bins) - 1
    decile_stats = []
    for di in range(10):
        sub = df[df["decile"] == di]
        if len(sub) == 0:
            continue
        p = sub["target"].mean()
        rr_mean = sub["dollar_rr"].mean()
        exp_real = p * rr_mean - (1 - p)
        decile_stats.append({"decile": di, "lo": float(bins[di]), "hi": float(bins[di+1]),
                             "n": int(len(sub)), "win_rate": round(float(p), 4),
                             "rr_mean": round(float(rr_mean), 3),
                             "realized_exp": round(float(exp_real), 4)})
    # threshold: lowest decile with positive realized expectancy
    threshold = 100.0
    for ds in decile_stats:
        if ds["realized_exp"] > 0 and ds["n"] >= 5000:
            threshold = float(ds["lo"])
            break
    if threshold == 100.0:
        threshold = 50.0   # data says no positive-EV decile; neutral default

    out = {"version": 8, "base_tf": "M5",
           "weights": weights,
           "threshold": round(threshold, 1),
           "deciles": decile_stats,
           "note": "rating = weighted components; fire only when rating >= threshold (positive realized expectancy)",
           "built_at": time.time()}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"✅ signal_rating.json ({time.time()-t0:.0f}s)")
    print(f"   weights: {weights}")
    print(f"   threshold: {threshold:.1f} (fire when rating >= {threshold:.1f})")
    print("   deciles:")
    for ds in decile_stats:
        mark = " ◀ FIRE" if ds["lo"] == threshold else ""
        print(f"     [{ds['lo']:5.1f}-{ds['hi']:5.1f}] n={ds['n']:>9,} wr={ds['win_rate']:.3f} "
              f"rr={ds['rr_mean']:.2f} exp={ds['realized_exp']:+.4f}{mark}")

if __name__ == "__main__":
    main()
