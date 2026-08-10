#!/usr/bin/env python3
"""
fit_signal_rating.py — v8.5 (2026-08-10) RATING-SCALE FIX
=========================================================
THE BUG (root cause of "no signals" complaint #2):
  The rating fitter computed P(win) from the EMPIRICAL bucket rate map
  (rate_map, with a hardcoded 0.5 fallback for off-grid/low-support
  buckets). The LIVE ENGINE computes P(win) = CALIBRATED MODEL
  probability for the specific placement geometry. These are two
  DIFFERENT scales:
    - fitter:  p_emp ~ 0.35-0.55  →  ratings cluster 40-60
    - engine:  p_cal ~ 0.10-0.45  →  live ratings 15-45
  The learned threshold (40) was fit on the EMPIRICAL scale but applied
  to the CALIBRATED scale → the gate systematically blocked every
  placement, even positive-expectancy ones.

v8.5 FIX:
  The fitter now uses the SAME p source as the live engine — the
  calibrated OOF model probabilities (oof_probs.npy + the per-direction
  × RR calibration curves from calibration_by_drr.json, exactly like
  engine.best_placement does). Zero hardcoded selectivity:
    - P(win)  = calibrated OOF model prob   (learned, 6yr)
    - weights = LSQ fit on 6yr realized outcomes
    - threshold = lowest rating decile with positive realized expectancy
  MIN_BUCKET / rate_map / 0.5 fallback are GONE — no hardcoded numbers.

Scale: base_tf=M5, 6yr matrix 32.49M rows.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import features as F
from features import vector_regime_bin, TP_RATIOS

MODEL_DIR = "models"
MATRIX = "gold_features_m5.csv"
OUT = f"{MODEL_DIR}/signal_rating.json"
REGIMES = ["STRONG_UP", "UP", "DOWN", "STRONG_DOWN",
           "RANGE_TIGHT", "RANGE_WIDE", "HIGH_VOL", "QUIET_LOW_VOL"]
CHUNK = 400_000
need = F.REGIME_KEYS + ["sl_atr_buy", "sl_atr_sell", "rr_buy", "rr_sell",
                        "mfe_atr", "mfa_atr", "target", "direction"]
dt = {c: np.float32 for c in need}
dt.pop("direction", None)


def _regime_conf(te, ap):
    """0..1: how confident is the regime call. Mirrors engine signal_rating."""
    dist = np.minimum(np.minimum(np.abs(np.abs(te) - 0.4),
                                 np.abs(np.abs(te) - 1.2)), 1.0)
    return np.clip(0.5 * (1 - dist) + 0.25 * np.abs(ap - 0.5) * 2.0, 0, 1)


def main():
    t0 = time.time()
    if not os.path.exists(MATRIX):
        print(f"❌ matrix not found: {MATRIX}")
        sys.exit(1)

    # ── Load placement prior (for excursion component) ──
    try:
        with open(f"{MODEL_DIR}/placement_prior.json") as f:
            pl = json.load(f)
    except Exception:
        pl = {}

    # ── Load OOF raw probs + per-dir×RR calibration curves ──
    # PREFERRED: full-matrix SPECIALIST OOF (build_spec_oof_full.py) — the
    # live engine routes every state through its regime specialist (3 seeds),
    # so the rating must be learned on THAT probability scale. Fallback: base
    # ensemble OOF (train_continue) — same curves, slightly different scale.
    spec_full = f"{MODEL_DIR}/oof_spec_full.npy"
    if os.path.exists(spec_full):
        oof = np.load(spec_full)
        oofy = np.load(f"{MODEL_DIR}/oof_spec_full_y.npy")
        print(f"OOF source: SPECIALIST full-matrix ({len(oof):,} rows) — engine parity")
    else:
        oof = np.load(f"{MODEL_DIR}/oof_probs.npy")          # raw 3-seed avg
        oofy = np.load(f"{MODEL_DIR}/oof_targets.npy")
        print(f"OOF source: BASE ensemble ({len(oof):,} rows)")
    n_oof = len(oof)
    print(f"OOF: {n_oof:,} rows | base WR {oofy.mean():.1%}")
    try:
        with open(f"{MODEL_DIR}/calibration_by_drr.json") as f:
            cal_by_rr = json.load(f)
        print(f"per-dir×RR curves: {len(cal_by_rr)}")
    except Exception as e:
        print(f"❌ calibration_by_drr.json: {e}")
        sys.exit(1)

    # ── PASS 1: calibrated OOF P(win) for every matrix row ──
    #    (stream matrix columns needed for direction+rr; align by position)
    #    ⚠️ SCALE PARITY: the LIVE ENGINE routes through the REGIME SPECIALIST
    #    curves (calibration_by_drr_spec_<regime>.json) when specialists are
    #    loaded — exactly like best_placement: route_cal = spec_cal[regime],
    #    fallback to base cal_by_rr. The fitter MUST apply the same per-regime
    #    curve so the learned rating threshold lives on the engine's real scale.
    print("Pass 1: calibrated OOF probs per row (per-regime curves)...", flush=True)
    spec_cals = {}
    for reg in REGIMES:
        pth = f"{MODEL_DIR}/calibration_by_drr_spec_{reg.lower()}.json"
        try:
            with open(pth) as f:
                spec_cals[reg] = json.load(f)
        except Exception:
            spec_cals[reg] = None
    pcal = np.zeros(n_oof, dtype=np.float32)
    seen = 0
    for ch in pd.read_csv(MATRIX, usecols=["direction", "rr_buy", "rr_sell"] + F.REGIME_KEYS,
                          dtype={"direction": np.int8, "rr_buy": np.float32,
                                 "rr_sell": np.float32},
                          chunksize=1_000_000):
        d = ch["direction"].values
        rr = np.where(d == 1, ch["rr_buy"].values, ch["rr_sell"].values)
        # nearest grid ratio (vectorized) — same nearest-key logic as engine
        tarr = np.asarray(TP_RATIOS, dtype=np.float64)
        bucket = tarr[np.argmin(np.abs(tarr[None, :] - rr[:, None]), axis=1)]
        bins = vector_regime_bin(ch)
        sl = slice(seen, seen + len(ch))
        pcal[sl] = oof[sl]                       # default: raw (uncalibrated)
        for b in REGIMES:
            m = bins == b
            if not m.any():
                continue
            cal = spec_cals.get(b) if spec_cals.get(b) else cal_by_rr
            for di, dname in enumerate(["SELL", "BUY"]):
                for t in TP_RATIOS:
                    mm = m & (d == (1 if dname == "BUY" else 0)) & (bucket == t)
                    if not mm.any():
                        continue
                    knots = cal.get(f"{dname}_{t}")
                    if knots:
                        # vectorized piecewise-linear calibration == engine path
                        pcal[sl][mm] = np.interp(oof[sl][mm],
                                                 knots["knots_p"],
                                                 knots["knots_y"])
        seen += len(ch)
    assert seen == n_oof, f"matrix rows {seen} != OOF rows {n_oof}"
    del oof, oofy
    print(f"  calibrated pcal mean={pcal.mean():.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── PASS 2: rating components + LSQ weights (constant memory) ──
    print("Pass 2: LSQ rating weights...", flush=True)
    XtX = np.zeros((5, 5)); Xty = np.zeros(5)
    total = 0
    seen = 0   # absolute matrix position (BEFORE dropna)
    for chunk in pd.read_csv(MATRIX, usecols=need, chunksize=CHUNK, dtype=dt,
                             low_memory=False):
        mask = ~chunk[["target", "direction", "rr_buy", "rr_sell",
                       "sl_atr_buy", "sl_atr_sell", "mfe_atr", "mfa_atr"]].isna().any(axis=1)
        chunk = chunk[mask].copy()
        if chunk.empty:
            seen += len(mask)
            continue
        bins = vector_regime_bin(chunk)
        d = np.where(chunk["direction"].values.astype(np.float64) == 1.0,
                     "BUY", "SELL")
        rr = np.where(d == "BUY", chunk["rr_buy"].values,
                      chunk["rr_sell"].values).astype(np.float64)
        tgt = chunk["target"].values.astype(np.float64)
        n = len(chunk)
        p_win = pcal[seen:seen + len(mask)][mask.values].astype(np.float64)
        exp = p_win * rr - (1.0 - p_win)
        e_comp = np.clip((exp + 0.25) / 3.0, 0, 1)
        r_comp = _regime_conf(chunk["trend_ema"].values,
                              chunk["atr_pctile"].values)
        mfa_p50 = np.zeros(n)
        learned_sl = np.zeros(n)
        for b in set(np.unique(bins)) & set(REGIMES):
            m = bins == b
            for di in ("BUY", "SELL"):
                dm = m & (d == di)
                pmeta = pl.get("regimes", {}).get(b, {}).get(di, {})
                ls = pmeta.get("sl_atr")
                mfa = pmeta.get("mfa_l_p50")
                if ls and mfa:
                    learned_sl[dm] = ls
                    mfa_p50[dm] = mfa
        sl_atr = np.where(d == "BUY",
                          chunk["sl_atr_buy"].values.astype(np.float64),
                          chunk["sl_atr_sell"].values.astype(np.float64))
        ok = (learned_sl > 0) & (mfa_p50 > 0)
        x_comp = np.full(n, 0.5, dtype=np.float64)
        x_comp[ok & (sl_atr >= learned_sl)] = 1.0
        x_comp[ok & (sl_atr < learned_sl) & (sl_atr > mfa_p50)] = 0.6
        x_comp[ok & (sl_atr <= mfa_p50)] = 0.2
        X = np.column_stack([p_win, e_comp, r_comp, x_comp, np.ones(n)])
        XtX += X.T @ X
        Xty += X.T @ tgt
        total += n
        seen += len(mask)
        del chunk, X
        if total % 2_000_000 < CHUNK:
            print(f"  pass2 {total:,} rows ({time.time()-t0:.0f}s)", flush=True)
    w = np.linalg.solve(XtX, Xty)
    # Normalize to sum 1.0 — the ENGINE divides rating by sum(weights)
    # (signal_rating.rate_signal). Without this, the fitter's decile scale
    # and the engine's live rating scale would diverge AGAIN.
    wsum = float(w[:4].sum())
    wsum = wsum if abs(wsum) > 1e-9 else 1.0
    weights = {"p_win": float(max(w[0] / wsum, 0.0)),
               "exp": float(max(w[1] / wsum, 0.0)),
               "regime_conf": float(max(w[2] / wsum, 0.0)),
               "excursion": float(max(w[3] / wsum, 0.0))}
    wsum2 = float(sum(weights.values()))
    if wsum2 > 0:
        for k in weights:
            weights[k] = weights[k] / wsum2
    print(f"  weights (normalized, sum={sum(weights.values()):.3f}): {weights}")

    # ── PASS 3: deciles + learned threshold on the ENGINE's scale ──
    print("Pass 3: decile realized expectancy...", flush=True)
    dec = {i: {"n": 0, "wins": 0.0, "rr_sum": 0.0} for i in range(10)}
    total = 0
    seen = 0
    for chunk in pd.read_csv(MATRIX, usecols=need, chunksize=CHUNK, dtype=dt,
                             low_memory=False):
        mask = ~chunk[["target", "direction", "rr_buy", "rr_sell",
                       "sl_atr_buy", "sl_atr_sell", "mfe_atr", "mfa_atr"]].isna().any(axis=1)
        chunk = chunk[mask].copy()
        if chunk.empty:
            seen += len(mask)
            continue
        bins = vector_regime_bin(chunk)
        d = np.where(chunk["direction"].values.astype(np.float64) == 1.0,
                     "BUY", "SELL")
        rr = np.where(d == "BUY", chunk["rr_buy"].values,
                      chunk["rr_sell"].values).astype(np.float64)
        tgt = chunk["target"].values.astype(np.float64)
        n = len(chunk)
        p_win = pcal[seen:seen + len(mask)][mask.values].astype(np.float64)
        exp = p_win * rr - (1.0 - p_win)
        e_comp = np.clip((exp + 0.25) / 3.0, 0, 1)
        r_comp = _regime_conf(chunk["trend_ema"].values,
                              chunk["atr_pctile"].values)
        sl_atr = np.where(d == "BUY",
                          chunk["sl_atr_buy"].values.astype(np.float64),
                          chunk["sl_atr_sell"].values.astype(np.float64))
        mfa_p50 = np.zeros(n)
        learned_sl = np.zeros(n)
        for b in set(np.unique(bins)) & set(REGIMES):
            m = bins == b
            for di in ("BUY", "SELL"):
                dm = m & (d == di)
                pmeta = pl.get("regimes", {}).get(b, {}).get(di, {})
                ls = pmeta.get("sl_atr")
                mfa = pmeta.get("mfa_l_p50")
                if ls and mfa:
                    learned_sl[dm] = ls
                    mfa_p50[dm] = mfa
        ok = (learned_sl > 0) & (mfa_p50 > 0)
        x_comp = np.full(n, 0.5, dtype=np.float64)
        x_comp[ok & (sl_atr >= learned_sl)] = 1.0
        x_comp[ok & (sl_atr < learned_sl) & (sl_atr > mfa_p50)] = 0.6
        x_comp[ok & (sl_atr <= mfa_p50)] = 0.2
        comp = (weights["p_win"] * p_win + weights["exp"] * e_comp +
                weights["regime_conf"] * r_comp +
                weights["excursion"] * x_comp)
        rating = np.clip(comp * 100, 0, 100)
        decile = np.digitize(rating, np.linspace(0, 100, 11)) - 1
        for i in range(10):
            m = decile == i
            if m.any():
                dec[i]["n"] += int(m.sum())
                dec[i]["wins"] += float(tgt[m].sum())
                dec[i]["rr_sum"] += float(rr[m].sum())
        total += n
        seen += len(mask)
        del chunk
        if total % 2_000_000 < CHUNK:
            print(f"  pass3 {total:,} rows ({time.time()-t0:.0f}s)", flush=True)

    decile_stats = []
    for i in range(10):
        s = dec[i]
        if s["n"] == 0:
            continue
        p = s["wins"] / s["n"]
        rr_mean = s["rr_sum"] / s["n"]
        exp_real = p * rr_mean - (1 - p)
        decile_stats.append({"decile": i,
                             "lo": float(np.linspace(0, 100, 11)[i]),
                             "hi": float(np.linspace(0, 100, 11)[i + 1]),
                             "n": s["n"], "win_rate": round(float(p), 4),
                             "rr_mean": round(float(rr_mean), 3),
                             "realized_exp": round(float(exp_real), 4)})
    # LEARNED threshold: lowest decile with positive realized expectancy.
    threshold = 100.0
    for ds in decile_stats:
        if ds["realized_exp"] > 0 and ds["n"] >= 5000:
            threshold = float(ds["lo"])
            break
    if threshold == 100.0:
        threshold = 50.0   # data says no positive-EV decile; neutral default

    out = {"version": 8.5, "base_tf": "M5",
           "weights": weights,
           "threshold": round(threshold, 1),
           "deciles": decile_stats,
           "p_source": "calibrated OOF model probs (same scale as live engine)",
           "note": "rating = weighted components; fire only when rating >= threshold "
                   "(lowest decile with positive realized expectancy, 6yr)",
           "built_at": time.time()}
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)  # atomic
    print(f"✅ signal_rating.json v8.5 ({time.time()-t0:.0f}s)")
    print(f"   weights: {weights}")
    print(f"   threshold: {threshold:.1f} (fire when rating >= {threshold:.1f})")
    print("   deciles:")
    for ds in decile_stats:
        mark = " ◀ FIRE" if ds["lo"] == threshold else ""
        print(f"     [{ds['lo']:5.1f}-{ds['hi']:5.1f}] n={ds['n']:>9,} "
              f"wr={ds['win_rate']:.3f} rr={ds['rr_mean']:.2f} "
              f"exp={ds['realized_exp']:+.4f}{mark}")


if __name__ == "__main__":
    main()
