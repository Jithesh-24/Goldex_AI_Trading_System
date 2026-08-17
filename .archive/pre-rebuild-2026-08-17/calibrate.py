#!/usr/bin/env python3
"""Learned probability calibration for the placement model (v4.1).

Diagnosis (2026-08-01): the LightGBM placement model is systematically
OVERCONFIDENT — model says P=0.60 -> actual WR 0.47, P=0.80 -> 0.62 (and
under-confident at the low end: P=0.15 -> actual 0.37). Raw probabilities fed
into EV = P*TP - (1-P)*(SL+spread) inflate EV, so the max-EV sweep fires on
nearly every bar (49,716 backtest trades, PF 0.99, Net -$391).

Fix: fit an isotonic (PAVA) calibration curve on OUT-OF-SAMPLE predictions
(the walk-forward OOF probs — never in-sample), persist knots to
models/calibration.json, and have the live engine + backtest map raw P -> cal P
before computing EV. This is LEARNED (data-driven), not a hardcoded gate —
consistent with "harness, not harden".

Usage:
  from calibrate import fit_calibration, apply_calibration
  knots = fit_calibration(p_oof, y_oof)      # dict
  p_cal = apply_calibration(p, knots)         # array or float
"""
import json
import numpy as np
import os

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "calibration.json")


def fit_calibration(p, y, n_bins=100):
    """PAVA isotonic regression of y on p, binned for stability. Returns knots.

    p: 1-D array of raw model probabilities (OOS/OOF only!)
    y: 1-D array of 0/1 targets
    n_bins: quantile bins of p; PAVA runs on bin centers (stable, few knots)
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # quantile bins — equal counts, not equal width
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.searchsorted(qs, p, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(qs) - 2)
    agg = {}
    for b in np.unique(bin_ids):
        m = bin_ids == b
        agg[b] = (p[m].mean(), y[m].sum(), int(m.sum()))  # sum_y + count
    bs = sorted(agg.keys())
    ps = np.array([agg[b][0] for b in bs])
    ys = np.array([agg[b][1] for b in bs])   # sum of y per bin
    ws = np.array([agg[b][2] for b in bs], dtype=np.float64)  # count per bin

    # PAVA: pool adjacent violators until monotone non-decreasing.
    # Each block: (p_edge, sum_y, w) — p_edge = weighted mean of merged bins.
    blocks = [[ps[i], ys[i], ws[i]] for i in range(len(ps))]
    i = 0
    while i < len(blocks) - 1:
        mean_i = blocks[i][1] / blocks[i][2]
        mean_j = blocks[i + 1][1] / blocks[i + 1][2]
        if mean_j < mean_i - 1e-12:
            tot_w = blocks[i][2] + blocks[i + 1][2]
            blocks[i][0] = (blocks[i][0] * blocks[i][2] + blocks[i + 1][0] * blocks[i + 1][2]) / tot_w
            blocks[i][1] += blocks[i + 1][1]
            blocks[i][2] = tot_w
            del blocks[i + 1]
            if i > 0:
                i -= 1
            continue
        i += 1
    ps_f = np.array([b[0] for b in blocks])
    ys_f = np.array([b[1] / b[2] for b in blocks])
    # monotone snap + clamp to [0.05, 0.95]
    ys_f = np.maximum.accumulate(ys_f)
    ys_f = np.clip(ys_f, 0.05, 0.95)
    out = {"knots_p": ps_f.tolist(), "knots_y": ys_f.tolist(),
           "n": int(len(p)), "min_p": float(p.min()), "max_p": float(p.max())}
    return out


def apply_calibration(p, knots):
    """Piecewise-linear (monotone) interpolation of raw p through the knots.

    v7.11 (2026-08-05): FLAT-CLAMP the tails. np.interp linearly extrapolates
    past the last knot — a raw P=0.83 on the SELL_1.3 curve extrapolated to a
    calibrated 0.999, and raw ~0.9 became >1.0. That is how the engine printed
    P=92% SELL into a +$47 rally whose true 6yr base rate is ~52%. The
    calibration curve's reliability ends at the last observed knot; beyond it
    the honest answer is "no better than the best empirically-observed rate",
    not an invented extrapolation. Values below the first knot clamp to the
    first knot's calibrated rate (never below), values above the last knot
    clamp to the last knot's rate (never above). Still fully data-driven —
    the curve itself is unchanged.
    """
    kp = np.asarray(knots["knots_p"], dtype=np.float64)
    ky = np.asarray(knots["knots_y"], dtype=np.float64)
    scalar = np.isscalar(p) or (hasattr(p, "ndim") and p.ndim == 0)
    p = np.asarray(p, dtype=np.float64)
    out = np.interp(p, kp, ky)
    # v7.11 tail clamp: no extrapolation beyond the fitted knot range.
    out = np.where(p < kp[0], ky[0], out)
    out = np.where(p > kp[-1], ky[-1], out)
    if scalar:
        return float(out)
    return out


def save_calibration(knots, path=CALIB_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(knots, f)
    os.replace(path + ".tmp", path)


def load_calibration(path=CALIB_FILE):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


if __name__ == "__main__":
    # self-test on synthetic overconfident data
    rng = np.random.default_rng(7)
    n = 200000
    x = rng.normal(size=(n, 5))
    y = (rng.random(n) < 1 / (1 + np.exp(-(x[:, 0] * 1.2 - 0.3)))).astype(float)
    # fake overconfident model: p = clip(0.5 + 0.45*sign-ish, ...)
    p = np.clip(0.35 + 0.55 * (2 * y - 1) + 0.1 * rng.normal(size=n), 0.02, 0.98)
    k = fit_calibration(p, y)
    cal = apply_calibration(p[:100000], k)
    raw_acc = ((p[:100000] > 0.5) == (y[:100000] > 0.5)).mean()
    print(f"n={k['n']} knots={len(k['knots_p'])}")
    print("sample: raw 0.80 -> cal", round(apply_calibration(0.80, k), 3),
          "| raw 0.60 -> cal", round(apply_calibration(0.60, k), 3),
          "| raw 0.30 -> cal", round(apply_calibration(0.30, k), 3))
    save_calibration(k, "/tmp/cal_test.json")
    print("saved + reloaded OK:", load_calibration("/tmp/cal_test.json")["n"] == n)
