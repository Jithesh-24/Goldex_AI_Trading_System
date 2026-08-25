"""research/phase5b_diagnostics/_stats_utils.py
Shared statistics helpers for Phase 5 Batch 1 diagnostics (D1-D6). Every
diagnostic that reports a point-biserial correlation, a calibration
slope/intercept, or a population's sample size uses these, so the
n/CI/population-tagging convention is identical across all six modules
instead of six independent reimplementations that could silently drift
apart (see docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch1-
diagnostics-design.md section 2a).
"""
import numpy as np
from scipy.stats import pointbiserialr


def pointbiserial_with_ci(y_true: np.ndarray, p: np.ndarray, z: float = 1.96) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = len(y_true)
    if n < 4:
        return {"r": None, "n": n, "ci_lo": None, "ci_hi": None}
    r, _ = pointbiserialr(y_true, p)
    r = float(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    z_r = np.arctanh(r)
    lo, hi = np.tanh(z_r - z * se), np.tanh(z_r + z * se)
    return {"r": r, "n": n, "ci_lo": float(lo), "ci_hi": float(hi)}


def fit_calibration_slope_intercept(y_true: np.ndarray, p_raw: np.ndarray) -> dict:
    y_c = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_raw, dtype=np.float64), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    n = len(y_c)
    a, b = 0.0, 1.0
    h_aa = h_bb = h_ab = -1.0
    for _ in range(50):
        z_lin = a + b * logit_p
        pr = 1 / (1 + np.exp(-z_lin))
        w = np.clip(pr * (1 - pr), 1e-6, None)
        grad_a = np.sum(y_c - pr)
        grad_b = np.sum((y_c - pr) * logit_p)
        h_aa = -np.sum(w)
        h_bb = -np.sum(w * logit_p ** 2)
        h_ab = -np.sum(w * logit_p)
        det = h_aa * h_bb - h_ab ** 2
        if abs(det) < 1e-12:
            break
        da = (grad_a * h_bb - grad_b * h_ab) / det
        db = (grad_b * h_aa - grad_a * h_ab) / det
        a -= da
        b -= db
    det = h_aa * h_bb - h_ab ** 2
    if abs(det) < 1e-12:
        intercept_se = slope_se = float("nan")
    else:
        cov_aa = -h_bb / det
        cov_bb = -h_aa / det
        intercept_se = float(np.sqrt(max(cov_aa, 0.0)))
        slope_se = float(np.sqrt(max(cov_bb, 0.0)))
    return {"intercept": float(a), "slope": float(b),
            "intercept_se": intercept_se, "slope_se": slope_se, "n": n}


def population_label(name: str, n: int) -> dict:
    return {"population": name, "n": n}
