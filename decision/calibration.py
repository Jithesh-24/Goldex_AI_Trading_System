"""
Rolling probability calibration (Phase 2, step 6/7).

Phase 1A found the meta model's raw probabilities are well-calibrated
(slope~0.93) in aggregate, but the absolute scale drifts over calendar time
(mean predicted p: ~0.50 in 2022-2024 -> 0.48 in 2025 -> 0.47 in 2026), while
relative ranking (top-decile win rate) stays intact. A single global
threshold fit once therefore silently stops firing as the scale drifts --
that's the deployed 0.60 threshold's actual failure mode, not a vanished
edge. This module fixes the SCALE without touching the model: a small
logistic (Platt) recalibration refit periodically on a trailing window of
resolved outcomes.

Why Platt over isotonic/beta here: the observed miscalibration is a smooth,
roughly monotonic scale/location shift (calibration slope near 1, intercept
drifting), not a complex non-monotonic distortion -- exactly what a 2-
parameter logistic correction fixes. Isotonic regression is nonparametric
and needs many samples per bin to avoid a noisy step function; a rolling
window sized to react to drift on a monthly-to-yearly timescale (see
ROLLING_WINDOW_DAYS below) won't reliably have that per-bin density. Beta
calibration's extra shape parameter isn't justified by anything in the
audit -- the audit found a scale problem, not a skew problem.

Causality is enforced by construction: `fit()` only ever sees rows the
caller passes in, and `fit_rolling()` filters to outcomes with t1 <= asof
before calling fit() -- no future observation can influence an earlier
calibration mapping.
"""
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

CALIBRATION_VERSION = "platt-rolling-v1"


@dataclass
class RollingCalibrationConfig:
    window_days: int = 180          # trailing window of RESOLVED outcomes used to fit
    min_samples: int = 500          # below this, fall back to the global (all-history) fit
    refit_frequency: str = "nightly"  # piggybacks on core/retrain_daily.py's existing cadence
    stale_days: int = 14            # warn (not silently trust) a calibrator older than this
    max_newton_iters: int = 50


class PlattCalibrator:
    """2-parameter logistic recalibration: calibrated_p = sigmoid(a + b*logit(raw_p))."""

    def __init__(self, a: float = 0.0, b: float = 1.0, n_samples: int = 0,
                 window_start: str = None, window_end: str = None, fit_at_utc: str = None):
        self.a = a
        self.b = b
        self.n_samples = n_samples
        self.window_start = window_start
        self.window_end = window_end
        self.fit_at_utc = fit_at_utc

    @classmethod
    def identity(cls):
        """a=0,b=1 -- exactly reproduces the raw model probability. Used as the
        cold-start / fallback state so 'no calibrator yet' never crashes or
        silently distorts scores."""
        return cls(a=0.0, b=1.0, n_samples=0)

    @classmethod
    def fit(cls, raw_p: np.ndarray, y: np.ndarray, max_iters: int = 50):
        """Newton's method logistic fit of y ~ sigmoid(a + b*logit(raw_p)).
        Same closed-form used in research/audit_edge.py's calibration check --
        kept identical so 'what did the audit measure' and 'what does
        production apply' can never silently diverge."""
        p = np.clip(np.asarray(raw_p, dtype=np.float64), 1e-6, 1 - 1e-6)
        logit_p = np.log(p / (1 - p))
        y = np.asarray(y, dtype=np.float64)
        a, b = 0.0, 1.0
        for _ in range(max_iters):
            z = a + b * logit_p
            pr = 1 / (1 + np.exp(-z))
            w = np.clip(pr * (1 - pr), 1e-6, None)
            grad_a = np.sum(y - pr)
            grad_b = np.sum((y - pr) * logit_p)
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
        return cls(a=float(a), b=float(b), n_samples=int(len(y)))

    def apply(self, raw_p):
        p = np.clip(np.asarray(raw_p, dtype=np.float64), 1e-6, 1 - 1e-6)
        logit_p = np.log(p / (1 - p))
        z = self.a + self.b * logit_p
        return 1 / (1 + np.exp(-z))

    def to_dict(self):
        return {"version": CALIBRATION_VERSION, "a": self.a, "b": self.b,
                "n_samples": self.n_samples, "window_start": self.window_start,
                "window_end": self.window_end, "fit_at_utc": self.fit_at_utc}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(a=d["a"], b=d["b"], n_samples=d.get("n_samples", 0),
                    window_start=d.get("window_start"), window_end=d.get("window_end"),
                    fit_at_utc=d.get("fit_at_utc"))


def fit_rolling(outcomes: pd.DataFrame, asof: pd.Timestamp, cfg: RollingCalibrationConfig,
                 global_fallback: "PlattCalibrator" = None) -> PlattCalibrator:
    """outcomes: DataFrame with columns [t0_time, t1_time, raw_proba, label].
    Only rows with t1_time <= asof (fully resolved by `asof`, no partial/open
    trades) and t0_time >= asof - window_days are used -- causal by
    construction; nothing after `asof` can enter the fit.

    Cold-start / min-sample fallback: if fewer than cfg.min_samples resolved
    rows fall in the window, fall back to `global_fallback` (a calibrator fit
    once on all history available up to `asof`) if given, else identity
    (raw model probability, uncalibrated) -- never refuses to return a
    usable calibrator.
    """
    window_start = asof - pd.Timedelta(days=cfg.window_days)
    mask = (outcomes["t1_time"] <= asof) & (outcomes["t0_time"] >= window_start)
    window = outcomes.loc[mask]
    if len(window) < cfg.min_samples:
        if global_fallback is not None:
            return global_fallback
        return PlattCalibrator.identity()
    cal = PlattCalibrator.fit(window["raw_proba"].to_numpy(), window["label"].to_numpy(),
                               max_iters=cfg.max_newton_iters)
    cal.window_start = str(window_start)
    cal.window_end = str(asof)
    cal.fit_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return cal


def is_stale(cal: PlattCalibrator, now: pd.Timestamp, cfg: RollingCalibrationConfig) -> bool:
    if cal.fit_at_utc is None:
        return True
    fit_at = pd.Timestamp(cal.fit_at_utc)
    return (now - fit_at) > pd.Timedelta(days=cfg.stale_days)
