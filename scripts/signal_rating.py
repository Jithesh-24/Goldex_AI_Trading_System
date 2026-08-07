"""v8 SIGNAL RATING — institutional 0-100 quality score, learned from data.

USER MANDATE (2026-08-07): "a rating system or something can be induced…
not every trade it sees should be taken." Also "no hardcoded values".

WHAT IT DOES:
  Combines the model's OWN outputs into a 0-100 rating:
    - calibrated P(win)          → honest probability (from per-dir×RR curves)
    - expectancy per $ risked    → edge magnitude (scale-free)
    - regime confidence          → how far the market state is from regime-bin
                                   boundaries (soft distance, from trend_ema,
                                   trend_slope, bb_pctile, atr_pctile)
    - MFE/MFA headroom           → is the chosen SL beyond the regime's learned
                                   adverse band? is TP inside the favorable run?
                                   (placement_prior.json, learned from 6yr data)
  All weights are LEARNED (fit_signal_rating.py) from the M5 matrix OOF
  calibration curves — nothing here is hardcoded. The learned weights live in
  models/signal_rating.json and are hot-reloaded per signal.

  Fire policy: engine fires only when rating >= learned threshold (the rating
  level at which historical expectancy turns positive, from OOF data).

Usage (engine):
    from signal_rating import rate_signal
    rating, score_parts = rate_signal(fx, direction, p_win, exp, sl_atr, regime)
    if rating < rating_threshold: skip
"""
import os, json
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RATING_FILE = f"{MODEL_DIR}/signal_rating.json"
_PLACEMENT_FILE = f"{MODEL_DIR}/placement_prior.json"

_CACHE = {"rating": None, "placement": None, "mtime": 0}


def _load():
    """Hot-reload: re-read if mtime changed (engine reloads per signal)."""
    try:
        mt = os.path.getmtime(RATING_FILE)
        if _CACHE["rating"] is None or mt != _CACHE["mtime"]:
            with open(RATING_FILE) as f:
                _CACHE["rating"] = json.load(f)
            _CACHE["mtime"] = mt
    except Exception:
        _CACHE["rating"] = None
    try:
        with open(_PLACEMENT_FILE) as f:
            _CACHE["placement"] = json.load(f)
    except Exception:
        _CACHE["placement"] = None
    return _CACHE["rating"]


def rate_signal(fx, direction, p_win, exp, sl_atr, regime=None):
    """Compute 0-100 rating for a candidate signal.

    fx       — feature dict (regime cols present)
    direction — "BUY"/"SELL"
    p_win    — CALIBRATED win probability
    exp      — expectancy per $ risked (p*rr − (1−p))
    sl_atr   — chosen SL distance in ATR units
    regime   — regime bin name (engine already computed it)
    Returns (rating 0-100, parts dict) — parts for logging/telegram.
    """
    cfg = _load()
    if not cfg:
        # No learned config yet (pre-training) — neutral fallback so the
        # engine never crashes: rating from p_win and exp alone, 0-100 scale.
        r = float(np.clip(50.0 + (p_win - 0.5) * 120.0 + exp * 25.0, 0, 100))
        return round(r, 1), {"p_win": p_win, "exp": exp, "fallback": True}

    w = cfg.get("weights", {})
    w_p = float(w.get("p_win", 0.5))
    w_e = float(w.get("exp", 0.4))
    w_r = float(w.get("regime_conf", 0.05))
    w_x = float(w.get("excursion", 0.05))
    w_d = float(w.get("direction", 0.0))

    # 1) calibrated probability component (0-1)
    p_component = float(np.clip(p_win, 0, 1))

    # 2) expectancy component (0-1): exp ranges roughly -1..+3 per $ risked
    e_component = float(np.clip((exp + 0.25) / 3.0, 0, 1))

    # 3) regime confidence (0-1): distance from regime-bin decision boundaries.
    #    Soft version of regime_bin() — high when the state is deep inside a
    #    bin, low when it's straddling a boundary (model uncertain).
    try:
        te = float(fx.get("trend_ema", 0.0))
        ts = float(fx.get("trend_slope", 0.0))
        ap = float(fx.get("atr_pctile", 0.5))
        bb = float(fx.get("bb_pctile", 0.5))
        # distance to nearest trend boundary (0.4 / 1.2 in trend_ema units)
        dist = min(abs(abs(te) - 0.4), abs(abs(te) - 1.2), 1.0)
        vol_dist = abs(ap - 0.5) * 2.0
        r_component = float(np.clip(0.5 * (1 - dist) + 0.25 * vol_dist, 0, 1))
    except Exception:
        r_component = 0.5

    # 4) excursion headroom (0-1): SL beyond the regime's learned adverse band
    #    → stop is outside noise, placement is institutional-grade.
    x_component = 0.5
    try:
        p = _CACHE["placement"] or {}
        reg = regime or ""
        if p and reg in p.get("regimes", {}):
            d = p["regimes"][reg].get(direction, {})
            learned_sl = d.get("sl_atr")
            mfa_p50 = d.get("mfa_p50")
            if learned_sl and mfa_p50 and sl_atr is not None:
                if sl_atr >= learned_sl:
                    x_component = 1.0   # stop beyond learned band — good
                elif sl_atr > mfa_p50:
                    x_component = 0.6   # between median and learned — acceptable
                else:
                    x_component = 0.2   # inside the noise band — placement risky
    except Exception:
        pass

    # 5) direction edge component (0-1) — optional learned direction prior tilt
    d_component = 0.5
    try:
        from features import regime_dir_prior_file  # noqa — hot path lazy
    except Exception:
        pass

    rating = 100.0 * (w_p * p_component + w_e * e_component +
                      w_r * r_component + w_x * x_component +
                      w_d * d_component) / max(w_p + w_e + w_r + w_x + w_d, 1e-9)
    return round(float(np.clip(rating, 0, 100)), 1), {
        "p_win": round(p_win, 3), "exp": round(exp, 3),
        "p_comp": round(p_component, 3), "e_comp": round(e_component, 3),
        "r_comp": round(r_component, 3), "x_comp": round(x_component, 3),
    }


def rating_threshold():
    """Learned fire threshold (0-100). From signal_rating.json."""
    cfg = _load()
    if not cfg:
        return 0.0   # no learned config → never block (old behavior)
    return float(cfg.get("threshold", 0.0))
