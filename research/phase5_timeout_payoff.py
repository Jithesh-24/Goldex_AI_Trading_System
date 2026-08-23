"""research/phase5_timeout_payoff.py
Spec section 9: timeout_R was a documented midpoint proxy
(0.5*(MFE_q50-MAE_q50)). This script computes the ACTUAL realized R at
timeout directly from historical outcomes -- events whose direction label
is 0 (neither barrier touched within max_holding) are a data fact, not a
model prediction, so no OOF/CV machinery is needed: this is a descriptive
statistic over historical timeout events, used as an EV engine constant.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_timeout_payoff
"""
import numpy as np

from research.phase4_dataset import assemble_v3_dataset
from research.audit_edge import _mae_mfe_core
from features.labeling import TripleBarrierConfig, triple_barrier_labels

MIN_TIMEOUT_SAMPLES = 200


def estimate_timeout_payoff(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    timeout_mask = y == 0
    n_timeout = int(timeout_mask.sum())

    directional_mask = y != 0
    side_directional = y[directional_mask]
    t0_dir = t0_idx[directional_mask]
    t1_dir = labels["t1"].to_numpy()[directional_mask]
    vol_dir = vol_tb[t0_dir]
    mae_dir, mfe_dir = _mae_mfe_core(close, high, low, t0_dir, t1_dir, side_directional.astype(float), vol_dir)
    proxy_mean = float(0.5 * (np.nanmedian(mfe_dir) - np.nanmedian(mae_dir)))

    if n_timeout < MIN_TIMEOUT_SAMPLES:
        return {"n_timeout_events": n_timeout, "timeout_R_mean": proxy_mean,
                "timeout_R_q25": None, "timeout_R_q75": None, "provisional_proxy": True}

    t0_to = t0_idx[timeout_mask]
    t1_to = labels["t1"].to_numpy()[timeout_mask]
    vol_to = vol_tb[t0_to]
    side_to = np.ones(n_timeout)  # side is undefined for a timeout event with no primary label; use symmetric +1 as the reference frame, magnitude only matters for R computation here
    mae_to, mfe_to = _mae_mfe_core(close, high, low, t0_to, t1_to, side_to, vol_to)
    realized_R = mfe_to - mae_to  # net directional excursion at timeout, in R-multiples

    return {"n_timeout_events": n_timeout,
            "timeout_R_mean": float(np.nanmean(realized_R)),
            "timeout_R_q25": float(np.nanpercentile(realized_R, 25)),
            "timeout_R_q75": float(np.nanpercentile(realized_R, 75)),
            "provisional_proxy": False}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = estimate_timeout_payoff(h)
        print(f"h={h}: {r}")
