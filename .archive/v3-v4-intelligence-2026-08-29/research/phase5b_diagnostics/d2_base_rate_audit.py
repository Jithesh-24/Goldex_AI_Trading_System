"""research/phase5b_diagnostics/d2_base_rate_audit.py
Batch 1, D2: model-free directional base rate from triple-barrier labels
alone (side=None, symmetric barriers), overall and by calendar year.
Answers whether h=15's 24-long/107,611-short replay skew could be
explained by the raw event population before any model exists. See
docs/superpowers/specs/2026-08-26-golex-v3-phase5-batch1-diagnostics-
design.md section D2.
"""
import numpy as np
import pandas as pd
from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels


def _fracs(touch: np.ndarray) -> dict:
    n = len(touch)
    if n == 0:
        return {"n": 0, "up_frac": None, "down_frac": None, "timeout_frac": None}
    return {"n": n,
            "up_frac": float((touch == 1).mean()),
            "down_frac": float((touch == -1).mean()),
            "timeout_frac": float((touch == 0).mean())}


def run_d2(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    touch = labels["touch"].to_numpy()
    times = pd.to_datetime(ds["feat_v3"]["time"].to_numpy())[ds["t0_idx"]]
    years = times.year

    overall = _fracs(touch)
    by_year = []
    for yr in sorted(set(years.tolist())):
        m = years == yr
        row = _fracs(touch[m])
        row["year"] = int(yr)
        by_year.append(row)

    return {"horizon": max_holding, "overall": overall, "by_year": by_year}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = run_d2(max_holding=h)
        print(f"D2 h={h}: {r['overall']}")
