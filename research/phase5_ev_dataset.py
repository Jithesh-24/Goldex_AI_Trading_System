"""research/phase5_ev_dataset.py
Assembles a per-event replay dataset: for each historical event, the
specialist outputs an EV engine would have seen (Direction/Opportunity/
Barrier OOF probabilities, MAE/MFE realized quantile-equivalent values,
a synthetic MarketState using a fixed representative spread since Phase 4
did not persist historical tick-level spread), plus the realized R
outcome for expected-vs-realized comparison.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_dataset
"""
from datetime import datetime, timezone

import numpy as np

from research.phase4_dataset import assemble_v3_dataset
from research.phase5_calibration import _oof_for_direction, _oof_for_opportunity, _oof_for_barrier
from research.audit_edge import _mae_mfe_core
from features.labeling import TripleBarrierConfig, triple_barrier_labels

REPRESENTATIVE_SPREAD = 0.015  # documented placeholder: Phase 4 did not persist historical tick spread; real
                                # live spread is used in decision/ev_engine.py's live path (Task 12) -- this
                                # constant is a research-only stand-in for OOS replay, not a live value.


def assemble_replay_dataset(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    nz = y != 0
    t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]

    y_dir, p_dir = _oof_for_direction(max_holding, rows=rows)
    y_opp, p_opp = _oof_for_opportunity(max_holding, rows=rows)
    y_bar, p_bar = _oof_for_barrier(max_holding, rows=rows)
    n = min(len(p_dir), len(p_opp), len(p_bar), nz.sum())

    side_nz = y[nz][:n].astype(float)
    vol_nz = vol_tb[t0_nz][:n]
    mae_r, mfe_r = _mae_mfe_core(close, high, low, t0_nz[:n], t1_nz[:n], side_nz, vol_nz)
    realized_r = np.where(y_bar[:n] == 1, mfe_r, -mae_r)

    return {"n": n, "p_direction": p_dir[:n], "p_opportunity": p_opp[:n], "p_barrier_win": p_bar[:n],
            "mae_r": mae_r, "mfe_r": mfe_r, "realized_r": realized_r,
            "spread": np.full(n, REPRESENTATIVE_SPREAD)}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        d = assemble_replay_dataset(h)
        print(f"h={h}: n={d['n']}")
