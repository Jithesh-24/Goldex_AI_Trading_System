"""research/phase5_ev_dataset.py
Assembles a per-event replay dataset: for each historical event, the
specialist outputs an EV engine would have seen (Direction/Opportunity/
Barrier OOF probabilities, MAE/MFE OOF-PREDICTED q75 values -- see FIX 1
below --, a synthetic MarketState using a fixed representative spread
since Phase 4 did not persist historical tick-level spread), plus the
realized R outcome for expected-vs-realized comparison.

FIX 1 (C1+C2, final-review fix wave): mae_r/mfe_r used to be the event's
OWN realized future MAE/MFE excursion (look-ahead leakage -- the
"specialist prediction" already knew the true future outcome). They are
now genuine out-of-fold PREDICTED q75 values from
research.phase5_calibration._oof_predicted_mae_mfe, the same OOF-fold
methodology research/phase4_mae_quantile.py already uses for its
persisted candidate model. realized_mae/realized_mfe/realized_r are kept
SEPARATELY, computed from the true _mae_mfe_core outcome -- legitimate
ONLY for the "expected vs realized" comparison metric, never as an input
to the engine's own decision.

FIX 2 (C3, final-review fix wave): the four OOF streams (direction,
opportunity, barrier, mae/mfe) each have independent OOF-availability
subsets. They are combined here via one AND-ed mask over their shared
t0_nz base index (see research/phase5_calibration.py's module docstring
for the alignment convention each _oof_for_* function follows), not via
the old buggy independent-length `[:n]` positional slicing.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_dataset
"""
import numpy as np

from research.phase4_dataset import assemble_v3_dataset
from research.phase5_calibration import (
    _oof_for_direction, _oof_for_opportunity, _oof_for_barrier, _oof_predicted_mae_mfe,
)
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

    t0_dir, y_dir, p_dir, m_dir = _oof_for_direction(max_holding, rows=rows)
    t0_opp, y_opp, p_opp, m_opp = _oof_for_opportunity(max_holding, rows=rows)
    t0_bar, y_bar, p_bar, m_bar = _oof_for_barrier(max_holding, rows=rows)
    t0_mm, mae_pred, mfe_pred, m_mm = _oof_predicted_mae_mfe(max_holding, rows=rows)

    # All four OOF-producing calls build their base event set from the exact same
    # assemble_v3_dataset(max_holding=...) + triple_barrier_labels(...) construction
    # with an identical TripleBarrierConfig, so t0_nz (this function's own base
    # index) must be identical in length and order to each function's own
    # returned t0 array. Assert this rather than trust it silently -- a silent
    # divergence here is exactly the C3 misalignment bug this fix addresses.
    assert np.array_equal(t0_nz, t0_dir), "direction OOF base index mismatch"
    assert np.array_equal(t0_nz, t0_opp), "opportunity OOF base index mismatch"
    assert np.array_equal(t0_nz, t0_bar), "barrier OOF base index mismatch"
    assert np.array_equal(t0_nz, t0_mm), "mae/mfe OOF base index mismatch"

    combined = m_dir & m_opp & m_bar & m_mm
    n = int(combined.sum())

    side_sel = y[nz][combined].astype(float)
    vol_sel = vol_tb[t0_nz][combined]
    t0_sel, t1_sel = t0_nz[combined], t1_nz[combined]
    realized_mae, realized_mfe = _mae_mfe_core(close, high, low, t0_sel, t1_sel, side_sel, vol_sel)
    realized_r = np.where(y_bar[combined] == 1, realized_mfe, -realized_mae)

    return {"n": n, "p_direction": p_dir[combined], "p_opportunity": p_opp[combined],
            "p_barrier_win": p_bar[combined],
            "mae_r": mae_pred[combined], "mfe_r": mfe_pred[combined],
            "realized_mae": realized_mae, "realized_mfe": realized_mfe, "realized_r": realized_r,
            "spread": np.full(n, REPRESENTATIVE_SPREAD)}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        d = assemble_replay_dataset(h)
        print(f"h={h}: n={d['n']}")
