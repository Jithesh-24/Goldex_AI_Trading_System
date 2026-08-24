"""research/phase5_ev_dataset.py
Assembles a per-event replay dataset: for each historical event, the
specialist outputs an EV engine would have seen (Direction/Opportunity/
Barrier OOF probabilities, MAE/MFE OOF-PREDICTED q75 values), plus real
historical excursions for expected-vs-realized comparison.

FIX (targeted correction pass, 2026-08-24): realized_r used to be computed
from the historically-realized winning side (`y[nz]`, the true triple-
barrier label) regardless of which direction the EV engine actually
decided to trade -- meaning "expected vs realized" validation numbers
measured the wrong thing whenever the engine's chosen side disagreed with
history's actual winning side. Fixed by computing realized MAE/MFE/R
independently for a fixed-LONG and a fixed-SHORT hypothesis at every
event (using the direction-agnostic `touch` column -- which barrier was
hit first is identical under either hypothesis because barrier widths are
symmetric, pt_mult==sl_mult==1.0; only the favorable/adverse sign
convention differs), and exposing `realized_r_for_direction` so the
consuming replay engine (research/phase5_ev_engine.py) picks the stream
matching its OWN decision, per event.

FIX (targeted correction pass, 2026-08-24): also exposes real per-event
`mid` (close price at entry) and `vol_60s_proxy` (the per-bar EWMA vol
underlying vol_tb, recovered by un-scaling: vol_tb = ewma_vol *
sqrt(max_holding) * HORIZON_VOL_SCALE) so research/phase5_ev_engine.py's
replay cost model no longer uses two hardcoded constants.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_ev_dataset
"""
import numpy as np

from research.phase4_dataset import assemble_v3_dataset, HORIZON_VOL_SCALE
from research.phase5_calibration import (
    _oof_for_direction, _oof_for_opportunity, _oof_for_barrier, _oof_predicted_mae_mfe,
)
from research.audit_edge import _mae_mfe_core
from features.labeling import TripleBarrierConfig, triple_barrier_labels

REPRESENTATIVE_SPREAD = 0.015  # documented placeholder: Phase 4 did not persist historical tick spread; real
                                # live spread is used in decision/ev_engine.py's live path (Task 12) -- this
                                # constant is a research-only stand-in for OOS replay, not a live value.


def realized_r_for_direction(direction: str, i: int, data: dict) -> float:
    """Picks the realized-R stream matching the direction actually traded.
    `direction` must be "long" or "short" (an EVDecision.direction value) --
    never called for a NO_TRADE event, which has no direction."""
    if direction == "long":
        return float(data["realized_r_long"][i])
    if direction == "short":
        return float(data["realized_r_short"][i])
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def assemble_replay_dataset(max_holding: int, rows: int = None) -> dict:
    ds = assemble_v3_dataset(max_holding=max_holding, rows=rows)
    close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=max_holding, min_vol=1e-6)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
    y = labels["label"].to_numpy()
    touch = labels["touch"].to_numpy()
    nz = y != 0
    t0_nz, t1_nz, touch_nz = t0_idx[nz], labels["t1"].to_numpy()[nz], touch[nz]

    t0_dir, y_dir, p_dir, m_dir = _oof_for_direction(max_holding, rows=rows)
    t0_opp, y_opp, p_opp, m_opp = _oof_for_opportunity(max_holding, rows=rows)
    t0_bar, y_bar, p_bar, m_bar = _oof_for_barrier(max_holding, rows=rows)
    t0_mm, mae_pred, mfe_pred, m_mm = _oof_predicted_mae_mfe(max_holding, rows=rows)

    assert np.array_equal(t0_nz, t0_dir), "direction OOF base index mismatch"
    assert np.array_equal(t0_nz, t0_opp), "opportunity OOF base index mismatch"
    assert np.array_equal(t0_nz, t0_bar), "barrier OOF base index mismatch"
    assert np.array_equal(t0_nz, t0_mm), "mae/mfe OOF base index mismatch"

    combined = m_dir & m_opp & m_bar & m_mm
    n = int(combined.sum())

    vol_sel = vol_tb[t0_nz][combined]
    t0_sel, t1_sel = t0_nz[combined], t1_nz[combined]
    touch_sel = touch_nz[combined]

    # Real historical excursions for BOTH a fixed-long and a fixed-short
    # hypothesis at every event -- NOT the historically-realized winning side.
    side_long = np.ones(n, dtype=np.float64)
    side_short = -np.ones(n, dtype=np.float64)
    mae_long, mfe_long = _mae_mfe_core(close, high, low, t0_sel, t1_sel, side_long, vol_sel)
    mae_short, mfe_short = _mae_mfe_core(close, high, low, t0_sel, t1_sel, side_short, vol_sel)

    realized_r_long = np.where(touch_sel == 1, mfe_long, -mae_long)
    realized_r_short = np.where(touch_sel == -1, mfe_short, -mae_short)

    # Real historical inputs for the replay cost model, replacing two
    # previously-hardcoded constants (realized_vol_60s=0.0006, mid=2350.0).
    mid = close[t0_sel]
    vol_60s_proxy = vol_sel / (np.sqrt(max_holding) * HORIZON_VOL_SCALE)

    return {"n": n, "p_direction": p_dir[combined], "p_opportunity": p_opp[combined],
            "p_barrier_win": p_bar[combined],
            "mae_r": mae_pred[combined], "mfe_r": mfe_pred[combined],
            "touch": touch_sel,
            "realized_r_long": realized_r_long, "realized_r_short": realized_r_short,
            "mid": mid, "vol_60s_proxy": vol_60s_proxy,
            "spread": np.full(n, REPRESENTATIVE_SPREAD)}


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        d = assemble_replay_dataset(h)
        print(f"h={h}: n={d['n']}")
