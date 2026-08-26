"""Synthetic-path validation for the direction-dependent realized_r fix
(targeted correction pass, 2026-08-24). Hand-computed expected R for a
known LONG-favorable path and a known SHORT-favorable path, verified
independently for both directions."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.audit_edge import _mae_mfe_core
from research.phase5_ev_dataset import realized_r_for_direction, assemble_replay_dataset


def test_realized_r_rising_path_long_wins():
    close = np.array([100., 101., 102., 104., 106., 106.])
    high = close.copy()
    low = close.copy()
    t0 = np.array([0])
    t1 = np.array([4])
    vol = np.array([1.0])  # vol=1 so R-multiples equal raw fractional moves

    mae_long, mfe_long = _mae_mfe_core(close, high, low, t0, t1, np.array([1.0]), vol)
    mae_short, mfe_short = _mae_mfe_core(close, high, low, t0, t1, np.array([-1.0]), vol)

    assert mae_long[0] == 0.0
    assert abs(mfe_long[0] - 0.06) < 1e-9
    assert mfe_short[0] == 0.0
    assert abs(mae_short[0] - 0.06) < 1e-9

    touch = 1  # rising path touches the upper barrier first -> favors long
    realized_r_long = mfe_long[0] if touch == 1 else -mae_long[0]
    realized_r_short = mfe_short[0] if touch == -1 else -mae_short[0]

    assert abs(realized_r_long - 0.06) < 1e-9
    assert abs(realized_r_short - (-0.06)) < 1e-9

    data = {"realized_r_long": np.array([realized_r_long]), "realized_r_short": np.array([realized_r_short])}
    assert realized_r_for_direction("long", 0, data) == realized_r_long
    assert realized_r_for_direction("short", 0, data) == realized_r_short


def test_realized_r_falling_path_short_wins():
    close = np.array([100., 99., 98., 96., 94., 94.])
    high = close.copy()
    low = close.copy()
    t0 = np.array([0])
    t1 = np.array([4])
    vol = np.array([1.0])

    mae_long, mfe_long = _mae_mfe_core(close, high, low, t0, t1, np.array([1.0]), vol)
    mae_short, mfe_short = _mae_mfe_core(close, high, low, t0, t1, np.array([-1.0]), vol)

    touch = -1  # falling path touches the lower barrier first -> favors short
    realized_r_long = mfe_long[0] if touch == 1 else -mae_long[0]
    realized_r_short = mfe_short[0] if touch == -1 else -mae_short[0]

    assert realized_r_long < 0
    assert realized_r_short > 0

    data = {"realized_r_long": np.array([realized_r_long]), "realized_r_short": np.array([realized_r_short])}
    assert realized_r_for_direction("long", 0, data) < 0
    assert realized_r_for_direction("short", 0, data) > 0


def test_realized_r_for_direction_rejects_bad_direction():
    data = {"realized_r_long": np.array([0.1]), "realized_r_short": np.array([-0.1])}
    try:
        realized_r_for_direction("sideways", 0, data)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_assemble_replay_dataset_exposes_direction_conditioned_mae_mfe():
    data = assemble_replay_dataset(max_holding=15, rows=600000)
    assert "mae_dir" in data and "mfe_dir" in data
    assert len(data["mae_dir"]) == data["n"]
    assert len(data["mfe_dir"]) == data["n"]
    assert (data["mae_dir"] >= 0).all()  # MAE is a magnitude (adverse excursion size), never negative
    assert (data["mfe_dir"] >= 0).all()  # MFE is a magnitude (favorable excursion size), never negative


if __name__ == "__main__":
    test_realized_r_rising_path_long_wins()
    test_realized_r_falling_path_short_wins()
    test_realized_r_for_direction_rejects_bad_direction()
    test_assemble_replay_dataset_exposes_direction_conditioned_mae_mfe()
    print("tests/test_phase5_ev_dataset_realized_r.py: OK")
