"""tests/test_direction_side.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.direction_side import compute_direction_oof


def test_compute_direction_oof_shapes_and_side_values():
    out = compute_direction_oof(max_holding=15, rows=600000)
    n = len(out["t0_nz"])
    assert len(out["p_direction_raw"]) == n
    assert len(out["p_direction_cal"]) == n
    assert len(out["side"]) == n
    assert len(out["has_oof"]) == n
    assert out["model_id"] == "direction_v3_candidate_h15"
    assert out["has_oof"].sum() > 50, "too few OOF events in dry run to trust anything downstream"
    side_valid = out["side"][out["has_oof"]]
    assert set(side_valid.tolist()) <= {1.0, -1.0}
    p_valid = out["p_direction_cal"][out["has_oof"]]
    assert ((p_valid >= 0.0) & (p_valid <= 1.0)).all()


def test_side_matches_probability_threshold():
    out = compute_direction_oof(max_holding=15, rows=600000)
    m = out["has_oof"]
    expected_side = (out["p_direction_raw"][m] >= 0.5).astype(float) * 2 - 1
    assert (out["side"][m] == expected_side).all(), \
        "side must be derived from p_direction_raw >= 0.5, matching ev_engine.py's own direction_gate_ok rule"


if __name__ == "__main__":
    test_compute_direction_oof_shapes_and_side_values()
    test_side_matches_probability_threshold()
    print("tests/test_direction_side.py: OK")
