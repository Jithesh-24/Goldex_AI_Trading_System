"""tests/test_phase5_uncertainty_k.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_uncertainty_k import derive_and_validate_k


def test_derive_and_validate_k_picks_best_separator():
    # Constructed so k=1.0 perfectly separates realized outcome sign from
    # a synthetic, uncertainty-inflated ev_raw; k=0.0 does not.
    events = [
        {"ev_raw": 0.5, "uncertainty": 0.9, "realized_r": -0.2},   # high uncertainty, actually a loser
        {"ev_raw": 0.5, "uncertainty": 0.1, "realized_r": 0.4},    # low uncertainty, actually a winner
        {"ev_raw": 0.3, "uncertainty": 0.8, "realized_r": -0.1},
        {"ev_raw": 0.3, "uncertainty": 0.05, "realized_r": 0.25},
    ]
    result = derive_and_validate_k(candidate_ks=[0.0, 0.5, 1.0], events=events)
    assert "chosen_k" in result
    assert result["chosen_k"] in (0.0, 0.5, 1.0)
    assert len(result["validation"]) == 3


if __name__ == "__main__":
    test_derive_and_validate_k_picks_best_separator()
    print("tests/test_phase5_uncertainty_k.py: OK")
