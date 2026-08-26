"""tests/test_phase5b_d8_selection_calibration.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.d8_selection_calibration import run_d8


def test_run_d8_shape_and_monotonic_stage_shrinkage():
    result = run_d8(max_holding=15, rows=600000)
    assert result["horizon"] == 15
    names = [s["stage"] for s in result["stages"]]
    assert names == ["stage_0_full_oos", "stage_1_after_opportunity_veto", "stage_2_after_ev_gate_final_traded"]
    ns = [s["n"] for s in result["stages"]]
    assert ns[0] >= ns[1] >= ns[2]  # each gate can only shrink the population, never grow it
    assert result["degradation_begins_at"] in names
    assert "honesty_note" in result and len(result["honesty_note"]) > 0


if __name__ == "__main__":
    test_run_d8_shape_and_monotonic_stage_shrinkage()
    print("tests/test_phase5b_d8_selection_calibration.py: OK")
