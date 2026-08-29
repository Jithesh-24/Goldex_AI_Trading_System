"""tests/test_phase5b_run_all.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5b_diagnostics.run_all import run_batch1, apply_attribution_framework, write_report


def test_apply_attribution_framework_always_returns_all_five_explanations():
    # minimal synthetic horizon_results shaped like one horizon's real output
    horizon_results = {
        "d1": {"oos": {"point_biserial": {"r": 0.02, "n": 100000, "ci_lo": 0.01, "ci_hi": 0.03}}},
        "d2": {"overall": {"up_frac": 0.02, "down_frac": 0.97, "timeout_frac": 0.01}},
        "d3": {"opportunity": {"point_biserial": {"r": 0.15, "n": 50000, "ci_lo": 0.14, "ci_hi": 0.16}},
               "barrier": {"point_biserial": {"r": 0.15, "n": 50000, "ci_lo": 0.14, "ci_hi": 0.16}}},
        "d4": {"contradiction_barrier_vs_reward_risk": {"rate": 0.3, "ci_lo": 0.29, "ci_hi": 0.31},
               "contradiction_opportunity_vs_barrier": {"rate": 0.1, "ci_lo": 0.09, "ci_hi": 0.11}},
        "d5": {"global": {"calibration": {"slope": 0.6, "intercept": 0.2}},
               "calibration_vs_meta_label": {"calibration": {"slope": 0.6, "intercept": 0.2}},
               "traded_subset": "N/A (zero trades at this horizon)"},
        "d6": {},
    }
    explanations = apply_attribution_framework(horizon_results)
    names = {e["explanation"] for e in explanations}
    assert names == {"market/labels", "direction", "downstream_specialists", "calibration", "disagreement"}
    for e in explanations:
        assert "evidence" in e and "decisive" in e


def test_run_batch1_and_write_report_smoke(monkeypatch):
    # small rows= slice for a fast smoke test of the orchestration wiring itself,
    # NOT the real full-history run (that's a separate, long-running research run,
    # not a unit test -- see Step 6 below).
    result = run_batch1(rows=600000)
    assert set(result["horizons"].keys()) == {15, 45, 90}
    for h in (15, 45, 90):
        assert set(result["horizons"][h].keys()) == {"d1", "d2", "d3", "d4", "d5", "d6"}
        assert len(result["attribution"][h]) == 5

    with tempfile.TemporaryDirectory() as out_dir:
        json_path, md_path = write_report(result, out_dir)
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        with open(json_path) as f:
            reloaded = json.load(f)
        assert "horizons" in reloaded


if __name__ == "__main__":
    test_apply_attribution_framework_always_returns_all_five_explanations()
    test_run_batch1_and_write_report_smoke()
    print("tests/test_phase5b_run_all.py: OK")
