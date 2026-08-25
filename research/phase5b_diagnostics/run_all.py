"""research/phase5b_diagnostics/run_all.py
Batch 1 orchestrator: runs D1-D6 for all three horizons, applies the
attribution framework (docs/superpowers/specs/2026-08-26-golex-v3-phase5-
batch1-diagnostics-design.md section 4), writes one JSON+markdown report.
No diagnostic is ever skipped based on another's result -- "decisive" is a
label attached when the report is assembled, never a control-flow branch.
"""
import json
import os

from research.phase4_dataset import HORIZONS
from research.phase5b_diagnostics.d1_direction_quality import run_d1
from research.phase5b_diagnostics.d2_base_rate_audit import run_d2
from research.phase5b_diagnostics.d3_specialist_oof_quality import run_d3
from research.phase5b_diagnostics.d4_cross_specialist_consistency import run_d4
from research.phase5b_diagnostics.d5_calibration_reliability import run_d5
from research.phase5b_diagnostics.d6_long_short_conditioning import run_d6


def apply_attribution_framework(horizon_results: dict) -> list:
    d1, d2, d3, d4, d5 = (horizon_results["d1"], horizon_results["d2"], horizon_results["d3"],
                           horizon_results["d4"], horizon_results["d5"])

    explanations = []

    d2_skew = max(d2["overall"]["up_frac"] or 0.0, d2["overall"]["down_frac"] or 0.0)
    market_decisive = d2_skew >= 0.9
    explanations.append({
        "explanation": "market/labels",
        "evidence": f"D2 raw label base rate: up={d2['overall']['up_frac']}, down={d2['overall']['down_frac']} "
                    f"(dominant-side frac={d2_skew:.4f})",
        "decisive": market_decisive,
    })

    dir_r = d1["oos"]["point_biserial"]["r"]
    direction_weak = dir_r is not None and abs(dir_r) < 0.02
    explanations.append({
        "explanation": "direction",
        "evidence": f"D1 Direction point-biserial r={dir_r} (n={d1['oos']['point_biserial']['n']})",
        "decisive": direction_weak,
    })

    opp_r = d3["opportunity"]["point_biserial"]["r"]
    downstream_weaker = (dir_r is not None and opp_r is not None and abs(opp_r) < abs(dir_r) * 0.5)
    explanations.append({
        "explanation": "downstream_specialists",
        "evidence": f"D3 Opportunity point-biserial r={opp_r} vs D1 Direction r={dir_r}",
        "decisive": downstream_weaker,
    })

    slope = d5["global"]["calibration"]["slope"]
    calibration_off = slope is not None and abs(slope - 1.0) > 0.3
    explanations.append({
        "explanation": "calibration",
        "evidence": f"D5 global calibration slope={slope} (ideal=1.0), traded_subset={d5['traded_subset'] if isinstance(d5['traded_subset'], str) else 'computed'}",
        "decisive": calibration_off,
    })

    rate1 = d4["contradiction_barrier_vs_reward_risk"]["rate"]
    rate2 = d4["contradiction_opportunity_vs_barrier"]["rate"]
    disagreement_high = (rate1 or 0) > 0.2 or (rate2 or 0) > 0.2
    explanations.append({
        "explanation": "disagreement",
        "evidence": f"D4 contradiction rates: barrier_vs_reward_risk={rate1}, opportunity_vs_barrier={rate2}",
        "decisive": disagreement_high,
    })

    return explanations


def run_batch1(rows: int = None, registry_dir: str = None) -> dict:
    horizons = {}
    attribution = {}
    for h in HORIZONS:
        h_result = {
            "d1": run_d1(max_holding=h, rows=rows),
            "d2": run_d2(max_holding=h, rows=rows),
            "d3": run_d3(max_holding=h, rows=rows),
            "d4": run_d4(max_holding=h, rows=rows),
            "d5": run_d5(max_holding=h, rows=rows, registry_dir=registry_dir),
            "d6": run_d6(max_holding=h, rows=rows),
        }
        horizons[h] = h_result
        attribution[h] = apply_attribution_framework(h_result)
    return {"horizons": horizons, "attribution": attribution}


def _render_markdown(result: dict) -> str:
    lines = ["# GOLEX V3 Phase 5 Batch 1 — Diagnostic Foundation Report", ""]
    for h, h_result in result["horizons"].items():
        lines.append(f"## Horizon h={h}")
        lines.append("")
        lines.append(f"- D1 Direction point-biserial: {h_result['d1']['oos']['point_biserial']}")
        lines.append(f"- D2 base rate: {h_result['d2']['overall']}")
        lines.append(f"- D3 Opportunity point-biserial: {h_result['d3']['opportunity']['point_biserial']}, "
                      f"win_rate={h_result['d3']['opportunity']['win_rate']}")
        lines.append(f"- D4 contradiction rates: {h_result['d4']['contradiction_barrier_vs_reward_risk']}, "
                      f"{h_result['d4']['contradiction_opportunity_vs_barrier']}")
        lines.append(f"- D5 global calibration: {h_result['d5']['global']['calibration']}, "
                      f"traded_subset={'N/A' if isinstance(h_result['d5']['traded_subset'], str) else 'computed'}")
        lines.append(f"- D6: {h_result['d6']}")
        lines.append("")
        lines.append("### Attribution")
        for e in result["attribution"][h]:
            marker = "DECISIVE" if e["decisive"] else "not decisive"
            lines.append(f"- **{e['explanation']}** ({marker}): {e['evidence']}")
        lines.append("")
    return "\n".join(lines)


def write_report(result: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "batch1_report.json")
    md_path = os.path.join(out_dir, "batch1_report.md")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write(_render_markdown(result))
    return json_path, md_path


if __name__ == "__main__":
    result = run_batch1()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    json_path, md_path = write_report(result, out_dir)
    print(f"Batch 1 report written: {json_path}, {md_path}")
