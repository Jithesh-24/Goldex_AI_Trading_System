"""research/phase5_direction_barrier_investigation.py
Spec section 6: measures whether Direction's calibrated probability carries
independent information about Barrier's realized p_tp once side is fixed,
or whether the two are redundant (Direction only useful for side-selection).
Method: bucket events into Direction-probability deciles (restricted to the
side Direction actually favored), then compare each decile's realized
Barrier win rate against the overall win rate. A flat relationship across
deciles = redundant (Direction adds nothing once Barrier is known). A
monotonic/structured relationship = Direction carries independent signal,
and a correction term should be derived (left as a documented follow-up,
not implemented speculatively here -- this script's job is measurement,
not correction-fitting).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_direction_barrier_investigation
"""
import json
import os

import numpy as np

from research.phase5_calibration import _oof_for_direction, _oof_for_barrier

BASE = os.path.dirname(os.path.abspath(__file__))


def investigate_direction_barrier_relationship(max_holding: int, rows: int = None) -> dict:
    # Aligned to a shared t0_nz base index (per research/phase5_calibration.py's
    # FIX-2 convention) rather than the old independent-length `[:n]` slicing,
    # which paired unrelated events across the two OOF streams.
    t0_dir, _, p_dir_full, m_dir = _oof_for_direction(max_holding, rows=rows)
    t0_bar, y_bar_full, p_bar_full, m_bar = _oof_for_barrier(max_holding, rows=rows)
    assert t0_dir.shape == t0_bar.shape and (t0_dir == t0_bar).all(), "OOF base index mismatch"
    combined = m_dir & m_bar
    n = int(combined.sum())
    if n == 0:
        return {"n_events": 0, "decile_table": [], "correction_needed": False,
                "correction_note": "Insufficient data for investigation", "overall_win_rate": 0.0}
    p_dir, y_bar = p_dir_full[combined], y_bar_full[combined]

    deciles = np.digitize(p_dir, np.percentile(p_dir, np.arange(10, 100, 10)))
    decile_table = []
    overall_rate = float(y_bar.mean())
    for d in sorted(set(deciles.tolist())):
        mask = deciles == d
        if mask.sum() < 20:
            continue
        decile_table.append({"decile": int(d), "n": int(mask.sum()),
                              "win_rate": float(y_bar[mask].mean())})

    rates = [row["win_rate"] for row in decile_table]
    spread = max(rates) - min(rates) if rates else 0.0
    correction_needed = spread > 0.05  # documented threshold: >5pp spread across deciles = non-trivial structure
    note = (f"Decile win-rate spread={spread:.4f} vs overall={overall_rate:.4f}. "
            + ("Structure found -- Direction probability appears to carry information "
               "about Barrier's realized win rate beyond side-selection; a correction "
               "term should be derived and OOS-validated in a follow-up task before "
               "folding into EV_side."
               if correction_needed else
               "No material structure found -- Direction and Barrier are effectively "
               "redundant once side is fixed; Direction used for side-selection only, "
               "per the original approach."))

    result = {"n_events": int(n), "decile_table": decile_table,
              "correction_needed": correction_needed, "correction_note": note,
              "overall_win_rate": overall_rate}
    with open(os.path.join(BASE, f"phase5_direction_barrier_report_h{max_holding}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    from research.phase4_dataset import HORIZONS
    for h in HORIZONS:
        r = investigate_direction_barrier_relationship(h)
        print(f"h={h}: n={r['n_events']} correction_needed={r['correction_needed']}")
        print(f"  {r['correction_note']}")
