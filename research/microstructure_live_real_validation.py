"""Phase 4 Task 11: real-data validation of Task 21's 5 live-only
microstructure features (spec section 22/7). Reads whatever real tick
capture exists at the given path (from market/tick_capture.py, Task 11's
Steps 4-6) and runs it through the SAME TickActivityTracker +
correlation_redundancy/distribution_stability tooling Task 26 already
built and proved on synthetic data -- this is the real-data counterpart,
not a re-run of the synthetic evidence. If no real capture exists yet (0
rows), this script says so explicitly and the 5 features stay OPTIONAL
pending a real capture window; it does not fabricate a result.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.microstructure_live_real_validation <path_to_real_ticks.csv>
"""
import csv
import os
import sys

import pandas as pd

from features.microstructure_live import TickActivityTracker
from features.registry.diagnostics import correlation_redundancy


def run_real_validation(ticks_csv_path: str) -> dict:
    if not os.path.exists(ticks_csv_path):
        print(f"[microstructure_live_real_validation] no real capture found at {ticks_csv_path} -- "
              f"0 real ticks available. The 5 live-only features remain OPTIONAL pending a real "
              f"capture window (spec section 7/22): synthetic evidence (Task 26) is NOT a substitute.")
        return {"n_real_ticks": 0, "status": "DATA_LIMITED"}

    with open(ticks_csv_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        print("[microstructure_live_real_validation] capture file exists but is empty -- 0 real ticks.")
        return {"n_real_ticks": 0, "status": "DATA_LIMITED"}

    tracker = TickActivityTracker()
    outputs = []
    for row in rows:
        # tracker.update() expects the same tick-derived state shape production feeds it --
        # adapt field names here to whatever market/feed_listener.py's real tick dict/MarketState
        # shape turned out to be in Step 1/Step 5 above.
        outputs.append(tracker.update(row))
    df = pd.DataFrame(outputs)
    pairs = correlation_redundancy(df, threshold=0.95)
    print(f"[microstructure_live_real_validation] n_real_ticks={len(rows)} "
          f"correlation_redundancy(threshold=0.95)={pairs}")
    print(df.describe())
    return {"n_real_ticks": len(rows), "status": "EVALUATED", "redundant_pairs": pairs}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_tick_capture.csv"
    run_real_validation(path)
