"""
Phase 3B -- build the full v3 candidate dataset (v2's 26 base features +
92 new candidates = 118 total) at the SAME CUSUM events v2 uses (same
CUSUM_K, same base feature set -- spread/tick_volume excluded from the
predictive matrix per the Phase 2 data-semantics fix), so the "current vs
expanded" comparison (Part C) is apples-to-apples.

Memory note: candidate features are computed over the full 2.4M-bar series
(needed for correct rolling/causal context) but only the ~313k CUSUM-event
ROWS are kept afterward -- the full-length matrix is dropped immediately.
Candidate-feature NaNs (e.g. jump_magnitude_mean_60 when no jump occurred
in the window) are LEFT IN, not dropped/imputed -- CatBoost/LightGBM/
XGBoost all natively treat NaN as a legitimate "missing" split value, and
for several of these features NaN is itself informative (no jump = 0 jumps
observed, a real state, not garbage). Only the base 26 (already NaN-free
past warmup by construction) gate row inclusion, matching v2 exactly.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.build_v3_dataset
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd

from learning.train import assemble_dataset, label_events
from research.features_v3 import build_candidate_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
EXCLUDE = frozenset({"spread", "tick_volume"})


def main():
    t_start = time.time()
    print("== base (v2, 26-feature) dataset assembly ==")
    feat, close, high, low, vol_tb, t0_idx, base_cols = assemble_dataset(exclude=EXCLUDE)
    print(f"base_cols ({len(base_cols)}): {base_cols}")

    print("\n== raw M1 reload for candidate feature construction ==")
    from learning.data import load_raw_m1
    raw = load_raw_m1()
    cand = build_candidate_features(raw, feat)
    cand_cols = [c for c in cand.columns if c != "time"]
    print(f"{len(cand_cols)} candidate columns built")
    del raw
    gc.collect()

    print("\n== labeling (same triple-barrier direction target as v2) ==")
    X_base, y_bin, t0, t1, t0_nz = label_events(close, high, low, vol_tb, t0_idx, base_cols, feat)

    # candidate values at the SAME t0_nz rows, downcast to float32 to bound memory
    X_cand = cand.loc[t0_nz, cand_cols].reset_index(drop=True).astype(np.float32)
    del cand
    gc.collect()

    X_full = pd.concat([X_base.astype(np.float32), X_cand], axis=1)
    all_cols = base_cols + cand_cols
    print(f"final event matrix: {X_full.shape} ({len(all_cols)} total features, "
          f"{len(t0_nz):,} directional events)")

    out_dir = os.path.join(OUT, "v3_dataset")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_v3.npy"), X_full.to_numpy(dtype=np.float32))
    np.save(os.path.join(out_dir, "y_bin.npy"), y_bin.to_numpy())
    np.save(os.path.join(out_dir, "t0.npy"), t0.to_numpy())
    np.save(os.path.join(out_dir, "t1.npy"), t1.to_numpy())
    np.save(os.path.join(out_dir, "t0_nz.npy"), t0_nz)
    np.save(os.path.join(out_dir, "close.npy"), close)
    np.save(os.path.join(out_dir, "high.npy"), high)
    np.save(os.path.join(out_dir, "low.npy"), low)
    np.save(os.path.join(out_dir, "vol_tb.npy"), vol_tb)
    with open(os.path.join(out_dir, "columns.json"), "w") as f:
        json.dump({"base_cols": base_cols, "cand_cols": cand_cols, "all_cols": all_cols}, f, indent=2)

    print(f"\nsaved -> {out_dir} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
