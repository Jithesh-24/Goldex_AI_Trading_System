"""python3 tests/test_specialist_inference_performance.py -- [SYNTHETIC-ROWS,
REAL-MODEL] benchmark of single-row inference latency for each Phase 4
specialist trained on a capped dry-run dataset. Two-pass pattern (timing,
then separate tracemalloc pass) matches Phase 2/3's test_performance.py /
test_feature_performance.py."""
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from research.phase4_dataset import assemble_v3_dataset
from features.labeling import TripleBarrierConfig, triple_barrier_labels
from catboost import CatBoostClassifier

ROWS = 20000
N_INFER = 200


def _train_small_direction_model():
    ds = assemble_v3_dataset(max_holding=45, rows=ROWS)
    cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=45, min_vol=1e-6)
    labels = triple_barrier_labels(ds["close"], ds["high"], ds["low"], ds["t0_idx"], ds["vol_tb"], cfg, side=None)
    nz = labels["label"].to_numpy() != 0
    cols = ds["baseline_cols"] + ds["useful_cols"]
    X = ds["feat_v3"].loc[labels.index.to_numpy()[nz], cols].reset_index(drop=True)
    y = (labels["label"].to_numpy()[nz] == 1).astype(int)
    model = CatBoostClassifier(depth=4, iterations=200, learning_rate=0.05, verbose=False, random_seed=42)
    model.fit(X, y)
    return model, X


def test_direction_candidate_single_row_inference_latency():
    model, X = _train_small_direction_model()
    row = X.iloc[[0]]
    latencies_us = []
    for _ in range(N_INFER):
        t0 = time.perf_counter()
        model.predict_proba(row)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(latencies_us)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"[direction_v3_candidate] single-row predict_proba latency over {N_INFER} calls: "
          f"p50={p50:.0f}us p95={p95:.0f}us p99={p99:.0f}us")
    assert p99 < 50_000, f"single-row inference p99={p99:.0f}us exceeds 50ms budget"


def test_direction_candidate_memory():
    model, X = _train_small_direction_model()
    row = X.iloc[[0]]
    tracemalloc.start()
    for _ in range(20):
        model.predict_proba(row)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[direction_v3_candidate] peak traced memory over 20 inference calls: {peak / 1024:.1f}KB")


if __name__ == "__main__":
    test_direction_candidate_single_row_inference_latency()
    test_direction_candidate_memory()
    print("tests/test_specialist_inference_performance.py: OK")
