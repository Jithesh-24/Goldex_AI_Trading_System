"""Phase 4, Role G: Execution/signal decay. No real human-execution-latency
data exists anywhere in this repo (verified: Telegram delivery in
app/engine.py is one-way, fire-and-forget, no fill/ack channel is ever
logged) -- per spec section 2G/27, this is built as real infrastructure
with an explicit DATA_LIMITED status, not fabricated. What CAN be computed
honestly from bars alone: a post-signal price-drift proxy -- the adverse
move from the signal bar's close at fixed delays, which bounds how much a
manually-executed trade's entry could already have decayed by the time a
human acts, independent of how long that human actually took.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_execution_decay
"""
import os

import numpy as np
import pandas as pd

from learning.data import load_raw_m1
from features.features import build_tier1_features
from features.labeling import cusum_filter
from contracts.model_registry import ModelRegistryEntry, ModelLineage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")
CUSUM_K = 2.5
DELAYS_BARS = {"30s": 0, "60s": 1, "120s": 2}  # M1 resolution: 30s rounds to same-bar close


def run_execution_decay_proxy(rows: int = None, registry_dir: str = None) -> dict:
    if registry_dir is None:
        registry_dir = REGISTRY_DIR
    df = load_raw_m1()
    if rows:
        df = df.tail(rows).reset_index(drop=True)
    base_feat = build_tier1_features(df)
    close = df["close"].to_numpy(dtype=np.float64)
    vol = base_feat["ewma_vol"].to_numpy(dtype=np.float64)
    vol_filled = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[np.isfinite(vol)]))
    threshold = np.clip(CUSUM_K * vol_filled * close, 1e-6, None)
    event_mask = cusum_filter(close, threshold)
    t0_idx = np.where(event_mask)[0]
    t0_idx = t0_idx[t0_idx < len(close) - max(DELAYS_BARS.values()) - 1]

    drift_by_delay = {}
    for label, bars in DELAYS_BARS.items():
        p0 = close[t0_idx]
        p_delay = close[t0_idx + bars]
        drift = (p_delay - p0) / p0
        drift_by_delay[label] = {"mean_abs_drift": float(np.mean(np.abs(drift))),
                                  "std_drift": float(np.std(drift)), "n": int(len(drift))}

    entry = ModelRegistryEntry(
        model_id="execution_decay_v3_stub", family="execution_decay", algorithm="none_data_limited",
        artifact_path="registry/execution_decay_v3_stub.json",
        target_definition=(
            "TRUE target (human-execution-latency-conditioned adverse move) is DATA_LIMITED: no "
            "execution fill/ack timestamps exist anywhere in this repo (Telegram delivery is "
            "one-way, fire-and-forget). PROXY target reported in metrics: post-signal price drift "
            "from the CUSUM event bar's close at fixed delays {30s, 60s, 120s}, at M1 (60s) bar "
            "resolution -- a market-data-only descriptive statistic, not a prediction of any "
            "specific human's execution latency."
        ),
        created_at=pd.Timestamp.utcnow().isoformat(),
        status="candidate",
        metrics={"data_limited": True, "n_events": int(len(t0_idx)), "drift_by_delay": drift_by_delay,
                 "reason": "no real execution/fill timestamp data exists; see Task 11 for the "
                           "real-tick-capture infra this could eventually be fed from"},
        lineage=ModelLineage(data_snapshot="data/gold_seed_merged_full6yr.csv"),
    )
    os.makedirs(registry_dir, exist_ok=True)
    with open(os.path.join(registry_dir, f"{entry.model_id}.json"), "w") as f:
        f.write(entry.model_dump_json(indent=2))

    print(f"[execution_decay] DATA_LIMITED -- n_events={len(t0_idx):,} drift_by_delay={drift_by_delay}")
    return {"data_limited": True, "n_events": int(len(t0_idx)), "drift_by_delay": drift_by_delay}


if __name__ == "__main__":
    run_execution_decay_proxy()
