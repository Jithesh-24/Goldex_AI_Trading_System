"""tests/test_ev_engine_performance.py
Mirrors tests/test_specialist_inference_performance.py's two-pass
(timing, then separate memory) pattern (spec section 30)."""
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contracts.specialist_output import DirectionOutput, OpportunityOutput, BarrierOutput, MAEOutput, MFEOutput
from decision.ev_engine import evaluate

N_CALLS = 200


class _FakeMarketState:
    def __init__(self):
        self.spread = 0.01
        self.market_timestamp = datetime.now(timezone.utc)
        self.realized_vol_60s = 0.0006
        self.mid = 2350.0


def _inputs():
    ms = _FakeMarketState()
    direction = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                                 model_status="VALIDATED", probability_long=0.6,
                                 probability_short=0.4, calibrated=True)
    opportunity = OpportunityOutput(model_id="opportunity_meta_v3_candidate_h15", horizon=15,
                                     model_status="VALIDATED", probability_take=0.55, calibrated=True)
    barrier = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                             model_status="VALIDATED", p_tp=0.55, calibrated=True)
    mae = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.3, q75=0.5, q90=0.8)
    mfe = MFEOutput(model_id="mfe_quantile_v3_candidate_h15", horizon=15, model_status="VALIDATED",
                     q50=0.5, q75=0.9, q90=1.4)
    return ms, direction, opportunity, barrier, mae, mfe


def test_ev_engine_single_decision_latency():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    latencies_us = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(latencies_us)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"[ev_engine] single-decision latency over {N_CALLS} calls: p50={p50:.0f}us p95={p95:.0f}us p99={p99:.0f}us")
    assert p99 < 50_000, f"single-decision p99={p99:.0f}us exceeds 50ms budget"


def test_ev_engine_memory():
    ms, direction, opportunity, barrier, mae, mfe = _inputs()
    tracemalloc.start()
    for _ in range(20):
        evaluate(ms, direction, opportunity, barrier, 0.5, mae, mfe, timeout_r=0.1, timeout_r_provisional_proxy=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[ev_engine] peak traced memory over 20 calls: {peak / 1024:.1f}KB")


if __name__ == "__main__":
    test_ev_engine_single_decision_latency()
    test_ev_engine_memory()
    print("tests/test_ev_engine_performance.py: OK")
