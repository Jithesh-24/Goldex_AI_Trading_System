"""Batch/replay feature engine -- composes every family's compute_<family>
function into one DataFrame, replacing research/features_v3.py's former
build_candidate_features (spec section 4/6). Used for historical dataset
building (Phase 4) and as the reference implementation live_engine.py's
per-M1-close recompute is checked against (Task 23)."""
import pandas as pd

from features._shared import build_shared_inputs
from features.returns_dynamics import compute_returns_dynamics
from features.volatility_dynamics import compute_volatility_dynamics
from features.jump_detection import compute_jump_detection
from features.distribution_info import compute_distribution_info
from features.market_geometry import compute_market_geometry
from features.persistence import compute_persistence
from features.temporal import compute_temporal
from features.microstructure_history import compute_microstructure_history
from features.regime_state import compute_regime_state
from features.first_passage import compute_first_passage

CUSUM_K = 2.5  # confirmed via learning/train.py:74 (Task 7's cusum_k parameter)


def build_candidate_features(df: pd.DataFrame, base_feat: pd.DataFrame, cusum_k: float = CUSUM_K) -> pd.DataFrame:
    shared = build_shared_inputs(df, base_feat)
    a = compute_returns_dynamics(shared)
    b = compute_volatility_dynamics(shared)
    c = compute_jump_detection(shared, cusum_k)
    d = compute_distribution_info(shared)
    e = compute_market_geometry(shared)
    fam_f = compute_persistence(shared)
    g = compute_temporal(shared)
    h = compute_microstructure_history(shared)
    upstream = {**c, **d, **h}
    i = compute_regime_state(shared, upstream)
    j = compute_first_passage(shared)

    merged = {}
    for fam in (a, b, c, d, e, fam_f, g, h, i, j):
        for k, v in fam.items():
            if k.startswith("_"):
                continue
            merged[k] = v
    out = pd.DataFrame(merged, index=df.index)
    out.insert(0, "time", df["time"].to_numpy())
    return out
