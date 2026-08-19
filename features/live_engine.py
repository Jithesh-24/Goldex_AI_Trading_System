"""Trigger-driven live feature engine on top of MarketState (spec section
6). M1_CLOSE-triggered families recompute via the SAME compute_<family>
functions replay_engine.py uses, against a bounded window pulled from
StateEngine.completed_m1_window() -- O(window), not O(1), not
O(history), matching Phase 2's own established performance bar.
TICK-triggered features use small dedicated ring buffers
(TickActivityTracker) or StatefulKalman. Additive only: never called from
app/engine.py's decision loop.

Composition below mirrors features/replay_engine.py's build_candidate_
features EXACTLY (Task 16's proven wiring) rather than reinventing it: the
`upstream` dict handed to compute_regime_state is the full jump_detection
+ distribution_info + microstructure_history dicts merged together
(**c, **d, **h), not a hand-picked subset of keys. compute_regime_state
requires jump_detection's underscore-prefixed internal key
(_bars_since_last_changepoint_internal) which only exists in the raw `c`
dict -- picking a couple of named keys back out of an already-merged,
already-"_"-stripped `merged` dict (as an earlier draft of this file did)
silently drops that key and breaks regime_state's jump_state/
changepoint_state outputs. See task-22-report.md."""
import numpy as np
import pandas as pd

from features._shared import build_shared_inputs
from features.features import build_tier1_features
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
from features.microstructure_live import TickActivityTracker
from features.daily_buffer import DailyBuffer
from features.registry import load_all

# StatefulKalman (Task 19) is NOT used here: kalman_residual_z etc. are
# baseline_v1 features (features/features.py, unmodified), out of
# live_engine.py's scope -- they're already computed live via
# app/engine.py's existing build_features() call, separately. Persistence
# family's residual_mean_reversion_60 gets kalman_residual_z from the
# bounded-window batch build_tier1_features() call below (O(window),
# same as every other M1_CLOSE family), which is already fast enough at
# window<=480 -- no O(1) path is needed for that. StatefulKalman remains
# a tested, standalone utility (Task 19) for a future consumer that
# genuinely needs per-tick O(1) kalman state; nothing in this plan calls
# it yet, so it is not instantiated here.

CUSUM_K_LIVE = 2.5  # verified identical to features.replay_engine.CUSUM_K (2026-08-19)


class LiveFeatureEngine:
    def __init__(self, state_engine, daily_bootstrap_csv: str = None, daily_buffer_size: int = 252):
        self.state_engine = state_engine
        self.tick_tracker = TickActivityTracker()
        self.daily_buffer = DailyBuffer(size=daily_buffer_size)
        if daily_bootstrap_csv:
            self.daily_buffer.bootstrap_from_csv(daily_bootstrap_csv, value_cols=["close", "spread"])
        self._descriptors = {d.feature_id: d for d in load_all()}
        self._last_tick_snapshot: dict = {}
        self._last_m1_snapshot: dict = {}

    def on_tick(self, state) -> dict:
        live_vals = self.tick_tracker.update(state)
        out = {}
        for feature_id, value in live_vals.items():
            quality = "WARMING_UP" if value is None else "VALID"
            out[feature_id] = (value, quality)
        self._last_tick_snapshot = out
        return {**self._last_tick_snapshot, **self._last_m1_snapshot}

    def on_m1_close(self, bars: list) -> dict:
        if len(bars) < 2:
            # build_tier1_features/build_shared_inputs need at least 2 rows
            # (np.diff/np.roll-based ret1 construction); genuinely nothing
            # useful to compute yet, stay WARMING_UP via last snapshot.
            return {**self._last_tick_snapshot, **self._last_m1_snapshot}

        df = pd.DataFrame({
            "time": [b.start_time for b in bars],
            "open": [b.open for b in bars], "high": [b.high for b in bars],
            "low": [b.low for b in bars], "close": [b.close for b in bars],
            "tick_volume": [b.tick_count for b in bars],
            "spread": [np.nan] * len(bars),  # not tracked per-bar live -- spread-history
                                              # features stay UNAVAILABLE via live_compatible=False below
        })
        base_feat = build_tier1_features(df)
        shared = build_shared_inputs(df, base_feat)

        # Exact replay_engine.py composition (Task 16), not a reinvention:
        a = compute_returns_dynamics(shared)
        b = compute_volatility_dynamics(shared)
        c = compute_jump_detection(shared, CUSUM_K_LIVE)
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

        # DailyBuffer wiring: compute_volatility_dynamics's own internal
        # daily resample (inside compute_fn above) only sees the bounded
        # ~8h window, so its vol_percentile_252 is structurally always NaN
        # live -- that's honest (WARMING_UP forever) but wastes the
        # DailyBuffer built in Task 20. Override with the real daily-
        # buffer-backed value once enough days have accumulated. Keyed as
        # "ewma_vol" (not "close") since that's what's actually recorded --
        # the brief's sketch named this key "close" while storing
        # ewma_vol, which would collide with a real close-price series
        # bootstrapped under the same key name.
        today = bars[-1].start_time.date()
        if len(shared.ewma_vol) and not np.isnan(shared.ewma_vol[-1]):
            self.daily_buffer.record(today, {"ewma_vol": float(shared.ewma_vol[-1])})
        vol_hist = self.daily_buffer.series("ewma_vol")
        if len(vol_hist) >= 60 and "vol_percentile_252" in merged:
            vp = np.asarray(merged["vol_percentile_252"])
            merged["vol_percentile_252"] = np.append(vp[:-1], vol_hist.rank(pct=True).iloc[-1])

        out = {}
        for feature_id, values in merged.items():
            descriptor = self._descriptors.get(feature_id)
            if descriptor is not None and not descriptor.live_compatible:
                out[feature_id] = (None, "UNAVAILABLE")
                continue
            last_val = values[-1] if hasattr(values, "__len__") and len(values) else None
            if last_val is None or (isinstance(last_val, float) and last_val != last_val):  # NaN check
                out[feature_id] = (None, "WARMING_UP")
            else:
                out[feature_id] = (float(last_val), "VALID")
        self._last_m1_snapshot = out
        return {**self._last_tick_snapshot, **self._last_m1_snapshot}
