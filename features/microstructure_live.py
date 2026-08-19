"""New live-only microstructure family (spec section 7) -- exists only
because Phase 2 provides a real tick-level bid/ask stream; no 6.7-year
historical analogue exists or can exist. Implemented now, evaluated
against real targets in Phase 4 (registry status=OPTIONAL, no
evidence_ref). Same small-ring-buffer pattern as
market/state_engine.py's _tick_times_60s/_spreads."""
from collections import deque

TICK_WINDOW_SEC = 60.0


class TickActivityTracker:
    def __init__(self):
        self._spreads = deque()
        self._times = deque()
        self._last_spread = None

    def update(self, state) -> dict:
        ts = state.market_timestamp.timestamp()
        spread = state.spread

        spread_change = None if self._last_spread is None else spread - self._last_spread
        self._last_spread = spread

        self._times.append(ts)
        self._spreads.append(spread)
        while self._times and ts - self._times[0] > TICK_WINDOW_SEC:
            self._times.popleft()
            self._spreads.popleft()

        spread_shock = None
        if len(self._spreads) > 1:
            mean = sum(self._spreads) / len(self._spreads)
            var = sum((x - mean) ** 2 for x in self._spreads) / len(self._spreads)
            std = var ** 0.5
            spread_shock = (spread - mean) / std if std > 1e-12 else 0.0

        interarrivals = [t2 - t1 for t1, t2 in zip(self._times, list(self._times)[1:])]
        interarrival_mean = sum(interarrivals) / len(interarrivals) if interarrivals else None
        interarrival_std = None
        burstiness = None
        if len(interarrivals) > 1:
            m = interarrival_mean
            var = sum((x - m) ** 2 for x in interarrivals) / len(interarrivals)
            interarrival_std = var ** 0.5
            burstiness = (interarrival_std - m) / (interarrival_std + m) if (interarrival_std + m) > 1e-12 else 0.0

        return {
            "spread_change_live": spread_change,
            "spread_shock_zscore_live": spread_shock,
            "tick_interarrival_mean_60s": interarrival_mean,
            "tick_interarrival_std_60s": interarrival_std,
            "tick_arrival_burstiness_60s": burstiness,
        }
