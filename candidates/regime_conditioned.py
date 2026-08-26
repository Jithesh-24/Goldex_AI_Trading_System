"""candidates/regime_conditioned.py
Tests classical regime-gating (design doc Section 2, Candidate D) without any
learned feature representation -- a volatility-percentile gate over a naive
momentum trigger, trading only in the HIGH_VOL regime."""
from candidates.base import CandidateMetadata


class RegimeConditionedCandidate:
    def __init__(self, vol_lookback_bars: int = 60, high_vol_percentile: float = 0.7):
        self.metadata = CandidateMetadata(
            candidate_id="regime_conditioned_momentum", version="v1",
            description="Volatility-regime-gated naive momentum rule, no ML.",
            mechanism_family="regime-statistical",
        )
        self.vol_lookback_bars = vol_lookback_bars
        self.high_vol_percentile = high_vol_percentile
        self._vols = []
        self._closes = []

    def _update_and_classify(self, market_state):
        if market_state.realized_vol_60s is not None:
            self._vols.append(market_state.realized_vol_60s)
            if len(self._vols) > self.vol_lookback_bars:
                self._vols.pop(0)
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > 3:
                self._closes.pop(0)
        if len(self._vols) < self.vol_lookback_bars or len(self._closes) < 3:
            return None
        sorted_vols = sorted(self._vols)
        rank = sum(1 for v in sorted_vols if v < self._vols[-1]) / len(sorted_vols)
        regime = "HIGH_VOL" if rank >= self.high_vol_percentile else "LOW_VOL"
        momentum = self._closes[-1] - self._closes[0]
        return regime, momentum

    def decide(self, market_state, account):
        result = self._update_and_classify(market_state)
        if result is None:
            return ("NO_TRADE", None, None)
        regime, momentum = result
        if regime != "HIGH_VOL":
            return ("NO_TRADE", None, None)
        if momentum > 0:
            return ("LONG", None, None)
        if momentum < 0:
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        result = self._update_and_classify(market_state)
        if result is None:
            return "HOLD"
        regime, _ = result
        if regime != "HIGH_VOL":
            return "EXIT"
        return "HOLD"
