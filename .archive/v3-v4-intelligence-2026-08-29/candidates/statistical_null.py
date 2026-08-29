"""candidates/statistical_null.py
A transparent, no-ML statistical rule (design doc Section 2, Candidate B) --
gives "no real edge" a legitimate, cheap null hypothesis to compare every
other candidate against."""
import statistics

from candidates.base import CandidateMetadata


class MomentumMeanReversionCandidate:
    def __init__(self, lookback_bars: int = 20, z_threshold: float = 1.5):
        self.metadata = CandidateMetadata(
            candidate_id="statistical_null_mean_reversion", version="v1",
            description="Volatility-normalized mean-reversion z-score rule, no ML.",
            mechanism_family="rule-based",
        )
        self.lookback_bars = lookback_bars
        self.z_threshold = z_threshold
        self._closes = []

    def _record_and_zscore(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > self.lookback_bars:
                self._closes.pop(0)
        if len(self._closes) < self.lookback_bars:
            return None
        mean = statistics.mean(self._closes)
        stdev = statistics.pstdev(self._closes)
        if stdev <= 0:
            return None
        return (market_state.mid - mean) / stdev

    def decide(self, market_state, account):
        z = self._record_and_zscore(market_state)
        if z is None:
            return ("NO_TRADE", None, None)
        if z <= -self.z_threshold:
            return ("LONG", None, None)
        if z >= self.z_threshold:
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        z = self._record_and_zscore(market_state)
        if z is None:
            return "HOLD"
        if abs(z) <= self.z_threshold / 2.0:
            return "EXIT"
        return "HOLD"
