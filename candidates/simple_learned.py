"""candidates/simple_learned.py
A simple learned linear model over a raw-OHLC feature family deliberately
DIFFERENT from V3's 125 hand-engineered features (design doc Section 2,
Candidate C) -- tests whether hand-engineering is actually necessary,
without introducing a new deep-learning dependency (design doc Section 6's
explicit caution). weights are fit OFFLINE by a separate training script;
this module is inference-only."""
import math

from candidates.base import CandidateMetadata

MIN_HISTORY = 20


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SimpleLearnedCandidate:
    def __init__(self, weights: dict, threshold: float = 0.5):
        self.metadata = CandidateMetadata(
            candidate_id="simple_learned_linear", version="v1",
            description="Logistic-style linear model over raw OHLC-derived features.",
            mechanism_family="learned-linear",
        )
        self.weights = weights
        self.threshold = threshold
        self._closes = []

    def _features(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > MIN_HISTORY:
                self._closes.pop(0)
        if len(self._closes) < MIN_HISTORY:
            return None
        short_window = self._closes[-5:]
        medium_window = self._closes
        short_return = (short_window[-1] - short_window[0]) / short_window[0]
        medium_return = (medium_window[-1] - medium_window[0]) / medium_window[0]
        diffs = [medium_window[i] - medium_window[i - 1] for i in range(1, len(medium_window))]
        gains = sum(d for d in diffs if d > 0)
        losses = sum(-d for d in diffs if d < 0)
        rsi_like = gains / (gains + losses) if (gains + losses) > 0 else 0.5
        return {"short_return": short_return, "medium_return": medium_return, "rsi_like": rsi_like}

    def _score(self, market_state):
        features = self._features(market_state)
        if features is None:
            return None
        raw = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        return _sigmoid(raw)

    def decide(self, market_state, account):
        score = self._score(market_state)
        if score is None:
            return ("NO_TRADE", None, None)
        if score > self.threshold:
            return ("LONG", None, None)
        if score < (1.0 - self.threshold):
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        score = self._score(market_state)
        if score is None:
            return "HOLD"
        if position_view is not None and position_view.side.name == "LONG" and score <= 0.5:
            return "EXIT"
        if position_view is not None and position_view.side.name == "SHORT" and score >= 0.5:
            return "EXIT"
        return "HOLD"
