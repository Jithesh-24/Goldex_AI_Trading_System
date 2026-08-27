"""candidates/sequence_history.py
Design doc Section 2.4: a minimal, honest test of "learning from the
candidate's own trading history" (principle #6) rather than only the market
-- combines a market momentum feature with a rolling window of the
candidate's OWN recent win/loss outcomes. No deep learning dependency: the
learned part is a 2-weight logistic model updated by plain gradient descent
in learn()."""
import math

from candidates.base import CandidateMetadata

MIN_HISTORY = 10


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class SequenceHistoryCandidate:
    def __init__(self, n_recent_trades: int = 5, learning_rate: float = 0.05):
        self.metadata = CandidateMetadata(
            candidate_id="sequence_history", version="v1",
            description="Logistic model over market momentum + own recent trade outcomes.",
            mechanism_family="sequence-history",
        )
        self.n_recent_trades = n_recent_trades
        self.learning_rate = learning_rate
        self.weights = {"momentum": 1.0, "recent_form": 1.0}
        self._closes = []
        self._recent_outcomes = [0.5] * n_recent_trades
        self._last_score_features = None

    def _features(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > MIN_HISTORY:
                self._closes.pop(0)
        if len(self._closes) < MIN_HISTORY:
            return None
        momentum = (self._closes[-1] - self._closes[0]) / self._closes[0]
        recent_form = sum(self._recent_outcomes) / len(self._recent_outcomes)
        return {"momentum": momentum, "recent_form": recent_form - 0.5}

    def _score(self, features):
        return _sigmoid(sum(self.weights[k] * v for k, v in features.items()))

    def decide(self, market_state, account):
        features = self._features(market_state)
        if features is None:
            return ("NO_TRADE", None, None)
        self._last_score_features = features
        score = self._score(features)
        if score > 0.55:
            return ("LONG", None, None)
        if score < 0.45:
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        features = self._features(market_state)
        if features is None:
            return "HOLD"
        score = self._score(features)
        if 0.45 <= score <= 0.55:
            return "EXIT"
        return "HOLD"

    def learn(self, training_experience: list) -> None:
        closed = [r for r in training_experience if r.get("event_type") == "POSITION_CLOSED"]
        for record in closed:
            won = 1.0 if float(record.get("realized_pnl") or 0.0) > 0 else 0.0
            self._recent_outcomes.append(won)
            if len(self._recent_outcomes) > self.n_recent_trades:
                self._recent_outcomes.pop(0)
            if self._last_score_features is None:
                continue
            prediction = self._score(self._last_score_features)
            error = prediction - won
            for key in self.weights:
                gradient = error * self._last_score_features.get(key, 0.0)
                self.weights[key] -= self.learning_rate * gradient
