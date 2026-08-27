"""candidates/bayesian_online.py
Design doc Section 2.2: principled uncertainty quantification via a
Beta-Bernoulli belief over whether a momentum signal precedes a winning
trade, updated only from SIMULATED_TRAINING experience via learn(). Tests
whether this beats a fixed-threshold rule (Phase 2's RegimeConditionedCandidate)
without any gradient-based learning."""
from candidates.base import CandidateMetadata

MIN_HISTORY = 10


class BayesianOnlineCandidate:
    def __init__(self, confidence_threshold: float = 0.65, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.metadata = CandidateMetadata(
            candidate_id="bayesian_online", version="v1",
            description="Beta-Bernoulli belief over momentum-signal win rate, updated via learn().",
            mechanism_family="bayesian-online",
        )
        self.confidence_threshold = confidence_threshold
        self.long_alpha, self.long_beta = prior_alpha, prior_beta
        self.short_alpha, self.short_beta = prior_alpha, prior_beta
        self._closes = []
        self._last_side = None
        self._open_sides = []

    def _momentum(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > MIN_HISTORY:
                self._closes.pop(0)
        if len(self._closes) < MIN_HISTORY:
            return None
        return self._closes[-1] - self._closes[0]

    def decide(self, market_state, account):
        momentum = self._momentum(market_state)
        if momentum is None:
            return ("NO_TRADE", None, None)
        long_belief = self.long_alpha / (self.long_alpha + self.long_beta)
        short_belief = self.short_alpha / (self.short_alpha + self.short_beta)
        if momentum > 0 and long_belief > self.confidence_threshold:
            self._last_side = "long"
            self._open_sides.append("long")
            return ("LONG", None, None)
        if momentum < 0 and short_belief > self.confidence_threshold:
            self._last_side = "short"
            self._open_sides.append("short")
            return ("SHORT", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        if self._last_side == "long":
            belief = self.long_alpha / (self.long_alpha + self.long_beta)
        elif self._last_side == "short":
            belief = self.short_alpha / (self.short_alpha + self.short_beta)
        else:
            return "HOLD"
        return "EXIT" if belief <= self.confidence_threshold else "HOLD"

    def learn(self, training_experience: list) -> None:
        for record in training_experience:
            if record.get("event_type") != "POSITION_CLOSED":
                continue
            if not self._open_sides:
                continue
            side = self._open_sides.pop(0)
            pnl = float(record.get("realized_pnl") or 0.0)
            won = pnl > 0
            if side == "long":
                if won:
                    self.long_alpha += 1
                else:
                    self.long_beta += 1
            elif side == "short":
                if won:
                    self.short_alpha += 1
                else:
                    self.short_beta += 1
