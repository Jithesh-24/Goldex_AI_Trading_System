"""candidates/controls.py
Mandatory sanity-floor candidates (design doc Section 6). These are NEVER
ranked as real competing intelligence -- research/phase2_tournament.py uses
their evidence profiles only as a validity gate on the harness itself. If
RandomCandidate shows meaningful persistent profitability after realistic
costs, that means the simulator/harness has a bug, not that random trading
works."""
import random

from candidates.base import CandidateMetadata


class NoTradeCandidate:
    def __init__(self):
        self.metadata = CandidateMetadata(
            candidate_id="control_no_trade", version="v1",
            description="Always NO_TRADE -- sanity floor.", mechanism_family="control",
        )

    def decide(self, market_state, account):
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"


class RandomCandidate:
    def __init__(self, seed: int = 0):
        self.metadata = CandidateMetadata(
            candidate_id="control_random", version="v1",
            description="Uniform-random actions -- sanity floor.", mechanism_family="control",
        )
        self._rng = random.Random(seed)

    def decide(self, market_state, account):
        action = self._rng.choice(["NO_TRADE", "LONG", "SHORT"])
        return (action, None, None)

    def manage(self, market_state, position_view, account):
        return self._rng.choice(["HOLD", "EXIT"])
