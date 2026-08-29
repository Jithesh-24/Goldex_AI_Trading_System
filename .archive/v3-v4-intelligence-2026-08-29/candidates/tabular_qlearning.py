"""candidates/tabular_qlearning.py
Design doc Section 2.1: the smallest genuinely sequential RL-family method
that can be honestly tried without the sample-inefficiency that ruled out
full RL in the original V4 architecture research. State space is
deliberately small (a handful of discretized bins) to keep visitation
counts meaningful given this project's single 6.7-year data window --
Section 4's research step checks whether even this small a state space
gets enough visitation to mean anything."""
import random
import statistics

from candidates.base import CandidateMetadata


class TabularQLearningCandidate:
    def __init__(self, n_vol_bins: int = 3, n_momentum_bins: int = 3, learning_rate: float = 0.1,
                 discount: float = 0.9, exploration_epsilon: float = 0.1, seed: int = 0):
        self.metadata = CandidateMetadata(
            candidate_id="tabular_qlearning", version="v1",
            description="Discretized-state tabular Q-learning agent.", mechanism_family="tabular-rl",
        )
        self.n_vol_bins = n_vol_bins
        self.n_momentum_bins = n_momentum_bins
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_epsilon = exploration_epsilon
        self._rng = random.Random(seed)
        self.q_table = {}
        self._vols = []
        self._closes = []
        self._in_position = False
        self._open_state_actions = []

    def _bin(self, value, edges):
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)

    def _current_state(self, market_state):
        if market_state.realized_vol_60s is not None:
            self._vols.append(market_state.realized_vol_60s)
            if len(self._vols) > 60:
                self._vols.pop(0)
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > 5:
                self._closes.pop(0)
        if len(self._vols) < 5 or len(self._closes) < 5:
            return None
        vol_edges = [statistics.median(self._vols) * 0.8, statistics.median(self._vols) * 1.2]
        vol_bin = self._bin(self._vols[-1], vol_edges)
        momentum = self._closes[-1] - self._closes[0]
        momentum_edges = [-1e-6, 1e-6]
        momentum_bin = self._bin(momentum, momentum_edges)
        return (vol_bin, momentum_bin, self._in_position)

    def _q_values(self, state, actions):
        if state not in self.q_table:
            self.q_table[state] = {}
        for a in actions:
            if a not in self.q_table[state]:
                self.q_table[state][a] = 0.0
        return self.q_table[state]

    def _epsilon_greedy(self, state, actions):
        if self._rng.random() < self.exploration_epsilon:
            return self._rng.choice(actions)
        q = self._q_values(state, actions)
        return max(actions, key=lambda a: q[a])

    def decide(self, market_state, account):
        state = self._current_state(market_state)
        if state is None:
            return ("NO_TRADE", None, None)
        actions = ["NO_TRADE", "LONG", "SHORT"]
        action = self._epsilon_greedy(state, actions)
        if action in ("LONG", "SHORT"):
            self._in_position = True
            self._open_state_actions.append((state, action))
        return (action, None, None)

    def manage(self, market_state, position_view, account):
        state = self._current_state(market_state)
        if state is None:
            return "HOLD"
        actions = ["HOLD", "EXIT"]
        action = self._epsilon_greedy(state, actions)
        if action == "EXIT":
            self._in_position = False
        return action

    def learn(self, training_experience: list) -> None:
        closed = [r for r in training_experience if r.get("event_type") == "POSITION_CLOSED"]
        for record in closed:
            if not self._open_state_actions:
                continue
            state, action = self._open_state_actions.pop(0)
            reward = float(record.get("realized_pnl") or 0.0)
            q = self._q_values(state, ["NO_TRADE", "LONG", "SHORT"])
            best_next = max(q.values()) if q else 0.0
            q[action] += self.learning_rate * (reward + self.discount * best_next - q[action])
