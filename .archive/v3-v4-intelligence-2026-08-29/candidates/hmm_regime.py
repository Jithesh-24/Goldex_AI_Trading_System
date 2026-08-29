"""candidates/hmm_regime.py
Design doc Section 2.3: a real EM-fit 2-state Gaussian HMM over per-bar
returns, fit ONLY on SIMULATED_TRAINING experience via learn() -- the honest
version of the classical-quant regime research the V4 architecture document
flagged. Unlike Phase 2's RegimeConditionedCandidate (a percentile heuristic
needing no fitting), this candidate is genuinely non-functional until
trained -- decide()/manage() are NO_TRADE/HOLD before learn() is called."""
import numpy as np

from candidates.base import CandidateMetadata


def _gaussian_pdf(x, mean, var):
    var = max(var, 1e-10)
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


class HMMRegimeCandidate:
    def __init__(self, n_states: int = 2, max_em_iterations: int = 20):
        self.metadata = CandidateMetadata(
            candidate_id="hmm_regime", version="v1",
            description="Hand-written 2-state Gaussian HMM regime model, EM-fit on training data only.",
            mechanism_family="regime-generative",
        )
        self.n_states = n_states
        self.max_em_iterations = max_em_iterations
        self.is_trained = False
        self.means = None
        self.variances = None
        self.transition = None
        self.initial = None
        self._current_state_belief = None
        self._closes = []

    def learn(self, training_experience: list) -> None:
        mids = [
            r["market_state_snapshot"]["mid"] for r in training_experience
            if r.get("event_type") in ("DECIDE", "MANAGE") and r.get("market_state_snapshot", {}).get("mid") is not None
        ]
        if len(mids) < self.n_states * 5:
            return
        returns = np.diff(np.array(mids, dtype=np.float64))
        n_obs = len(returns)

        means = np.percentile(returns, np.linspace(10, 90, self.n_states))
        variances = np.full(self.n_states, np.var(returns) + 1e-8)
        transition = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        initial = np.full(self.n_states, 1.0 / self.n_states)

        for _ in range(self.max_em_iterations):
            emission = np.array([_gaussian_pdf(returns, means[s], variances[s]) for s in range(self.n_states)]).T
            emission = np.clip(emission, 1e-300, None)

            alpha = np.zeros((n_obs, self.n_states))
            alpha[0] = initial * emission[0]
            alpha[0] /= alpha[0].sum()
            for t in range(1, n_obs):
                alpha[t] = (alpha[t - 1] @ transition) * emission[t]
                alpha[t] /= alpha[t].sum() + 1e-300

            beta = np.zeros((n_obs, self.n_states))
            beta[-1] = 1.0
            for t in range(n_obs - 2, -1, -1):
                beta[t] = (transition @ (emission[t + 1] * beta[t + 1]))
                beta[t] /= beta[t].sum() + 1e-300

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            initial = gamma[0]
            for s in range(self.n_states):
                weight = gamma[:, s].sum() + 1e-300
                means[s] = (gamma[:, s] * returns).sum() / weight
                variances[s] = (gamma[:, s] * (returns - means[s]) ** 2).sum() / weight + 1e-8

            xi_sum = np.zeros((self.n_states, self.n_states))
            for t in range(n_obs - 1):
                xi = np.outer(alpha[t], beta[t + 1] * emission[t + 1]) * transition
                xi /= xi.sum() + 1e-300
                xi_sum += xi
            for s in range(self.n_states):
                denom = xi_sum[s].sum() + 1e-300
                transition[s] = xi_sum[s] / denom

        self.means, self.variances, self.transition, self.initial = means, variances, transition, initial
        self._current_state_belief = gamma[-1]
        self.is_trained = True

    def _update_belief(self, market_state):
        if market_state.completed_m1 is not None:
            self._closes.append(market_state.completed_m1.close)
            if len(self._closes) > 2:
                self._closes.pop(0)
        if not self.is_trained or len(self._closes) < 2:
            return None
        obs = self._closes[-1] - self._closes[-2]
        emission = np.array([_gaussian_pdf(obs, self.means[s], self.variances[s]) for s in range(self.n_states)])
        belief = (self._current_state_belief @ self.transition) * emission
        belief /= belief.sum() + 1e-300
        self._current_state_belief = belief
        high_vol_state = int(np.argmax(self.variances))
        regime = "HIGH_VOL" if np.argmax(belief) == high_vol_state else "LOW_VOL"
        momentum = self._closes[-1] - self._closes[0]
        return regime, momentum

    def decide(self, market_state, account):
        result = self._update_belief(market_state)
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
        result = self._update_belief(market_state)
        if result is None:
            return "HOLD"
        regime, _ = result
        return "EXIT" if regime != "HIGH_VOL" else "HOLD"
