"""The Bayesian adaptive-trust mechanism at the core of the GOLDEX Fast Tier
(mandate Section 1: explicitly not a static per-source weighted average).

For each `EvidenceSource`, `ToolTrust` maintains a Beta-distributed posterior
belief (`scipy.stats.beta`) over "this source's directional signal agreed
with the eventual realized outcome," conditioned on a small number of
continuous context buckets derived from the recursive state-space sources
themselves (GARCH conditional variance, Kalman velocity) -- never a
hardcoded regime label (mandate Section 3). `context_bucket()` bins a
continuous scalar built from those two state-space outputs; there is no
if/elif chain classifying "trend" vs "range" vs "breakout" anywhere in this
module.

`FastTierReasoner` owns the refit-cadence caching for the expensive
recursive sources (GARCH, Kalman) that Task 3's wrappers deliberately do
NOT cache (they always compute fresh from whatever array they're given).
This reasoner calls the O(n) GARCH/Kalman wrappers only every
`refit_interval` bars, reusing the last computed EvidenceValue between
refits; cheap sources are recomputed every call as normal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist

from contracts.market_state import MarketState
from intelligence.applicability import apply_applicability
from intelligence.evidence import EvidenceRegistry, EvidenceValue

# The three registered source names (see intelligence/evidence_sources.py::
# build_default_registry) backed by an O(n) recursive fit -- GARCH(1,1) and
# the constant-velocity Kalman filter (velocity and innovation share one
# underlying filter run). These are this task's refit-caching targets per
# the controller ruling; every other registered source is cheap and is
# recomputed fresh on every call, same as Task 3's wrappers already do.
EXPENSIVE_SOURCE_NAMES = frozenset({
    "garch_conditional_variance",
    "kalman_filtered_velocity",
    "kalman_innovation",
})

# Small, fixed number of continuous-valued context buckets. Not regime
# labels -- just a discretization resolution for the binned continuous
# GARCH-variance / Kalman-velocity magnitude scalar computed below.
N_CONTEXT_BUCKETS = 5


class ToolTrust:
    """Per-(source_name, context_bucket) Beta(alpha, beta) posterior over
    "this source agreed with the realized outcome in this context." Every
    unseen (source_name, context_bucket) pair starts from the uninformative
    Beta(1, 1) prior (uniform on [0, 1]) -- deliberately no informative
    prior favoring or penalizing any source before it has been observed.
    """

    def __init__(self) -> None:
        self._params: dict[tuple[str, int], list[float]] = {}

    def _get(self, source_name: str, context_bucket: int) -> list[float]:
        key = (source_name, context_bucket)
        if key not in self._params:
            self._params[key] = [1.0, 1.0]  # Beta(1, 1) uninformative prior
        return self._params[key]

    def update(self, source_name: str, context_bucket: int, agreed: bool) -> None:
        alpha_beta = self._get(source_name, context_bucket)
        if agreed:
            alpha_beta[0] += 1.0
        else:
            alpha_beta[1] += 1.0

    def posterior_mean(self, source_name: str, context_bucket: int) -> float:
        a, b = self._get(source_name, context_bucket)
        return float(a / (a + b))

    def posterior_uncertainty(self, source_name: str, context_bucket: int) -> float:
        """The Beta posterior's variance (not std) -- chosen because
        variance is what combines additively/linearly in the weighted
        aggregate-uncertainty combination `FastTierReasoner.hypothesis`
        performs below, avoiding a sqrt-then-resquare round trip."""
        a, b = self._get(source_name, context_bucket)
        return float(beta_dist(a, b).var())


def _evidence_scalar(evidence: dict[str, EvidenceValue], name: str) -> float:
    ev = evidence.get(name)
    if ev is None or ev.value is None or not math.isfinite(ev.value):
        return 0.0
    return float(ev.value)


def context_bucket(evidence: dict[str, EvidenceValue]) -> int:
    """Derives a discrete context bucket in [0, N_CONTEXT_BUCKETS - 1] from a
    continuous scalar built out of the GARCH conditional-variance and Kalman
    velocity evidence values -- never a named regime category. The scalar is
    `log1p(GARCH sigma^2) + log1p(|Kalman velocity|)`: a monotonically
    increasing "how turbulent/fast-moving is the market right now" magnitude
    that combines a volatility-scale term and a trend-speed-scale term on
    comparable (log) footing. That magnitude is squashed through a logistic
    sigmoid into (0, 1) (so no fixed scale needs to be guessed -- the
    binning is scale-free) and multiplied into N_CONTEXT_BUCKETS equal-width
    bins. Missing/inapplicable evidence contributes 0.0 to the magnitude
    (sigmoid(0) = 0.5 -> a mid bucket), never raises.
    """
    sigma2 = max(_evidence_scalar(evidence, "garch_conditional_variance"), 0.0)
    velocity = abs(_evidence_scalar(evidence, "kalman_filtered_velocity"))

    magnitude = math.log1p(sigma2) + math.log1p(velocity)
    scaled = 1.0 / (1.0 + math.exp(-magnitude))  # logistic squash into (0, 1)

    bucket = int(scaled * N_CONTEXT_BUCKETS)
    return min(bucket, N_CONTEXT_BUCKETS - 1)


@dataclass
class Hypothesis:
    """The Fast Tier's directional output for one decision point."""

    net_directional_belief: float  # signed, roughly in [-1, 1]: net bullish/bearish belief
    aggregate_uncertainty: float  # roughly in [0, 1]: how much to distrust net_directional_belief
    load_bearing_sources: list[tuple[str, int, float]] = field(default_factory=list)
    # (source_name, context_bucket, signed contribution) for every source whose
    # applicability-gated, trust-weighted contribution exceeded the reasoner's
    # load-bearing floor -- feeds Task 7's thesis memory.


class FastTierReasoner:
    """Combines the evidence registry, Task 4's applicability gate, and a
    ToolTrust posterior into a per-decision Hypothesis, while caching the
    expensive GARCH/Kalman sources between refits rather than recomputing
    them on every single bar.
    """

    def __init__(
        self,
        registry: EvidenceRegistry,
        refit_interval: int = 50,
        load_bearing_floor: float = 0.05,
    ) -> None:
        self.registry = registry
        self.refit_interval = refit_interval
        self.load_bearing_floor = load_bearing_floor
        # {source_name: (bar_index_at_last_refit, EvidenceValue)} -- only
        # ever populated for EXPENSIVE_SOURCE_NAMES.
        self._cache: dict[str, tuple[int, EvidenceValue]] = {}

    def _compute_evidence(self, closes_so_far: np.ndarray) -> dict[str, EvidenceValue]:
        bar = len(closes_so_far)
        results: dict[str, EvidenceValue] = {}
        for name, spec in self.registry.specs().items():
            if name in EXPENSIVE_SOURCE_NAMES:
                cached = self._cache.get(name)
                if cached is not None and (bar - cached[0]) < self.refit_interval:
                    results[name] = cached[1]
                    continue
                try:
                    value = spec.compute(closes_so_far)
                except Exception:
                    value = EvidenceValue(None, 0.0, name)
                self._cache[name] = (bar, value)
                results[name] = value
            else:
                try:
                    results[name] = spec.compute(closes_so_far)
                except Exception:
                    results[name] = EvidenceValue(None, 0.0, name)
        return results

    def hypothesis(
        self,
        closes_so_far: np.ndarray,
        market_state: Optional[MarketState],
        trust: ToolTrust,
    ) -> Hypothesis:
        raw_evidence = self._compute_evidence(closes_so_far)

        gated_evidence = {
            name: apply_applicability(name, ev, closes_so_far, market_state)
            for name, ev in raw_evidence.items()
        }

        bucket = context_bucket(gated_evidence)

        contributions: list[tuple[float, float, float]] = []  # (signed_contribution, uncertainty, weight)
        load_bearing: list[tuple[str, int, float]] = []

        for name, ev in gated_evidence.items():
            if ev.value is None or not math.isfinite(ev.value) or ev.confidence <= 0.0:
                continue
            trust_mean = trust.posterior_mean(name, bucket)
            trust_unc = trust.posterior_uncertainty(name, bucket)
            weight = trust_mean * ev.confidence
            if weight <= 0.0:
                continue
            direction = math.copysign(1.0, ev.value) if ev.value != 0.0 else 0.0
            contribution = direction * weight
            contributions.append((contribution, trust_unc, weight))
            if abs(contribution) >= self.load_bearing_floor:
                load_bearing.append((name, bucket, contribution))

        if not contributions:
            # Genuine abstention: no applicable, weight-bearing evidence at
            # all -- net belief flat at 0.0 and uncertainty pinned at its
            # maximum so a downstream consumer reads this as NO_TRADE.
            return Hypothesis(0.0, 1.0, [])

        total_weight = sum(c[2] for c in contributions)
        if total_weight <= 0.0:
            return Hypothesis(0.0, 1.0, [])

        net_belief = sum(c[0] for c in contributions) / total_weight
        weighted_trust_unc = sum(c[1] * c[2] for c in contributions) / total_weight

        # Disagreement term: fraction of weight pointing the "wrong" way
        # relative to the net sign. 0.0 when every contributing source
        # agrees on direction; approaches 1.0 as opposing weight balances
        # out the net -- this is what prevents contradictory sources from
        # being silently averaged into a falsely confident midpoint.
        net_sign = math.copysign(1.0, net_belief) if net_belief != 0.0 else 0.0
        if net_sign == 0.0:
            disagreement = 1.0
        else:
            opposing_weight = sum(
                c[2] for c in contributions
                if math.copysign(1.0, c[0]) != net_sign and c[0] != 0.0
            )
            disagreement = opposing_weight / total_weight

        aggregate_uncertainty = min(1.0, weighted_trust_unc + disagreement)

        return Hypothesis(net_belief, aggregate_uncertainty, load_bearing)
