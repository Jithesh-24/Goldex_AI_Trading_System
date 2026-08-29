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

# Explicit sentinel context bucket for "no usable context evidence at all"
# (both the GARCH-variance and the Kalman-velocity readings are missing,
# non-finite, or applicability-gated to zero confidence). Deliberately
# OUTSIDE [0, N_CONTEXT_BUCKETS - 1] so it can never collide with a genuine
# reading, however low: pooling "we know nothing about the current regime"
# with "we measured a genuinely quiet regime" would train one Beta posterior
# on two categorically different situations. ToolTrust keys on
# (source_name, context_bucket) tuples, so a negative bucket is a perfectly
# valid, permanently distinct key.
GATED_OUT_CONTEXT_BUCKET = -1

# Re-centering constants for the context magnitude (see context_bucket()).
# The raw magnitude log1p(sigma2) + log1p(|velocity|) is non-negative by
# construction, so squashing it directly through a logistic gives a value
# that is ALWAYS >= 0.5 -- structurally stranding the lower half of the
# bucket range (with N_CONTEXT_BUCKETS=5, buckets 0 and 1 were literally
# unreachable, leaving only 3 live buckets instead of 5). The fix is to
# squash the LOG of that magnitude, centered on a typical observed value,
# so the sigmoid input is genuinely two-sided.
#
# MAGNITUDE_LOG_CENTER: log of a typical mid-regime magnitude. Measured
# empirically over synthetic random-walk closes at this repo's XAUUSD price
# scale (median raw magnitude ~0.15 across sampled decision points).
# MAGNITUDE_LOG_SCALE: how many e-folds of magnitude span the bucket range.
# At 1.0, roughly a factor-of-e change in magnitude moves one bucket over
# the middle of the range, so magnitudes from ~0.02 (bucket 0) to ~1.0
# (bucket 4) are all reachable -- a realistic quiet-to-turbulent span.
# Both are documented round numbers, not fit to profitable data.
MAGNITUDE_LOG_CENTER = math.log(0.15)
MAGNITUDE_LOG_SCALE = 1.0


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


def _evidence_scalar(evidence: dict[str, EvidenceValue], name: str) -> Optional[float]:
    """The usable scalar for `name`, or None if there is no usable reading
    at all (missing, non-finite, or applicability-gated to zero confidence).
    None is distinct from 0.0 on purpose -- see GATED_OUT_CONTEXT_BUCKET."""
    ev = evidence.get(name)
    if ev is None or ev.value is None or not math.isfinite(ev.value):
        return None
    # A source Task 4's applicability gate has zeroed out (confidence <= 0,
    # e.g. insufficient history or invalid MarketState) must not steer
    # bucket selection -- it's not trustworthy evidence of anything,
    # including "how turbulent the market is right now."
    if ev.confidence <= 0.0:
        return None
    return float(ev.value)


def context_bucket(evidence: dict[str, EvidenceValue]) -> int:
    """Derives a discrete context bucket in [0, N_CONTEXT_BUCKETS - 1] from a
    continuous scalar built out of the GARCH conditional-variance and Kalman
    velocity evidence values -- never a named regime category. The scalar is
    `log1p(GARCH sigma^2) + log1p(|Kalman velocity|)`: a monotonically
    increasing "how turbulent/fast-moving is the market right now" magnitude
    that combines a volatility-scale term and a trend-speed-scale term on
    comparable (log) footing.

    That magnitude is then squashed through a logistic sigmoid in LOG space,
    centered on MAGNITUDE_LOG_CENTER and scaled by MAGNITUDE_LOG_SCALE,
    before being multiplied into N_CONTEXT_BUCKETS equal-width bins. The log
    re-centering matters: the raw magnitude is non-negative by construction,
    so squashing it directly gave sigmoid >= 0.5 ALWAYS and made buckets 0
    and 1 structurally unreachable (only 3 of 5 buckets were ever live).
    Centering in log space makes the sigmoid input genuinely two-sided, so
    the full bucket range is reachable over realistic vol/velocity ranges.

    If BOTH context readings are unusable (missing, non-finite, or
    applicability-gated to zero confidence), this returns the explicit
    GATED_OUT_CONTEXT_BUCKET sentinel rather than any in-range bucket -- "we
    have no idea what regime this is" must not pool with a genuine
    low-magnitude reading. If only ONE reading is usable, the other
    contributes 0.0 to the magnitude and a real bucket is still returned:
    that is a genuine (partial) measurement, not an absence of one. Never
    raises.
    """
    sigma2_raw = _evidence_scalar(evidence, "garch_conditional_variance")
    velocity_raw = _evidence_scalar(evidence, "kalman_filtered_velocity")

    if sigma2_raw is None and velocity_raw is None:
        return GATED_OUT_CONTEXT_BUCKET

    sigma2 = max(sigma2_raw, 0.0) if sigma2_raw is not None else 0.0
    velocity = abs(velocity_raw) if velocity_raw is not None else 0.0

    magnitude = math.log1p(sigma2) + math.log1p(velocity)
    if magnitude <= 0.0:
        # Both terms are exactly zero -- a real, measured, maximally quiet
        # reading. Floor it into the lowest bucket rather than taking a log
        # of zero.
        return 0
    z = (math.log(magnitude) - MAGNITUDE_LOG_CENTER) / MAGNITUDE_LOG_SCALE
    scaled = 1.0 / (1.0 + math.exp(-z))  # logistic squash into (0, 1)

    bucket = int(scaled * N_CONTEXT_BUCKETS)
    return min(max(bucket, 0), N_CONTEXT_BUCKETS - 1)


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
        # NOTE on load_bearing_floor: a source's contribution magnitude is
        # trust.posterior_mean(...) * ev.confidence. An unseen (source,
        # bucket) pair starts at the Beta(1,1) prior mean of 0.5, and
        # ev.confidence is typically 1.0 once applicability passes -- so
        # essentially every applicable source clears a 0.05 floor by an
        # order of magnitude on its very first observation, before any
        # trust has actually been learned. At this default,
        # `load_bearing_sources` is effectively "every applicable source,"
        # not a meaningfully discriminating subset -- it does NOT yet
        # distinguish a well-trusted source from a brand-new, untested one.
        # A more meaningful floor (e.g. one set relative to the
        # distribution of observed weights, or requiring a minimum number
        # of prior observations before a source can be load-bearing at all)
        # needs real data to calibrate and is explicitly deferred to
        # Task 12's measurement pass / a later tuning task. Task 7's thesis
        # memory should treat `load_bearing_sources` as "applicable and
        # non-trivially weighted," not "proven reliable," until that
        # tuning happens.
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

        specs = self.registry.specs()
        for name, ev in gated_evidence.items():
            if ev.value is None or not math.isfinite(ev.value) or ev.confidence <= 0.0:
                continue
            # C1 fix: only sources whose VALUE SIGN encodes a price
            # direction may cast a directional vote. A variance or a
            # variance ratio is non-negative by construction and would
            # otherwise cast a permanent, unfalsifiable LONG vote that no
            # amount of trust learning could ever cancel (a Beta posterior
            # mean is strictly in (0, 1), never 0, never negative); a skew
            # or kurtosis reading is signed but its sign means "tail shape",
            # not "price up/down". Non-directional sources are still
            # computed and still applicability-gated above, and still feed
            # context_bucket() (which conditions on exactly the GARCH
            # variance and Kalman velocity magnitude) -- they simply take no
            # part in net_directional_belief.
            #
            # They are consequently also excluded from load_bearing_sources:
            # that list exists to drive credit assignment
            # (intelligence/credit_assignment.py), whose whole semantics --
            # and ToolTrust's own documented semantics above -- is "did THIS
            # source's directional call agree with the realized outcome."
            # That question is undefined for a source that never made a
            # directional call, so crediting one either way would be
            # fabricated evidence.
            spec = specs.get(name)
            if spec is None or not spec.is_directional:
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

        # Disagreement term: twice the fraction of total weight pointing the
        # "wrong" way relative to the net sign. Since net_sign is defined by
        # whichever side holds the majority of weight, opposing_weight can
        # never exceed half of total_weight by construction -- so the raw
        # ratio opposing/total is capped at 0.5 and only reaches 1.0 exactly
        # at a 50/50 tie, an artificial discontinuity (a 49/51 split would
        # read as ~0.49, then a 50/50 split would jump straight to 1.0).
        # Doubling it makes the term continuous and reach 1.0 exactly at the
        # tie: 0.0 when every source agrees, up to 1.0 as the split
        # approaches even -- this is what prevents contradictory sources
        # from being silently averaged into a falsely confident midpoint.
        net_sign = math.copysign(1.0, net_belief) if net_belief != 0.0 else 0.0
        if net_sign == 0.0:
            disagreement = 1.0
        else:
            opposing_weight = sum(
                c[2] for c in contributions
                if math.copysign(1.0, c[0]) != net_sign and c[0] != 0.0
            )
            disagreement = min(1.0, 2.0 * opposing_weight / total_weight)

        aggregate_uncertainty = min(1.0, weighted_trust_unc + disagreement)

        return Hypothesis(net_belief, aggregate_uncertainty, load_bearing)
