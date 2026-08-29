"""Wraps the 9 validated Phase 3A/Phase 4 representation functions as
EvidenceSourceSpec instances registered in a single EvidenceRegistry.

Each wrapper is deliberately simple: no caching, no periodic-refit logic
(that is Task 5's job). Every call re-invokes the underlying batch function
on the full `closes_so_far` array it is given and returns the LAST finite
value as the EvidenceValue. If there isn't enough history for the source's
own window/lookback, the wrapper returns EvidenceValue(None, 0.0, name)
rather than raising.

Because every wrapper recomputes fresh from `closes_so_far` alone (never
touching indices beyond what it's handed), these sources are causal by
construction: computing on closes_so_far[:i] always agrees with computing
on the full array and then reading index i-1 -- see
tests/intelligence/test_evidence_sources.py::test_no_look_ahead_*.
"""
from __future__ import annotations

import numpy as np

from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue
from research.phase3a_representation_experiments import (
    momentum_scalar,
    path_pca_projection,
    multiscale_vol_summary,
    vol_regime_transition,
    MOMENTUM_LOOKBACK,
    PATH_WINDOW,
    VOL_WINDOWS,
)
from research.phase4_garch_volatility_mechanism import fit_garch11
from research.phase4_kalman_trend_mechanism import kalman_level_trend_filter
from research.phase4_distributional_mechanism import _rolling_moment, WINDOW as DIST_WINDOW


def _last_finite(arr: np.ndarray) -> "float | None":
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return None
    return float(finite[-1])


def _make_momentum_scalar_compute(lookback: int = MOMENTUM_LOOKBACK):
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "momentum_scalar"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < lookback + 1:
            return EvidenceValue(None, 0.0, name)
        mom = momentum_scalar(closes, lookback=lookback)
        value = _last_finite(mom)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_path_pca_projection_compute(window: int = PATH_WINDOW):
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "path_pca_projection"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < window + 1:
            return EvidenceValue(None, 0.0, name)
        proj = path_pca_projection(closes, window=window)
        value = _last_finite(proj)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_multiscale_vol_ratio_compute(windows=VOL_WINDOWS):
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "multiscale_vol_ratio"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < max(windows) + 1:
            return EvidenceValue(None, 0.0, name)
        ratio, _vols = multiscale_vol_summary(closes, windows=windows)
        value = _last_finite(ratio)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_vol_regime_transition_compute(windows=VOL_WINDOWS, n_bins: int = 3):
    """Chained: calls multiscale_vol_summary first to get vols_dict, then
    feeds vols_dict[short_window] (NOT raw closes) into vol_regime_transition,
    exactly matching how research/phase3a_representation_experiments.py:main()
    composes these two functions."""

    short_w = min(windows)

    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "vol_regime_transition"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        # vol_regime_transition additionally needs n_bins * 5 finite vol
        # points, so require some cushion beyond the raw window minimum.
        if len(closes) < max(windows) + n_bins * 5:
            return EvidenceValue(None, 0.0, name)
        _ratio, vols = multiscale_vol_summary(closes, windows=windows)
        transition = vol_regime_transition(vols[short_w], n_bins=n_bins)
        value = _last_finite(transition)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_garch_conditional_variance_compute():
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "garch_conditional_variance"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        # GARCH(1,1) needs a reasonable number of return observations to
        # produce a meaningful fit; require at least 30 returns.
        if len(closes) < 31:
            return EvidenceValue(None, 0.0, name)
        returns = np.diff(closes, prepend=closes[0])
        (_omega, _alpha, _beta), sigma2 = fit_garch11(returns)
        value = _last_finite(sigma2)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_kalman_velocity_compute():
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "kalman_filtered_velocity"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < 2:
            return EvidenceValue(None, 0.0, name)
        _levels, velocities, _innovations = kalman_level_trend_filter(closes)
        value = _last_finite(velocities)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_kalman_innovation_compute():
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "kalman_innovation"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < 2:
            return EvidenceValue(None, 0.0, name)
        _levels, _velocities, innovations = kalman_level_trend_filter(closes)
        value = _last_finite(innovations)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_rolling_skew_compute(window: int = DIST_WINDOW):
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "rolling_skew"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < window + 1:
            return EvidenceValue(None, 0.0, name)
        returns = np.diff(closes, prepend=closes[0])
        skew = _rolling_moment(returns, window, order=3)
        value = _last_finite(skew)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def _make_rolling_excess_kurtosis_compute(window: int = DIST_WINDOW):
    def compute(closes_so_far: np.ndarray) -> EvidenceValue:
        name = "rolling_excess_kurtosis"
        closes = np.asarray(closes_so_far, dtype=np.float64)
        if len(closes) < window + 1:
            return EvidenceValue(None, 0.0, name)
        returns = np.diff(closes, prepend=closes[0])
        kurt = _rolling_moment(returns, window, order=4)
        value = _last_finite(kurt)
        if value is None:
            return EvidenceValue(None, 0.0, name)
        return EvidenceValue(value, 1.0, name)

    return compute


def build_default_registry() -> EvidenceRegistry:
    """Builds and returns an EvidenceRegistry with all 9 validated
    representation functions registered as EvidenceSourceSpecs.

    DIRECTIONALITY CLASSIFICATION (`is_directional`, see
    intelligence/evidence.py for why the field exists):

      Directional (sign == expected price direction):
        - momentum_scalar          close[t] - close[t-lookback]: sign IS
                                   "price went up / down".
        - path_pca_projection      slope of the normalized price path: sign
                                   IS "path sloping up / down".
        - kalman_filtered_velocity filtered trend velocity: sign IS "trend
                                   up / down".
        - kalman_innovation        observation minus one-step-ahead
                                   prediction: sign IS "price surprised to
                                   the upside / downside".

      NOT directional (sign means something else, or cannot be negative):
        - multiscale_vol_ratio        std/std, non-negative BY CONSTRUCTION
                                      -- can only ever vote "up".
        - garch_conditional_variance  a variance, non-negative BY
                                      CONSTRUCTION -- same permanent-long
                                      failure mode.
        - vol_regime_transition       signed, but the sign means "volatility
                                      regime rose / fell", not "price rose /
                                      fell" (see vol_regime_transition in
                                      research/phase3a_representation_experiments.py:
                                      it differences a quantile-binned
                                      volatility series).
        - rolling_skew                signed, but the sign means "returns
                                      distribution is right/left tailed" --
                                      a shape statistic, not a price call.
        - rolling_excess_kurtosis     signed, but the sign means "fatter /
                                      thinner tails than Gaussian".

    The non-directional five are still computed, still applicability-gated,
    and still feed `context_bucket()` (GARCH variance and Kalman velocity
    magnitude are exactly what it conditions on) -- they simply never cast a
    directional vote.
    """
    registry = EvidenceRegistry()

    registry.register(EvidenceSourceSpec(
        name="momentum_scalar",
        mathematical_formulation="momentum[t] = close[t] - close[t - lookback], lookback=10 (Phase-3-style single momentum scalar).",
        required_inputs=["closes"],
        assumptions="Requires at least lookback+1 closes. Uses only closes up to and including the current index -- no look-ahead.",
        known_failure_conditions="Insufficient history (fewer than lookback+1 points) yields no finite value.",
        compute=_make_momentum_scalar_compute(),
        is_directional=True,
        computational_cost_hint=(
            "Cheap: O(n) vectorized array subtraction. ~20us at 1,000 closes (Task 12 per-source measurement) -- effectively free next to the recursive sources."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="path_pca_projection",
        mathematical_formulation=(
            "Collapse a raw normalized price-path window (last `window` closes, "
            "mean/first-value normalized) to a single scalar: the slope of a "
            "simple linear fit over the normalized window (a first "
            "principal-component-like projection)."
        ),
        required_inputs=["closes"],
        assumptions=(
            "Requires at least window+1 closes (window=15). This deliberately "
            "loses information relative to using the full window in a real "
            "model -- it exists to give an apples-to-apples single-scalar "
            "comparison against other representations."
        ),
        known_failure_conditions="Insufficient history yields no finite value; degenerate windows (w[0]==0) fall back to an un-normalized slope.",
        compute=_make_path_pca_projection_compute(),
        is_directional=True,
        computational_cost_hint=(
            "Moderate: O(n * window) Python-loop linear fit per bar. ~45ms at 1,000 closes (Task 12) -- the most expensive of the non-recursive sources."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="multiscale_vol_ratio",
        mathematical_formulation=(
            "Ratio of shortest-window realized vol to longest-window realized "
            "vol across windows=(10,30,100): ratio[t] = std(returns[t-10:t]) / "
            "std(returns[t-100:t]) -- a 'vol-of-vol regime' summary scalar."
        ),
        required_inputs=["closes"],
        assumptions="Requires at least max(windows)+1 closes. Only past returns are used for each window's rolling std -- no look-ahead.",
        known_failure_conditions="Insufficient history, or a zero/near-zero long-window vol causing a non-finite ratio.",
        compute=_make_multiscale_vol_ratio_compute(),
        is_directional=False,
        computational_cost_hint=(
            "Moderate: O(n * max(window)) rolling std over 3 windows. ~35ms at 1,000 closes (Task 12)."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="vol_regime_transition",
        mathematical_formulation=(
            "Chained with multiscale_vol_summary: first computes the "
            "short-window (10-bar) realized-vol series via multiscale_vol_summary, "
            "then discretizes it into n_bins=3 regimes by quantile and returns "
            "the bin-to-bin transition (current regime minus previous regime) "
            "as a single scalar representing volatility-state transition."
        ),
        required_inputs=["closes"],
        assumptions=(
            "Requires at least max(windows)+1 closes AND at least n_bins*5 finite "
            "short-window vol points to form quantile bins. Depends on "
            "vols_dict[short_window] from multiscale_vol_summary, not raw closes -- "
            "the two functions must be called in sequence, which this wrapper does "
            "internally."
        ),
        known_failure_conditions="Insufficient history for either the vol windows or the minimum bin population yields no finite value.",
        compute=_make_vol_regime_transition_compute(),
        is_directional=False,
        computational_cost_hint=(
            "Moderate: dominated by the multiscale_vol_summary call it chains onto, plus an O(n log n) quantile. ~35ms at 1,000 closes (Task 12)."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="garch_conditional_variance",
        mathematical_formulation=(
            "GARCH(1,1) one-step-ahead conditional variance forecast: "
            "sigma2[t] = omega + alpha*eps[t-1]^2 + beta*sigma2[t-1], fit by "
            "maximizing the Gaussian log-likelihood over (omega, alpha, beta) "
            "on demeaned returns. Uses only returns up to and including t-1 -- "
            "no look-ahead."
        ),
        required_inputs=["closes"],
        assumptions=(
            "This is a from-scratch, textbook GARCH(1,1) implementation (the "
            "`arch` library is not installed in this venv) fit via scipy's "
            "Nelder-Mead when available, or a coordinate-descent fallback "
            "otherwise -- not sourced from a battle-tested library, which is a "
            "known limitation. Wrapper refits fresh on every call (no refit-interval "
            "caching in this task -- Task 5's job)."
        ),
        known_failure_conditions=(
            "Insufficient history (fewer than 31 closes). A diverged fit "
            "(alpha+beta >= 1, or non-positive omega/alpha/beta) falls back to "
            "a stable stationary initial guess rather than an invalid estimate."
        ),
        compute=_make_garch_conditional_variance_compute(),
        is_directional=False,
        computational_cost_hint=(
            "EXPENSIVE: O(n) full-history likelihood refit per call, and the single dominant cost in the registry -- ~260ms of compute_all's ~350ms mean at 1,000 closes (Task 12). Cached by FastTierReasoner every refit_interval=50 bars for exactly this reason."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="kalman_filtered_velocity",
        mathematical_formulation=(
            "Constant-velocity Kalman filter over price closes, state = "
            "[level, velocity], observation = close price. Returns the "
            "filtered velocity v[t], a trend estimate. A proper forward filter "
            "-- x[t] is the posterior after observing close[t], derived only "
            "from close[0..t] -- no smoother/backward pass, no look-ahead."
        ),
        required_inputs=["closes"],
        assumptions=(
            "Uses the function's own default process/observation variance "
            "parameters (process_var_level=1e-4, process_var_velocity=1e-6, "
            "obs_var=1.0). Requires at least 2 closes. Wrapper recomputes fresh "
            "on every call (no refit caching in this task)."
        ),
        known_failure_conditions="Insufficient history (fewer than 2 closes).",
        compute=_make_kalman_velocity_compute(),
        is_directional=True,
        computational_cost_hint=(
            "EXPENSIVE (recursive): O(n) full-history forward filter per call, ~19ms at 1,000 closes (Task 12). Cached by FastTierReasoner every refit_interval=50 bars. Cheaper than GARCH but grows linearly with history, unlike the windowed sources."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="kalman_innovation",
        mathematical_formulation=(
            "Same constant-velocity Kalman filter as kalman_filtered_velocity; "
            "returns the innovation (observation minus one-step-ahead "
            "prediction) at the current index -- a 'surprise' signal."
        ),
        required_inputs=["closes"],
        assumptions=(
            "Uses the function's own default process/observation variance "
            "parameters. Requires at least 2 closes. Wrapper recomputes fresh "
            "on every call (no refit caching in this task)."
        ),
        known_failure_conditions="Insufficient history (fewer than 2 closes).",
        compute=_make_kalman_innovation_compute(),
        is_directional=True,
        computational_cost_hint=(
            "EXPENSIVE (recursive): re-runs the same O(n) Kalman filter as kalman_filtered_velocity, ~19ms at 1,000 closes (Task 12). Cached by FastTierReasoner every refit_interval=50 bars."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="rolling_skew",
        mathematical_formulation=(
            "Rolling standardized 3rd moment (skewness) of the trailing "
            "WINDOW=30 bar-to-bar returns, ending strictly before the current "
            "index: rolling_stat[t] computed from returns[t-window:t], never "
            "including returns[t] -- no look-ahead."
        ),
        required_inputs=["closes"],
        assumptions="Requires at least window+1 closes (window=30). A zero-std trailing window yields a defined 0.0 rather than a division error.",
        known_failure_conditions="Insufficient history yields no finite value.",
        compute=_make_rolling_skew_compute(),
        is_directional=False,
        computational_cost_hint=(
            "Moderate: O(n * window) rolling 3rd moment, ~23ms at 1,000 closes (Task 12)."
        ),
    ))

    registry.register(EvidenceSourceSpec(
        name="rolling_excess_kurtosis",
        mathematical_formulation=(
            "Rolling standardized 4th moment minus 3 (excess kurtosis) of the "
            "trailing WINDOW=30 bar-to-bar returns, ending strictly before the "
            "current index -- same no-look-ahead trailing-window construction "
            "as rolling_skew."
        ),
        required_inputs=["closes"],
        assumptions="Requires at least window+1 closes (window=30). A zero-std trailing window yields a defined 0.0 rather than a division error.",
        known_failure_conditions="Insufficient history yields no finite value.",
        compute=_make_rolling_excess_kurtosis_compute(),
        is_directional=False,
        computational_cost_hint=(
            "Moderate: O(n * window) rolling 4th moment, ~23ms at 1,000 closes (Task 12)."
        ),
    ))

    return registry
