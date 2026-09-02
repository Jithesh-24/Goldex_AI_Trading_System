"""Calibrates MAGNITUDE_LOG_CENTER / MAGNITUDE_LOG_SCALE in intelligence/fast_tier.py
against the real evidence registry's output distribution on TRAINING-partition
synthetic data only (never protected OOS -- see mandate Section on calibration).
Run manually; not part of the test suite. Prints the values to hand-copy into
fast_tier.py along with the sample statistics that justify them.
"""
import math
import numpy as np

from intelligence.evidence_sources import build_default_registry
from intelligence.fast_tier import FastTierReasoner, _evidence_scalar

def main():
    rng = np.random.default_rng(42)
    registry = build_default_registry()
    reasoner = FastTierReasoner(registry)
    # A single constant-sigma i.i.d. random walk makes GARCH conditional variance
    # converge to one flat steady-state value, starving MAGNITUDE_LOG_SCALE of real
    # spread (p10/p90 collapse near the median). Build the walk instead from
    # alternating quiet/turbulent sub-windows of differing noise scale so GARCH's
    # conditional variance genuinely varies across the run, like real gold-tick
    # data alternates between calm ranges and volatile breakouts. Segment length
    # 750 bars (8 segments over 6000 bars), cycling through 3 sigma regimes:
    # 0.03 (quiet), 0.15 (normal -- the prior single-regime value), 0.5 (turbulent).
    segment_len = 750
    sigmas = [0.03, 0.15, 0.5]
    steps = np.empty(6000)
    for i in range(0, 6000, segment_len):
        sigma = sigmas[(i // segment_len) % len(sigmas)]
        end = min(i + segment_len, 6000)
        steps[i:end] = rng.normal(0, sigma, end - i)
    closes = np.cumsum(steps) + 2000.0  # gold-tick-like noise scale, regime-varying
    magnitudes = []
    for bar in range(200, len(closes), 5):
        evidence = reasoner._compute_evidence(closes[:bar])
        sigma2 = _evidence_scalar(evidence, "garch_conditional_variance")
        velocity = _evidence_scalar(evidence, "kalman_filtered_velocity")
        if sigma2 is None and velocity is None:
            continue
        sigma2 = max(sigma2, 0.0) if sigma2 is not None else 0.0
        velocity = abs(velocity) if velocity is not None else 0.0
        magnitude = math.log1p(sigma2) + math.log1p(velocity)
        if magnitude > 0.0:
            magnitudes.append(magnitude)
    arr = np.array(magnitudes)
    median = float(np.median(arr))
    p10, p90 = float(np.percentile(arr, 10)), float(np.percentile(arr, 90))
    center = math.log(median)
    coverage_zspan = 3.0  # span the p10-p90 range over roughly a +/-1.5 sigmoid-input range
    scale = math.log(p90 / p10) / coverage_zspan if p90 > p10 else 1.0
    print(f"n={len(arr)} median={median:.5f} p10={p10:.5f} p90={p90:.5f}")
    print(f"MAGNITUDE_LOG_CENTER = math.log({median:.5f})  # = {center:.5f}")
    print(f"MAGNITUDE_LOG_SCALE = {scale:.5f}")

if __name__ == "__main__":
    main()
