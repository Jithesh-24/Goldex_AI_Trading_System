"""Task 12 -- shared invalid-price / spread-anomaly detection, used
identically by both the historical path (simulator/market_state_builder.py)
and the live path (market/state_engine.py) so replay and live trading agree
on what "invalid" and "anomalous" mean. Pure, stateless functions -- no
structural coupling to either caller, hence living here in contracts/
alongside the DataQuality enum they exist to help populate, rather than in
simulator/ or market/ where only one side would naturally own them."""
import math

# 5 std devs above the trailing spread_mean_60s/spread_std_60s is a
# conservative "this is not noise" bar; the x10-of-mean fallback covers the
# case where std is 0 or unavailable (e.g. too early in a run/replay for a
# window to have formed), scaled down from the task brief's own example
# ("spread suddenly 100x normal") to something that won't false-positive on
# ordinary widening.
SPREAD_ANOMALY_STD_MULT = 5.0
SPREAD_ANOMALY_MEAN_RATIO = 10.0

# HISTORICAL-PATH NOTE: simulator/market_state_builder.py's 60-second
# trailing window contains exactly one prior M1 bar at bar granularity, so
# the spread_std it passes here is structurally always 0.0 -- the 5-sigma
# branch below never fires on the historical path; only the 10x-mean-ratio
# fallback provides real spread-anomaly detection for replay. The live path
# (market/state_engine.py) has a genuine multi-tick std and both branches
# are live there.


def is_invalid_price(x) -> bool:
    """True for a zero/negative/NaN/inf/missing price."""
    return x is None or math.isnan(x) or math.isinf(x) or x <= 0


def is_anomalous_spread(spread, spread_mean, spread_std) -> bool:
    """True if `spread` is itself corrupted (negative/NaN/inf), or is far
    outside the trailing-window norm given by `spread_mean`/`spread_std`
    (both computed by the caller from real prior history, excluding the
    sample being judged). With no usable mean (None or <=0) there is no
    baseline to compare against, so only the corrupted-value case applies."""
    if spread is None or math.isnan(spread) or math.isinf(spread) or spread < 0:
        return True
    if spread_mean is None or spread_mean <= 0:
        return False
    if spread_std and spread_std > 0:
        return spread > spread_mean + SPREAD_ANOMALY_STD_MULT * spread_std
    return spread > spread_mean * SPREAD_ANOMALY_MEAN_RATIO
