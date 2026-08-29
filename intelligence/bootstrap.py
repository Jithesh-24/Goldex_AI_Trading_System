"""intelligence/bootstrap.py -- Task 11: analytical SL/TP/sizing bootstrap.

EXPLICITLY NOT a learned SL/TP or sizing head. Both the mandate and the
architecture spec defer a learned sizing/exit head to a later phase, once
entry/exit trust has been validated (Section L reasoning). This module is
pure, documented, hand-fit arithmetic -- a volatility-scaled SL/TP distance
and a thin pass-through wrapper around Phase 1's existing risk-fraction
sizing mechanism. No statistics/ML beyond that.

These are REAL implementations of the two callables
`intelligence.decision_engine.FastTierDecisionEngine` already expects,
constructor-injected as `sltp_bootstrap`/`sizing_bootstrap` (Task 6). Task
6's own tests used trivial fixed-value test doubles for those two slots
purely because this task's real implementation didn't exist yet -- see
`tests/intelligence/test_decision_engine.py::_fixed_sltp` /
`_fixed_sizing`. This module is a drop-in replacement for those doubles: it
does not change, and must not change, the callable shapes Task 6 already
established:

    sltp_bootstrap(hyp, market_state) -> (sl_price, tp_price)
    sizing_bootstrap(hyp, account) -> size

(confirmed straight from `FastTierDecisionEngine.decide`, which calls
`self.sltp_bootstrap(hyp, market_state)` and, further down,
`self.sizing_bootstrap(hyp, account)` -- intelligence/decision_engine.py
lines 188 and 211.)
"""
from __future__ import annotations

from typing import Callable, Optional

from intelligence.fast_tier import Hypothesis
from simulator.contracts import AccountState, SimulatedExecutionConfig

# --- SL/TP volatility-scaling design (documented, deliberately simple) ----
# Volatility source: market_state.realized_vol_60s (contracts/market_state.py),
# NOT a GARCH conditional-variance evidence source. Reasoning: by the time
# `decide()` calls sltp_bootstrap (intelligence/decision_engine.py:188),
# only `hyp` (the Hypothesis) and `market_state` are in scope -- no evidence
# registry/tool-trust snapshot is threaded through this call, and
# `Hypothesis` (intelligence/fast_tier.py) does not carry a raw GARCH
# variance figure, only aggregated belief/uncertainty. Reaching into the
# registry for a specific evidence source's raw internal state from inside
# a bootstrap callable would reintroduce a dependency this seam was
# designed to avoid (module docstring: "invents no new interface").
# realized_vol_60s is already the field decision_engine.py itself uses
# immediately after this call to normalize sl_distance_price into R-multiples
# (line 198) and the same field simulator.cost_model.round_trip_cost_r uses
# for its own vol normalization -- so using it here keeps a single
# consistent volatility convention across the whole decide() path, rather
# than mixing two different vol estimates in one decision.
#
# SL distance = SL_VOL_MULTIPLIER * realized_vol_60s * mid (a price
# distance; realized_vol_60s is a return-fraction volatility estimate, so
# multiplying by mid converts it to a price distance, matching
# cost_model.py's own convention).
#
# SL_VOL_MULTIPLIER = 2.0: a conservative round-number "2 standard
# deviations" stop -- wide enough that ordinary 60s-scale noise does not
# stop the position out immediately, without being fit to any data (this is
# a seam-integration task, not a calibration task; Task 12's later
# measurement pass may revisit this).
#
# TP_VOL_MULTIPLIER = 3.0, i.e. a 1.5:1 reward:risk ratio (TP distance is
# 1.5x the SL distance). A reward:risk ratio > 1 is required for the
# strategy to be viable even at a sub-50% hit rate; 1.5:1 is a conservative
# round-number starting point (not a max-Sharpe-fit ratio) that still
# clears round-trip transaction costs comfortably in the common case,
# consistent with the EV/cost gate already enforced upstream in
# decision_engine.py's decide(). Also a candidate for Task 12's later
# tuning pass.
SL_VOL_MULTIPLIER = 2.0
TP_VOL_MULTIPLIER = 3.0


def analytical_sltp_bootstrap(hyp: Hypothesis, market_state) -> tuple:
    """Real (non-learned) SL/TP bootstrap. Matches the exact callable shape
    Task 6 already established for `sltp_bootstrap` (see module docstring).

    Direction: derived from `hyp.net_directional_belief`'s sign (>= 0 =>
    LONG-shaped SL/TP placement, below entry/above entry respectively;
    < 0 => SHORT-shaped) -- decision_engine.py's own `decide()` derives the
    actual trade direction from the identical sign convention via
    `_direction_from_hypothesis`, so this mirrors that without needing a
    separate `direction` parameter (Task 6's established signature doesn't
    carry one; `hyp` alone is sufficient and consistent with the caller).

    Entry-price proxy: `market_state.mid`, the same convention Task 6's own
    `_fixed_sltp` test double used -- the *actual* fill price (adjusted for
    half-spread/slippage by `simulator.execution.entry_fill_price`) isn't
    known until `simulator.engine.open_position` runs, downstream of this
    call. Using mid as the SL/TP anchor is consistent with how
    decision_engine.py itself immediately re-derives `sl_distance_price` as
    `abs(market_state.mid - sl_price)` right after this call returns (line
    197) -- that line would be systematically wrong if this function anchored
    to anything other than mid.

    Returns (None, None) if `realized_vol_60s` is missing/non-positive/NaN --
    conservative, matching the "never fabricate a volatility estimate"
    principle already used by `simulator.cost_model.round_trip_cost_r` and
    already handled safely by `decide()`'s own post-call guard (line 195),
    which turns a None sl_price into NO_TRADE rather than erroring.
    """
    vol = market_state.realized_vol_60s
    if vol is None or not (vol > 0) or vol != vol:  # None, <=0, or NaN
        return None, None

    mid = market_state.mid
    sl_distance = SL_VOL_MULTIPLIER * vol * mid
    tp_distance = TP_VOL_MULTIPLIER * vol * mid

    if hyp.net_directional_belief >= 0:
        # LONG-shaped: SL below mid, TP above mid.
        return mid - sl_distance, mid + tp_distance
    # SHORT-shaped: SL above mid, TP below mid.
    return mid + sl_distance, mid - tp_distance


# --- Sizing bootstrap design (documented, deliberately simple) ------------
# Reuses Phase 1's EXISTING risk-fraction-of-equity sizing mechanism
# (simulator/contracts.py:57 `SimulatedExecutionConfig.risk_fraction_of_equity`)
# rather than reinventing a sizing formula. `simulator.engine.open_position`
# already computes this exact formula as ITS OWN default whenever the caller
# passes `size=None` (simulator/engine.py:34-35:
# `size = account.equity * config.risk_fraction_of_equity`). This bootstrap
# is a THIN wrapper that reproduces that identical formula explicitly, as an
# injected callable, rather than relying on open_position's implicit
# size=None default -- the point is making the sizing decision explicit and
# swappable at the FastTierDecisionEngine seam (Task 6's constructor already
# requires *some* sizing_bootstrap; this is that function made real), not
# changing what gets computed. `hyp` is intentionally unused here: Task 11's
# mandate explicitly excludes belief/conviction-scaled sizing (that would be
# a step toward a learned/edge-scaled sizing head, deferred to a later
# phase) -- sizing stays flat risk-fraction regardless of hypothesis
# strength, exactly matching Phase 1's own default.
#
# `sizing_bootstrap`'s established call shape is `(hyp, account) -> size`
# (Task 6, confirmed at decision_engine.py:211) -- it does not receive
# `config`. `SimulatedExecutionConfig` is needed to read
# `risk_fraction_of_equity`, so `analytical_sizing_bootstrap` is a FACTORY:
# call it once with the config at construction time and pass the returned
# closure as `sizing_bootstrap`. This keeps the injected callable's shape
# identical to what Task 6 already established while still letting the
# bootstrap reuse Phase 1's config-driven risk fraction rather than
# hardcoding a literal.
def analytical_sizing_bootstrap(config: SimulatedExecutionConfig) -> Callable:
    """Returns a `sizing_bootstrap(hyp, account) -> float` closure over
    `config`. See module-level design note above: this reproduces
    `simulator.engine.open_position`'s own default sizing formula
    (`account.equity * config.risk_fraction_of_equity`) explicitly, as a
    swappable injected callable, rather than a new sizing formula.
    `simulator.engine.open_position` still independently enforces
    INSUFFICIENT_MARGIN against whatever size this proposes -- this
    function only ever *proposes* a size, it never bypasses that check."""

    def sizing_bootstrap(hyp: Hypothesis, account: AccountState) -> float:
        return account.equity * config.risk_fraction_of_equity

    return sizing_bootstrap
