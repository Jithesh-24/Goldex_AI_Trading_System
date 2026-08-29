"""intelligence/decision_engine.py -- Fast Tier's plug into Phase 1's
existing, unmodified DecideFn/ManageFn seam (simulator/replay.py:20-24).

This module invents no new interface. `FastTierDecisionEngine.decide` is a
valid `DecideFn` (returns `(action, sl_price, tp_price, size)`, a
backward-compatible 4-tuple) and `FastTierDecisionEngine.manage` is a valid
`ManageFn` (returns "HOLD"|"EXIT").

DESIGN NOTE -- closes_so_far: `FastTierReasoner.hypothesis()` needs a raw
`closes_so_far` array, but `DecideFn`'s contract (simulator/replay.py:20)
only hands `decide()` a single bar's `MarketState`, not a history array --
and `MarketState` itself (contracts/market_state.py) carries only the
current/completed *single* M1 bar, never a rolling window. `simulator.replay
.run_replay` calls `decide_fn` exactly once per bar, in chronological order,
only while flat (simulator/replay.py:93-94). So this engine accumulates its
own `closes_so_far` buffer across successive `decide()` calls, appending
`market_state.mid` each time it is invoked. This is a deliberate, documented
scope choice: the buffer only grows on bars where the engine is flat and
gets asked to decide (never on MANAGE bars, since a position being open
means `decide()` isn't called for those bars) -- for Task 6's seam-
integration purpose this is an acceptable approximation; a later task that
wants a true unbroken closes history would need `simulator.replay` to hand
it the full window directly, which is out of this task's scope to change.

DESIGN NOTE -- direction/threshold and EV/cost gate: see the two constants
and `decide()`'s docstring below for the exact documented rule.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from intelligence.evidence import EvidenceRegistry
from intelligence.fast_tier import FastTierReasoner, Hypothesis, ToolTrust
from intelligence.thesis import Thesis

# --- Direction/threshold design (documented, deliberately simple) ---------
# A Hypothesis is only acted on if its net_directional_belief is clearly
# non-trivial AND the reasoner isn't mostly unsure about it. Both floors are
# conservative round numbers, not fit to data -- this is a seam-integration
# task, not a calibration task (that's later, per Task 12's measurement
# pass). Below either floor: NO_TRADE.
MIN_ABS_BELIEF_TO_ACT = 0.10
MAX_UNCERTAINTY_TO_ACT = 0.75

# --- EV/cost gate design (documented, deliberately simple) ----------------
# `simulator.cost_model.round_trip_cost_r` returns the round-trip cost of a
# candidate trade in R-multiples (normalized by market_state.realized_vol_60s
# and mid, mirroring how the SL distance itself must already be expressed in
# R -- see cost_model.py's own docstring). This engine proxies the trade's
# "expected edge" in the same R-like units as
# `abs(net_directional_belief) * (1 - aggregate_uncertainty)` -- a crude,
# dimensionless in-[0, 1] stand-in for real expected R, since the actual
# analytical SL/TP/edge bootstrap is explicitly Task 11's job, not this
# task's. The gate is the simplest possible EV check: trade only if the
# proxied edge exceeds the round-trip cost. If cost can't be computed at all
# (round_trip_cost_r returns None -- missing vol, stale snapshot, etc.) this
# engine is conservative and returns NO_TRADE rather than fabricating a
# cost, matching the "never fabricate a cost" principle documented in
# cost_model.py itself.
EdgeGateFn = Callable[[object, float], Optional[float]]  # (market_state, candidate_sl_distance_r) -> cost_r | None


def _direction_from_hypothesis(hyp: Hypothesis) -> Optional[str]:
    """Returns "LONG"/"SHORT"/None (None == not-directional-enough)."""
    if hyp.aggregate_uncertainty > MAX_UNCERTAINTY_TO_ACT:
        return None
    if abs(hyp.net_directional_belief) < MIN_ABS_BELIEF_TO_ACT:
        return None
    return "LONG" if hyp.net_directional_belief > 0 else "SHORT"


class FastTierDecisionEngine:
    """Wires FastTierReasoner into Phase 1's DecideFn/ManageFn seam.

    All of sizing, SL/TP placement, and the cost-model call itself are
    constructor-injected (dependency injection, not hardcoded logic) so
    Task 11's real analytical SL/TP/sizing bootstrap can be swapped in later
    without touching this class."""

    def __init__(
        self,
        registry: EvidenceRegistry,
        trust: ToolTrust,
        reasoner: FastTierReasoner,
        ev_cost_gate: EdgeGateFn,
        sizing_bootstrap: Callable,
        sltp_bootstrap: Callable,
    ) -> None:
        self.registry = registry
        self.trust = trust
        self.reasoner = reasoner
        self.ev_cost_gate = ev_cost_gate
        self.sizing_bootstrap = sizing_bootstrap
        self.sltp_bootstrap = sltp_bootstrap
        # See module docstring's closes_so_far design note.
        self._closes: list[float] = []
        # Task 7: thesis memory. At most one Thesis at a time, held as a
        # private instance attribute -- never a module-level dict -- so
        # there is no structural way for one position's load-bearing
        # sources to leak into another's. None while flat.
        self._open_thesis: Optional[Thesis] = None

    @property
    def open_thesis(self) -> Optional[Thesis]:
        """The Thesis for the currently open position, or None while flat."""
        return self._open_thesis

    def clear_thesis(self) -> None:
        """Discards the retained thesis. Called below whenever `decide()`
        is entered with a leftover thesis still set (see note in `decide()`
        for why that's the general-purpose exit-detection point given this
        engine's DecideFn/ManageFn seam). Also exposed as a public method
        so a future MANAGE-driven exit (Task 8's real continuous
        reassessment, once `manage()` can itself decide "EXIT") can clear
        the thesis at the exact bar it triggers the exit, rather than
        waiting for the next `decide()` call."""
        self._open_thesis = None

    def decide(self, market_state, account) -> tuple:
        # `simulator.replay.run_replay` only ever calls decide() while flat
        # (simulator/replay.py:93-94) -- so entering decide() with
        # self._open_thesis still set means the previously open position
        # closed (SL/TP hit, liquidation, or a POLICY_EXIT) via a path this
        # engine's DecideFn/ManageFn seam never observes directly (manage()
        # isn't even called on a bar where price already closed the
        # position -- see simulator/replay.py's same-bar-ambiguity check
        # running before manage_fn). This is therefore the one place that's
        # structurally guaranteed to run exactly once between any position's
        # close and the next one's possible open, making it the correct
        # place to enforce "no leakage across positions" unconditionally,
        # regardless of which path closed the prior position.
        if self._open_thesis is not None:
            self.clear_thesis()

        self._closes.append(float(market_state.mid))
        closes_so_far = np.asarray(self._closes, dtype=np.float64)

        hyp = self.reasoner.hypothesis(closes_so_far, market_state, self.trust)

        direction = _direction_from_hypothesis(hyp)
        if direction is None:
            return ("NO_TRADE", None, None, None)

        sl_price, tp_price = self.sltp_bootstrap(hyp, market_state)

        # Convert the price-distance SL into the same R-multiple convention
        # round_trip_cost_r expects (see cost_model.py's docstring: an
        # R-multiple is a price-return-fraction normalized by a volatility
        # estimate). mirrors the identical normalization cost_model.py
        # itself applies to market_state.spread.
        if sl_price is None or market_state.realized_vol_60s is None or market_state.realized_vol_60s <= 0:
            return ("NO_TRADE", None, None, None)
        sl_distance_price = abs(market_state.mid - sl_price)
        sl_distance_r = sl_distance_price / (market_state.realized_vol_60s * market_state.mid)
        if sl_distance_r <= 0:
            return ("NO_TRADE", None, None, None)

        cost_r = self.ev_cost_gate(market_state, sl_distance_r)
        if cost_r is None:
            # Can't compute cost -- conservative, no fabricated cost.
            return ("NO_TRADE", None, None, None)

        edge_proxy_r = abs(hyp.net_directional_belief) * (1.0 - hyp.aggregate_uncertainty)
        if edge_proxy_r <= cost_r:
            return ("NO_TRADE", None, None, None)

        size = self.sizing_bootstrap(hyp, account)

        # Task 7: retain this entry's load-bearing sources for as long as
        # the position stays open. Only set on an actual LONG/SHORT (never
        # on NO_TRADE, since no position is opening).
        self._open_thesis = Thesis(
            load_bearing_sources=list(hyp.load_bearing_sources),
            entry_belief=hyp.net_directional_belief,
            entry_timestamp=market_state.market_timestamp,
        )
        return (direction, sl_price, tp_price, size)

    def manage(self, market_state, position_view, account) -> str:
        """Stub: always HOLD for this task. Task 8 builds the real
        thesis-based continuous reassessment on top of this stub -- this
        engine's own scope (per the task brief) is only the DECIDE seam plus
        a minimal, contract-honoring MANAGE stub. When Task 8's reassessment
        logic decides to return "EXIT" here, it should also call
        `self.clear_thesis()` at that point so the thesis is discarded on
        the exact bar that triggers the exit; `decide()`'s own self-heal
        (see its docstring) is the fallback that still guarantees clearing
        even for exits this stub never sees (SL/TP hit, liquidation)."""
        return "HOLD"
