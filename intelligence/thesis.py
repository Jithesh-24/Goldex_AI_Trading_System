"""intelligence/thesis.py -- Task 7: thesis memory (mandate Section 6/12
item 7).

Retains, only while a position is open, the specific
`(source_name, context_bucket, contribution)` tuples that were load-bearing
at entry (Task 5's `Hypothesis.load_bearing_sources`) -- discarded
immediately once that position closes. `Thesis` itself is a plain,
inert value object; the scoping guarantee (no persistence beyond one
position's lifetime, no leakage across positions) comes from *where* it is
held, not from anything in this class. `FastTierDecisionEngine`
(intelligence/decision_engine.py) holds at most one `Thesis` at a time as a
private instance attribute (`self._open_thesis`) -- never a module-level
dict or any other structure a second, unrelated position could accidentally
read from. See decision_engine.py's docstring for exactly how that
attribute is populated (on `decide()` returning LONG/SHORT) and cleared (on
the next `decide()` call, which `simulator.replay.run_replay` only ever
invokes while flat -- i.e. only after any prior position has already
closed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Thesis:
    """The retained justification for one currently-open position."""

    load_bearing_sources: list[tuple[str, int, float]] = field(default_factory=list)
    # (source_name, context_bucket, signed contribution) -- copied verbatim
    # from the Hypothesis that justified this position's entry.
    entry_belief: float = 0.0  # Hypothesis.net_directional_belief at entry
    entry_timestamp: Optional[datetime] = None  # market_state.market_timestamp at entry
