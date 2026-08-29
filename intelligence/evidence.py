"""The EvidenceSource contract for the GOLDEX intelligence layer.

This is the "quantitative knowledge/tool interface" (mandate Section 12, item 1):
each evidence source is not a bare (value, confidence) tuple but a full spec that
carries its mathematical formulation, required inputs, assumptions, and known
failure conditions alongside the compute callable. A later reasoning layer audits
*why* a source might be unreliable using this metadata, not just whether it
recently performed well.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class EvidenceValue:
    """The result of evaluating one evidence source."""

    value: Optional[float]
    confidence: float
    source_name: str


@dataclass
class EvidenceSourceSpec:
    """Full metadata + compute callable for one evidence source."""

    name: str
    mathematical_formulation: str
    required_inputs: list[str]
    assumptions: str
    known_failure_conditions: str
    compute: Callable[[np.ndarray], EvidenceValue]
    computational_cost_hint: Optional[str] = None  # populated by Task 12's latency
                                                     # instrumentation, not guessed here
    is_directional: bool = False
    # Whether the SIGN of this source's `EvidenceValue.value` encodes an
    # expected PRICE DIRECTION (positive == up/bullish, negative ==
    # down/bearish). Only sources with is_directional=True may contribute a
    # directional vote to `Hypothesis.net_directional_belief`.
    #
    # This field exists because treating every source's sign as a
    # directional vote is a hard structural bias: a variance (GARCH
    # conditional variance) or a variance ratio (multiscale vol ratio) is
    # non-negative BY CONSTRUCTION, so it can only ever vote "LONG," and a
    # Beta posterior mean lives strictly in (0, 1) -- trust learning can
    # shrink such a source's weight but can never zero or flip its permanent
    # long vote. Distribution-shape statistics (skew, excess kurtosis) and a
    # volatility-regime transition are signed, but their sign means
    # "left/right-tailed", "fatter/thinner tails", "vol rising/falling" --
    # not "price up/down".
    #
    # Defaults to False deliberately: a new source must OPT IN to being
    # treated as a directional vote, so the failure mode of forgetting to
    # classify is "ignored for direction" (safe) rather than "silently biases
    # the aggregate" (the bug this field was added to fix).


class EvidenceRegistry:
    """Holds registered evidence sources and evaluates them safely."""

    def __init__(self) -> None:
        self._specs: dict[str, EvidenceSourceSpec] = {}

    def register(self, spec: EvidenceSourceSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Evidence source already registered: {spec.name!r}")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def specs(self) -> dict[str, EvidenceSourceSpec]:
        return self._specs

    def compute_all(self, closes_so_far: np.ndarray) -> dict[str, EvidenceValue]:
        results: dict[str, EvidenceValue] = {}
        for name, spec in self._specs.items():
            try:
                results[name] = spec.compute(closes_so_far)
            except Exception:
                results[name] = EvidenceValue(None, 0.0, name)
        return results
