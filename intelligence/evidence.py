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
