"""Tests for intelligence/evidence.py — the EvidenceSource contract."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from intelligence.evidence import EvidenceRegistry, EvidenceSourceSpec, EvidenceValue


def _make_spec(name, compute):
    return EvidenceSourceSpec(
        name=name,
        mathematical_formulation="f(x) = mean(x)",
        required_inputs=["closes"],
        assumptions="closes is a 1-D array of recent prices",
        known_failure_conditions="empty input array",
        compute=compute,
    )


def test_evidence_value_fields():
    ev = EvidenceValue(value=1.5, confidence=0.9, source_name="mean_source")
    assert ev.value == 1.5
    assert ev.confidence == 0.9
    assert ev.source_name == "mean_source"


def test_evidence_source_spec_fields():
    def compute(closes):
        return EvidenceValue(float(np.mean(closes)), 1.0, "mean_source")

    spec = _make_spec("mean_source", compute)
    assert spec.name == "mean_source"
    assert spec.mathematical_formulation == "f(x) = mean(x)"
    assert spec.required_inputs == ["closes"]
    assert spec.assumptions == "closes is a 1-D array of recent prices"
    assert spec.known_failure_conditions == "empty input array"
    assert spec.computational_cost_hint is None
    assert callable(spec.compute)


def test_register_and_names():
    registry = EvidenceRegistry()
    spec = _make_spec("source_a", lambda closes: EvidenceValue(1.0, 1.0, "source_a"))
    registry.register(spec)
    assert registry.names() == ["source_a"]


def test_register_duplicate_name_raises():
    registry = EvidenceRegistry()
    spec1 = _make_spec("dup", lambda closes: EvidenceValue(1.0, 1.0, "dup"))
    spec2 = _make_spec("dup", lambda closes: EvidenceValue(2.0, 1.0, "dup"))
    registry.register(spec1)
    with pytest.raises(ValueError):
        registry.register(spec2)


def test_specs_returns_full_metadata_unmodified():
    registry = EvidenceRegistry()
    spec = _make_spec("source_a", lambda closes: EvidenceValue(1.0, 1.0, "source_a"))
    registry.register(spec)

    specs = registry.specs()
    assert set(specs.keys()) == {"source_a"}
    returned = specs["source_a"]
    assert returned is spec
    assert returned.mathematical_formulation == "f(x) = mean(x)"
    assert returned.required_inputs == ["closes"]
    assert returned.assumptions == "closes is a 1-D array of recent prices"
    assert returned.known_failure_conditions == "empty input array"


def test_compute_all_calls_every_source_with_same_array():
    seen_arrays = {}

    def make_compute(name):
        def compute(closes):
            seen_arrays[name] = closes
            return EvidenceValue(float(np.sum(closes)), 1.0, name)
        return compute

    registry = EvidenceRegistry()
    registry.register(_make_spec("a", make_compute("a")))
    registry.register(_make_spec("b", make_compute("b")))

    closes = np.array([1.0, 2.0, 3.0])
    results = registry.compute_all(closes)

    assert set(results.keys()) == {"a", "b"}
    assert results["a"].value == 6.0
    assert results["b"].value == 6.0
    assert seen_arrays["a"] is closes
    assert seen_arrays["b"] is closes


def test_compute_all_isolates_exceptions():
    def broken_compute(closes):
        raise RuntimeError("boom")

    def good_compute(closes):
        return EvidenceValue(42.0, 1.0, "good")

    registry = EvidenceRegistry()
    registry.register(_make_spec("broken", broken_compute))
    registry.register(_make_spec("good", good_compute))

    results = registry.compute_all(np.array([1.0, 2.0, 3.0]))

    assert results["broken"] == EvidenceValue(None, 0.0, "broken")
    assert results["good"].value == 42.0


def test_compute_all_empty_registry():
    registry = EvidenceRegistry()
    assert registry.compute_all(np.array([1.0, 2.0])) == {}
