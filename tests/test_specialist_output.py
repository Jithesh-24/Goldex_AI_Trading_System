"""tests/test_specialist_output.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError

from contracts.specialist_output import (
    ModelStatus, DirectionOutput, OpportunityOutput, RegimeOutput,
    MAEOutput, MFEOutput, BarrierOutput, ExecutionOutput,
)


def test_direction_output_requires_status():
    with pytest.raises(ValidationError):
        DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15)


def test_direction_output_unavailable_omits_probabilities():
    out = DirectionOutput(model_id="direction_v3_candidate_h90", horizon=90,
                           model_status="UNAVAILABLE")
    assert out.probability_long is None
    assert out.probability_short is None


def test_direction_output_validated_carries_probabilities():
    out = DirectionOutput(model_id="direction_v3_candidate_h15", horizon=15,
                           model_status="VALIDATED", probability_long=0.55,
                           probability_short=0.45, calibrated=True)
    assert out.probability_long == 0.55
    assert out.calibrated is True


def test_barrier_output_fields():
    out = BarrierOutput(model_id="barrier_v3_candidate_h15", horizon=15,
                         model_status="VALIDATED", p_tp=0.5, p_sl=0.3,
                         p_timeout=0.2, calibrated=True)
    assert abs((out.p_tp + out.p_sl + out.p_timeout) - 1.0) < 1e-9


def test_mae_output_no_q95_field_exists():
    out = MAEOutput(model_id="mae_quantile_v3_candidate_h15", horizon=15,
                     model_status="VALIDATED", q50=0.3, q75=0.6, q90=0.9)
    assert not hasattr(out, "q95")


def test_execution_output_data_limited():
    out = ExecutionOutput(model_id="execution_decay_v3_stub", model_status="DATA_LIMITED",
                           data_limited=True)
    assert out.drift_60s is None


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        DirectionOutput(model_id="x", horizon=15, model_status="NOT_A_REAL_STATUS")


if __name__ == "__main__":
    test_direction_output_requires_status()
    test_direction_output_unavailable_omits_probabilities()
    test_direction_output_validated_carries_probabilities()
    test_barrier_output_fields()
    test_mae_output_no_q95_field_exists()
    test_execution_output_data_limited()
    test_invalid_status_rejected()
    print("tests/test_specialist_output.py: OK")
