"""tests/test_phase4_mae_mfe_quantile_v3b.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_mae_quantile import run_mae_quantile_candidate_v3b
from research.phase4_mfe_quantile import run_mfe_quantile_candidate_v3b


def test_mae_quantile_v3b():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_mae_quantile_candidate_v3b(max_holding=45, rows=600000,
                                             registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert result["status"] in ("validated", "rejected")
    assert os.path.exists(os.path.join(tmp_registry.name, "mae_quantile_v3b_candidate_h45.json"))
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


def test_mfe_quantile_v3b():
    tmp_registry = tempfile.TemporaryDirectory()
    tmp_schemas = tempfile.TemporaryDirectory()
    result = run_mfe_quantile_candidate_v3b(max_holding=45, rows=600000,
                                             registry_dir=tmp_registry.name, schemas_dir=tmp_schemas.name)
    assert result["n_events"] > 50
    assert result["status"] in ("validated", "rejected")
    assert os.path.exists(os.path.join(tmp_registry.name, "mfe_quantile_v3b_candidate_h45.json"))
    tmp_registry.cleanup()
    tmp_schemas.cleanup()


if __name__ == "__main__":
    test_mae_quantile_v3b()
    test_mfe_quantile_v3b()
    print("tests/test_phase4_mae_mfe_quantile_v3b.py: OK")
