"""python3 tests/test_router.py -- requires Task 11's models/registry/*.json
to exist. Run for real verification after Task 11, not after Task 10."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.router import ModelRouter
from config.loader import load_config


def test_router_resolves_configured_roles():
    cfg = load_config()
    router = ModelRouter(role_map=cfg.models.model_dump())
    direction = router.resolve("direction")
    meta = router.resolve("opportunity_meta")
    assert direction is not None and direction.status == "active"
    assert meta is not None and meta.status == "active"
    assert os.path.exists(router.artifact_path("direction"))
    assert os.path.exists(router.artifact_path("opportunity_meta"))


def test_router_returns_none_for_unconfigured_role():
    cfg = load_config()
    router = ModelRouter(role_map=cfg.models.model_dump())
    assert router.resolve("regime") is None
    assert router.resolve("mae_quantile") is None


if __name__ == "__main__":
    test_router_resolves_configured_roles()
    test_router_returns_none_for_unconfigured_role()
    print("decision/router.py: OK")
