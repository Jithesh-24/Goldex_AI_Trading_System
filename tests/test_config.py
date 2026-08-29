"""Minimal config-loading smoke test. Only asserts that load_config()
correctly parses config/risk.yaml into RiskConfig -- not a resurrection of
the old archived V3-coupled test_config.py."""
import os

import yaml

from config.loader import load_config, CONFIG_DIR


def test_risk_config_matches_risk_yaml():
    with open(os.path.join(CONFIG_DIR, "risk.yaml")) as f:
        raw = yaml.safe_load(f)

    risk = load_config().risk

    assert risk.currency == raw["currency"]
    assert risk.leverage == raw["leverage"]
    assert risk.margin_call_level == raw["margin_call_level"]


def test_load_config_is_cached():
    """load_config() is @functools.lru_cache'd (Finding 4 fix) -- repeated
    calls with the same config_dir must return the identical object, not
    re-parse the 6 YAML files each time."""
    first = load_config()
    second = load_config()
    assert first is second
