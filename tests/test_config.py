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
