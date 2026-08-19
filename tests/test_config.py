"""python3 tests/test_config.py"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_config


def test_load_config_valid():
    cfg = load_config()
    assert cfg.market.symbol == "XAUUSD"
    assert cfg.decision.meta_prob_threshold == 0.6
    assert cfg.models.direction == "direction_catboost_20260818"
    assert cfg.models.regime is None
    assert cfg.learning.acc_regression_tolerance == 0.01
    assert cfg.market.feed_host == "127.0.0.1"
    assert cfg.market.feed_port == 47115
    assert cfg.market.feed_mode == "managed_socket_feed"
    assert cfg.features.registry_dir == "features/registry"
    assert cfg.features.daily_buffer_size == 252


if __name__ == "__main__":
    test_load_config_valid()
    print("config/: OK")
