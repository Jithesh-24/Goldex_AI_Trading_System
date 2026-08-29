"""One entry point for every config category. load_config() returns a
fully validated Config; nothing downstream re-parses YAML itself."""
import functools
import os

import yaml

from config.schema import (
    Config, MarketConfig, FeaturesConfig,
    RiskConfig, TelegramConfig, JournalConfig, RuntimeConfig,
)

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


@functools.lru_cache(maxsize=None)
def load_config(config_dir: str = CONFIG_DIR) -> Config:
    """Cached: parses all 6 YAML files once per distinct config_dir, then
    returns the same Config object on every subsequent call. Uncached, this
    was measured at 3.36ms per call and was being invoked twice per
    SimulatedExecutionConfig() construction (simulator/contracts.py's
    _default_leverage/_default_currency factories) -- i.e. every construction
    re-parsed the entire config tree just to read risk.yaml. Safe to cache:
    the YAML files are static for the process lifetime, and config_dir (a
    str) is hashable so lru_cache's key works unmodified."""
    def _load(name):
        path = os.path.join(config_dir, name)
        with open(path) as f:
            return yaml.safe_load(f) or {}

    return Config(
        market=MarketConfig(**_load("market.yaml")),
        features=FeaturesConfig(**_load("features.yaml")),
        risk=RiskConfig(**_load("risk.yaml")),
        telegram=TelegramConfig(**_load("telegram.yaml")),
        journal=JournalConfig(**_load("journal.yaml")),
        runtime=RuntimeConfig(**_load("runtime.yaml")),
    )
