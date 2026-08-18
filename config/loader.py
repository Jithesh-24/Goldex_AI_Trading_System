"""One entry point for every config category. load_config() returns a
fully validated Config; nothing downstream re-parses YAML itself."""
import os

import yaml

from config.schema import (
    Config, MarketConfig, FeaturesConfig, ModelRoleConfig, DecisionConfig,
    RiskConfig, TelegramConfig, JournalConfig, LearningConfig, RuntimeConfig,
)

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_dir: str = CONFIG_DIR) -> Config:
    def _load(name):
        path = os.path.join(config_dir, name)
        with open(path) as f:
            return yaml.safe_load(f) or {}

    return Config(
        market=MarketConfig(**_load("market.yaml")),
        features=FeaturesConfig(**_load("features.yaml")),
        models=ModelRoleConfig(**_load("models.yaml")),
        decision=DecisionConfig(**_load("decision.yaml")),
        risk=RiskConfig(**_load("risk.yaml")),
        telegram=TelegramConfig(**_load("telegram.yaml")),
        journal=JournalConfig(**_load("journal.yaml")),
        learning=LearningConfig(**_load("learning.yaml")),
        runtime=RuntimeConfig(**_load("runtime.yaml")),
    )
