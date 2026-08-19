"""Single source of truth for every runtime setting. No Python file
outside this package should hardcode a threshold, path, model ID, or
feature list -- if it needs one, it reads it from here."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class MarketConfig(BaseModel):
    symbol: str
    feed_mode: Literal["managed_socket_feed"] = "managed_socket_feed"
    feed_host: str = "127.0.0.1"
    feed_port: int = 47115
    state_dir: str
    tick_state_file: str
    active_signal_file: str
    bars_file: str
    legacy_note: str = (
        "Market-data path (bid/ask/M1) is now the managed feed "
        "(market/mt5_feed.py -> market/feed_listener.py -> MarketState), "
        "Phase 2. tick_state_file/active_signal_file/bars_file remain "
        "referenced ONLY for the untouched verdict-tracking channel "
        "(trade-management, deliberately out of Phase 2 scope) -- not "
        "for market-data reads anymore."
    )


class FeaturesConfig(BaseModel):
    schema_version: str
    registry_dir: str
    daily_buffer_bootstrap_csv: str
    daily_buffer_size: int


class ModelRoleConfig(BaseModel):
    direction: Optional[str] = None
    opportunity_meta: Optional[str] = None
    regime: Optional[str] = None
    mae_quantile: Optional[str] = None
    mfe_quantile: Optional[str] = None
    barrier_probability: Optional[str] = None


class DecisionConfig(BaseModel):
    meta_prob_threshold: float = Field(ge=0.0, le=1.0)


class RiskConfig(BaseModel):
    pass


class TelegramConfig(BaseModel):
    env_path: str


class JournalConfig(BaseModel):
    schema_version: str
    output_dir: str
    legacy_note: str = (
        "TEMPORARY: journal event files live in an external directory "
        "outside this repo (output_dir). Only the schema is versioned "
        "here in Phase 1."
    )


class LearningConfig(BaseModel):
    acc_regression_tolerance: float


class RuntimeConfig(BaseModel):
    base_dir: str
    outdir: str


class Config(BaseModel):
    market: MarketConfig
    features: FeaturesConfig
    models: ModelRoleConfig
    decision: DecisionConfig
    risk: RiskConfig
    telegram: TelegramConfig
    journal: JournalConfig
    learning: LearningConfig
    runtime: RuntimeConfig
