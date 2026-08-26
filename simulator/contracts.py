"""simulator/contracts.py
GOLDEX V4 Phase 1 core data contracts. Neither sl_price nor tp_price is ever
mandatory on a Position -- a policy may run with either, both, or neither
(see docs/superpowers/specs/2026-08-26-goldex-v4-phase1-simulator-design.md
Section 13). No fixed holding horizon appears anywhere in this module."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EnvironmentTag(str, Enum):
    SIMULATED_TRAINING = "SIMULATED_TRAINING"
    SIMULATED_VALIDATION = "SIMULATED_VALIDATION"
    SIMULATED_OOS_TEST = "SIMULATED_OOS_TEST"
    LIVE_DEMO = "LIVE_DEMO"
    LIVE_REAL = "LIVE_REAL"


class PositionOutcome(str, Enum):
    POLICY_EXIT = "POLICY_EXIT"
    SL_HIT = "SL_HIT"
    TP_HIT = "TP_HIT"
    LIQUIDATION = "LIQUIDATION"
    END_OF_REPLAY_FORCED_CLOSE = "END_OF_REPLAY_FORCED_CLOSE"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class SimulatedExecutionConfig:
    starting_balance: float = 10000.0
    leverage: float = 100.0
    slippage_fraction_of_spread: float = 0.5
    latency_ms: float = 0.0
    margin_call_threshold: float = 0.5
    liquidation_threshold: float = 0.2
    risk_fraction_of_equity: float = 0.01
    max_staleness_seconds: float = 5.0


@dataclass
class AccountState:
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    exposure: float
    open_position_id: Optional[str]
    simulation_timestamp: datetime

    @classmethod
    def initial(cls, config: SimulatedExecutionConfig, timestamp: datetime) -> "AccountState":
        return cls(balance=config.starting_balance, equity=config.starting_balance,
                    margin_used=0.0, margin_free=config.starting_balance,
                    exposure=0.0, open_position_id=None, simulation_timestamp=timestamp)


@dataclass
class Position:
    position_id: str
    side: Side
    entry_time: datetime
    entry_price: float
    size: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    margin_used: float

    def unrealized_pnl(self, current_mid: float) -> float:
        direction = 1.0 if self.side == Side.LONG else -1.0
        return direction * (current_mid - self.entry_price) * self.size


@dataclass
class PositionView:
    position_id: str
    side: Side
    entry_time: datetime
    entry_price: float
    size: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    unrealized_pnl: float
    bars_held: int
