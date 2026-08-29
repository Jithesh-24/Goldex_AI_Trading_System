"""simulator/contracts.py
GOLDEX V4 Phase 1 core data contracts. Neither sl_price nor tp_price is ever
mandatory on a Position -- a policy may run with either, both, or neither
(see docs/superpowers/specs/2026-08-26-goldex-v4-phase1-simulator-design.md
Section 13). No fixed holding horizon appears anywhere in this module."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


def _default_leverage() -> float:
    """Sourced from config/risk.yaml (RiskConfig.leverage) rather than a
    hardcoded literal. Imported lazily to avoid a module-load-time
    dependency between simulator and config."""
    from config.loader import load_config
    return load_config().risk.leverage


def _default_currency() -> str:
    """Sourced from config/risk.yaml (RiskConfig.currency)."""
    from config.loader import load_config
    return load_config().risk.currency


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
    # Defaults sourced from config/risk.yaml (RiskConfig.leverage /
    # RiskConfig.currency) via lazy factories -- see _default_leverage /
    # _default_currency above -- rather than hardcoded literals.
    leverage: float = field(default_factory=_default_leverage)
    slippage_fraction_of_spread: float = 0.5
    latency_ms: float = 0.0
    margin_call_threshold: float = 0.5
    liquidation_threshold: float = 0.2
    risk_fraction_of_equity: float = 0.01
    currency: str = field(default_factory=_default_currency)
    # BUGFIX (whole-branch review): decision.ev_cost.round_trip_cost_r rejects a
    # market_state whose market_timestamp is older than max_staleness_seconds
    # relative to wall-clock now(). Historical replay timestamps are years old,
    # so the live default of 5.0s made round_trip_cost_r return None on EVERY
    # bar -- the entire cost model was silently inert. Replay is offline: the
    # staleness guard is a live-feed-health check with no meaning here.
    max_staleness_seconds: float = float("inf")


@dataclass
class AccountState:
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    exposure: float
    open_position_id: Optional[str]
    simulation_timestamp: datetime
    # Sum of net_pnl across every close_position() call so far this run.
    realized_pnl_total: float = 0.0
    # Highest equity observed so far this run (monotonically non-decreasing).
    # None until the first equity value is known (AccountState.initial() sets
    # it to starting_balance immediately, so in practice this is only None for
    # ad-hoc AccountState(...) construction, e.g. in tests).
    peak_equity: Optional[float] = None
    # Convention: drawdown = (peak_equity - equity) / peak_equity, i.e. the
    # fractional retracement from the running peak. 0.0 at/above peak, grows
    # toward 1.0 as equity falls toward zero. Recomputed any time equity
    # changes (open-position mark_to_market and close-time settlement).
    drawdown: float = 0.0
    currency: str = "USD"

    @classmethod
    def initial(cls, config: SimulatedExecutionConfig, timestamp: datetime) -> "AccountState":
        return cls(balance=config.starting_balance, equity=config.starting_balance,
                    margin_used=0.0, margin_free=config.starting_balance,
                    exposure=0.0, open_position_id=None, simulation_timestamp=timestamp,
                    realized_pnl_total=0.0, peak_equity=config.starting_balance, drawdown=0.0,
                    currency=config.currency)

    @staticmethod
    def compute_drawdown(prior_peak_equity: Optional[float], equity: float) -> tuple:
        """Returns (peak_equity, drawdown) for the given new equity value,
        given the running peak equity before this update. peak_equity only
        ever increases or stays flat; drawdown = (peak - equity) / peak."""
        peak_equity = equity if prior_peak_equity is None else max(prior_peak_equity, equity)
        drawdown = 0.0 if peak_equity <= 0 else (peak_equity - equity) / peak_equity
        return peak_equity, drawdown


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
    # Money cost already embedded in entry_price by entry_fill_price()
    # (half-spread + slippage). Recorded so close_position can report the true
    # round-trip execution cost without charging it a second time.
    entry_cost_amount: float = 0.0

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
