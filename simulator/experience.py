"""simulator/experience.py
Records the raw ingredients of every decide()/manage()/close event -- PnL,
cost, account snapshot, environment tag -- with NO reward formula computed
here. Reward shaping (R-multiple, Sharpe-like, drawdown-penalized, etc.) is
explicitly a Phase 2+ research question (see design doc Section 13.B)."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from simulator.contracts import EnvironmentTag, PositionOutcome


@dataclass
class ExperienceRecord:
    environment_tag: EnvironmentTag
    timestamp: datetime
    event_type: str  # "DECIDE" | "MANAGE" | "POSITION_CLOSED"
    market_state_snapshot: dict
    position_view: Optional[dict]
    action: Optional[str]
    account_state: dict
    realized_pnl: Optional[float]
    cost_amount: Optional[float]
    outcome: Optional[PositionOutcome]
    gap_type: str  # "NORMAL" | "WEEKEND_CLOSURE" | "DATA_GAP" -- gap classification for this bar
    # Raw R-space cost ingredient from decision.ev_cost.round_trip_cost_r.
    # Recorded, never applied -- reward shaping is Phase 2+.
    cost_r: Optional[float] = None


class ExperienceRecorder:
    def __init__(self):
        self._records: list[ExperienceRecord] = []

    def record(self, record: ExperienceRecord) -> None:
        self._records.append(record)

    def all_records(self) -> list[ExperienceRecord]:
        return list(self._records)


def write_tag_guard(active_partition: EnvironmentTag, record: ExperienceRecord) -> None:
    if record.environment_tag != active_partition:
        raise ValueError(
            f"Experience record tagged {record.environment_tag} written during "
            f"active partition {active_partition} -- cross-partition contamination rejected."
        )
