"""simulator/closure.py
Reuses the weekend-gap convention already established in learning/data.py's
gap-detection code (Friday >=20:00 marks the start of the weekend close) so
the simulator's closure handling matches the rest of the codebase rather than
inventing a second convention."""
from datetime import datetime, timedelta

NORMAL_BAR_SECONDS = 60
DATA_GAP_TOLERANCE_SECONDS = 90


def is_weekend_close_start(timestamp: datetime) -> bool:
    return timestamp.weekday() == 4 and timestamp.hour >= 20


def classify_gap(prev_timestamp: datetime, current_timestamp: datetime) -> str:
    gap_seconds = (current_timestamp - prev_timestamp).total_seconds()
    if gap_seconds <= DATA_GAP_TOLERANCE_SECONDS:
        return "NORMAL"
    if is_weekend_close_start(prev_timestamp) or prev_timestamp.weekday() in (5, 6):
        return "WEEKEND_CLOSURE"
    return "DATA_GAP"
