"""Bounded daily-resampled ring buffer for the handful of ~252-observation
features (vol_percentile_252, spread_percentile_252) that need more
history than the live process's bounded M1 buffer holds -- WITHOUT
loading the 6.7-year historical CSV into the live process (spec section
6). Bootstrapped once at live_engine startup from the small rolling
gold_seed.csv (~2.5mo), refreshed once/day thereafter from live values."""
from collections import deque

import pandas as pd


class DailyBuffer:
    def __init__(self, size: int):
        self.size = size
        self._days = deque(maxlen=size)  # general day-tracking / bootstrap ordering only
        self._values: dict = {}  # key -> deque(maxlen=size) of (day, value) pairs, own index per key

    def bootstrap_from_csv(self, csv_path: str, value_cols: list) -> None:
        df = pd.read_csv(csv_path, parse_dates=["time"])
        daily = df.set_index("time")[value_cols].resample("1D").last().dropna(how="all")
        daily = daily.tail(self.size)
        for day, row in daily.iterrows():
            self.record(day.date(), {c: float(row[c]) for c in value_cols if pd.notna(row[c])})

    def record(self, day, values: dict) -> None:
        if not self._days or self._days[-1] != day:
            self._days.append(day)
        for k, v in values.items():
            dq = self._values.setdefault(k, deque(maxlen=self.size))
            if dq and dq[-1][0] == day:
                dq[-1] = (day, v)  # same-day update-in-place for this key
            else:
                dq.append((day, v))

    def series(self, key: str) -> pd.Series:
        dq = self._values.get(key, deque())
        if not dq:
            return pd.Series(dtype=float)
        days, vals = zip(*dq)
        return pd.Series(list(vals), index=list(days))
