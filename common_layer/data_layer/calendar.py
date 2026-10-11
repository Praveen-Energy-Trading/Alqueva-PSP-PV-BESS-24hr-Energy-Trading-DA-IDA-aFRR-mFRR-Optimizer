"""
calendar.py — true market time for MIBEL delivery days.

A delivery day runs from local midnight to the next local midnight in market time (Europe/Madrid,
CET/CEST). On the two daylight-saving days it has 23 or 25 hours, i.e. 92 or 100 quarter-hours.
Period indices follow the files: 1..N in chronological order, so the autumn day has two "hour 3"
slots at different UTC times. Everything is stored with the UTC start of each period, which is
unambiguous.

The optimiser uses a fixed 24-hour / 96-period grid. `to_model_grid` is the ONE place that maps a
true day onto that grid, with a documented policy (below), so the policy is not repeated in loaders.

Model-grid policy (kept identical to the original pipeline):
    spring day (23 h): the missing hour 3 is filled with the mean of hours 2 and 4;
    autumn day (25 h): the two hour-3 slots are averaged into one hour 3.
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional, Sequence, Tuple

from common_layer.utilities.timezone_utils import MARKET_TZ

UTC = dt.timezone.utc


def _local_midnight(day: dt.date) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, tzinfo=MARKET_TZ)


def day_bounds_utc(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    """UTC start and end of the delivery day (local midnight to local midnight)."""
    start = _local_midnight(day).astimezone(UTC)
    nxt = day + dt.timedelta(days=1)
    return start, _local_midnight(nxt).astimezone(UTC)


def hours_in_day(day: dt.date) -> int:
    """23, 24 or 25."""
    s, e = day_bounds_utc(day)
    return int(round((e - s).total_seconds() / 3600))


def expected_periods(day: dt.date, resolution_min: int) -> int:
    """Number of periods in the delivery day at the given resolution (60 or 15 minutes)."""
    return hours_in_day(day) * 60 // resolution_min


def period_start_utc(day: dt.date, index: int, resolution_min: int) -> dt.datetime:
    """UTC start of period `index` (1-based) of the delivery day."""
    n = expected_periods(day, resolution_min)
    if not 1 <= index <= n:
        raise ValueError(f"period {index} outside 1..{n} for {day} at {resolution_min} min")
    s, _ = day_bounds_utc(day)
    return s + dt.timedelta(minutes=resolution_min * (index - 1))


def detect_resolution_min(day: dt.date, n_values: int) -> Optional[int]:
    """60 if n_values is a full hourly day, 15 if a full quarter-hour day, else None."""
    h = hours_in_day(day)
    if n_values == h:
        return 60
    if n_values == 4 * h:
        return 15
    return None


def to_model_grid(values: Sequence[float], day: dt.date) -> List[float]:
    """Map a full true day (hourly 23/24/25 or quarterly 92/96/100 values) onto the fixed
    24-hour or 96-period grid, using the policy in the module docstring."""
    n = len(values)
    res = detect_resolution_min(day, n)
    if res is None:
        raise ValueError(f"{n} values do not form a full day on {day}")
    vals = list(values)
    if res == 60:
        if n == 23:
            vals = vals[:2] + [(vals[1] + vals[2]) / 2.0] + vals[2:]
        elif n == 25:
            vals = vals[:2] + [(vals[2] + vals[3]) / 2.0] + vals[4:]
        return vals
    if n == 92:
        fill = [(vals[4 + q] + vals[8 + q]) / 2.0 for q in range(4)]
        vals = vals[:8] + fill + vals[8:]
    elif n == 100:
        merged = [(vals[8 + q] + vals[12 + q]) / 2.0 for q in range(4)]
        vals = vals[:8] + merged + vals[16:]
    return vals
