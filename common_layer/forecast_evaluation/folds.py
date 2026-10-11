"""
folds.py — rolling-origin folds with a retraining schedule and a gap.

A fold is one model fit and the consecutive delivery days it predicts. A model is refitted every
`retrain_every_days` days, exactly as it would be live. Its training targets end `gap_days` + 1 days before
the first day it predicts (the gap covers the retraining and checking time), and, when `train_window_days` is
set, start that many days earlier (a rolling window; None = expanding from the first available day).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Fold:
    fold_id: int
    test_days: List[dt.date]
    train_start: dt.date
    train_end: dt.date

    @property
    def train_days(self) -> List[dt.date]:
        n = (self.train_end - self.train_start).days + 1
        return [self.train_start + dt.timedelta(days=i) for i in range(n)]


def make_folds(test_start: dt.date, test_end: dt.date, first_data_day: dt.date, schedule: dict) -> List[Fold]:
    """Folds covering [test_start, test_end]; folds without `min_train_days` of history are an error."""
    every, gap = schedule["retrain_every_days"], schedule["gap_days"]
    window: Optional[int] = schedule.get("train_window_days")
    folds: List[Fold] = []
    day, fid = test_start, 0
    while day <= test_end:
        block = [day + dt.timedelta(days=i) for i in range(every) if day + dt.timedelta(days=i) <= test_end]
        train_end = day - dt.timedelta(days=gap + 1)
        train_start = first_data_day if window is None else max(first_data_day, train_end - dt.timedelta(days=window - 1))
        if (train_end - train_start).days + 1 < schedule["min_train_days"]:
            raise ValueError(f"fold {fid} starting {day} has only {(train_end - train_start).days + 1} training days "
                             f"(minimum {schedule['min_train_days']})")
        folds.append(Fold(fid, block, train_start, train_end))
        day += dt.timedelta(days=every)
        fid += 1
    return folds
