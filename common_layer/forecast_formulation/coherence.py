"""
coherence.py — hourly and 15-minute forecasts that agree.

The day-ahead market has priced 15-minute products since 2025-10-01 but only hourly prices before, so a model
trained on the long hourly history produces hourly forecasts and the 96-period optimiser needs quarters.
Coherence rule: the four quarter forecasts of an hour must average to that hour's forecast. `reconcile`
applies the smallest additive correction that enforces it; `within_hour_shape` measures the quarter pattern
inside an hour that a quarter-hour model can add (or that a flat split throws away).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def hourly_from_quarters(q: np.ndarray) -> np.ndarray:
    """Mean of each block of four quarters; len(q) must be a multiple of 4."""
    q = np.asarray(q, dtype=float)
    if len(q) % 4:
        raise ValueError("quarter series must contain whole hours")
    return q.reshape(-1, 4).mean(axis=1)


def reconcile(quarters: np.ndarray, hourly: np.ndarray) -> np.ndarray:
    """Shift each hour's four quarters by the same amount so their mean equals `hourly`.

    Keeps the within-hour shape of `quarters` exactly; idempotent; the correction is the least-squares one."""
    q = np.asarray(quarters, dtype=float).reshape(-1, 4)
    h = np.asarray(hourly, dtype=float)
    if len(h) != len(q):
        raise ValueError("one hourly value per block of four quarters is required")
    return (q + (h - q.mean(axis=1))[:, None]).ravel()


def flat_split(hourly: np.ndarray) -> np.ndarray:
    """Each hour's forecast repeated for its four quarters (the baseline a quarter-hour model must beat)."""
    return np.repeat(np.asarray(hourly, dtype=float), 4)


def within_hour_shape(df: pd.DataFrame, price_col: str = "price") -> Dict[str, pd.DataFrame]:
    """Quarter pattern inside the hour. `df` needs columns local_hour, quarter (0-3) and price_col, with every
    row belonging to a complete hour. Returns the mean deviation from the hourly mean by (hour, quarter), and
    the error (MAE, RMSE) of a flat split by hour."""
    d = df.copy()
    d["hour_mean"] = d.groupby(d.index.floor("h") if isinstance(d.index, pd.DatetimeIndex) else d["hour_key"])[price_col].transform("mean")
    d["dev"] = d[price_col] - d["hour_mean"]
    shape = d.pivot_table(index="local_hour", columns="quarter", values="dev", aggfunc="mean")
    err = d.groupby("local_hour")["dev"].agg(mae=lambda s: s.abs().mean(), rmse=lambda s: float(np.sqrt((s ** 2).mean())))
    err.loc["all", "mae"] = d["dev"].abs().mean()
    err.loc["all", "rmse"] = float(np.sqrt((d["dev"] ** 2).mean()))
    return {"shape": shape, "flat_split_error": err}
