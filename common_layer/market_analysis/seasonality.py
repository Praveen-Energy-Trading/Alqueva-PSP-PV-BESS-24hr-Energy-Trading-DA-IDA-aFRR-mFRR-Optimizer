"""
seasonality.py — how much of the price is daily, weekly and annual structure.

`variance_shares` removes one component at a time, in a fixed order, and reports how much of the total
variance each one explained:

    level           year-month mean (the slow level: fuel prices, hydrology, regime)
    hour_of_day     hour-of-day profile within that level
    hour x season   how the daily shape changes with the calendar month (solar, demand)
    day_of_week     weekday effect on what is left
    residual        everything else

The order matters (later components only get what earlier ones left), so shares are an ordered
decomposition, not independent effects. "share_of_within_month" divides by the variance that remains
after the slow level, which is the part a day-ahead model has to explain.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _remove(x: pd.Series, keys: List[pd.Series]) -> pd.Series:
    """Group-mean fit of x on the given keys (the fitted part)."""
    return x.groupby(keys).transform("mean")


def variance_shares(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ordered variance decomposition of df[col]; df needs year, month, local_hour, dow columns."""
    d = df[[col, "year", "month", "local_hour", "dow"]].dropna(subset=[col])
    x = d[col].astype(float)
    total = float(((x - x.mean()) ** 2).sum())
    if total == 0 or len(x) < 100:
        return pd.DataFrame(columns=["component", "share_of_total", "share_of_within_month"])
    ym = [d["year"], d["month"]]
    level = _remove(x, ym)
    r1 = x - level
    hour = _remove(r1, [d["local_hour"]])
    r2 = r1 - hour
    shape = _remove(r2, [d["local_hour"], d["month"]])
    r3 = r2 - shape
    dow = _remove(r3, [d["dow"]])
    r4 = r3 - dow
    ss = {
        "level (year-month)": float(((level - x.mean()) ** 2).sum()),
        "hour of day": float((hour ** 2).sum()),
        "hour x season": float((shape ** 2).sum()),
        "day of week": float((dow ** 2).sum()),
        "residual": float((r4 ** 2).sum()),
    }
    within = total - ss["level (year-month)"]
    rows = [{"component": k, "share_of_total": v / total,
             "share_of_within_month": (v / within if k != "level (year-month)" and within > 0 else np.nan)}
            for k, v in ss.items()]
    return pd.DataFrame(rows)


def profile(df: pd.DataFrame, col: str, by: str) -> pd.DataFrame:
    """Mean, median, std and count of df[col] by a calendar column (local_hour, dow, month)."""
    g = df.dropna(subset=[col]).groupby(by)[col]
    return g.agg(mean="mean", median="median", std="std", n="count")


def hour_by_month(df: pd.DataFrame, col: str, year: int = None) -> pd.DataFrame:
    """Median price by local hour (rows) and month (columns), optionally for one year."""
    d = df if year is None else df[df["year"] == year]
    return d.dropna(subset=[col]).pivot_table(index="local_hour", columns="month", values=col, aggfunc="median")


def weekend_effect(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Weekend minus weekday mean price per year (EUR/MWh) and as share of the weekday mean."""
    d = df.dropna(subset=[col])
    m = d.groupby(["year", "is_weekend"])[col].mean().unstack()
    out = pd.DataFrame({"weekday_mean": m[False], "weekend_mean": m[True]})
    out["difference"] = out["weekend_mean"] - out["weekday_mean"]
    out["difference_pct"] = out["difference"] / out["weekday_mean"].abs() * 100.0
    return out


def annual_shape(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calendar-month effect with each year's own mean removed (EUR/MWh), one column per year."""
    d = df.dropna(subset=[col]).copy()
    d["dev"] = d[col] - d.groupby("year")[col].transform("mean")
    return d.pivot_table(index="month", columns="year", values="dev", aggfunc="mean")


def annual_share(df: pd.DataFrame, col: str) -> float:
    """Share of within-year variance explained by the calendar month (ordered after the yearly mean)."""
    d = df.dropna(subset=[col]).copy()
    dev = d[col] - d.groupby("year")[col].transform("mean")
    fit = dev.groupby(d["month"]).transform("mean")
    tot = float((dev ** 2).sum())
    return float((fit ** 2).sum() / tot) if tot > 0 else float("nan")
