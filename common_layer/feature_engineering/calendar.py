"""
calendar.py — what the calendar says about a delivery day, known long before the day.

Holidays follow the Portuguese national list: 1 Jan, Good Friday, Easter Sunday, 25 Apr, 1 May, Corpus Christi,
10 Jun, 15 Aug, 5 Oct, 1 Nov, 1 Dec, 8 Dec, 25 Dec; Corpus Christi, 5 Oct, 1 Nov and 1 Dec were suspended
2013-2015. Regional and local holidays are not included.

Bridge day (puente): a working weekday sandwiched between a holiday and the weekend, i.e. a Monday followed by a
Tuesday holiday or a Friday preceded by a Thursday holiday.
"""
from __future__ import annotations

import datetime as dt
import functools
import math
from typing import Dict, List, Set

import pandas as pd

from common_layer.data_layer import calendar as market_calendar
from common_layer.data_layer import regimes


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


@functools.lru_cache(maxsize=None)
def holidays_pt(year: int) -> Set[dt.date]:
    e = easter(year)
    h = {dt.date(year, 1, 1), e - dt.timedelta(days=2), e, dt.date(year, 4, 25), dt.date(year, 5, 1),
         dt.date(year, 6, 10), dt.date(year, 8, 15), dt.date(year, 12, 8), dt.date(year, 12, 25)}
    if not 2013 <= year <= 2015:
        h |= {e + dt.timedelta(days=60), dt.date(year, 10, 5), dt.date(year, 11, 1), dt.date(year, 12, 1)}
    return h


def _holidays(country: str, year: int) -> Set[dt.date]:
    if country != "pt":
        raise ValueError("only the Portuguese calendar is kept (country='pt')")
    return holidays_pt(year)


def is_holiday(day: dt.date, country: str = "pt") -> bool:
    return day in _holidays(country, day.year)


def is_bridge(day: dt.date, country: str) -> bool:
    if day.weekday() >= 5 or is_holiday(day, country):
        return False
    nxt, prv = day + dt.timedelta(days=1), day - dt.timedelta(days=1)
    return (day.weekday() == 0 and is_holiday(nxt, country)) or (day.weekday() == 4 and is_holiday(prv, country))


def days_to_next_holiday(day: dt.date, country: str, cap: int = 30) -> int:
    for k in range(1, cap + 1):
        if is_holiday(day + dt.timedelta(days=k), country):
            return k
    return cap


def days_since_holiday(day: dt.date, country: str, cap: int = 30) -> int:
    for k in range(1, cap + 1):
        if is_holiday(day - dt.timedelta(days=k), country):
            return k
    return cap


def day_features(day: dt.date, events=None) -> Dict[str, float]:
    """All date-level calendar and market-rule features of one delivery day."""
    hours = market_calendar.hours_in_day(day)
    rg = regimes.regime_at(day, events)
    doy = day.timetuple().tm_yday
    return {
        "dow": float(day.weekday()), "month": float(day.month), "is_weekend": float(day.weekday() >= 5),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25), "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "is_holiday_pt": float(is_holiday(day, "pt")), "is_bridge_pt": float(is_bridge(day, "pt")),
        "days_to_holiday_pt": float(days_to_next_holiday(day, "pt")),
        "days_since_holiday_pt": float(days_since_holiday(day, "pt")),
        "dst_day_type": float({23: -1, 24: 0, 25: 1}[hours]),
        "da_resolution_min": float(rg["da_resolution_min"]), "price_floor_eur_mwh": float(rg["price_floor_eur_mwh"]),
    }


def hour_features(hours) -> Dict[str, list]:
    h = [float(x) for x in hours]
    return {"hour": h, "hour_sin": [math.sin(2 * math.pi * x / 24) for x in h],
            "hour_cos": [math.cos(2 * math.pi * x / 24) for x in h]}
