"""
weather_forecast.py - the numerical weather forecast of irradiance and air temperature that the PV forecast is built on (Open-Meteo, free, no key).

For a delivery day D the forecast is taken, in this order:
  1. the ARCHIVE: the two-day-old forecast as it was issued (model_development/plant_inputs/open_meteo_forecasts.csv.gz, written by
     tools/fetch_open_meteo.py). It is point-in-time and reproducible: it was available at 11:00 on D-1 for every hour of D, and it is what a replay of a
     past day (a backtest, an evaluation) must use;
  2. the LIVE forecast service, only for a delivery day that has not started yet (today or later): the newest forecast, which is the one a real run has
     at the decision time. A past day is never asked of the live service, because it would return a forecast that already knows the day (hindsight);
  3. None. The caller then falls back to its own method.

Hours are UTC, numbered 1..24 like the PV workbook (hour h covers h-1:00 to h:00 UTC). Irradiance is clipped at 0.

recent() is different: it returns the newest COMPLETE days of weather as it happened (the analysis of the weather model), for the days between the end of
the reanalysis (published with a delay of about five days) and today. It is real data, not a forecast and not a synthetic curve.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, Optional, Tuple

import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ARCHIVE = os.path.join(_REPO, "model_development", "plant_inputs", "open_meteo_forecasts.csv.gz")
LIVE_URL = "https://api.open-meteo.com/v1/forecast"
LAT, LON = 38.20, -7.49                                      # Alqueva
Forecast = Tuple[Dict[int, float], Dict[int, float]]         # ({hour: GHI W/m2}, {hour: T_amb degC})

_archive_cache: dict = {}
_live_cache: dict = {}


def _archive() -> Optional[pd.DataFrame]:
    if not os.path.isfile(ARCHIVE):
        return None
    mtime = os.path.getmtime(ARCHIVE)
    if _archive_cache.get("mtime") != mtime:
        df = pd.read_csv(ARCHIVE, parse_dates=["datetime_utc"])
        df["date"] = df["datetime_utc"].dt.date
        df["hour"] = df["datetime_utc"].dt.hour + 1
        _archive_cache.update(mtime=mtime, df=df)
    return _archive_cache["df"]


def archived(delivery_date: str) -> Optional[Forecast]:
    """The archived two-day-old forecast of delivery_date, or None unless all 24 hours of both quantities are present."""
    df = _archive()
    if df is None:
        return None
    day = df[df["date"] == dt.date.fromisoformat(str(delivery_date)[:10])].drop_duplicates("hour").set_index("hour")
    if len(day) < 24 or day[["ghi_fc_d2", "tamb_fc_d2"]].isna().any().any():
        return None
    return ({h: max(0.0, float(day.at[h, "ghi_fc_d2"])) for h in range(1, 25)}, {h: float(day.at[h, "tamb_fc_d2"]) for h in range(1, 25)})


def live(delivery_date: str, timeout: float = 20.0) -> Optional[Forecast]:
    """The newest forecast of delivery_date from the live service, or None if it cannot be reached. Only for a day that has not started."""
    day = dt.date.fromisoformat(str(delivery_date)[:10])
    if day < dt.date.today():
        return None
    if str(day) in _live_cache:
        return _live_cache[str(day)]
    try:
        import requests
        r = requests.get(LIVE_URL, params=dict(latitude=LAT, longitude=LON, start_date=str(day), end_date=str(day),
                                               hourly="shortwave_radiation,temperature_2m", timezone="UTC"), timeout=timeout)
        r.raise_for_status()
        h = r.json()["hourly"]
        ghi, tamb = h["shortwave_radiation"], h["temperature_2m"]
        if len(ghi) < 24 or any(v is None for v in ghi[:24]) or any(v is None for v in tamb[:24]):
            return None
        out = ({i + 1: max(0.0, float(ghi[i])) for i in range(24)}, {i + 1: float(tamb[i]) for i in range(24)})
    except Exception:
        return None
    _live_cache[str(day)] = out
    return out


def recent(start_date, end_date=None, timeout: float = 30.0) -> Optional[pd.DataFrame]:
    """Complete days of weather as it happened, from start_date up to end_date (default: yesterday), or None if they cannot be had.

    Columns: datetime (naive UTC), Hour (1..24), GHI (W/m2), T_amb (degC). Only days with all 24 hours of both quantities are returned, and never today
    or later (a day that has not finished is not data). The source is the recent-days data of Open-Meteo's forecast service (at most 90 days back)."""
    today = dt.date.today()
    start = dt.date.fromisoformat(str(start_date)[:10])
    end = min(dt.date.fromisoformat(str(end_date)[:10]) if end_date else today - dt.timedelta(days=1), today - dt.timedelta(days=1))
    if end < start or (today - start).days > 90:
        return None
    try:
        import requests
        for attempt in range(3):
            r = requests.get(LIVE_URL, params=dict(latitude=LAT, longitude=LON, hourly="shortwave_radiation,temperature_2m", timezone="UTC",
                                                   past_days=(today - start).days, forecast_days=1), timeout=timeout)
            if r.status_code == 200 and "hourly" in r.json():
                break
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
    except Exception:
        return None
    h["datetime"] = pd.to_datetime(h["time"])
    h["date"] = h["datetime"].dt.date
    h = h[(h["date"] >= start) & (h["date"] <= end)].dropna(subset=["shortwave_radiation", "temperature_2m"])
    full = h.groupby("date")["datetime"].transform("count") == 24
    h = h[full]
    if h.empty:
        return None
    return pd.DataFrame({"datetime": h["datetime"].to_numpy(), "Hour": h["datetime"].dt.hour.to_numpy() + 1, "GHI": h["shortwave_radiation"].clip(lower=0).round(1).to_numpy(),
                         "T_amb": h["temperature_2m"].round(2).to_numpy()}).reset_index(drop=True)


def forecast(delivery_date: str, allow_live: bool = True) -> Optional[Forecast]:
    """The weather forecast for delivery_date: the archive, else (for a day not yet started) the live service, else None."""
    return archived(delivery_date) or (live(delivery_date) if allow_live else None)
