"""
fetch_open_meteo.py - archived numerical weather forecasts (as issued) for the Alqueva site, from Open-Meteo's Previous Runs API.

    python tools/fetch_open_meteo.py                       # forecasts as issued, 2022-01-01 up to yesterday
    python tools/fetch_open_meteo.py --start 2024-01-01 --out model_development/plant_inputs/open_meteo_forecasts.csv.gz
    python tools/fetch_open_meteo.py --reanalysis          # what happened: ERA5 reanalysis, 2015-01-01 up to about a week ago

What is stored, per hour (UTC): the forecast of global horizontal irradiance (W/m2) and air temperature at 2 m (degC) that the weather
model had ONE and TWO days before the hour (the `previous_day1` / `previous_day2` fields). They are forecasts as they were issued, not
reanalysis of what happened, so they carry no information from the future. The two-day-old forecast is available at 11:00 on D-1 for
every hour of day D; the one-day-old forecast is not (for the afternoon hours it is issued after 11:00), so the evaluation uses the two-day one.

With --reanalysis the tool stores instead the ERA5 reanalysis of the same two quantities (the usual stand-in for on-site measurements when there is
no verified pyranometer): irradiance and air temperature as they happened, from 2015. ERA5 is a weather-model reanalysis fitted to observations, not a
sensor at the plant, and the output is labelled that way.

The file is a source file (a download, not a result of this repository): its origin is https://open-meteo.com (free, no key). Re-running
this tool refreshes it; the first hours where a field is missing stay empty (the archive of irradiance forecasts starts in 2024).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
URL_ERA5 = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 38.20, -7.49                                   # Alqueva (the PV forecaster's coordinates)
FIELDS = ["shortwave_radiation_previous_day1", "shortwave_radiation_previous_day2", "temperature_2m_previous_day1", "temperature_2m_previous_day2"]
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def fetch(start: dt.date, end: dt.date) -> pd.DataFrame:
    frames, a = [], start
    while a <= end:
        b = min(dt.date(a.year, 12, 31), end)
        for attempt in range(4):
            r = requests.get(URL, params=dict(latitude=LAT, longitude=LON, start_date=a.isoformat(), end_date=b.isoformat(),
                                              hourly=",".join(FIELDS), timezone="UTC"), timeout=120)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]))
        print(f"{a} to {b}: {len(frames[-1])} hours", flush=True)
        a = b + dt.timedelta(days=1)
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out.rename(columns={"time": "datetime_utc", "shortwave_radiation_previous_day1": "ghi_fc_d1", "shortwave_radiation_previous_day2": "ghi_fc_d2",
                               "temperature_2m_previous_day1": "tamb_fc_d1", "temperature_2m_previous_day2": "tamb_fc_d2"})


def fetch_era5(start: dt.date, end: dt.date) -> pd.DataFrame:
    frames, a = [], start
    while a <= end:
        b = min(dt.date(a.year, 12, 31), end)
        for attempt in range(4):
            r = requests.get(URL_ERA5, params=dict(latitude=LAT, longitude=LON, start_date=a.isoformat(), end_date=b.isoformat(), models="era5",
                                                   hourly="shortwave_radiation,temperature_2m", timezone="UTC"), timeout=120)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]))
        print(f"{a} to {b}: {len(frames[-1])} hours", flush=True)
        a = b + dt.timedelta(days=1)
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out.rename(columns={"time": "datetime_utc", "shortwave_radiation": "ghi", "temperature_2m": "tamb"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reanalysis", action="store_true", help="store the ERA5 reanalysis (what happened) instead of the forecasts")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    folder = os.path.join(_REPO, "model_development", "plant_inputs")
    if a.reanalysis:
        start = dt.date.fromisoformat(a.start or "2015-01-01")
        end = dt.date.fromisoformat(a.end) if a.end else dt.date.today() - dt.timedelta(days=7)         # ERA5 is published with a delay of about five days
        out_path = a.out or os.path.join(folder, "era5_alqueva.csv.gz")
        df = fetch_era5(start, end)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False, compression="gzip")
        print(f"wrote {out_path}: {len(df):,} hours, {df['datetime_utc'].min()} to {df['datetime_utc'].max()}, missing GHI {int(df['ghi'].isna().sum())}, missing T {int(df['tamb'].isna().sum())}")
        return 0
    a.start = a.start or "2022-01-01"
    a.end = a.end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    a.out = a.out or os.path.join(folder, "open_meteo_forecasts.csv.gz")
    df = fetch(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False, compression="gzip")
    cov = df.set_index("datetime_utc").notna().resample("MS").mean().round(2)
    print(f"wrote {a.out}: {len(df):,} hours")
    print("share of hours present, first month with >=95%:", {c: (str(cov.index[cov[c] >= 0.95][0])[:7] if (cov[c] >= 0.95).any() else None) for c in cov.columns})
    return 0


if __name__ == "__main__":
    sys.exit(main())
