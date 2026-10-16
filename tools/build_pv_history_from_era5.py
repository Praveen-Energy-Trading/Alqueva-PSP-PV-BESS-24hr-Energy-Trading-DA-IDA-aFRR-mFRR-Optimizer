"""build_pv_history_from_era5.py - the PV workbook (irradiance and air temperature, hourly) rebuilt from the ERA5 reanalysis.

    python tools/build_pv_history_from_era5.py

The workbook phase_2.../pv_training_data_from_2015.xlsx used to carry the label PLANT_SENSOR for every hour since 2015. That label cannot be literal
(the floating PV pilot started in July 2022), and in 2025 the "measurements" agree with a weather-model forecast to within one W/m2 of spread, so the
rows are of mixed or unverified origin (findings F8 and D2). They are replaced by the ERA5 reanalysis of the Alqueva site (tools/fetch_open_meteo.py
--reanalysis): what the atmosphere did, from a weather model fitted to observations, labelled ERA5_REANALYSIS. It is not an on-site measurement and is
never called one. The days after the end of the reanalysis (it is published with a delay of about five days) come from tools/fetch_open_meteo.py --recent: real weather as it
happened, labelled OPEN_METEO_RECENT, and replaced by the reanalysis the next time this tool runs on a newer reanalysis file. What is still missing
(today) is added by the PV forecaster's own gap fill, labelled SYNTHETIC.
Hours are UTC, numbered 1..24 (hour h covers h-1:00 to h:00), the convention of the workbook and of the weather forecast.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ERA5 = os.path.join(ROOT, "model_development", "plant_inputs", "era5_alqueva.csv.gz")
WORKBOOK = os.path.join(ROOT, "phase_2_da_day_ahead_bidding", "da_price_pv_inflow_forecasting", "pv_training_data_from_2015.xlsx")
SHEET = "PV_Weather_2015_2026"


def main() -> int:
    era = pd.read_csv(ERA5, parse_dates=["datetime_utc"])
    t = era["datetime_utc"].dt.tz_localize(None)
    out = pd.DataFrame({"Date": t.dt.normalize(), "Hour": t.dt.hour + 1, "GHI": era["ghi"].clip(lower=0).round(1), "T_amb": era["tamb"].round(2),
                        "source": "ERA5_REANALYSIS"}).sort_values(["Date", "Hour"]).reset_index(drop=True)
    recent_path = os.path.join(os.path.dirname(ERA5), "open_meteo_recent.csv.gz")
    if os.path.isfile(recent_path):
        rec = pd.read_csv(recent_path, parse_dates=["datetime_utc"])
        t2 = rec["datetime_utc"].dt.tz_localize(None)
        extra = pd.DataFrame({"Date": t2.dt.normalize(), "Hour": t2.dt.hour + 1, "GHI": rec["ghi"].clip(lower=0).round(1), "T_amb": rec["tamb"].round(2),
                              "source": "OPEN_METEO_RECENT"})
        out = pd.concat([out, extra[extra["Date"] > out["Date"].max()]], ignore_index=True).sort_values(["Date", "Hour"]).reset_index(drop=True)
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl", mode="w") as writer:
        out.to_excel(writer, sheet_name=SHEET, index=False)
    print(f"wrote {WORKBOOK}: {len(out):,} rows, {out['Date'].min().date()} to {out['Date'].max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
