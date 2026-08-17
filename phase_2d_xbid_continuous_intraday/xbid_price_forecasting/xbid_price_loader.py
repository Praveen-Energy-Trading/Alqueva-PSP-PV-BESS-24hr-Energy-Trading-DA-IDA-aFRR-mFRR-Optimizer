"""
xbid_price_loader.py — XBID continuous intraday prices.

The delivery-day forecast always comes from the ML spread model
(fetch_xbid_prices / forecast_xbid_prices, trained on IDA3 + OU noise) — no
order book (live or historical) publishes tomorrow's continuous-market price
in advance, live feed or not. What CAN be live is the model's TRAINING data:
update_training_data() below backfills real settled XBID continuous-market
prices for past dates from OMIE's own public per-period min/max/weighted-mean
report, falling back to synthetic on any download failure.

(An earlier version of this docstring claimed no public XBID data exists at
all and that a commercial EPEX SPOT subscription would be required — that
was wrong. OMIE itself publishes a genuine, free, no-auth continuous-market
price summary; see _download_omie_xbid() below, verified against a live
response.)

XBID closes 1 hour before each delivery period; tradable_hours_for_window() computes
the still-open hours for ANY check window's real trigger time (generalised so the
number/timing of windows is fully config-driven — see config/market.yaml
gates.XBID.check_windows).
"""
from __future__ import annotations

import datetime as dt
import os
import random
from typing import Dict, List

import pandas as pd

from common_layer.utilities.logging_utils import get_logger
from common_layer.utilities.timezone_utils import resolve_gate_time

from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster import (
    forecast_da_prices, forecast_da_prices_isp,
)
from phase_2d_xbid_continuous_intraday.xbid_price_forecasting.xbid_price_forecaster import (
    forecast_xbid_prices,
)

log = get_logger(__name__)

_HERE          = os.path.dirname(os.path.abspath(__file__))
_REPO          = os.path.dirname(os.path.dirname(_HERE))
_TRAINING_XLSX = os.path.join(_HERE, "xbid_training_data_2024_2025.xlsx")
_TRAINING_SHEET = "XBID_2024_2025"
_DA_XLSX       = os.path.join(_REPO, "phase_1_da_day_ahead_bidding",
                              "da_price_pv_inflow_forecasting",
                              "da_training_data_2020_2026.xlsx")
_DA_SHEET      = "DA_Price_2020_2026"
_HOURS         = list(range(1, 25))
_N_QUARTERS    = 96


def fetch_xbid_prices(hours: List[int], delivery_date: str, window_id: str,
                      use_synthetic: bool = True) -> Dict[int, float]:
    """XBID proxy prices for a check window (e.g. 'W1'..'W6', see market.yaml).

    Uses the XBID ML spread model as the base, then adds a small per-window drift
    (±1.5 EUR/MWh) to approximate order-book movement between windows.
    """
    if not use_synthetic:
        try:
            update_training_data(delivery_date)
        except Exception as exc:
            log.warning(f"XBID training-data live update failed ({exc}); "
                        f"using existing/cached history.")
    da_prices  = forecast_da_prices(hours, delivery_date)
    base       = forecast_xbid_prices(hours, delivery_date, da_prices)
    rng        = random.Random(f"xbid-{window_id}-{delivery_date}")
    return {h: round(max(1.0, base[h] + rng.uniform(-1.5, 1.5)), 2) for h in hours}


def fetch_xbid_prices_isp(isps: List[int], hours: List[int], day, delivery_date: str,
                          window_id: str, use_synthetic: bool = True) -> Dict[int, float]:
    """XBID proxy prices at real ISP resolution. The XBID spread model
    (trained on IDA3 + OU noise, see module docstring) has no genuine 15-min
    training history -- same honest treatment as the IDA gates: real
    ISP-level DA price, hourly-predicted XBID spread on top of it, applied
    flat within the hour, plus per-window order-book drift now seeded
    per-ISP (a real improvement here, not an approximation -- continuous
    intraday quotes genuinely move faster than once per hour)."""
    if not use_synthetic:
        try:
            update_training_data(delivery_date)
        except Exception as exc:
            log.warning(f"XBID training-data live update failed ({exc}); "
                        f"using existing/cached history.")
    da_hourly = forecast_da_prices(hours, delivery_date)
    base_hourly = forecast_xbid_prices(hours, delivery_date, da_hourly)
    da_isp = forecast_da_prices_isp(isps, delivery_date)

    from common_layer.utilities import date_utils as du
    out: Dict[int, float] = {}
    for h in hours:
        spread = base_hourly.get(h, da_hourly.get(h, 55.0)) - da_hourly.get(h, 55.0)
        for isp in du.hour_to_isps(h, day):
            rng = random.Random(f"xbid-{window_id}-{delivery_date}-{isp}")
            price = da_isp.get(isp, 55.0) + spread + rng.uniform(-1.5, 1.5)
            out[isp] = round(max(1.0, price), 2)
    return out


# ---------------------------------------------------------------------------
# Training-data live update — backfills real settled XBID continuous prices
# ---------------------------------------------------------------------------

def update_training_data(delivery_date: str) -> None:
    """Fill xbid_training_data Excel with real (or synthetic) XBID prices up
    to yesterday. Requires DA training data already current for the same
    dates (run_da's update_training_data runs first in the pipeline)."""
    target_dt = pd.Timestamp(delivery_date)
    yesterday = target_dt - pd.Timedelta(days=1)

    existing  = _load_excel(_TRAINING_XLSX, _TRAINING_SHEET)
    last_date = existing["Date"].max() if not existing.empty else pd.Timestamp("2024-06-12")
    if last_date >= yesterday:
        log.info(f"XBID training data already current up to "
                 f"{last_date.date()} — no update needed")
        return

    missing = pd.date_range(start=last_date + pd.Timedelta(days=1),
                            end=yesterday, freq="D")
    log.info(f"XBID: filling {len(missing)} missing date(s): "
             f"{missing[0].date()} -> {missing[-1].date()}")

    da_df = _load_excel(_DA_XLSX, _DA_SHEET)
    new_rows = []
    for d in missing:
        date_str  = d.strftime("%Y-%m-%d")
        da_prices = _da_prices_for_date(da_df, d)

        try:
            xbid_prices = _download_omie_xbid(d)
            source = "OMIE_LIVE"
            log.info(f"  {date_str} XBID -> OMIE_LIVE downloaded")
        except Exception as exc:
            xbid_prices = _synthetic_xbid_prices(date_str, _HOURS, da_prices)
            source = "SYNTHETIC"
            log.warning(f"  {date_str} XBID -> OMIE failed ({exc}), using SYNTHETIC")

        for h in _HOURS:
            da_p   = da_prices.get(h, 55.0)
            xbid_p = xbid_prices.get(h, da_p)
            new_rows.append({
                "Date"                  : d,
                "Hour"                  : h,
                "price_DA_PT_EUR_MWh"   : da_p,
                "price_XBID_PT_EUR_MWh" : xbid_p,
                "spread_EUR_MWh"        : round(xbid_p - da_p, 2),
                "source"                : source,
            })

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated = updated.sort_values(["Date", "Hour"]).reset_index(drop=True)
    _save_excel(updated, _TRAINING_XLSX, _TRAINING_SHEET)
    log.info(f"XBID Excel updated: {len(missing)} date(s) added, "
             f"last date now {updated['Date'].max().date()}")


def _da_prices_for_date(da_df: pd.DataFrame, d: pd.Timestamp) -> Dict[int, float]:
    day = da_df[da_df["Date"] == d] if not da_df.empty else da_df
    if day.empty:
        return {h: 55.0 for h in _HOURS}
    return dict(zip(day["Hour"], day["price_DA_PT_EUR_MWh"]))


# ---------------------------------------------------------------------------
# OMIE download — continuous-market (XBID) min/max/weighted-mean report
# ---------------------------------------------------------------------------

def _parse_eur_comma(s: str) -> float:
    """Parse a European-format price string, e.g. "1.234,56" or "184,99"."""
    return float(s.replace(".", "").replace(",", "."))


def _download_omie_xbid(d: pd.Timestamp) -> Dict[int, float]:
    """Download and parse OMIE's continuous-intraday (XBID) price report.

    Public, no-auth file: "precios_pibcic_<YYYYMMDD>.1" — 96 quarter-hour
    periods/day, decimal-comma numbers, columns include MedioPT (Portugal
    volume-weighted average price) which we use as the XBID proxy price.
    Averages the 4 quarters within each hour. Raises on any failure.
    """
    import requests
    filename = f"precios_pibcic_{d.strftime('%Y%m%d')}.1"
    url = f"https://www.omie.es/es/file-download?parents=precios_pibcic&filename={filename}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    header_cols = None
    quarters: Dict[int, float] = {}
    for line in resp.text.splitlines():
        parts = [p.strip() for p in line.split(";") if p.strip() != ""]
        if not parts:
            continue
        if parts[0].lower() in ("año", "ańo", "ano"):
            header_cols = parts
            continue
        if header_cols is None or len(parts) < len(header_cols):
            continue
        row = dict(zip(header_cols, parts))
        try:
            period  = int(row["Periodo"])
            medio_pt = _parse_eur_comma(row["MedioPT"])
        except (KeyError, ValueError):
            continue
        quarters[period] = medio_pt

    if len(quarters) < _N_QUARTERS:
        raise ValueError(f"Expected {_N_QUARTERS} quarter-hour periods in "
                         f"{filename}, got {len(quarters)}")

    return {h: round(sum(quarters[(h - 1) * 4 + q] for q in range(1, 5)) / 4.0, 2)
            for h in range(1, 25)}


# ---------------------------------------------------------------------------
# Synthetic fallback (deterministic spread around DA price)
# ---------------------------------------------------------------------------

def _synthetic_xbid_prices(delivery_date: str, hours: List[int],
                           da_prices: Dict[int, float]) -> Dict[int, float]:
    rng = random.Random(f"xbid-hist-{delivery_date}")
    return {
        h: round(max(-600.0, da_prices.get(h, 55.0) + rng.uniform(-14.0, 14.0)), 2)
        for h in hours
    }


# ---------------------------------------------------------------------------
# Excel read / write
# ---------------------------------------------------------------------------

def _load_excel(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _save_excel(df: pd.DataFrame, path: str, sheet: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)


def tradable_hours_for_window(all_periods: List[int], delivery_date: str,
                              window_trigger: str, period_duration_min: int = 60) -> List[int]:
    """Periods still open at a check window (XBID closes 1h before delivery).

    Generic version: resolves `window_trigger` (e.g. "D-1 18:30") to a real
    CET datetime, then compares against each period's real delivery start —
    works for any number of windows at any time, and at any period duration
    (60 min hourly or 15-min real ISP), not just two fixed hourly windows.
    """
    day = dt.datetime.strptime(delivery_date, "%Y-%m-%d").date()
    trigger_dt = resolve_gate_time(window_trigger, day)
    day_start  = dt.datetime.combine(day, dt.time(0, 0), tzinfo=trigger_dt.tzinfo)

    open_periods = []
    for p in all_periods:
        delivery_start = day_start + dt.timedelta(minutes=(p - 1) * period_duration_min)
        if delivery_start - trigger_dt > dt.timedelta(hours=1):
            open_periods.append(p)
    return open_periods
