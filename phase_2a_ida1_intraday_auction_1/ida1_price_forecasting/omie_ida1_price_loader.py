"""
omie_ida1_price_loader.py — IDA1 intraday auction price (D-1 15:00 CET close).

IDA1 clears the whole delivery day (all 24 hours tradable) with fresher
information than DA. The delivery-day forecast itself always comes from the
ML intraday spread model (fetch_ida1_prices / forecast_ida1_prices) — OMIE
never publishes IDA1's own future clearing price. What CAN be live is the
model's TRAINING data: update_training_data() below backfills real settled
IDA1 clearing prices for past dates (same "keep history current" role as
omie_da_price_loader.update_training_data for DA), falling back to synthetic
spreads on any download failure.

Live source (verified against a live OMIE response):
    https://www.omie.es/es/file-download?parents=marginalpibcpt&filename=marginalpibcpt_<YYYYMMDD><SS>.1
    SS = session: 01=IDA1, 02=IDA2, 03=IDA3 (all three share the same
    "marginalpibcpt" directory — Portugal intraday-auction marginal prices).
    Format: "YYYY;MM;DD;PERIOD;price_ES;price_PT;" — 96 quarter-hour periods
    per day, decimal point (not comma), one row per period.

Returned shape: {hour: EUR/MWh} for hours 1..24.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster import (
    forecast_da_prices,
)
from phase_2a_ida1_intraday_auction_1.ida1_price_forecasting.ida1_price_forecaster import (
    forecast_ida1_prices,
)

try:
    from common_layer.utilities.logging_utils import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)

_HERE          = os.path.dirname(os.path.abspath(__file__))
_REPO          = os.path.dirname(os.path.dirname(_HERE))
_TRAINING_XLSX = os.path.join(_HERE, "ida1_training_data_2024_2025.xlsx")
_TRAINING_SHEET = "IDA1_2024_2025"
_DA_XLSX       = os.path.join(_REPO, "phase_1_da_day_ahead_bidding",
                              "da_price_pv_inflow_forecasting",
                              "da_training_data_2020_2026.xlsx")
_DA_SHEET      = "DA_Price_2020_2026"
_SESSION       = "01"
_HOURS         = list(range(1, 25))
_N_QUARTERS    = 96


def fetch_ida1_prices(hours: List[int], delivery_date: str) -> Dict[int, float]:
    """Return {hour: EUR/MWh} IDA1 clearing price forecast.

    DA prices computed first (already cached if run_da ran earlier in session),
    then the IDA1-specific ML spread model adds the intraday deviation
    (gate closes D-1 15:00 CET, covers H1-H24).
    """
    da_prices = forecast_da_prices(hours, delivery_date)
    return forecast_ida1_prices(hours, delivery_date, da_prices)


# ---------------------------------------------------------------------------
# Training-data live update — backfills real settled IDA1 clearing prices
# ---------------------------------------------------------------------------

def update_training_data(delivery_date: str) -> None:
    """Fill ida1_training_data Excel with real (or synthetic) IDA1 prices up
    to yesterday. Requires DA training data already current for the same
    dates (run_da's update_training_data runs first in the pipeline)."""
    target_dt = pd.Timestamp(delivery_date)
    yesterday = target_dt - pd.Timedelta(days=1)

    existing  = _load_excel(_TRAINING_XLSX, _TRAINING_SHEET)
    last_date = existing["Date"].max() if not existing.empty else pd.Timestamp("2024-06-12")
    if last_date >= yesterday:
        log.info(f"IDA1 training data already current up to "
                 f"{last_date.date()} — no update needed")
        return

    missing = pd.date_range(start=last_date + pd.Timedelta(days=1),
                            end=yesterday, freq="D")
    log.info(f"IDA1: filling {len(missing)} missing date(s): "
             f"{missing[0].date()} -> {missing[-1].date()}")

    da_df = _load_excel(_DA_XLSX, _DA_SHEET)
    new_rows = []
    for dt in missing:
        date_str  = dt.strftime("%Y-%m-%d")
        da_prices = _da_prices_for_date(da_df, dt)

        try:
            ida_prices = _download_omie_ida(date_str, _SESSION, _HOURS)
            source = "OMIE_LIVE"
            log.info(f"  {date_str} IDA1 -> OMIE_LIVE downloaded")
        except Exception as exc:
            ida_prices = _synthetic_ida_prices(date_str, _HOURS, da_prices)
            source = "SYNTHETIC"
            log.warning(f"  {date_str} IDA1 -> OMIE failed ({exc}), using SYNTHETIC")

        for h in _HOURS:
            da_p  = da_prices.get(h, 55.0)
            ida_p = ida_prices.get(h, da_p)
            new_rows.append({
                "Date"                 : dt,
                "Hour"                 : h,
                "price_DA_PT_EUR_MWh"  : da_p,
                "price_IDA_PT_EUR_MWh" : ida_p,
                "spread_EUR_MWh"       : round(ida_p - da_p, 2),
                "source"               : source,
            })

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated = updated.sort_values(["Date", "Hour"]).reset_index(drop=True)
    _save_excel(updated, _TRAINING_XLSX, _TRAINING_SHEET)
    log.info(f"IDA1 Excel updated: {len(missing)} date(s) added, "
             f"last date now {updated['Date'].max().date()}")


def _da_prices_for_date(da_df: pd.DataFrame, dt: pd.Timestamp) -> Dict[int, float]:
    day = da_df[da_df["Date"] == dt] if not da_df.empty else da_df
    if day.empty:
        return {h: 55.0 for h in _HOURS}
    return dict(zip(day["Hour"], day["price_DA_PT_EUR_MWh"]))


# ---------------------------------------------------------------------------
# OMIE download (shared file format across IDA1/IDA2/IDA3)
# ---------------------------------------------------------------------------

_MIN_QUARTERS = 8   # sanity floor (2 complete hours) — reject empty/garbage files only


def _download_omie_ida(delivery_date: str, session: str,
                       hours: List[int]) -> Dict[int, float]:
    """Download and parse OMIE's marginalpibcpt intraday-auction file.

    File: up to 96 rows "YYYY;MM;DD;PERIOD;price_ES;price_PT;" (decimal
    point, quarter-hour periods 1-96). Averages the 4 quarters within each
    hour. Builds the hourly dict from whichever hours have a complete set of
    4 quarters rather than requiring all 96 up front (IDA3's real file only
    ever contains periods 49-96 — see omie_ida3_price_loader.py for details;
    IDA1/IDA2 normally do have the full 96, but this stays tolerant of any
    partial file rather than needlessly failing). Raises only if the file is
    empty/near-empty (network issue, wrong filename, garbage response).
    """
    import requests
    yyyymmdd = delivery_date.replace("-", "")
    filename = f"marginalpibcpt_{yyyymmdd}{session}.1"
    url = f"https://www.omie.es/es/file-download?parents=marginalpibcpt&filename={filename}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    quarter_prices: Dict[int, float] = {}
    for line in resp.text.splitlines():
        parts = [p.strip() for p in line.split(";") if p.strip() != ""]
        if len(parts) < 6:
            continue
        try:
            period   = int(parts[3])
            price_pt = float(parts[5])
        except ValueError:
            continue
        quarter_prices[period] = price_pt

    if len(quarter_prices) < _MIN_QUARTERS:
        raise ValueError(
            f"Too few quarter-hour periods in {filename}: {len(quarter_prices)} "
            f"(need at least {_MIN_QUARTERS})"
        )

    hourly = {}
    for h in range(1, 25):
        qs = [quarter_prices.get((h - 1) * 4 + q) for q in range(1, 5)]
        if all(q is not None for q in qs):
            hourly[h] = round(sum(qs) / 4.0, 2)

    return {h: hourly[h] for h in hours if h in hourly}


# ---------------------------------------------------------------------------
# Synthetic fallback (deterministic spread around DA price)
# ---------------------------------------------------------------------------

def _synthetic_ida_prices(delivery_date: str, hours: List[int],
                          da_prices: Dict[int, float]) -> Dict[int, float]:
    import random
    rng = random.Random(f"ida1-{delivery_date}")
    return {
        h: round(max(-600.0, da_prices.get(h, 55.0) + rng.uniform(-8.0, 8.0)), 2)
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
