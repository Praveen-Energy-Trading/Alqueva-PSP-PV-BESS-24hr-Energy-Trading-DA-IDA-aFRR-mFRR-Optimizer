"""
historical_data_loader.py — dates and 'actual' realisations for backtesting.

Offline we replay a span of delivery days. For forecast validation and for
computing a genuinely realised P&L, we need the REALISED series to compare
against (and re-settle) the forecast the DA gate actually bid on.

Two sources, used deliberately differently -- never conflated:
  - DA price: real_omie_price() reads genuine archived OMIE prices (the same
    cache omie_da_price_loader.py already maintains). Real data exists for
    any date up to "yesterday" relative to when this cache was last
    refreshed; a genuinely future date (or an unrefreshed gap) has no real
    row and real_omie_price() returns None rather than padding with a
    synthetic value silently mixed in.
  - Everything else (PV) has no real-data source anywhere in this project
    (no generation-telemetry loader exists) and stays on
    realised_from_forecast()'s synthetic perturbation, labelled as such by
    every caller.
"""
from __future__ import annotations

import datetime as dt
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

_FORECASTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "phase_1_da_day_ahead_bidding", "da_price_pv_inflow_forecasting")

# The DA gate (_assemble_inputs in run_da.py) always builds inputs at real
# ISP (15-min) resolution -- MIBEL's actual settlement granularity since
# 2025-10-01 -- so real-price lookup must match that resolution, not the
# legacy 24-hourly file. Real quarter-hour OMIE data (confirmed genuinely
# varying within each hour, not a flat hourly-expanded placeholder) is
# cached under two source labels from different code paths -- both real.
_OMIE_ISP_EXCEL_PATH = os.path.join(_FORECASTER_DIR, "da_training_data_isp_2025_2026.xlsx")
_OMIE_ISP_SHEET = "DA_Price_ISP_2025_2026"
_REAL_ISP_SOURCES = {"OMIE_LIVE", "OMIE_ISP_LIVE"}

_omie_isp_cache: Optional[pd.DataFrame] = None

# Reserve capacity: aFRR/mFRR training caches are hourly (1-24), maintained
# by picasso_afrr_price_loader.py / mari_mfrr_price_loader.py respectively.
# Only the "REN_LIVE" rows are genuine archived REN capacity prices; all
# other rows in these files are SYNTHETIC placeholders for un-backfilled
# dates, exactly like the OMIE cache above.
_AFRR_EXCEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "phase_3a_afrr_automatic_frequency_reserve", "afrr_price_forecasting",
    "afrr_training_data_2019_2025.xlsx")
_AFRR_SHEET = "AFRR_2019_2025"

_MFRR_EXCEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "phase_3b_mfrr_manual_frequency_reserve", "mfrr_price_forecasting",
    "mfrr_training_data_2024_2025.xlsx")
_MFRR_SHEET = "MFRR_2024_2025"

_REAL_RESERVE_SOURCE = "REN_LIVE"

_afrr_cache: Optional[pd.DataFrame] = None
_mfrr_cache: Optional[pd.DataFrame] = None

# Intraday auction price archives: hourly (1-24), maintained by each gate's
# own omie_ida{1,2,3}_price_loader.py / xbid_price_loader.py. IDA1/IDA2/
# IDA3/XBID all wired for real-price backtesting. IDA3 has only 7 real
# archived dates today (real coverage started 2026-08-15) -- genuinely
# thin, reported honestly as such rather than padded.
_IDA_PRICE_PATH: Dict[str, Tuple[str, str, str]] = {
    "IDA1": ("phase_2a_ida1_intraday_auction_1", "ida1_price_forecasting",
             "ida1_training_data_2024_2025.xlsx"),
    "IDA2": ("phase_2b_ida2_intraday_auction_2", "ida2_price_forecasting",
             "ida2_training_data_2024_2025.xlsx"),
    "IDA3": ("phase_2c_ida3_intraday_auction_3", "ida3_price_forecasting",
             "ida3_training_data_2024_2025.xlsx"),
    "XBID": ("phase_2d_xbid_continuous_intraday", "xbid_price_forecasting",
             "xbid_training_data_2024_2025.xlsx"),
}
_IDA_PRICE_COLUMN: Dict[str, str] = {
    "IDA1": "price_IDA_PT_EUR_MWh",
    "IDA2": "price_IDA_PT_EUR_MWh",
    "IDA3": "price_IDA_PT_EUR_MWh",
    "XBID": "price_XBID_PT_EUR_MWh",
}
_IDA_SHEET = None  # these workbooks use the default (first/only) sheet

_ida_price_cache: Dict[str, pd.DataFrame] = {}


def date_range(start_date: str, n_days: int) -> List[str]:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    return [(start + dt.timedelta(days=i)).isoformat() for i in range(n_days)]


def realised_from_forecast(forecast: Dict[int, float], delivery_date: str,
                           rel_error: float = 0.10, tag: str = "px") -> Dict[int, float]:
    """Synthetic 'actual' = forecast * (1 +/- error). Deterministic per date."""
    rng = random.Random(f"{tag}-actual-{delivery_date}")
    return {h: round(v * (1.0 + rng.uniform(-rel_error, rel_error)), 4)
            for h, v in forecast.items()}


def real_omie_price(delivery_date: str, isps: List[int]) -> Optional[Dict[int, float]]:
    """Return genuine archived OMIE DA prices at ISP (15-min) resolution for
    delivery_date, or None.

    `isps` must be the actual ISP indices the caller is asking about
    (typically 1..96, matching _assemble_inputs' inputs["da_prices"].keys()
    -- the DA gate always builds inputs at real ISP resolution, not hourly,
    since MIBEL's settlement granularity moved to 15-min on 2025-10-01).

    Reads the same cached workbook omie_da_price_loader.py's
    update_isp_training_data() maintains (da_training_data_isp_2025_2026.xlsx)
    -- no new data source. Returns a result ONLY if every requested ISP has
    a real row (source in _REAL_ISP_SOURCES); otherwise returns None. Never
    returns a partial result silently padded with synthetic values --
    callers must treat None as "no real data for this date" (a genuinely
    future date, a date before the 2025-10-01 real-ISP-data start, or a
    real gap not yet backfilled), not as something to paper over.
    """
    global _omie_isp_cache
    if _omie_isp_cache is None:
        if not os.path.isfile(_OMIE_ISP_EXCEL_PATH):
            return None
        _omie_isp_cache = pd.read_excel(_OMIE_ISP_EXCEL_PATH, sheet_name=_OMIE_ISP_SHEET)

    target = pd.Timestamp(delivery_date)
    df = _omie_isp_cache
    rows = df[(df["Date"] == target) & (df["source"].isin(_REAL_ISP_SOURCES))]
    if rows.empty:
        return None

    by_isp = dict(zip(rows["ISP"].astype(int), rows["price_DA_PT_EUR_MWh"].astype(float)))
    if not all(i in by_isp for i in isps):
        return None   # partial real coverage for this date -- refuse, don't pad

    return {i: by_isp[i] for i in isps}


def real_ren_capacity_price(
    delivery_date: str, hours: List[int], product: str,
) -> Optional[Tuple[Dict[int, float], Dict[int, float]]]:
    """Return genuine archived REN reserve-capacity prices (cap_up, cap_dn) at
    hourly resolution for delivery_date, or None.

    `product`: "aFRR" reads the cache picasso_afrr_price_loader.py maintains
    (afrr_training_data_2019_2025.xlsx); "mFRR" reads the cache
    mari_mfrr_price_loader.py maintains (mfrr_training_data_2024_2025.xlsx).
    No new data source -- same caches the live forecasters already use.

    Same all-or-nothing rule as real_omie_price(): returns a result only if
    every requested hour has a genuine REN_LIVE row for BOTH cap_up and
    cap_dn; otherwise None. Never returns a partially-real result.
    """
    global _afrr_cache, _mfrr_cache
    if product == "aFRR":
        path, sheet = _AFRR_EXCEL_PATH, _AFRR_SHEET
    elif product == "mFRR":
        path, sheet = _MFRR_EXCEL_PATH, _MFRR_SHEET
    else:
        raise ValueError(f"unknown reserve product: {product!r}")

    if product == "aFRR":
        if _afrr_cache is None:
            if not os.path.isfile(path):
                return None
            _afrr_cache = pd.read_excel(path, sheet_name=sheet)
        df = _afrr_cache
    else:
        if _mfrr_cache is None:
            if not os.path.isfile(path):
                return None
            _mfrr_cache = pd.read_excel(path, sheet_name=sheet)
        df = _mfrr_cache

    target = pd.Timestamp(delivery_date)
    rows = df[(df["Date"] == target) & (df["source"] == _REAL_RESERVE_SOURCE)]
    if rows.empty:
        return None

    by_hour_up = dict(zip(rows["Hour"].astype(int), rows["cap_up_EUR_MW"].astype(float)))
    by_hour_dn = dict(zip(rows["Hour"].astype(int), rows["cap_dn_EUR_MW"].astype(float)))
    if not all(h in by_hour_up and h in by_hour_dn for h in hours):
        return None   # partial real coverage for this date -- refuse, don't pad

    return ({h: by_hour_up[h] for h in hours}, {h: by_hour_dn[h] for h in hours})


def real_ida_price(gate: str, delivery_date: str, hours: List[int]) -> Optional[Dict[int, float]]:
    """Return genuine archived intraday-auction clearing prices at hourly
    resolution for delivery_date, or None.

    `gate`: "IDA1" (203 real archived dates), "IDA2" (225 real dates),
    "IDA3" (only 7 real dates -- real coverage started 2026-08-15, a
    genuinely thin sample, not padded), or "XBID" (227 real dates,
    continuous-intraday proxy price).

    Reads the same cached workbook each gate's own omie_ida{n}_price_loader.py
    maintains -- no new data source. `hours` are real clock hours (1-24);
    each gate's spread model is trained at hourly resolution only (no real
    15-min IDA spread history exists to retrain on -- same documented
    limitation as ida_reoptimiser.py's _get_intraday_prices_isp), so this
    does not attempt to return ISP resolution -- callers expand hour->ISP
    flat themselves, same honesty convention already used elsewhere in this
    project for this same real-data resolution limit.

    Same all-or-nothing rule as real_omie_price(): a result only if every
    requested hour has a genuine OMIE_LIVE row; otherwise None.
    """
    if gate not in _IDA_PRICE_PATH:
        raise ValueError(f"real_ida_price: no price archive configured for gate {gate!r}")

    if gate not in _ida_price_cache:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            *_IDA_PRICE_PATH[gate])
        if not os.path.isfile(path):
            return None
        _ida_price_cache[gate] = pd.read_excel(path)

    df = _ida_price_cache[gate]
    price_col = _IDA_PRICE_COLUMN[gate]
    target = pd.Timestamp(delivery_date)
    rows = df[(df["Date"] == target) & (df["source"] == "OMIE_LIVE")]
    if rows.empty:
        return None

    by_hour = dict(zip(rows["Hour"].astype(int), rows[price_col].astype(float)))
    if not all(h in by_hour for h in hours):
        return None   # partial real coverage for this date -- refuse, don't pad

    return {h: by_hour[h] for h in hours}
