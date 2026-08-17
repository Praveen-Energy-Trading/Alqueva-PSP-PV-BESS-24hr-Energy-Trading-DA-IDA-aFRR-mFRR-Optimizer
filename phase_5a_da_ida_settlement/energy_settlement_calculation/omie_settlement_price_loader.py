"""
omie_settlement_price_loader.py — final cleared prices for energy settlement.

Settlement values each gate's traded volume at that gate's CLEARED price (a
price-taker is settled at the marginal price, not at its bid). Settlement
always looks BACKWARD in time — by the time Phase 5A runs, the real cleared
price for that delivery date either has already been live-fetched into the
training Excels (source == "OMIE_LIVE", see each gate's
omie_*_price_loader.update_training_data) or is fetchable now. There is never
a legitimate reason to re-run the ML *forecaster* for settlement: the
forecaster's job is bidding under genuine uncertainty about the future;
settlement's job is accounting for something that already happened and can
be looked up.

fetch_settlement_prices() therefore tries the real settled price first (all
requested hours must have source == "OMIE_LIVE" for that date — a partial
match falls through to the forecaster rather than mixing real and predicted
prices within the same day) and only falls back to the forecaster when the
real price genuinely isn't available yet (e.g. a synthetic-only historical
run, or OMIE hasn't published for a very recent date).

XBID settles at its own real cleared-price proxy (OMIE's continuous-market
MedioPT, see xbid_price_loader.py) when available, falling back to the IDA3
price as the best available proxy — the same "reuse the nearest real price"
logic, generalised.

Returns {hour: EUR/MWh} for the requested gate.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster import (
    forecast_da_prices,
)
from phase_2a_ida1_intraday_auction_1.ida1_price_forecasting.ida1_price_forecaster import (
    forecast_ida1_prices,
)
from phase_2b_ida2_intraday_auction_2.ida2_price_forecasting.ida2_price_forecaster import (
    forecast_ida2_prices,
)
from phase_2c_ida3_intraday_auction_3.ida3_price_forecasting.ida3_price_forecaster import (
    forecast_ida3_prices,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

# (excel_path, sheet, price_column) per gate — same training Excels the live
# loaders already backfill with real OMIE_LIVE prices.
_REAL_PRICE_SOURCES = {
    "DA": (
        os.path.join(_REPO, "phase_1_da_day_ahead_bidding",
                     "da_price_pv_inflow_forecasting", "da_training_data_2020_2026.xlsx"),
        "DA_Price_2020_2026", "price_DA_PT_EUR_MWh",
    ),
    "IDA1": (
        os.path.join(_REPO, "phase_2a_ida1_intraday_auction_1",
                     "ida1_price_forecasting", "ida1_training_data_2024_2025.xlsx"),
        "IDA1_2024_2025", "price_IDA_PT_EUR_MWh",
    ),
    "IDA2": (
        os.path.join(_REPO, "phase_2b_ida2_intraday_auction_2",
                     "ida2_price_forecasting", "ida2_training_data_2024_2025.xlsx"),
        "IDA2_2024_2025", "price_IDA_PT_EUR_MWh",
    ),
    "IDA3": (
        os.path.join(_REPO, "phase_2c_ida3_intraday_auction_3",
                     "ida3_price_forecasting", "ida3_training_data_2024_2025.xlsx"),
        "IDA3_2024_2025", "price_IDA_PT_EUR_MWh",
    ),
    "XBID": (
        os.path.join(_REPO, "phase_2d_xbid_continuous_intraday",
                     "xbid_price_forecasting", "xbid_training_data_2024_2025.xlsx"),
        "XBID_2024_2025", "price_XBID_PT_EUR_MWh",
    ),
}


def fetch_settlement_prices(delivery_date: str, gate: str,
                            hours: List[int]) -> Dict[int, float]:
    real = _real_settled_prices(delivery_date, gate, hours)
    if real is not None:
        return real
    return _forecast_settlement_prices(delivery_date, gate, hours)


def _real_settled_prices(delivery_date: str, gate: str,
                         hours: List[int]) -> Optional[Dict[int, float]]:
    """Return the real OMIE_LIVE cleared price for every requested hour, or
    None if any hour is missing / not backed by real data (XBID additionally
    falls back to IDA3's real price before giving up)."""
    source = _REAL_PRICE_SOURCES.get(gate)
    if source is None:
        return None
    prices = _load_real_prices(delivery_date, *source)
    if prices is not None and all(h in prices for h in hours):
        return {h: prices[h] for h in hours}
    if gate == "XBID":
        ida3_source = _REAL_PRICE_SOURCES["IDA3"]
        prices = _load_real_prices(delivery_date, *ida3_source)
        if prices is not None and all(h in prices for h in hours):
            return {h: prices[h] for h in hours}
    return None


def _load_real_prices(delivery_date: str, excel_path: str, sheet: str,
                      price_col: str) -> Optional[Dict[int, float]]:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet)
    except FileNotFoundError:
        return None
    df.columns = [c.strip() for c in df.columns]
    if "source" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    day = df[(df["Date"] == pd.Timestamp(delivery_date)) & (df["source"] == "OMIE_LIVE")]
    if day.empty:
        return None
    return dict(zip(day["Hour"].astype(int), day[price_col].astype(float)))


def _forecast_settlement_prices(delivery_date: str, gate: str,
                                hours: List[int]) -> Dict[int, float]:
    if gate == "DA":
        return forecast_da_prices(hours, delivery_date)
    # IDA and XBID prices are modelled relative to the DA price, so every IDA
    # forecaster takes the DA prices as its base (da_prices is a required arg).
    da_prices = forecast_da_prices(hours, delivery_date)
    if gate == "IDA1":
        return forecast_ida1_prices(hours, delivery_date, da_prices)
    if gate == "IDA2":
        return forecast_ida2_prices(hours, delivery_date, da_prices)
    if gate == "IDA3":
        return forecast_ida3_prices(hours, delivery_date, da_prices)
    if gate == "XBID":
        # XBID settles at the IDA3 price (best available proxy for continuous clearing)
        return forecast_ida3_prices(hours, delivery_date, da_prices)
    raise ValueError(f"Unknown settlement gate {gate!r}")
