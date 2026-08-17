"""
mari_mfrr_price_loader.py — mFRR capacity prices (MARI platform).

REN joined the European mFRR platform MARI on 27 Nov 2024. The delivery-day
forecast always comes from the ML cap-price forecaster (fetch_mfrr_cap_prices
/ forecast_mfrr_cap_prices) — REN never publishes tomorrow's mFRR price any
more than OMIE publishes tomorrow's DA price. What CAN be live is the model's
TRAINING data: update_training_data() below backfills real settled mFRR
prices for past dates from REN's own public market-info site, falling back
to synthetic on any download failure.

mFRR is forecast independently of aFRR — the two markets (MARI vs PICASSO) have
separate supply/demand dynamics and prices can diverge significantly.

Live source (verified against a live REN response — endpoint and auth
extracted from REN's own public "mFRR Preco" SharePoint web part JS bundle,
loaded from
https://mercado.ren.pt/PT/Electr/InfoMercado/InfSistema/Energia-Reserva/mFRR/Paginas/mFRR-Preco.aspx):
    GET https://mercadoservices.ren.pt/api/MFRRPreco/GetMFRRPreco
        ?language=PT&dayQuery=<D>&monthQuery=<M>&yearQuery=<Y>&sWhere=
    Header: X-ApiKey: base64("mercado_mL273BtiLeRcqfqBqImWBf5uvPTmdW4VHxb4EeD6")
        (same static public key as the aFRR endpoint — see
        picasso_afrr_price_loader.py's module docstring.)
    Response: JSON array, 96 quarter-hour periods/day, fields PERIODO,
    HORA_INI_FIM, AP_PRECO (single activation price — REN does not publish a
    separate mFRR up/down capacity split via this endpoint; AD_PRECO_Q0/Q1_
    SUBIR/DESCER are present but observed null on every date checked), and
    PRECO_MER_DIARIO (DA reference price). We use AP_PRECO as a documented
    proxy for BOTH cap_up and cap_dn — the same "best available proxy"
    pattern the pipeline already uses for XBID settlement (see
    phase_5a_da_ida_settlement/.../omie_settlement_price_loader.py).
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

from common_layer.configuration.config_loader import AppConfig
from common_layer.utilities.logging_utils import get_logger
from phase_3b_mfrr_manual_frequency_reserve.mfrr_price_forecasting.mfrr_price_forecaster import (
    forecast_mfrr_cap_prices,
)

log = get_logger(__name__)

_HERE          = os.path.dirname(os.path.abspath(__file__))
_REPO          = os.path.dirname(os.path.dirname(_HERE))
_TRAINING_XLSX = os.path.join(_HERE, "mfrr_training_data_2024_2025.xlsx")
_TRAINING_SHEET = "MFRR_2024_2025"
_DA_XLSX       = os.path.join(_REPO, "phase_1_da_day_ahead_bidding",
                              "da_price_pv_inflow_forecasting",
                              "da_training_data_2020_2026.xlsx")
_DA_SHEET      = "DA_Price_2020_2026"
_HOURS         = list(range(1, 25))
_N_QUARTERS    = 96
_REN_API_KEY_B64 = "bWVyY2Fkb19tTDI3M0J0aUxlUmNxZnFCcUltV0JmNXV2UFRtZFc0Vkh4YjRFZUQ2"


def fetch_mfrr_cap_prices(hours: List[int], delivery_date: str, cfg: AppConfig,
                          use_synthetic: bool = True
                          ) -> Tuple[Dict[int, float], Dict[int, float], str]:
    if not use_synthetic:
        try:
            update_training_data(delivery_date)
        except Exception as exc:
            log.warning(f"mFRR training-data live update failed ({exc}); "
                        f"using existing/cached history.")
    cap_max = cfg.market.afrr.cap_price_max_eur_mw  # mFRR shares the same REN 250 EUR/MW ceiling; no separate mFRR config field
    cap_up, cap_dn = forecast_mfrr_cap_prices(hours, delivery_date, cap_max)
    return cap_up, cap_dn, "MARI_ML_FORECAST"


# ---------------------------------------------------------------------------
# Training-data live update — backfills real settled mFRR prices
# ---------------------------------------------------------------------------

def update_training_data(delivery_date: str) -> None:
    """Fill mfrr_training_data Excel with real (or synthetic) mFRR prices up
    to yesterday. Requires DA training data already current for the same
    dates (run_da's update_training_data runs first in the pipeline)."""
    target_dt = pd.Timestamp(delivery_date)
    yesterday = target_dt - pd.Timedelta(days=1)

    existing  = _load_excel(_TRAINING_XLSX, _TRAINING_SHEET)
    last_date = existing["Date"].max() if not existing.empty else pd.Timestamp("2024-11-26")
    if last_date >= yesterday:
        log.info(f"mFRR training data already current up to "
                 f"{last_date.date()} — no update needed")
        return

    missing = pd.date_range(start=last_date + pd.Timedelta(days=1),
                            end=yesterday, freq="D")
    log.info(f"mFRR: filling {len(missing)} missing date(s): "
             f"{missing[0].date()} -> {missing[-1].date()}")

    da_df = _load_excel(_DA_XLSX, _DA_SHEET)
    new_rows = []
    for dt in missing:
        date_str  = dt.strftime("%Y-%m-%d")
        da_prices = _da_prices_for_date(da_df, dt)

        try:
            price = _download_ren_mfrr(dt)
            source = "REN_LIVE"
            log.info(f"  {date_str} mFRR -> REN_LIVE downloaded")
        except Exception as exc:
            price = _synthetic_mfrr_prices(date_str, _HOURS)
            source = "SYNTHETIC"
            log.warning(f"  {date_str} mFRR -> REN failed ({exc}), using SYNTHETIC")

        for h in _HOURS:
            p = price.get(h, 10.0)
            new_rows.append({
                "Date"                : dt,
                "Hour"                : h,
                "price_DA_PT_EUR_MWh" : da_prices.get(h, 55.0),
                "cap_up_EUR_MW"       : p,
                "cap_dn_EUR_MW"       : p,
                "source"              : source,
            })

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated = updated.sort_values(["Date", "Hour"]).reset_index(drop=True)
    _save_excel(updated, _TRAINING_XLSX, _TRAINING_SHEET)
    log.info(f"mFRR Excel updated: {len(missing)} date(s) added, "
             f"last date now {updated['Date'].max().date()}")


def _da_prices_for_date(da_df: pd.DataFrame, dt: pd.Timestamp) -> Dict[int, float]:
    day = da_df[da_df["Date"] == dt] if not da_df.empty else da_df
    if day.empty:
        return {h: 55.0 for h in _HOURS}
    return dict(zip(day["Hour"], day["price_DA_PT_EUR_MWh"]))


# ---------------------------------------------------------------------------
# REN download
# ---------------------------------------------------------------------------

def _download_ren_mfrr(dt: pd.Timestamp) -> Dict[int, float]:
    """Download and parse REN's mFRR price API for one delivery date.

    Averages the 4 quarter-hour periods within each hour. Raises on any
    failure (network, missing/short response, unexpected format).
    """
    import requests
    url = (f"https://mercadoservices.ren.pt/api/MFRRPreco/GetMFRRPreco"
           f"?language=PT&dayQuery={dt.day}&monthQuery={dt.month}&yearQuery={dt.year}&sWhere=")
    resp = requests.get(url, headers={"X-ApiKey": _REN_API_KEY_B64,
                                       "Accept": "application/json"}, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    if isinstance(rows, str):
        import json as _json
        rows = _json.loads(rows)   # REN's API double-encodes the JSON body
    if not isinstance(rows, list) or len(rows) < _N_QUARTERS:
        raise ValueError(f"Expected {_N_QUARTERS} quarter-hour rows, got "
                          f"{len(rows) if isinstance(rows, list) else type(rows)}")

    quarters: Dict[int, float] = {}
    for r in rows:
        period = int(r["PERIODO"])
        p = r.get("AP_PRECO")
        if p in (None, ""):
            continue
        quarters[period] = float(p)

    if len(quarters) < _N_QUARTERS:
        raise ValueError(f"Incomplete mFRR price data: {len(quarters)} periods "
                          f"(need {_N_QUARTERS})")

    return {h: round(sum(quarters[(h - 1) * 4 + q] for q in range(1, 5)) / 4.0, 2)
            for h in range(1, 25)}


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _synthetic_mfrr_prices(delivery_date: str, hours: List[int]) -> Dict[int, float]:
    import random
    rng = random.Random(f"mfrr-{delivery_date}")
    return {h: round(max(0.0, rng.gauss(10.0, 5.0)), 2) for h in hours}


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
