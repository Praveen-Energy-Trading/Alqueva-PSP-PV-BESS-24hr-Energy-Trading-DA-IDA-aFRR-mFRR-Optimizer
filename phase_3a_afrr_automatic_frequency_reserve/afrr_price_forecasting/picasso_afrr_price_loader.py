"""
picasso_afrr_price_loader.py — aFRR capacity prices (national platform / PICASSO).

As of 2026 Portugal runs aFRR on a national platform; REN's PICASSO accession is
expected ~Q3 2026 (config afrr.platform). The delivery-day forecast always comes
from the ML cap-price forecaster (fetch_afrr_cap_prices / forecast_afrr_cap_prices)
— REN never publishes tomorrow's aFRR band price any more than OMIE publishes
tomorrow's DA price. What CAN be live is the model's TRAINING data:
update_training_data() below backfills real settled aFRR band prices for past
dates from REN's own public market-info site, falling back to synthetic on any
download failure. Prices bounded by the REN cap-price ceiling
(config afrr.cap_price_max_eur_mw = 250 EUR/MW).

Live source (verified against a live REN response — endpoint and auth extracted
from REN's own public "BaFRR Preco" SharePoint web part JS bundle, loaded from
https://mercado.ren.pt/PT/Electr/InfoMercado/InfSistema/Bandas/Banda_aFRR/Paginas/bafrr-preco.aspx):
    GET https://mercadoservices.ren.pt/api/BaFRRPreco/GetBaFRRPreco
        ?language=PT&dayQuery=<D>&monthQuery=<M>&yearQuery=<Y>&sWhere=
    Header: X-ApiKey: base64("mercado_mL273BtiLeRcqfqBqImWBf5uvPTmdW4VHxb4EeD6")
        (a static public app key REN's own frontend embeds — not a personal
        credential; same key used by REN's mFRR price page.)
    Response: JSON array, 96 quarter-hour periods/day, fields PERIODO,
    HORA_INI_FIM, PRECO_INI_SUB/PRECO_INI_DES (initial up/down €/MW),
    PRECO_AJUST_SUB/PRECO_AJUST_DES (adjusted, preferred when present),
    MERCADO_DIARIO_PT (DA reference price).

Returns (cap_up, cap_dn) each {hour: EUR/MW} plus the platform label.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

from common_layer.configuration.config_loader import AppConfig
from common_layer.utilities.logging_utils import get_logger
from phase_3a_afrr_automatic_frequency_reserve.afrr_price_forecasting.afrr_price_forecaster import (
    forecast_afrr_cap_prices,
)

log = get_logger(__name__)

_HERE          = os.path.dirname(os.path.abspath(__file__))
_REPO          = os.path.dirname(os.path.dirname(_HERE))
_TRAINING_XLSX = os.path.join(_HERE, "afrr_training_data_2019_2025.xlsx")
_TRAINING_SHEET = "AFRR_2019_2025"
_DA_XLSX       = os.path.join(_REPO, "phase_1_da_day_ahead_bidding",
                              "da_price_pv_inflow_forecasting",
                              "da_training_data_2020_2026.xlsx")
_DA_SHEET      = "DA_Price_2020_2026"
_HOURS         = list(range(1, 25))
_N_QUARTERS    = 96
_REN_API_KEY_B64 = "bWVyY2Fkb19tTDI3M0J0aUxlUmNxZnFCcUltV0JmNXV2UFRtZFc0Vkh4YjRFZUQ2"


def fetch_afrr_cap_prices(hours: List[int], delivery_date: str, cfg: AppConfig,
                          use_synthetic: bool = True
                          ) -> Tuple[Dict[int, float], Dict[int, float], str]:
    platform = cfg.market.afrr.platform
    cap_max  = cfg.market.afrr.cap_price_max_eur_mw
    if not use_synthetic:
        try:
            update_training_data(delivery_date)
        except Exception as exc:
            log.warning(f"aFRR training-data live update failed ({exc}); "
                        f"using existing/cached history.")
    cap_up, cap_dn = forecast_afrr_cap_prices(hours, delivery_date, cap_max)
    return cap_up, cap_dn, f"{platform}_ML_FORECAST"


# ---------------------------------------------------------------------------
# Training-data live update — backfills real settled aFRR band prices
# ---------------------------------------------------------------------------

def update_training_data(delivery_date: str) -> None:
    """Fill afrr_training_data Excel with real (or synthetic) aFRR cap prices
    up to yesterday. Requires DA training data already current for the same
    dates (run_da's update_training_data runs first in the pipeline)."""
    target_dt = pd.Timestamp(delivery_date)
    yesterday = target_dt - pd.Timedelta(days=1)

    existing  = _load_excel(_TRAINING_XLSX, _TRAINING_SHEET)
    last_date = existing["Date"].max() if not existing.empty else pd.Timestamp("2018-12-31")
    if last_date >= yesterday:
        log.info(f"aFRR training data already current up to "
                 f"{last_date.date()} — no update needed")
        return

    missing = pd.date_range(start=last_date + pd.Timedelta(days=1),
                            end=yesterday, freq="D")
    log.info(f"aFRR: filling {len(missing)} missing date(s): "
             f"{missing[0].date()} -> {missing[-1].date()}")

    da_df = _load_excel(_DA_XLSX, _DA_SHEET)
    new_rows = []
    for dt in missing:
        date_str  = dt.strftime("%Y-%m-%d")
        da_prices = _da_prices_for_date(da_df, dt)

        try:
            cap_up, cap_dn = _download_ren_afrr(dt)
            source = "REN_LIVE"
            log.info(f"  {date_str} aFRR -> REN_LIVE downloaded")
        except Exception as exc:
            cap_up, cap_dn = _synthetic_afrr_prices(date_str, _HOURS)
            source = "SYNTHETIC"
            log.warning(f"  {date_str} aFRR -> REN failed ({exc}), using SYNTHETIC")

        for h in _HOURS:
            new_rows.append({
                "Date"                : dt,
                "Hour"                : h,
                "price_DA_PT_EUR_MWh" : da_prices.get(h, 55.0),
                "cap_up_EUR_MW"       : cap_up.get(h, 0.0),
                "cap_dn_EUR_MW"       : cap_dn.get(h, 0.0),
                "source"              : source,
            })

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated = updated.sort_values(["Date", "Hour"]).reset_index(drop=True)
    _save_excel(updated, _TRAINING_XLSX, _TRAINING_SHEET)
    log.info(f"aFRR Excel updated: {len(missing)} date(s) added, "
             f"last date now {updated['Date'].max().date()}")


def _da_prices_for_date(da_df: pd.DataFrame, dt: pd.Timestamp) -> Dict[int, float]:
    day = da_df[da_df["Date"] == dt] if not da_df.empty else da_df
    if day.empty:
        return {h: 55.0 for h in _HOURS}
    return dict(zip(day["Hour"], day["price_DA_PT_EUR_MWh"]))


# ---------------------------------------------------------------------------
# REN download
# ---------------------------------------------------------------------------

def _download_ren_afrr(dt: pd.Timestamp) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Download and parse REN's aFRR band-price API for one delivery date.

    Averages the 4 quarter-hour periods within each hour. Prefers the
    'Ajustado' (adjusted) price over 'Inicial' when present. Raises on any
    failure (network, missing/short response, unexpected format).
    """
    import requests
    url = (f"https://mercadoservices.ren.pt/api/BaFRRPreco/GetBaFRRPreco"
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

    def _num(v):
        return float(v) if v not in (None, "") else None

    quarters_up: Dict[int, float] = {}
    quarters_dn: Dict[int, float] = {}
    for r in rows:
        period = int(r["PERIODO"])
        up = _num(r.get("PRECO_AJUST_SUB")) or _num(r.get("PRECO_INI_SUB"))
        dn = _num(r.get("PRECO_AJUST_DES")) or _num(r.get("PRECO_INI_DES"))
        if up is None or dn is None:
            continue
        quarters_up[period] = up
        quarters_dn[period] = dn

    if len(quarters_up) < _N_QUARTERS or len(quarters_dn) < _N_QUARTERS:
        raise ValueError(f"Incomplete aFRR price data: {len(quarters_up)} up, "
                          f"{len(quarters_dn)} dn periods (need {_N_QUARTERS})")

    cap_up = {h: round(sum(quarters_up[(h - 1) * 4 + q] for q in range(1, 5)) / 4.0, 2)
              for h in range(1, 25)}
    cap_dn = {h: round(sum(quarters_dn[(h - 1) * 4 + q] for q in range(1, 5)) / 4.0, 2)
              for h in range(1, 25)}
    return cap_up, cap_dn


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _synthetic_afrr_prices(delivery_date: str,
                           hours: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    import random
    rng = random.Random(f"afrr-{delivery_date}")
    cap_up = {h: round(max(0.0, rng.gauss(25.0, 10.0)), 2) for h in hours}
    cap_dn = {h: round(max(0.0, rng.gauss(12.0, 6.0)), 2) for h in hours}
    return cap_up, cap_dn


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
