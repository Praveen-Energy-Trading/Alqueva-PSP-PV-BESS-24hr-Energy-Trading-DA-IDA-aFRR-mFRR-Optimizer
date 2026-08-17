"""
ren_imbalance_price_loader.py — REN dual imbalance (deviation) prices.

Dual pricing penalises being out of balance: a SHORT position (under-delivery) is
bought back at a premium to the energy price; a LONG position (over-delivery) is
sold at a discount.

Settlement always looks BACKWARD in time (same reasoning as
omie_settlement_price_loader.py) — by the time Phase 5C runs, REN has already
published the real settled deviation price for that delivery date, so we
always try that first and only fall back to the configured DA-price
multipliers when the live fetch fails (network down, date not yet published,
etc). No separate live/synthetic flag is needed here, unlike the bidding-time
forecasters — there is no "predicting the future" step for a settlement price.

Live source (verified against a live REN response — endpoint and auth
extracted from REN's own public "Preço Desvio" SharePoint web part JS
bundle, loaded from
https://mercado.ren.pt/PT/Electr/InfoMercado/InfSistema/Desvios/Paginas/Desvio-Preco.aspx):
    GET https://mercadoservices.ren.pt/api/DesvioPreco/GetDesvioPreco
        ?language=PT&dayQuery=<D>&monthQuery=<M>&yearQuery=<Y>&sWhere=
    Header: X-ApiKey: base64("mercado_mL273BtiLeRcqfqBqImWBf5uvPTmdW4VHxb4EeD6")
        (same static public key as the aFRR/mFRR endpoints.)
    Response: JSON array, 96 quarter-hour periods/day, fields PERIODO,
    HORA_INI_FIM, PRECO_DEFEITO (deficit/short price — premium, paid when
    the plant under-delivers), PRECO_EXCESSO (excess/long price — discount,
    received when the plant over-delivers), PRECO_MERC_DIARIO (DA reference).

Returns (short_price, long_price) each {hour: EUR/MWh}.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from common_layer.configuration.config_loader import AppConfig
from common_layer.utilities.logging_utils import get_logger
from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster import (
    forecast_da_prices,
)

log = get_logger(__name__)

_N_QUARTERS = 96
_REN_API_KEY_B64 = "bWVyY2Fkb19tTDI3M0J0aUxlUmNxZnFCcUltV0JmNXV2UFRtZFc0Vkh4YjRFZUQ2"


def fetch_imbalance_prices(delivery_date: str, hours: List[int], cfg: AppConfig
                           ) -> Tuple[Dict[int, float], Dict[int, float]]:
    try:
        return _download_ren_imbalance(delivery_date, hours)
    except Exception as exc:
        log.warning(f"REN imbalance price fetch failed ({exc}); "
                    f"falling back to DA-price multipliers.")
        return _fallback_imbalance_prices(delivery_date, hours, cfg)


def _fallback_imbalance_prices(delivery_date: str, hours: List[int], cfg: AppConfig
                               ) -> Tuple[Dict[int, float], Dict[int, float]]:
    da = forecast_da_prices(hours, delivery_date)
    imb = cfg.market.imbalance
    short = {h: round(da[h] * imb.fallback_short_factor, 2) for h in hours}
    long_ = {h: round(da[h] * imb.fallback_long_factor, 2) for h in hours}
    return short, long_


def _download_ren_imbalance(delivery_date: str, hours: List[int]
                            ) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Download and parse REN's imbalance (deviation) price API.

    Averages the 4 quarter-hour periods within each hour. Raises on any
    failure (network, missing/short response, unexpected format).
    """
    import requests
    from datetime import datetime as _dt
    d = _dt.strptime(delivery_date, "%Y-%m-%d")
    url = (f"https://mercadoservices.ren.pt/api/DesvioPreco/GetDesvioPreco"
           f"?language=PT&dayQuery={d.day}&monthQuery={d.month}&yearQuery={d.year}&sWhere=")
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

    quarters_short: Dict[int, float] = {}
    quarters_long: Dict[int, float] = {}
    for r in rows:
        period = int(r["PERIODO"])
        short_v = _num(r.get("PRECO_DEFEITO"))
        long_v  = _num(r.get("PRECO_EXCESSO"))
        if short_v is None or long_v is None:
            continue
        quarters_short[period] = short_v
        quarters_long[period] = long_v

    if len(quarters_short) < _N_QUARTERS or len(quarters_long) < _N_QUARTERS:
        raise ValueError(f"Incomplete imbalance price data: {len(quarters_short)} "
                          f"short, {len(quarters_long)} long periods (need {_N_QUARTERS})")

    short_hourly = {h: round(sum(quarters_short[(h - 1) * 4 + q] for q in range(1, 5)) / 4.0, 2)
                    for h in range(1, 25)}
    long_hourly  = {h: round(sum(quarters_long[(h - 1) * 4 + q] for q in range(1, 5)) / 4.0, 2)
                    for h in range(1, 25)}
    return ({h: short_hourly[h] for h in hours if h in short_hourly},
            {h: long_hourly[h] for h in hours if h in long_hourly})
