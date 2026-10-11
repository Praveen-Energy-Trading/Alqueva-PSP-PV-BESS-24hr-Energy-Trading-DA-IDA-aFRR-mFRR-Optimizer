"""
parsers.py — raw bytes -> tidy tables. Pure functions: same bytes in, same table out.

Every table has the columns
    delivery_date (YYYY-MM-DD), period_index (1-based, chronological), resolution_min (60 or 15),
    start_utc (YYYY-MM-DDTHH:MM:SSZ)
plus dataset-specific columns. The TRUE periods are kept: 23/25-hour days give 23/25 hourly or
92/100 quarter-hour rows. No value is interpolated, averaged or dropped here; a 24/96-slot view is
made only by calendar.to_model_grid.

Datasets:  da_price, ida_price (column `session`), xbid_price, afrr_price, mfrr_price, imbalance_price
PARSER_VERSION is stored with every clean row so a later parser change can be traced.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from typing import Dict, List, Optional

import pandas as pd

from common_layer.data_layer import calendar as cal
from common_layer.utilities.timezone_utils import MARKET_TZ

PARSER_VERSION = "1"
NAN = float("nan")


class ParseError(ValueError):
    """The raw content does not have the expected structure."""


# ------------------------------------------------------------------ helpers
def _num_comma(s: str) -> float:
    """Spanish number format: '1.234,56' -> 1234.56; '' -> NaN."""
    s = s.strip()
    if not s:
        return NAN
    return float(s.replace(".", "").replace(",", "."))


def _num_dot(s: str) -> float:
    s = s.strip()
    return float(s) if s else NAN


def _decode(raw: bytes) -> str:
    return raw.decode("latin-1")


def _frame(day: dt.date, n_slots: int, resolution_min: int, indices: List[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "delivery_date": day.isoformat(),
        "period_index": indices,
        "resolution_min": resolution_min,
        "start_utc": [cal.period_start_utc(day, i, resolution_min).strftime("%Y-%m-%dT%H:%M:%SZ")
                      for i in indices],
    })


def _issue_utc(header_line: str) -> Optional[str]:
    """'Fecha Emisión :26/01/2026 - 12:37' (market time) -> UTC ISO string."""
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})", header_line)
    if not m:
        return None
    d, mo, y, hh, mm = (int(x) for x in m.groups())
    local = dt.datetime(y, mo, d, hh, mm, tzinfo=MARKET_TZ)
    return local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolution_for(day: dt.date, max_period: int) -> int:
    return 15 if max_period > cal.hours_in_day(day) else 60


# ------------------------------------------------------------- OMIE day-ahead
_DA_COLUMNS = {
    ("precio marginal", "espa"): "price_es_eur_mwh",
    ("precio marginal", "portug"): "price_pt_eur_mwh",
    ("compra", "espa"): "buy_es_mw",
    ("venta", "espa"): "sell_es_mw",
    ("compra", "portug"): "buy_pt_mw",
    ("venta", "portug"): "sell_pt_mw",
    ("total del mercado", "ib"): "total_iberian_mw",
    ("bilaterales", ""): "total_with_bilaterals_mw",
    ("importaci", ""): "flow_pt_to_es_mw",
    ("exportaci", ""): "flow_es_to_pt_mw",
}


def parse_omie_da(raw: bytes, day: dt.date) -> pd.DataFrame:
    """OMIE day-ahead file (INT_PBC_EV_H_1): prices, volumes and Spain-Portugal flows per period.

    Volumes are MW (hourly files give MWh per hour, which is the same number)."""
    text = _decode(raw)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or "omie" not in lines[0].lower():
        raise ParseError("not an OMIE day-ahead file")
    head = lines[0].split(";")
    file_day = next((p.strip() for p in head if re.fullmatch(r"\d{2}/\d{2}/\d{4}", p.strip())), None)
    if file_day and dt.datetime.strptime(file_day, "%d/%m/%Y").date() != day:
        raise ParseError(f"file is for {file_day}, expected {day}")
    issued = _issue_utc(lines[0])

    series: Dict[str, List[float]] = {}
    for ln in lines[2:]:
        parts = ln.split(";")
        name = parts[0].lower()
        vals = [p for p in parts[1:] if p.strip() != ""]
        if not vals:
            continue
        for (k1, k2), col in _DA_COLUMNS.items():
            if k1 in name and k2 in name and col not in series:
                series[col] = [_num_comma(v) for v in vals]
                break
    if "price_es_eur_mwh" not in series and "price_pt_eur_mwh" not in series:
        raise ParseError("no price row found")

    flags = ""
    if "price_pt_eur_mwh" not in series:
        series["price_pt_eur_mwh"] = list(series["price_es_eur_mwh"])
        flags = "pt_row_missing_copied_from_es"
    if "price_es_eur_mwh" not in series:
        series["price_es_eur_mwh"] = list(series["price_pt_eur_mwh"])
        flags = "es_row_missing_copied_from_pt"
    n = len(series["price_pt_eur_mwh"])
    res = cal.detect_resolution_min(day, n)
    if res is None:
        raise ParseError(f"{n} prices do not form a full day on {day} (hours={cal.hours_in_day(day)})")
    df = _frame(day, n, res, list(range(1, n + 1)))
    for col in _DA_COLUMNS.values():
        vals = series.get(col)
        df[col] = vals if vals is not None and len(vals) == n else NAN
    df["issued_utc"] = issued or ""
    df["flags"] = flags
    return df


# ------------------------------------------------------------- OMIE intraday
def parse_omie_ida(raw: bytes, day: dt.date, session: int) -> pd.DataFrame:
    """OMIE intraday auction file (marginalpibcpt): 'year;month;day;period;price_ES;price_PT;'."""
    rows: Dict[int, tuple] = {}
    for ln in _decode(raw).splitlines():
        p = [x.strip() for x in ln.split(";")]
        if len(p) < 6 or not p[3].isdigit() or not p[0].isdigit():
            continue
        if dt.date(int(p[0]), int(p[1]), int(p[2])) != day:
            raise ParseError(f"line for {p[0]}-{p[1]}-{p[2]} in a file for {day}")
        rows[int(p[3])] = (_num_dot(p[4]), _num_dot(p[5]))
    if not rows:
        raise ParseError("no intraday price lines")
    res = _resolution_for(day, max(rows))
    idx = sorted(rows)
    df = _frame(day, len(idx), res, idx)
    df["session"] = session
    df["price_es_eur_mwh"] = [rows[i][0] for i in idx]
    df["price_pt_eur_mwh"] = [rows[i][1] for i in idx]
    return df


# ------------------------------------------------------- OMIE continuous
def parse_omie_xbid(raw: bytes, day: dt.date) -> pd.DataFrame:
    """OMIE continuous-market summary (precios_pibcic): max/min/volume-weighted mean per period.

    OMIE writes 0 for the Portuguese columns when there were no Portuguese trades; those periods
    get pt_traded = 0 and NaN prices, never a price of 0."""
    text = _decode(raw)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    issued = _issue_utc(lines[0]) if lines else None
    header = None
    rows: Dict[int, dict] = {}
    for ln in lines:
        p = [x.strip() for x in ln.split(";")]
        if p and p[0].lower() in ("año", "ano", "a\xf1o"):
            header = [h.lower() for h in p]
            continue
        if header is None or not p[0].isdigit() or len(p) < len(header) - 1:
            continue
        row = dict(zip(header, p))
        per = row.get("periodo") or row.get("hora")
        if not per or not per.isdigit():
            continue
        rows[int(per)] = row
    if not rows:
        raise ParseError("no continuous-market lines")
    res = _resolution_for(day, max(rows))
    idx = sorted(rows)
    df = _frame(day, len(idx), res, idx)

    def col(name_candidates):
        for c in name_candidates:
            if c in header:
                return c
        return None

    def get(c):
        return [(_num_comma(rows[i].get(c, "")) if c else NAN) for i in idx]

    c_max_pt = col(["máximopt", "maximopt", "m\xe1ximopt"])
    c_min_pt = col(["mínimopt", "minimopt", "m\xednimopt"])
    c_mean_pt = col(["mediopt"])
    c_mean_es = col(["medioes"])
    mx, mn, me = get(c_max_pt), get(c_min_pt), get(c_mean_pt)
    traded = [0 if (a == 0 and b == 0 and c == 0) or math.isnan(c) else 1 for a, b, c in zip(mx, mn, me)]
    df["pt_traded"] = traded
    df["price_pt_max_eur_mwh"] = [v if t else NAN for v, t in zip(mx, traded)]
    df["price_pt_min_eur_mwh"] = [v if t else NAN for v, t in zip(mn, traded)]
    df["price_pt_mean_eur_mwh"] = [v if t else NAN for v, t in zip(me, traded)]
    df["price_es_mean_eur_mwh"] = get(c_mean_es)
    df["issued_utc"] = issued or ""
    return df


# ---------------------------------------------------------------------- REN
_REN_MAPS = {
    "afrr_price": {"PRECO_INI_SUB": "price_initial_up", "PRECO_INI_DES": "price_initial_dn",
                   "PRECO_AJUST_SUB": "price_adjusted_up", "PRECO_AJUST_DES": "price_adjusted_dn",
                   "MERCADO_DIARIO_PT": "price_da_ref_eur_mwh"},
    "mfrr_price": {"AP_PRECO": "price_programmed_activation",
                   "AD_PRECO_Q0_SUBIR": "price_direct_q0_up", "AD_PRECO_Q0_DESCER": "price_direct_q0_dn",
                   "AD_PRECO_Q1_SUBIR": "price_direct_q1_up", "AD_PRECO_Q1_DESCER": "price_direct_q1_dn",
                   "PRECO_MER_DIARIO": "price_da_ref_eur_mwh"},
    "imbalance_price": {"PRECO_DEFEITO": "price_short_eur_mwh", "PRECO_EXCESSO": "price_long_eur_mwh",
                        "PRECO_MERC_DIARIO": "price_da_ref_eur_mwh"},
}


def parse_ren(raw: bytes, day: dt.date, dataset: str) -> pd.DataFrame:
    """REN market-service JSON (BaFRRPreco, MFRRPreco, DesvioPreco): one row per quarter-hour.

    REN gives the UTC start of every period (DATA_UTC); it is checked against the calendar."""
    if dataset not in _REN_MAPS:
        raise ParseError(f"unknown REN dataset {dataset}")
    payload = json.loads(_decode(raw))
    if isinstance(payload, str):                    # REN double-encodes the JSON body
        payload = json.loads(payload)
    if not isinstance(payload, list) or not payload:
        raise ParseError("empty or non-list REN response")
    rows = {int(r["PERIODO"]): r for r in payload}
    n = len(rows)
    if cal.detect_resolution_min(day, n) != 15:
        raise ParseError(f"{n} quarter-hour rows do not form a full day on {day}")
    idx = sorted(rows)
    df = _frame(day, n, 15, idx)
    for r_idx, i in enumerate(idx):
        if rows[i].get("DATA_UTC") and not df.loc[r_idx, "start_utc"].startswith(rows[i]["DATA_UTC"].replace(" ", "T")):
            raise ParseError(f"DATA_UTC {rows[i]['DATA_UTC']} disagrees with calendar {df.loc[r_idx, 'start_utc']}")
    for src, col in _REN_MAPS[dataset].items():
        df[col] = [_num_dot(str(rows[i].get(src) or "")) for i in idx]
    return df
