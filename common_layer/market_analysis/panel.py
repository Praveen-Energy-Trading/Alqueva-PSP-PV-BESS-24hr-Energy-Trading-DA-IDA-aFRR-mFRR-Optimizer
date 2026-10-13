"""
panel.py — all price series on one hourly UTC grid, read from the clean zone.

Quarter-hour values become hourly means only when all four quarters are present; hourly values pass
through. The hourly "day-ahead price" of the 15-minute era (from 2025-10-01) is therefore the mean of its
four quarters, which is how it compares with the earlier hourly prices. By default nothing is filled or
interpolated: a missing hour stays missing (the model-development evaluation scores real outcomes only).
`curate=True` returns the panel after data cleaning and filling (common_layer/data_quality/curation.py), which is what the
live pipeline and the champions' fits use.

Calendar features use market time (Europe/Madrid), the time zone of the auctions.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from common_layer.data_layer.clean_store import CleanStore
from common_layer.utilities.timezone_utils import MARKET_TZ

# name -> (dataset, column, session or None)
SERIES: Dict[str, Tuple[str, str, Optional[int]]] = {
    "da_pt": ("da_price", "price_pt_eur_mwh", None),
    "da_es": ("da_price", "price_es_eur_mwh", None),
    "ida1_pt": ("ida_price", "price_pt_eur_mwh", 1),
    "ida2_pt": ("ida_price", "price_pt_eur_mwh", 2),
    "ida3_pt": ("ida_price", "price_pt_eur_mwh", 3),
    "xbid_pt": ("xbid_price", "price_pt_mean_eur_mwh", None),
    "afrr_up": ("afrr_price", "price_adjusted_up", None),
    "afrr_dn": ("afrr_price", "price_adjusted_dn", None),
    "mfrr_ap": ("mfrr_price", "price_programmed_activation", None),
    "imb_short": ("imbalance_price", "price_short_eur_mwh", None),
    "imb_long": ("imbalance_price", "price_long_eur_mwh", None),
    # market results published together with the day-ahead price (Spain-Portugal interconnection, matched volumes)
    "da_vol_total": ("da_price", "total_iberian_mw", None),
    "da_vol_pt_buy": ("da_price", "buy_pt_mw", None),
    "da_vol_pt_sell": ("da_price", "sell_pt_mw", None),
    "da_flow_pt_to_es": ("da_price", "flow_pt_to_es_mw", None),
    "da_flow_es_to_pt": ("da_price", "flow_es_to_pt_mw", None),
}
TZ_NAME = "Europe/Madrid"


def to_hourly(df: pd.DataFrame, col: str) -> pd.Series:
    """Hourly UTC series from rows with start_utc, resolution_min and `col`."""
    if df.empty:
        return pd.Series(dtype=float, name=col)
    t = pd.to_datetime(df["start_utc"], utc=True, format="%Y-%m-%dT%H:%M:%SZ")
    d = pd.DataFrame({"v": df[col].astype(float).to_numpy(), "res": df["resolution_min"].to_numpy(),
                      "hour": t.dt.floor("h").to_numpy()})
    g = d.groupby("hour")
    val = g["v"].mean()
    full = g["v"].count() == (60 // g["res"].first())
    val[~full] = np.nan
    ix = pd.DatetimeIndex(val.index)
    val.index = (ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")).rename(None)
    val.name = col
    return val


def build_panel(clean: CleanStore, start: Optional[dt.date] = None, end: Optional[dt.date] = None,
                names: Optional[Iterable[str]] = None, curate: bool = False) -> pd.DataFrame:
    """Hourly UTC panel with one column per series name (see SERIES); `curate` cleans and fills it (see the module header)."""
    names = list(names) if names else list(SERIES)
    cache: Dict[str, pd.DataFrame] = {}
    cols: Dict[str, pd.Series] = {}
    for name in names:
        dataset, col, session = SERIES[name]
        if dataset not in cache:
            cache[dataset] = clean.read(dataset, start, end)
        df = cache[dataset]
        if session is not None and not df.empty:
            df = df[df["session"] == session]
        cols[name] = to_hourly(df, col)
    panel = pd.DataFrame(cols).sort_index()
    if curate:
        from common_layer.data_quality import curation
        panel = curation.curate(panel).panel
    return panel


def add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Add market-time calendar columns: local_date, local_hour, dow (0=Mon), month, year, is_weekend."""
    out = panel.copy()
    local = out.index.tz_convert(TZ_NAME)
    out["local_date"] = local.date
    out["local_hour"] = local.hour
    out["dow"] = local.dayofweek
    out["month"] = local.month
    out["year"] = local.year
    out["is_weekend"] = local.dayofweek >= 5
    return out


def daily_mean(s: pd.Series, min_hours: int = 20) -> pd.Series:
    """Mean per market-time day, only for days with at least `min_hours` hourly values.
    Index: python dates."""
    local = s.index.tz_convert(TZ_NAME)
    g = s.groupby(local.date)
    m = g.mean()
    m[g.count() < min_hours] = np.nan
    m.index = pd.to_datetime(m.index)
    return m.dropna()


def period_mask(index: pd.DatetimeIndex, start: dt.date, end: dt.date) -> np.ndarray:
    """Boolean mask for a (UTC or naive daily) index inside [start, end] in market-time days."""
    if index.tz is not None:
        days = index.tz_convert(TZ_NAME).tz_localize(None).normalize()
    else:
        days = index.normalize()
    return np.asarray((days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end)))
