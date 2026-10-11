"""
builder.py — builds the catalogue's features for one delivery day from an as-of view.

The same function serves training (one call per training day), evaluation (one call per test day) and live
prediction, so a feature is defined once and cannot differ between training and serving. The builder only reads
columns and days its features declare; the as-of view has already removed everything the target may not know, so
a feature that read too far would come out missing, which the availability test catches.

    builder = FeatureBuilder(iface, catalogue, feature_ids)
    X = builder.build(view, D)          # one row per hour of day D, one column per feature, in feature_ids order
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.feature_engineering import calendar as cal
from common_layer.feature_engineering.catalogue import availability, by_id
from common_layer.forecast_evaluation.asof import ModelInterface, matrix


class FeatureUnavailable(ValueError):
    """A requested feature cannot be built for this target at its as-of time."""


def _at_day(M: pd.DataFrame, d: int, hours: np.ndarray) -> np.ndarray:
    row = M.reindex([d])
    return row.to_numpy()[0][hours] if len(row) else np.full(len(hours), np.nan)


def _stat(values: pd.DataFrame, stat: str, axis: int) -> np.ndarray:
    return getattr(values, stat)(axis=axis, skipna=True).to_numpy()


class _DayContext:
    """Per-day cache of the matrices the features read."""

    def __init__(self, iface: ModelInterface, view: pd.DataFrame, D: dt.date):
        self.iface, self.view, self.D = iface, view, D
        self.d0 = D.toordinal()
        self.rows = view[view["local_date"] == self.d0]
        self.hours = self.rows["local_hour"].to_numpy()
        self._m: Dict[str, pd.DataFrame] = {}

    def mat(self, key: str, values: Optional[pd.Series] = None) -> pd.DataFrame:
        if key not in self._m:
            self._m[key] = matrix(self.view, values if values is not None else self.view[key])
        return self._m[key]

    def modelled(self) -> pd.DataFrame:
        return self.mat("__y__", self.iface.modelled(self.view))

    def base_series(self) -> pd.Series:
        return self.view[self.iface.spec["base"]]


def _calendar_day(ctx: _DayContext, p: dict, cache: dict) -> np.ndarray:
    if ctx.D not in cache:
        cache[ctx.D] = cal.day_features(ctx.D)
    return np.full(len(ctx.rows), cache[ctx.D][p["field"]])


def _calendar_hour(ctx: _DayContext, p: dict, cache: dict) -> np.ndarray:
    return np.asarray(cal.hour_features(ctx.hours)[p["field"]], dtype=float)


def _modelled_lag(ctx, p, cache):
    return _at_day(ctx.modelled(), ctx.d0 - int(p["lag_days"]), ctx.hours)


def _modelled_roll(ctx, p, cache):
    w, e = int(p["window_days"]), int(p.get("end_lag_days", 1))
    win = ctx.modelled().reindex(range(ctx.d0 - e - w + 1, ctx.d0 - e + 1))
    if p["scope"] == "hour":
        return _stat(win, p["stat"], 0)[ctx.hours]
    flat = pd.Series(win.to_numpy().ravel())
    return np.full(len(ctx.hours), getattr(flat, p["stat"])())


def _modelled_yday(ctx, p, cache):
    row = ctx.modelled().reindex([ctx.d0 - int(p.get("lag_days", 1))])
    v = getattr(pd.Series(row.to_numpy().ravel()), p["stat"])() if len(row) else np.nan
    return np.full(len(ctx.hours), v)


def _base_now(ctx, p, cache):
    b = ctx.rows[ctx.iface.spec["base"]]
    arr = b.to_numpy(dtype=float)
    f = p["field"]
    if f == "value":
        return arr
    if f == "dev":
        return arr - np.nanmean(arr)
    if f == "day_mean":
        return np.full(len(arr), np.nanmean(arr))
    if f == "day_max":
        return np.full(len(arr), np.nanmax(arr))
    if f == "day_min":
        return np.full(len(arr), np.nanmin(arr))
    if f == "rank":
        return b.rank(pct=True).to_numpy()
    if f == "shape":
        mid, eve = np.isin(ctx.hours, [11, 12, 13, 14, 15]), np.isin(ctx.hours, [19, 20, 21])
        return np.full(len(arr), np.nanmean(arr[mid]) - np.nanmean(arr[eve]))
    raise ValueError(f)


def _base_lag(ctx, p, cache):
    return _at_day(ctx.mat("__base__", ctx.base_series()), ctx.d0 - int(p["lag_days"]), ctx.hours)


def _base_change(ctx, p, cache):
    return ctx.rows[ctx.iface.spec["base"]].to_numpy(dtype=float) - _base_lag(ctx, p, cache)


def _series_value(ctx, p, cache):
    return _at_day(ctx.mat(p["col"]), ctx.d0 + int(p["day_offset"]), ctx.hours)


def _series_spread(ctx, p, cache):
    key = f"{p['col']}-{p['base_col']}"
    return _at_day(ctx.mat(key, ctx.view[p["col"]] - ctx.view[p["base_col"]]), ctx.d0 + int(p["day_offset"]), ctx.hours)


def _series_day_stat(ctx, p, cache):
    row = ctx.mat(p["col"]).reindex([ctx.d0 + int(p["day_offset"])])
    v = getattr(pd.Series(row.to_numpy().ravel()), p["stat"])() if len(row) else np.nan
    return np.full(len(ctx.hours), v)


_KIND = {"calendar_hour": _calendar_hour, "calendar_day": _calendar_day, "modelled_lag": _modelled_lag,
         "modelled_roll": _modelled_roll, "modelled_yday": _modelled_yday, "base_now": _base_now,
         "base_lag": _base_lag, "base_change": _base_change, "series_value": _series_value,
         "series_spread": _series_spread, "series_day_stat": _series_day_stat}


class FeatureBuilder:
    def __init__(self, iface: ModelInterface, catalogue: dict, feature_ids: List[str]):
        self.iface, self.ids = iface, list(feature_ids)
        self._defs = by_id(catalogue)
        self._cal_cache: dict = {}
        for fid in self.ids:
            ok, why = availability(self._defs[fid], iface.spec, iface.offsets)
            if not ok:
                raise FeatureUnavailable(f"{iface.target_id}: feature {fid} is not available: {why}")

    def build(self, view: pd.DataFrame, D: dt.date) -> pd.DataFrame:
        ctx = _DayContext(self.iface, view, D)
        cols = {fid: _KIND[self._defs[fid]["kind"]](ctx, self._defs[fid].get("params", {}), self._cal_cache)
                for fid in self.ids}
        return pd.DataFrame(cols, index=ctx.rows.index)


def point_in_time_check(view, builder: FeatureBuilder, days: List[dt.date], history_days: int = 400) -> pd.DataFrame:
    """Point-in-time correctness: every feature must have the same value whether it is built from the as-of view or
    from the same rows with nothing masked. A feature that depended on data the target may not know would differ
    (the masked view has it missing). Returns one row per (day, feature) that differs; empty means correct."""
    rows = []
    for D in days:
        masked = builder.build(view.at(D, history_days), D)
        full = builder.build(view.at(D, history_days, masked=False), D)
        for fid in builder.ids:
            a, b = masked[fid].to_numpy(dtype=float), full[fid].to_numpy(dtype=float)
            if not np.allclose(a, b, equal_nan=True):
                rows.append({"day": D, "feature": fid, "max_abs_diff": float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else float("inf")})
    return pd.DataFrame(rows, columns=["day", "feature", "max_abs_diff"])
