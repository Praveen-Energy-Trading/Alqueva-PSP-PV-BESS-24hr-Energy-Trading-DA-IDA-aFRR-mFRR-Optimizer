"""
evidence.py — measurements behind the Phase 5 formulation choices. Each function answers one question with
numbers; docs/forecasting/Phase5_Problem_Formulation.md quotes them.

    heteroscedasticity_by_scale   which asinh scale makes the day-over-day price change equally variable at all levels?
    day_shift_share               how much of a day's price error is one common shift (what the scenario fan assumes)?
    daily_shape_factors           how many factors describe the day's shape (one model per hour or one vector)?
    spread_vs_level               is the spread to day-ahead easier to model than the level?
    publication_lags              when are the files really issued, against the declared rules?
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from common_layer.utilities.timezone_utils import MARKET_TZ

TZ = "Europe/Madrid"


# ------------------------------------------------------------------ transform scale
def heteroscedasticity_by_scale(hourly: pd.Series, scales: Iterable[Optional[float]], block_days: int = 30,
                                min_level: float = 5.0) -> pd.DataFrame:
    """For each candidate asinh scale (None = untransformed): the elasticity of the spread of the day-over-day
    change (z_t - z_{t-24h}) with respect to the price level, over non-overlapping blocks, plus the shape of that
    change. Elasticity near 0 = equally variable at every level (what a squared-error loss assumes)."""
    x = hourly.dropna()
    x = x.asfreq("h")                                   # keeps gaps as NaN so the 24-hour shift is exact
    rows = []
    n = block_days * 24
    for s in scales:
        z = x if s is None else np.arcsinh(x / s)
        d = (z - z.shift(24)).dropna()
        pts = []
        for i in range(0, len(x) - n + 1, n):
            lvl = x.iloc[i:i + n].mean()
            vol = d.reindex(x.index[i:i + n]).std()
            if lvl > min_level and vol and vol > 0:
                pts.append((np.log(lvl), np.log(vol)))
        if len(pts) < 8:
            rows.append({"scale": s, "blocks": len(pts)})
            continue
        a = np.array(pts)
        r = stats.linregress(a[:, 0], a[:, 1])
        rows.append({"scale": s, "blocks": len(pts), "elasticity": r.slope, "r2": r.rvalue ** 2,
                     "skew_change": float(stats.skew(d)), "excess_kurtosis_change": float(stats.kurtosis(d)),
                     "std_change": float(d.std())})
    return pd.DataFrame(rows)


# ----------------------------------------------------- day shape: one shift, a few factors
def _day_matrix(resid: pd.Series, hours: int = 24) -> pd.DataFrame:
    """Rows = market-time days with all 24 hours present, columns = local hour 0..23."""
    local = resid.index.tz_convert(TZ)
    df = pd.DataFrame({"v": resid.to_numpy(), "d": local.date, "h": local.hour})
    m = df.pivot_table(index="d", columns="h", values="v", aggfunc="first")
    return m.dropna()


def day_shift_share(resid: pd.Series) -> Dict[str, float]:
    """Share of the residual variance that is a shift common to all 24 hours of the day.

    `resid` = hourly price minus its calendar profile (see market_analysis.dependence.deseasonalise). The
    scenario generator applies one offset to every hour of a scenario; this is how much of reality that covers."""
    m = _day_matrix(resid)
    total = float(np.var(m.to_numpy()))
    common = float(np.var(m.mean(axis=1).to_numpy()))
    return {"days": len(m), "total_variance": total, "common_shift_variance": common,
            "common_shift_share": common / total if total > 0 else np.nan}


def daily_shape_factors(resid: pd.Series, k: int = 5) -> pd.DataFrame:
    """Principal components of the day's 24-hour residual vector: share of variance of each factor."""
    m = _day_matrix(resid)
    x = m.to_numpy() - m.to_numpy().mean(axis=0)
    s = np.linalg.svd(x, compute_uv=False)
    share = s ** 2 / np.sum(s ** 2)
    return pd.DataFrame({"factor": range(1, k + 1), "variance_share": share[:k], "cumulative": np.cumsum(share)[:k]})


# --------------------------------------------------------------- spread versus level
def spread_vs_level(level: pd.Series, base: pd.Series) -> Dict[str, float]:
    """Compare modelling the level with modelling the spread to the base price (day-ahead).

    For a forecaster that knows the base price, the level error equals the spread error. What matters is how
    much of the level's variation the base already explains and how big the spread is next to the level."""
    d = pd.concat([level, base], axis=1, keys=["level", "base"]).dropna()
    if len(d) < 200:
        return {"n": len(d)}
    sp = d["level"] - d["base"]
    reg = stats.linregress(d["base"], d["level"])
    naive24 = (d["level"] - d["level"].shift(24)).abs().mean()
    return {"n": len(d), "std_level": float(d["level"].std()), "std_spread": float(sp.std()),
            "std_ratio": float(sp.std() / d["level"].std()), "r2_level_on_base": float(reg.rvalue ** 2),
            "slope_level_on_base": float(reg.slope), "mae_base_as_forecast": float(sp.abs().mean()),
            "mae_naive_same_hour_yesterday": float(naive24),
            "spread_acf1": float(sp.autocorr(1)), "spread_acf24": float(sp.autocorr(24)),
            "spread_skew": float(stats.skew(sp)), "spread_excess_kurtosis": float(stats.kurtosis(sp))}


# ------------------------------------------------------------------ publication times
def publication_lags(issued_utc: pd.Series, delivery_date: pd.Series) -> pd.DataFrame:
    """When a file for delivery day d was issued, relative to local midnight starting d (hours; negative = before
    the day starts), and its local time of day. Files re-issued long after delivery are reported separately."""
    t = pd.to_datetime(issued_utc, utc=True, errors="coerce").dt.tz_convert(TZ)
    d = pd.to_datetime(delivery_date)
    midnight = d.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    lag_h = (t - midnight).dt.total_seconds() / 3600.0
    out = pd.DataFrame({"delivery_date": d.dt.date, "issued_local": t, "hours_vs_day_start": lag_h,
                        "issue_time_of_day": t.dt.hour + t.dt.minute / 60.0})
    return out.dropna()


def summarise_publication(lags: pd.DataFrame, original_window_h: float = 48.0) -> Dict[str, float]:
    """Summary over files issued within `original_window_h` hours around the delivery day (i.e. the original
    issue, not a later re-issue)."""
    orig = lags[(lags["hours_vs_day_start"] > -original_window_h) & (lags["hours_vs_day_start"] < original_window_h)]
    if orig.empty:
        return {"files": len(lags), "original_files": 0}
    return {"files": len(lags), "original_files": len(orig),
            "median_hours_vs_day_start": float(orig["hours_vs_day_start"].median()),
            "p05_hours_vs_day_start": float(orig["hours_vs_day_start"].quantile(0.05)),
            "p95_hours_vs_day_start": float(orig["hours_vs_day_start"].quantile(0.95)),
            "median_issue_time_of_day": float(orig["issue_time_of_day"].median())}
