"""
extremes.py — spikes, negative prices, tails, solar cannibalisation and heteroscedasticity.

Spikes use a trailing window (only earlier prices), so a count here is the same thing a model could see
at forecast time. Negative prices and spikes are information about the market, not data errors.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def spike_mask(s: pd.Series, window_days: int = 14, z: float = 6.0, min_abs: float = 30.0,
               min_history: int = 96) -> Tuple[pd.Series, pd.Series]:
    """(up, down) boolean masks: |x - median| / (IQR/1.349) > z and |x - median| > min_abs, using the
    window_days before each point. `s` needs a DatetimeIndex."""
    roll = s.astype(float).rolling(f"{int(window_days)}D", closed="left", min_periods=min_history)
    med = roll.median()
    sigma = (roll.quantile(0.75) - roll.quantile(0.25)) / 1.349
    dev = s - med
    flag = (dev.abs() / sigma.where(sigma > 1e-6) > z) & (dev.abs() > min_abs)
    return (flag & (dev > 0)).fillna(False), (flag & (dev < 0)).fillna(False)


def distribution_summary(df: pd.DataFrame, col: str, by: str, quantiles: List[float]) -> pd.DataFrame:
    """Shape of df[col] per value of `by` (e.g. year): moments, quantiles, shares at or below zero."""
    rows = []
    for key, g in df.dropna(subset=[col]).groupby(by):
        x = g[col].astype(float)
        row = {by: key, "n": len(x), "mean": x.mean(), "std": x.std(), "skew": stats.skew(x),
               "excess_kurtosis": stats.kurtosis(x), "share_negative": float((x < 0).mean()),
               "share_at_or_below_zero": float((x <= 0).mean())}
        row.update({f"q{int(q * 100):02d}": float(x.quantile(q)) for q in quantiles})
        rows.append(row)
    return pd.DataFrame(rows)


def negative_by_hour(df: pd.DataFrame, col: str, threshold: float = 0.0) -> pd.DataFrame:
    """Share of hours with price <= threshold, by local hour (rows) and year (columns)."""
    d = df.dropna(subset=[col]).assign(neg=lambda t: (t[col] <= threshold).astype(float))
    return d.pivot_table(index="local_hour", columns="year", values="neg", aggfunc="mean")


def negative_by_month(df: pd.DataFrame, col: str, threshold: float = 0.0) -> pd.DataFrame:
    d = df.dropna(subset=[col]).assign(neg=lambda t: (t[col] <= threshold).astype(float))
    return d.pivot_table(index="month", columns="year", values="neg", aggfunc="mean")


def cannibalisation(df: pd.DataFrame, col: str, midday: List[int], evening: List[int] = (19, 20, 21)) -> pd.DataFrame:
    """Per year: mean price at the solar-peak hours versus all hours and versus the evening ramp.

    midday_ratio < 1 means solar hours are cheaper than average; evening_minus_midday is the depth of the
    daily 'duck' shape in EUR/MWh."""
    d = df.dropna(subset=[col])
    rows = []
    for year, g in d.groupby("year"):
        all_mean = g[col].mean()
        mid = g[g["local_hour"].isin(midday)][col].mean()
        eve = g[g["local_hour"].isin(list(evening))][col].mean()
        rows.append({"year": year, "mean_all": all_mean, "mean_midday": mid, "mean_evening": eve,
                     "midday_ratio": mid / all_mean if all_mean else np.nan, "evening_minus_midday": eve - mid})
    return pd.DataFrame(rows)


def hill_tail(x: np.ndarray, fraction: float = 0.05) -> Dict[str, float]:
    """Hill estimator of the upper-tail index alpha (smaller = heavier tail). Needs positive tail values."""
    v = np.sort(np.asarray(x, dtype=float))
    v = v[v > 0]
    k = max(10, int(len(v) * fraction))
    if len(v) < 3 * k:
        return {"alpha": np.nan, "k": k}
    tail, thr = v[-k:], v[-k - 1]
    return {"alpha": float(1.0 / np.mean(np.log(tail / thr))), "k": k}


def spike_counts(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    """Count of up and down spikes (trailing robust z) per year; index = year."""
    s = df[col].dropna()
    up, dn = spike_mask(s, params["window_days"], params["robust_z"], params["min_abs_deviation_eur"],
                        params["min_history_periods"])
    t = pd.DataFrame({"up": up, "down": dn, "year": df.loc[s.index, "year"], "hour": df.loc[s.index, "local_hour"]})
    return t.groupby("year")[["up", "down"]].sum()


def spike_by_hour(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    s = df[col].dropna()
    up, dn = spike_mask(s, params["window_days"], params["robust_z"], params["min_abs_deviation_eur"],
                        params["min_history_periods"])
    t = pd.DataFrame({"up": up, "down": dn, "hour": df.loc[s.index, "local_hour"]})
    return t.groupby("hour")[["up", "down"]].sum()


def heteroscedasticity(daily: pd.Series, block: int = 30, min_level: float = 5.0) -> Dict[str, object]:
    """Does volatility scale with the price level? Over non-overlapping blocks, regress log(std of daily
    changes) on log(mean level). An elasticity near 1 means volatility moves in proportion to the level
    (model log-returns or scale errors); near 0 means roughly constant absolute volatility."""
    d = daily.dropna()
    rows = []
    for i in range(0, len(d) - block + 1, block):
        seg = d.iloc[i:i + block]
        m, sd = seg.mean(), seg.diff().dropna().std()
        if m > min_level and sd > 0:
            rows.append((np.log(m), np.log(sd)))
    if len(rows) < 8:
        return {"blocks": len(rows), "elasticity": np.nan, "r2": np.nan}
    a = np.array(rows)
    res = stats.linregress(a[:, 0], a[:, 1])
    return {"blocks": len(rows), "elasticity": float(res.slope), "r2": float(res.rvalue ** 2),
            "p_value": float(res.pvalue)}
