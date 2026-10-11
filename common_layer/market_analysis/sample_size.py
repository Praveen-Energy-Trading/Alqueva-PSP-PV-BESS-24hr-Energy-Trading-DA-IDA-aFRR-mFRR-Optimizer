"""
sample_size.py — how much independent information the history holds, and which part still looks like today.

Effective sample size: n observations that are autocorrelated carry fewer than n independent pieces of
information. ESS = n / (1 + 2 * sum of autocorrelations), summing positive autocorrelations in pairs and
stopping at the first pair that is not positive (Geyer's initial positive sequence). It is computed on the
daily mean price: the unit that day-ahead model validation actually counts. It is also computed on the daily
*changes*, because the level of a near-unit-root series carries only a handful of independent observations
(it says how many independent price levels the history holds), while the changes are what a model that
already knows yesterday's price has left to predict.

Representativeness: for each calendar year and analysis period, how far is the distribution of hourly prices
from the reference (the most recent `reference_months`)? Reported as the Kolmogorov-Smirnov distance (0 =
identical, 1 = disjoint) and the ratio of means and standard deviations. A distance below the configured
threshold is called "similar". This describes the history; whether a model should still train on a year is a
modelling decision that the findings note makes.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from common_layer.market_analysis.panel import period_mask


def effective_sample_size(x: pd.Series) -> Dict[str, float]:
    v = x.dropna().astype(float).to_numpy()
    n = len(v)
    if n < 30 or v.std() == 0:
        return {"n": n, "rho1": np.nan, "ess": np.nan, "ess_ratio": np.nan}
    v = v - v.mean()
    max_lag = min(n // 3, 400)
    f = np.fft.rfft(v, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    ac = ac / ac[0]
    total, k = 0.0, 1
    while k + 1 <= max_lag:
        pair = ac[k] + ac[k + 1]
        if pair <= 0:
            break
        total += pair
        k += 2
    ess = n / (1.0 + 2.0 * total)
    return {"n": n, "rho1": float(ac[1]), "ess": float(ess), "ess_ratio": float(ess / n)}


def ess_by_period(daily: pd.Series, periods: List[dict]) -> pd.DataFrame:
    rows = []
    for p in periods:
        m = period_mask(daily.index, p["start"], p["end"])
        r = effective_sample_size(daily[m])
        rc = effective_sample_size(daily[m].diff().dropna())
        rows.append({"period": p["id"], "label": p["label"], "days": r["n"], "rho1_daily": r["rho1"],
                     "effective_days": r["ess"], "ess_ratio": r["ess_ratio"],
                     "rho1_changes": rc["rho1"], "effective_days_changes": rc["ess"],
                     "ess_ratio_changes": rc["ess_ratio"]})
    return pd.DataFrame(rows)


def representativeness(hourly: pd.Series, periods: List[dict], reference_months: int, ks_threshold: float
                       ) -> pd.DataFrame:
    """KS distance of each calendar year and analysis period to the most recent `reference_months`."""
    s = hourly.dropna()
    ref_end = s.index.max()
    ref = s[s.index > ref_end - pd.DateOffset(months=reference_months)]
    rows = []

    def row(label: str, kind: str, x: pd.Series) -> None:
        if len(x) < 200:
            return
        ks = stats.ks_2samp(x.to_numpy(), ref.to_numpy())
        in_ref = float(((x.index >= ref.index.min()) & (x.index <= ref.index.max())).mean())
        rows.append({"kind": kind, "group": label, "n_hours": len(x), "share_inside_reference": in_ref,
                     "mean": float(x.mean()), "std": float(x.std()),
                     "ks_distance": float(ks.statistic), "mean_ratio": float(x.mean() / ref.mean()) if ref.mean() else np.nan,
                     "std_ratio": float(x.std() / ref.std()) if ref.std() else np.nan,
                     "similar": bool(ks.statistic < ks_threshold)})

    local_year = s.index.tz_convert("Europe/Madrid").year
    for y in sorted(set(local_year)):
        row(str(y), "year", s[local_year == y])
    for p in periods:
        row(p["id"], "period", s[period_mask(s.index, p["start"], p["end"])])
    out = pd.DataFrame(rows)
    out.attrs["reference_start"] = str(ref.index.min())
    out.attrs["reference_end"] = str(ref.index.max())
    out.attrs["reference_mean"] = float(ref.mean())
    out.attrs["reference_std"] = float(ref.std())
    return out
