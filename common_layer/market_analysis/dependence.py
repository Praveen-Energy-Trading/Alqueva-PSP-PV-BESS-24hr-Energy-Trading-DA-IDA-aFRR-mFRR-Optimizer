"""
dependence.py — autocorrelation, volatility clustering and stationarity.

Stationarity is reported with two tests that have opposite null hypotheses, because either alone misleads:
    ADF   null: unit root (non-stationary).      Small p-value  -> evidence of stationarity.
    KPSS  null: level stationary.                 Small p-value  -> evidence of non-stationarity.
Verdict: "stationary" if ADF rejects and KPSS does not; "non-stationary" if the reverse; "ambiguous" otherwise
(typical for prices that move around a slowly changing level).
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss


def acf_series(x: pd.Series, nlags: int) -> pd.Series:
    """Autocorrelation at lags 0..nlags of a regularly spaced series (NaN gaps tolerated)."""
    v = acf(x.astype(float).to_numpy(), nlags=nlags, missing="conservative", fft=True)
    return pd.Series(v, index=range(nlags + 1), name="acf")


def key_lags(x: pd.Series, lags: Iterable[int]) -> Dict[int, float]:
    a = acf_series(x, max(lags))
    return {int(k): float(a[k]) for k in lags}


def ljung_box(x: pd.Series, lags: List[int]) -> pd.DataFrame:
    out = acorr_ljungbox(x.dropna().astype(float), lags=lags, return_df=True)
    out.index.name = "lag"
    return out.reset_index()


def arch_lm(x: pd.Series, nlags: int = 12) -> Dict[str, float]:
    """Engle's ARCH-LM test on a (de-meaned) series: small p-value = volatility clusters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)          # statsmodels announces a future return type
        lm, p, f, fp = het_arch(x.dropna().astype(float) - x.mean(), nlags=nlags)
    return {"lm_stat": float(lm), "p_value": float(p), "nlags": nlags}


def stationarity(x: pd.Series, name: str = "") -> Dict[str, object]:
    """ADF and KPSS on a series (regular spacing assumed, NaNs dropped)."""
    v = x.dropna().astype(float)
    if len(v) < 50 or v.std() == 0:
        return {"series": name, "n": len(v), "adf_stat": np.nan, "adf_p": np.nan, "kpss_stat": np.nan,
                "kpss_p": np.nan, "verdict": "too short"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # KPSS p-values are interpolated and clipped to [0.01, 0.1]
        adf = adfuller(v, autolag="AIC")
        kp = kpss(v, regression="c", nlags="auto")
    adf_reject, kpss_reject = adf[1] < 0.05, kp[1] < 0.05
    verdict = ("stationary" if adf_reject and not kpss_reject else
               "non-stationary" if (not adf_reject) and kpss_reject else "ambiguous")
    return {"series": name, "n": len(v), "adf_stat": float(adf[0]), "adf_p": float(adf[1]),
            "kpss_stat": float(kp[0]), "kpss_p": float(kp[1]), "verdict": verdict}


def deseasonalise(df: pd.DataFrame, col: str) -> pd.Series:
    """Remove the mean by (year, month, local hour, weekend) -- the profile that a calendar-aware model would
    capture -- leaving the part that must come from information. Index is kept."""
    d = df[[col, "year", "month", "local_hour", "is_weekend"]].dropna(subset=[col])
    fit = d.groupby(["year", "month", "local_hour", "is_weekend"])[col].transform("mean")
    return (d[col] - fit).rename(col + "_resid")


def dependence_summary(hourly: pd.Series, daily: pd.Series, resid: pd.Series, cfg: dict) -> Dict[str, pd.DataFrame]:
    """Tables used by the findings note."""
    lags_h = cfg["dependence"]["hourly_lags"]
    lags_d = cfg["dependence"]["daily_lags"]
    marks = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    marks = [m for m in marks if m <= lags_h]
    t_h = pd.DataFrame({"lag_hours": marks, "acf_price": [key_lags(hourly, [m])[m] for m in marks],
                        "acf_deseasonalised": [key_lags(resid, [m])[m] for m in marks]})
    dmarks = [m for m in [1, 2, 3, 7, 14, 30, 60] if m <= lags_d]
    t_d = pd.DataFrame({"lag_days": dmarks, "acf_daily_mean": [key_lags(daily, [m])[m] for m in dmarks]})
    sq = (resid - resid.mean()) ** 2
    t_v = pd.DataFrame({"lag_hours": [1, 24, 168], "acf_squared_residual": [key_lags(sq, [m])[m] for m in (1, 24, 168)]})
    lb = ljung_box(resid, cfg["dependence"]["ljung_box_lags"])
    return {"acf_hourly": t_h, "acf_daily": t_d, "volatility_clustering": t_v, "ljung_box_residual": lb}
