"""
dm.py — Diebold-Mariano test: is one forecaster's error significantly different from another's?

Two forecasters are compared on the same days. The 24 hourly errors of a day are strongly related, so the
test works on ONE loss difference per delivery day:  d_t = L(error of A) - L(error of B), where L is the
absolute or squared error averaged over the day's hours. The variance of the mean of d_t uses a Newey-West
(Bartlett kernel) estimate with `hac_lags` lags because the daily differences are autocorrelated, and the
Harvey-Leybourne-Newbold correction with a t distribution is applied for the modest number of days.

Reading the result: statistic > 0 means A has the LARGER loss (B is better); the p-value is two-sided.
This replaces the Wilcoxon test on at most four fold MAEs that the old model comparison used (finding F6).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def daily_losses(pred: pd.DataFrame, loss: str = "absolute") -> pd.Series:
    """Mean loss per delivery day of a prediction table (needs local_date, err)."""
    p = pred.dropna(subset=["err"])
    v = p["err"].abs() if loss == "absolute" else p["err"] ** 2
    return v.groupby(p["local_date"]).mean()


def diebold_mariano(pred_a: pd.DataFrame, pred_b: pd.DataFrame, loss: str = "absolute", hac_lags: int = 7,
                    small_sample: str = "hln") -> Dict[str, float]:
    la, lb = daily_losses(pred_a, loss), daily_losses(pred_b, loss)
    d = (la - lb).dropna()
    n = len(d)
    if n < 20 or d.std() == 0:
        return {"n_days": n, "mean_loss_a": float(la.mean()), "mean_loss_b": float(lb.mean()), "mean_diff": float(d.mean()) if n else np.nan,
                "statistic": np.nan, "p_value": np.nan}
    x = d.to_numpy() - d.mean()
    lags = min(hac_lags, n - 1)
    gamma0 = float(np.dot(x, x) / n)
    long_run = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        long_run += 2.0 * w * float(np.dot(x[k:], x[:-k]) / n)
    long_run = max(long_run, 1e-12)
    stat = d.mean() / np.sqrt(long_run / n)
    if small_sample == "hln":
        stat *= np.sqrt((n - 1) / n)                 # one-step-ahead (h = 1) form of the HLN factor
        p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    else:
        p = 2.0 * stats.norm.sf(abs(stat))
    return {"n_days": n, "mean_loss_a": float(la.mean()), "mean_loss_b": float(lb.mean()),
            "mean_diff": float(d.mean()), "statistic": float(stat), "p_value": float(p)}
