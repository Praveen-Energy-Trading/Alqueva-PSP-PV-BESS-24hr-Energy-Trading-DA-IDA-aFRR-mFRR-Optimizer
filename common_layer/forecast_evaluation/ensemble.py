"""
ensemble.py — forecast combinations of the tree models.

Simple averages are tested first because they are hard to beat. All methods use only what is known at the forecast
time: a weight for delivery day D comes from the daily absolute errors of days up to D - `lag_days` (the errors of the
most recent days are not yet known when day D is forecast), never from D itself.

    mean            average of the models
    median          median of the models
    trimmed_mean    average without the highest and the lowest forecast of each hour
    inverse_error   weights proportional to 1 / trailing MAE of each model
    top3_mean       average of the three models with the lowest trailing MAE

The inputs are the out-of-sample predictions of the base models from the rolling backtest, so a combination is as
out-of-sample as its members; trailing windows without enough history fall back to the plain average.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation import metrics

METHODS = ("mean", "median", "trimmed_mean", "inverse_error", "top3_mean")
_KEEP = ["y_true", "fold", "local_date", "local_hour", "month", "period"]


def _matrix(preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """y_pred of every model on the hours all of them predicted."""
    idx = None
    for p in preds.values():
        ok = p.dropna(subset=["y_true", "y_pred"]).index
        idx = ok if idx is None else idx.intersection(ok)
    return pd.DataFrame({m: p.loc[idx, "y_pred"] for m, p in preds.items()})


def trailing_error(preds: Dict[str, pd.DataFrame], window_days: int, lag_days: int) -> pd.DataFrame:
    """Per delivery day (ordinal) and model: mean of the daily MAE over the `window_days` days ending `lag_days` before it."""
    first = next(iter(preds.values()))
    Y = _matrix(preds)
    truth = first.loc[Y.index]
    err = Y.sub(truth["y_true"], axis=0).abs()
    daily = err.groupby(truth["local_date"]).mean()
    full = daily.reindex(range(int(daily.index.min()), int(daily.index.max()) + 1))
    return full.shift(lag_days).rolling(window_days, min_periods=7).mean()


def combine(preds: Dict[str, pd.DataFrame], method: str, window_days: int = 28, lag_days: int = 2) -> pd.DataFrame:
    """Prediction table (same columns as a backtest) of the combination of the models' predictions."""
    if method not in METHODS:
        raise KeyError(f"unknown ensemble method {method!r}; available: {METHODS}")
    if len(preds) < 2:
        raise ValueError("a combination needs at least two models")
    Y = _matrix(preds)
    first = next(iter(preds.values()))
    base = first.loc[Y.index, _KEEP].copy()
    A = Y.to_numpy()
    if method == "mean":
        out = A.mean(axis=1)
    elif method == "median":
        out = np.median(A, axis=1)
    elif method == "trimmed_mean":
        srt = np.sort(A, axis=1)
        out = srt[:, 1:-1].mean(axis=1) if A.shape[1] >= 3 else A.mean(axis=1)
    else:
        trail = trailing_error(preds, window_days, lag_days).reindex(base["local_date"].to_numpy())[list(Y.columns)].to_numpy()
        out = np.empty(len(A))
        known = np.isfinite(trail).all(axis=1) & (trail > 0).all(axis=1)
        out[~known] = A[~known].mean(axis=1)
        if method == "inverse_error":
            w = 1.0 / trail[known]
            out[known] = (A[known] * w).sum(axis=1) / w.sum(axis=1)
        else:                                               # top3_mean
            k = min(3, A.shape[1])
            best = np.argsort(trail[known], axis=1)[:, :k]
            out[known] = np.take_along_axis(A[known], best, axis=1).mean(axis=1)
    base["y_pred"] = out
    return metrics.add_error_columns(base)
