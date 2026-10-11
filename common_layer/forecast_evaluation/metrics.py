"""
metrics.py — SSE, MAE and RMSE, in price units.

    SSE   sum of squared errors   EUR^2/MWh^2   grows with the number of hours scored; compare only on the same hours
    MAE   mean |error|            EUR/MWh       (EUR/MW/h for the reserve band)
    RMSE  sqrt(mean error^2)      same unit as MAE; punishes large misses (spikes, negative-price hours)

SSE and RMSE rank forecasters identically on the same hours (RMSE = sqrt(SSE / n)); MAE can disagree with them
when the errors are heavy-tailed, and that disagreement is worth reading. Percentage errors are not used because
prices can be zero or negative.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def add_error_columns(pred: pd.DataFrame) -> pd.DataFrame:
    """Adds err (prediction minus outcome), abs_err and sq_err to a table with y_true and y_pred."""
    out = pred.copy()
    out["err"] = out["y_pred"] - out["y_true"]
    out["abs_err"] = out["err"].abs()
    out["sq_err"] = out["err"] ** 2
    return out


def score(pred: pd.DataFrame) -> Dict[str, float]:
    """SSE, MAE, RMSE and the number of points of a prediction table with the error columns."""
    p = pred.dropna(subset=["y_true", "y_pred"])
    if p.empty:
        return {"n": 0, "sse": np.nan, "mae": np.nan, "rmse": np.nan}
    return {"n": int(len(p)), "sse": float(p["sq_err"].sum()), "mae": float(p["abs_err"].mean()),
            "rmse": float(np.sqrt(p["sq_err"].mean()))}


def score_by(pred: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in pred.dropna(subset=["y_true", "y_pred"]).groupby(by):
        s = score(g)
        s[by] = key
        rows.append(s)
    cols = [by, "n", "sse", "mae", "rmse"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def all_scores(pred: pd.DataFrame, groupings: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for g in groupings:
        if g == "overall":
            out["overall"] = pd.DataFrame([score(pred)])
        else:
            col = {"hour_of_day": "local_hour", "month": "month", "period": "period"}[g]
            out[g] = score_by(pred, col)
    return out
