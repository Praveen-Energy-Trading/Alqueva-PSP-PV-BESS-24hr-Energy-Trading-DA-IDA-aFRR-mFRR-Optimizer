"""
backtest.py — run one forecaster over the rolling-origin folds and score it.

For each fold: build a fresh forecaster, fit it on the fold's training days (as-of views and labels of known days
only), then predict every delivery day of the fold from that day's as-of view. Predictions are converted to
prices (a spread forecast plus the day-ahead price known at the gate) and compared with the outcome in price
units. Hours a forecaster does not predict (NaN) are left out of its metrics and counted in `coverage`.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation import metrics
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.forecasters import FitContext, Forecaster, _rows_of
from common_layer.forecast_evaluation.folds import Fold, make_folds


@dataclass
class BacktestResult:
    target_id: str
    forecaster: str
    predictions: pd.DataFrame
    folds: pd.DataFrame
    meta: Dict[str, object] = field(default_factory=dict)

    def scores(self, groupings: List[str]) -> Dict[str, pd.DataFrame]:
        return metrics.all_scores(self.predictions, groupings)


def period_of(day: dt.date, periods: List[dict]) -> str:
    for p in periods:
        if p["start"] <= day <= p["end"]:
            return p["id"]
    return "other"


def run_backtest(factory: Callable[[], Forecaster], view: AsOfView, test_start: dt.date, test_end: dt.date,
                 schedule: dict, periods: List[dict], history_days: int = 400,
                 name: Optional[str] = None) -> BacktestResult:
    folds = make_folds(test_start, test_end, view.first_day(), schedule)
    spread = view.spec["kind"] == "spread"
    series = view.spec["series"]
    base = view.spec.get("base")
    frames, fold_rows = [], []
    fname = name
    for fold in folds:
        f = factory()
        fname = fname or getattr(f, "name", "forecaster")
        ctx = FitContext(view, fold.train_days, fold.train_end, history_days)
        f.fit(ctx)
        fold_rows.append({"fold": fold.fold_id, "first_test_day": fold.test_days[0], "last_test_day": fold.test_days[-1],
                          "train_start": fold.train_start, "train_end": fold.train_end})
        for D in fold.test_days:
            v = view.at(D, history_days)
            rows = _rows_of(v, D)
            if rows.empty:
                continue
            yhat = np.asarray(f.predict(v, D), dtype=float)
            if len(yhat) != len(rows):
                raise ValueError(f"{fname} returned {len(yhat)} values for {len(rows)} hours on {D}")
            truth = view.truth_day(D)
            price_pred = yhat + rows[base].to_numpy() if spread else yhat
            frames.append(pd.DataFrame({
                "y_true": truth[series].to_numpy(), "y_pred": price_pred, "fold": fold.fold_id,
                "local_date": rows["local_date"].to_numpy(), "local_hour": rows["local_hour"].to_numpy(),
                "month": D.month, "period": period_of(D, periods)}, index=rows.index))
    pred = metrics.add_error_columns(pd.concat(frames)) if frames else pd.DataFrame()
    have = pred.dropna(subset=["y_true"])
    coverage = float(have["y_pred"].notna().mean()) if len(have) else np.nan
    return BacktestResult(view.target_id, fname or "forecaster", pred, pd.DataFrame(fold_rows),
                          {"coverage": coverage, "n_folds": len(folds), "test_start": str(test_start),
                           "test_end": str(test_end)})
