"""
forecasters.py — the forecaster interface, what a forecaster is given, and the shared lag-feature model.

A forecaster is any object with
    fit(ctx)            learn from the training days of one fold; ctx gives as-of views and labels of KNOWN days only
    predict(view, D)    the modelled quantity (level, or spread to the base price) for the hours of delivery day D,
                        as a numpy array aligned with the rows of `view` whose local_date is D, in price units

`FeatureModel` trains a learner on the features of a feature set (config/feature_sets.yaml), built by
common_layer/feature_engineering.builder.FeatureBuilder: the SAME function that serves evaluation and live
prediction, so training and serving cannot differ. The features of a day depend only on that day's as-of view, so a
cache shared between folds avoids recomputation.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation.asof import AsOfView, ModelInterface, matrix
from common_layer.feature_engineering.builder import FeatureBuilder
from common_layer.forecast_formulation.transforms import make_transform


@dataclass
class FitContext:
    _view: AsOfView
    train_days: List[dt.date]
    train_end: dt.date
    history_days: int = 400

    def view_at(self, D: dt.date) -> pd.DataFrame:
        if D > self.train_end:
            raise PermissionError(f"day {D} is after the end of this fold's training data ({self.train_end})")
        return self._view.at(D, self.history_days)

    def label(self, D: dt.date) -> pd.DataFrame:
        """Known outcome of a TRAINING day: index = UTC hour, columns y (price units) and local_hour."""
        if D > self.train_end:
            raise PermissionError(f"the label of {D} is not known at the end of this fold's training data ({self.train_end})")
        t = self._view.truth_day(D)
        return pd.DataFrame({"y": self._view.modelled(t), "local_hour": t["local_hour"]})


class Forecaster:
    name = "forecaster"

    def fit(self, ctx: FitContext) -> None:
        return None

    def predict(self, view: pd.DataFrame, D: dt.date) -> np.ndarray:
        raise NotImplementedError


def _rows_of(view: pd.DataFrame, D: dt.date) -> pd.DataFrame:
    return view[view["local_date"] == D.toordinal()]


class FeatureModel(Forecaster):
    """Trains `make_learner()` on a feature set, in the formulation's transformed target space, and returns forecasts
    in price units. Features that are missing for every training row are dropped; remaining gaps are filled with
    the training means so every learner sees identical inputs."""

    def __init__(self, view: ModelInterface, make_learner: Callable[[], object], builder: FeatureBuilder,
                 set_version: str = "", cache: Optional[Dict] = None):
        self.view, self.builder, self.set_version = view, builder, set_version
        self._make_learner = make_learner
        tr = view.spec["transform"]
        self.tr = make_transform(tr["name"], tr.get("scale"))
        self.cache = cache if cache is not None else {}
        self.model = None
        self.fill: Optional[np.ndarray] = None
        self.cols: List[str] = []

    def _day_features(self, ctx: FitContext, D: dt.date) -> pd.DataFrame:
        key = (self.view.target_id, self.set_version, D)
        if key not in self.cache:
            self.cache[key] = self.builder.build(ctx.view_at(D), D)
        return self.cache[key]

    def _collect(self, ctx: FitContext) -> pd.DataFrame:
        """The training table: one row per hour of every training day with a known outcome, the features of the
        builder, the outcome in price units (`y_raw`) and the day (`local_date`, ordinal)."""
        frames = []
        for D in ctx.train_days:
            lab = ctx.label(D)
            if lab["y"].notna().sum() == 0:
                continue
            t = lab["y"].rename("y_raw").to_frame().join(self._day_features(ctx, D), how="inner").dropna(subset=["y_raw"])
            t["local_date"] = D.toordinal()
            frames.append(t)
        return pd.concat(frames)

    def fit(self, ctx: FitContext) -> None:
        T = self._collect(ctx)
        Xd = T[self.builder.ids]
        Xd = Xd.loc[:, Xd.notna().any()]
        self.cols = list(Xd.columns)
        self.fill = np.nanmean(Xd.to_numpy(), axis=0)
        A = np.where(np.isnan(Xd.to_numpy()), self.fill, Xd.to_numpy())
        self.model = self._make_learner().fit(A, self.tr.forward(T["y_raw"].to_numpy()))

    def predict(self, view: pd.DataFrame, D: dt.date) -> np.ndarray:
        f = self.builder.build(view, D)[self.cols]
        A = np.where(np.isnan(f.to_numpy()), self.fill, f.to_numpy())
        return self.tr.inverse(self.model.predict(A))
