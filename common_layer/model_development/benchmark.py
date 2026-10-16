"""
benchmark.py — the naive benchmark: is the champion better than a rule that needs no model? (extension of stage 4 and 6 of
docs/Model_Development_Plan.pdf)

The tree algorithms are compared with the simplest algorithm (the decision tree), which is the reference of the selection rule.
That says what the extra complexity buys, but not whether machine learning is worth having at all. The standard answer in
forecasting is a naive forecast, which has no parameters to fit or tune:

    naive_seasonal   the price of the same local hour on the previous day for delivery days Tuesday to Friday, and one week
                     earlier for Saturday, Sunday and Monday (the standard weekday-aware benchmark of the price-forecasting
                     literature: yesterday is a good guide in mid-week, but not for a weekend or the Monday after it)
    naive_day        the price of the same local hour on the previous day
    naive_week       the price of the same local hour one week earlier

For the intraday targets, which are modelled as a spread to the day-ahead price (known when the forecast is made), the naive forecast is
the day-ahead price plus the naive spread, so it has the same information as the algorithms; one more, `base_price`, forecasts no
spread at all (the intraday price equals the day-ahead price), which is the plain "no change" forecast of an intraday market.

All use only prices that are known when the forecast is made: a lag is never shorter than the number of days between the delivery day
and the latest outcome of the target's own series that the information set allows (one day for most targets, two for XBID, three for
mFRR), so they are fair competitors. "The previous day" below therefore means the latest day available. They are scored on the SAME hours as the champion, and the champion is tested against
each with the Diebold-Mariano test used everywhere else. The benchmark never takes part in the choice of a champion and never changes
one; it is reported next to the result. A naive forecast has nothing to fit, so scoring it on the locked window consumes nothing.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation import metrics
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.dm import diebold_mariano
from common_layer.model_development import comparison

BASE_ONLY = "base_price"
SEASONAL = "naive_seasonal"
NAIVE_LAGS = {SEASONAL: None, "naive_day": 1, "naive_week": 7}        # None: the lag depends on the weekday of the delivery day


def lag_for(local_date: int, lag_days: Optional[int]) -> int:
    """Lag in days for a delivery day (ordinal): the given lag, or for the seasonal naive 7 on Saturday, Sunday and Monday and 1 otherwise."""
    if lag_days is not None:
        return lag_days
    return 7 if dt.date.fromordinal(int(local_date)).weekday() in (5, 6, 0) else 1


def base_prices(view: AsOfView, pred: pd.DataFrame) -> pd.Series:
    """The base (day-ahead) price of every row of a prediction table, for a spread target; zero for a level target."""
    base = view.spec.get("base")
    if view.spec["kind"] != "spread" or not base:
        return pd.Series(0.0, index=pred.index)
    days = sorted(set(int(x) for x in pred["local_date"]))
    b = pd.concat([view.truth_day(dt.date.fromordinal(o))[base] for o in days])
    return b[~b.index.duplicated()].reindex(pred.index)


def naive_prediction(view: AsOfView, pred: pd.DataFrame, lag_days: Optional[int]) -> pd.Series:
    """The naive forecast, in price units, for every row of a prediction table (needs local_date and local_hour): the modelled quantity
    of the same local hour `lag_days` earlier (None: the weekday-aware seasonal naive), plus the base price of the row for a spread
    target (zero for a level target). NaN where that earlier value is missing. In the hour that occurs twice at the clock change the
    two values are averaged. The lag is at least the number of days the information set keeps back the target's own series."""
    min_lag = max(1, -int(view.offsets.get(view.spec["series"], -1)))
    by_day: Dict[int, pd.Series] = {}
    for o in sorted(set(int(x) for x in pred["local_date"])):
        t = view.truth_day(dt.date.fromordinal(o - max(lag_for(o, lag_days), min_lag)))
        by_day[o] = view.modelled(t).groupby(t["local_hour"]).mean() if len(t) else pd.Series(dtype=float)
    vals = [by_day[int(o)].get(int(h), np.nan) for o, h in zip(pred["local_date"], pred["local_hour"])]
    return pd.Series(vals, index=pred.index, dtype=float) + base_prices(view, pred)


def with_naive(view: AsOfView, champion: pd.DataFrame, reference: Optional[pd.DataFrame] = None,
               lags: Optional[Dict[str, int]] = None) -> Dict[str, pd.DataFrame]:
    """Prediction tables (same columns as the champion's) for the champion, the reference and each naive forecast."""
    out = {"champion": champion}
    if reference is not None:
        out["reference"] = reference.reindex(champion.index)
    for name, lag in (lags or NAIVE_LAGS).items():
        t = champion.copy()
        t["y_pred"] = naive_prediction(view, champion, lag)
        out[name] = t
    if view.spec["kind"] == "spread":                                   # no spread at all: the intraday price equals the day-ahead price
        t = champion.copy()
        t["y_pred"] = base_prices(view, champion)
        out[BASE_ONLY] = t
    return out


def assess(preds: Dict[str, pd.DataFrame], loss: str = "absolute", hac_lags: int = 7, small_sample: str = "hln") -> dict:
    """Scores of every table on the hours all of them predicted, the gain of the champion over each other table, and the DM tests."""
    prep = comparison.prepare(preds)
    scores = {n: {k: float(v) for k, v in metrics.score(p).items()} for n, p in prep.items()}
    champ = scores["champion"]
    gain, dm = {}, {}
    for n in prep:
        if n == "champion":
            continue
        gain[n] = float(1 - champ["mae"] / scores[n]["mae"]) if scores[n]["mae"] else None
        d = diebold_mariano(prep["champion"], prep[n], loss, hac_lags, small_sample)
        dm[n] = {k: float(v) for k, v in d.items()}
    return {"hours": int(champ["n"]), "scores": scores, "gain_of_champion_over": gain, "dm_champion_vs": dm, "loss": loss,
            "dropped_hours": int(len(next(iter(preds.values())).dropna(subset=["y_true", "y_pred"])) - champ["n"])}
