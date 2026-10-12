"""
test_benchmark.py — the naive benchmark: the right earlier price for each delivery day, scored on the champion's hours, and nothing else.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.model_development import benchmark as bm

from tests.test_forecast_evaluation import D0, synthetic_panel

FORM = load_forecast_formulation()


def table(view: AsOfView, first: dt.date, days: int, noise: float = 0.0, seed: int = 3) -> pd.DataFrame:
    """A prediction table for `days` delivery days: the outcome, and a forecast that is the outcome plus noise."""
    r = np.random.default_rng(seed)
    frames = []
    for i in range(days):
        D = first + dt.timedelta(days=i)
        t = view.truth_day(D)
        frames.append(pd.DataFrame({"y_true": t[view.spec["series"]].to_numpy(), "local_date": t["local_date"].to_numpy(),
                                    "local_hour": t["local_hour"].to_numpy()}, index=t.index))
    p = pd.concat(frames)
    p["y_pred"] = p["y_true"] + r.normal(0, noise, len(p)) if noise else p["y_true"]
    return p


@pytest.fixture(scope="module")
def view() -> AsOfView:
    return AsOfView(synthetic_panel(120), FORM, "da_price")


def value(view: AsOfView, day: dt.date, hour: int) -> float:
    t = view.truth_day(day)
    return float(t.loc[t["local_hour"] == hour, view.spec["series"]].mean())


def test_the_naive_forecast_is_the_price_of_the_same_hour_one_day_or_one_week_earlier(view):
    p = table(view, D0 + dt.timedelta(days=30), 3)
    for lag, name in ((1, "naive_day"), (7, "naive_week")):
        n = bm.naive_prediction(view, p, bm.NAIVE_LAGS[name])
        for ts in (p.index[0], p.index[30], p.index[-1]):
            D = dt.date.fromordinal(int(p.loc[ts, "local_date"]))
            assert n.loc[ts] == pytest.approx(value(view, D - dt.timedelta(days=lag), int(p.loc[ts, "local_hour"])))


def test_the_seasonal_naive_uses_yesterday_midweek_and_last_week_around_the_weekend(view):
    first = D0 + dt.timedelta(days=28)
    p = table(view, first, 7)                                         # seven consecutive delivery days
    n = bm.naive_prediction(view, p, None)
    for D in (first + dt.timedelta(days=i) for i in range(7)):
        rows = p[p["local_date"] == D.toordinal()].index
        lag = 7 if D.weekday() in (5, 6, 0) else 1                    # Saturday, Sunday, Monday: last week; Tuesday to Friday: yesterday
        assert bm.lag_for(D.toordinal(), None) == lag
        assert n.loc[rows[5]] == pytest.approx(value(view, D - dt.timedelta(days=lag), 5))
    assert {bm.lag_for(dt.date(2022, 3, d).toordinal(), None) for d in (1, 2, 3, 4)} == {1}                    # Tuesday to Friday
    assert {bm.lag_for(dt.date(2022, 3, d).toordinal(), None) for d in (5, 6, 7)} == {7}                       # Saturday to Monday


def test_a_missing_earlier_price_gives_no_forecast_and_the_hour_is_dropped_for_everyone(view):
    p = table(view, D0 + dt.timedelta(days=0), 12, noise=2.0)         # the first days have no earlier week in the data
    tables = bm.with_naive(view, p)
    out = bm.assess(tables)
    assert out["hours"] < len(p) and out["dropped_hours"] == len(p) - out["hours"]
    assert {s["n"] for s in out["scores"].values()} == {out["hours"]}     # every forecast is scored on the same hours


def test_the_gain_over_a_naive_forecast_and_its_test_follow_the_errors(view):
    p = table(view, D0 + dt.timedelta(days=40), 60, noise=0.5)
    ref = p.copy()
    ref["y_pred"] = ref["y_true"] + 3.0
    out = bm.assess(bm.with_naive(view, p, ref))
    sc = out["scores"]
    assert sc["champion"]["mae"] < sc["reference"]["mae"] and sc["champion"]["mae"] < sc["naive_week"]["mae"]
    for name, gain in out["gain_of_champion_over"].items():
        assert gain == pytest.approx(1 - sc["champion"]["mae"] / sc[name]["mae"])
    assert out["gain_of_champion_over"]["reference"] > 0.5
    assert out["dm_champion_vs"]["reference"]["p_value"] < 0.001 and out["dm_champion_vs"]["reference"]["n_days"] == 60
    assert set(out["scores"]) == {"champion", "reference", "naive_seasonal", "naive_day", "naive_week"}


def test_a_champion_worse_than_the_naive_forecast_has_a_negative_gain(view):
    p = table(view, D0 + dt.timedelta(days=40), 40)
    p["y_pred"] = p["y_true"] + 50.0                                   # a bad champion
    out = bm.assess(bm.with_naive(view, p))
    assert out["gain_of_champion_over"]["naive_day"] < 0 and out["dm_champion_vs"]["naive_day"]["statistic"] > 0


def test_for_a_spread_target_the_naive_forecast_has_the_day_ahead_price_and_the_earlier_spread():
    v = AsOfView(synthetic_panel(120), FORM, "ida1_price")
    first = D0 + dt.timedelta(days=40)                                   # a Tuesday or so: check against the rule, not the weekday
    p = table(v, first, 3)
    t0 = v.truth_day(first)
    ts = p.index[7]
    D = dt.date.fromordinal(int(p.loc[ts, "local_date"]))
    prev = v.truth_day(D - dt.timedelta(days=bm.lag_for(D.toordinal(), None)))
    spread_before = float(v.modelled(prev).groupby(prev["local_hour"]).mean().loc[int(p.loc[ts, "local_hour"])])
    base_now = float(v.truth_day(D).loc[ts, v.spec["base"]])
    n = bm.naive_prediction(v, p, None)
    assert n.loc[ts] == pytest.approx(base_now + spread_before)
    tables = bm.with_naive(v, p)
    assert bm.BASE_ONLY in tables and tables[bm.BASE_ONLY].loc[ts, "y_pred"] == pytest.approx(base_now)    # no spread: the day-ahead price
    assert len(t0) > 0
    assert bm.BASE_ONLY not in bm.with_naive(AsOfView(synthetic_panel(120), FORM, "da_price"), table(AsOfView(synthetic_panel(120), FORM, "da_price"), first, 2))
