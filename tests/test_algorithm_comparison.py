"""
test_algorithm_comparison.py — stage 4: comparing the algorithms by out-of-sample error, checked on constructed forecasts.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from common_layer.model_development import comparison as cmp
from common_layer.model_development import comparison_figures as figs

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
D0 = dt.date(2023, 1, 1).toordinal()


def make(errors: dict, days: int = 200, seed: int = 0) -> dict:
    """Prediction tables with truth 100 and y_pred = 100 + the given error per model (array, or a scalar sd)."""
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    out = {}
    for m, e in errors.items():
        err = rng.normal(0, e, days * 24) if np.isscalar(e) else np.asarray(e, float)
        out[m] = pd.DataFrame({"y_true": 100.0, "y_pred": 100.0 + err, "fold": 0, "local_date": np.repeat(D0 + np.arange(days), 24),
                               "local_hour": np.tile(np.arange(24), days)}, index=idx)
    return out


def test_holm_matches_a_hand_calculation_and_never_lowers_a_p_value():
    adj = cmp.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])             # sorted 0.01*3, max(0.03*2=0.06), max(0.06, 0.04*1)
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))
    assert np.isnan(cmp.holm([0.5, float("nan")])[1]) and cmp.holm([0.5, float("nan")])[0] == pytest.approx(0.5)


def test_scoreboard_orders_by_mae_and_counts_days_best():
    prep = cmp.prepare(make({"good": 2.0, "mid": 5.0, "bad": 12.0}))
    b = cmp.scoreboard(prep)
    assert list(b["model"]) == ["good", "mid", "bad"] and (b["days_best"].sum() == pytest.approx(1.0))
    assert b.iloc[0]["days_best"] > 0.7 and b["mae"].is_monotonic_increasing
    assert b.iloc[0]["rmse"] == pytest.approx(np.sqrt(b.iloc[0]["sse"] / b.iloc[0]["n"]))


def test_only_the_hours_all_algorithms_predicted_are_compared():
    p = make({"a": 3.0, "b": 4.0})
    p["b"].iloc[:48, p["b"].columns.get_loc("y_pred")] = np.nan                  # b has no forecast for the first two days
    prep = cmp.prepare(p)
    assert all(len(v) == 200 * 24 - 48 for v in prep.values())


def test_pairwise_finds_a_real_difference_and_not_a_fake_one():
    prep = cmp.prepare(make({"good": 2.0, "twin": 2.0, "bad": 6.0}, days=300, seed=3))
    pw = cmp.pairwise(prep).set_index(["a", "b"])
    assert pw.loc[("good", "bad"), "significant"] and pw.loc[("good", "bad"), "better"] == "good"
    assert pw.loc[("twin", "bad"), "significant"]
    assert not pw.loc[("good", "twin"), "significant"]                            # the same error level: cannot be separated
    assert (pw["p_holm"] >= pw["p_value"] - 1e-15).all()


def test_the_matrix_is_antisymmetric_in_the_statistic():
    prep = cmp.prepare(make({"a": 2.0, "b": 5.0, "c": 3.0}))
    pw = cmp.pairwise(prep)
    m = cmp.matrix(pw, ["a", "b", "c"])
    assert m.loc["a", "b"] == pytest.approx(-m.loc["b", "a"]) and m.loc["a", "b"] < 0        # a has the smaller loss
    p = cmp.matrix(pw, ["a", "b", "c"], "p_holm")
    assert p.loc["a", "b"] == pytest.approx(p.loc["b", "a"])


def test_versus_best_names_the_best_and_the_algorithms_that_cannot_be_separated_from_it():
    prep = cmp.prepare(make({"best": 2.0, "twin": 2.0, "bad": 8.0}, days=300, seed=5))
    vb = cmp.versus_best(prep)
    assert vb["best"].iloc[0] in ("best", "twin")
    tops = cmp.top_set(vb)
    assert "bad" not in tops and len(tops) == 2


def test_win_table_by_hour_month_and_regime():
    # model "night" is exact at hours 0-11 and bad later; "day" the reverse
    days = 120
    hour = np.tile(np.arange(24), days)
    e_night = np.where(hour < 12, 0.5, 10.0)
    e_day = np.where(hour < 12, 10.0, 0.5)
    prep = cmp.prepare(make({"night": e_night, "day": e_day}, days=days))
    h = cmp.win_table(prep, "hour").set_index("group")
    assert all(h.loc[range(12), "lowest_mae"] == "night") and all(h.loc[range(12, 24), "lowest_mae"] == "day")
    assert (h["winner_share"] == 1.0).all() and h.loc[0, "units"] == days
    m = cmp.win_table(prep, "month")
    assert set(m["group"]) == {1, 2, 3, 4}                                          # 120 days from 2023-01-01
    periods = [{"id": "early", "start": dt.date(2023, 1, 1), "end": dt.date(2023, 2, 15)},
               {"id": "late", "start": dt.date(2023, 2, 16), "end": dt.date(2023, 12, 31)}]
    r = cmp.win_table(prep, "regime", periods).set_index("group")
    assert r.loc["early", "units"] == 46 and r.loc["late", "units"] == days - 46
    with pytest.raises(ValueError):
        cmp.win_table(prep, "regime")
    with pytest.raises(ValueError):
        cmp.win_table(prep, "weekday")


def test_a_better_average_algorithm_can_still_lose_more_days():
    """`winner` (most units) and `lowest_mae` (average) are different questions; both are reported."""
    days = 100
    steady = np.full(days * 24, 3.0)                                       # always off by 3
    risky = np.where(np.repeat(np.arange(days), 24) % 10 == 0, 40.0, 2.0)  # best on 90% of days, terrible on 10%
    prep = cmp.prepare(make({"steady": steady, "risky": risky}, days=days))
    t = cmp.win_table(prep, "month").iloc[0]
    assert t["winner"] == "risky" and t["lowest_mae"] == "steady"


def test_figures_are_written(tmp_path):
    prep = cmp.prepare(make({"a": 2.0, "b": 5.0, "c": 3.0}, days=120))
    pw = cmp.pairwise(prep)
    order = ["a", "b", "c"]
    paths = [figs.dm_heatmap(cmp.matrix(pw, order), cmp.matrix(pw, order, "p_holm"), str(tmp_path)),
             figs.mae_by_hour(cmp.win_table(prep, "hour"), order, str(tmp_path)),
             figs.mae_by_group(cmp.win_table(prep, "month"), order, str(tmp_path), "m.png", "month")]
    assert all(os.path.getsize(p) > 2000 for p in paths)


def test_the_tool_runs_end_to_end_and_is_repeatable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    errs = {"decision_tree": 9.0, "random_forest": 5.0, "gradient_boosting": 4.0, "xgboost": 4.2, "lightgbm": 3.8}
    for m, df in make(errs, days=150).items():
        df.to_csv(run / f"predictions_{m}_tuned.csv")
    outs = []
    for i in (1, 2):
        out = tmp_path / f"out{i}"
        r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "compare_algorithms.py"), "--run", str(run), "--suffix", "_tuned",
                            "--out", str(out)], capture_output=True, text=True, cwd=_REPO, stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stderr[-1500:]
        outs.append(out)
    md = (outs[0] / "algorithm_comparison.md").read_text(encoding="utf-8")
    assert "## 3. Which algorithms cannot be told apart from the best" in md and "by market regime" in md.lower()
    for t in sorted(os.listdir(outs[0] / "tables")):
        assert (outs[0] / "tables" / t).read_bytes() == (outs[1] / "tables" / t).read_bytes()
    r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "compare_algorithms.py"), "--run", str(run), "--out", str(tmp_path / "o3")],
                       capture_output=True, text=True, cwd=_REPO, stdin=subprocess.DEVNULL)
    assert r.returncode != 0                                                # no predictions without the suffix: a clear refusal
