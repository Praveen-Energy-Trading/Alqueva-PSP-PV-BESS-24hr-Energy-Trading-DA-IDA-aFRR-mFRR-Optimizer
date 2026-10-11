"""
test_tuning_evidence.py — stage 3: the tables and figures that show what the tuning did, checked by hand on small data.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation.runner import evaluate
from common_layer.forecast_evaluation.tree_models import tree_candidates
from common_layer.model_development import evidence as ev
from common_layer.model_development import evidence_figures as figs

from tests.test_forecast_evaluation import PERIODS, cfg_small, synthetic_panel
from tests.test_tuning_ensembles import BIG, small_tcfg

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FORM = load_forecast_formulation()


def trials_table() -> pd.DataFrame:
    """One model, one round. Defaults 10.0; others 8.0 and 7.0 (a=1), 9.0 (a=2), and one pruned trial."""
    rows = [(0, "{}", 10.0, False, True), (1, "{'a': 1}", 8.0, False, False), (2, "{'a': 2}", 9.0, False, False),
            (3, "{'a': 2}", np.nan, True, False), (4, "{'a': 1}", 7.0, False, False)]
    df = pd.DataFrame([{"model": "m", "tuned_at": "2023-01-01", "trial": t, "params": p, "score": s, "pruned": pr, "is_default": d,
                        "duplicate": False, "selected": t == 4} for t, p, s, pr, d in rows])
    df.to_csv  # noqa: B018 (a plain frame; load_trials is tested through files below)
    return df


def _loaded(tmp_path) -> pd.DataFrame:
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    trials_table().to_csv(d / "tuning_trials.csv", index=False)
    return ev.load_trials(str(d))


def test_relative_scores_use_the_defaults_of_the_same_round(tmp_path):
    t = _loaded(tmp_path)
    assert list(t["rel"].round(3)[:3]) == [1.0, 0.8, 0.9] and np.isnan(t["rel"][3]) and t["rel"][4] == pytest.approx(0.7)
    assert t["params_d"][1] == {"a": 1} and t["params_d"][0] == {}


def test_convergence_is_the_running_best_and_a_pruned_trial_never_improves_it(tmp_path):
    c = ev.best_so_far(_loaded(tmp_path))
    assert list(c["best_rel"].round(3)) == [1.0, 0.8, 0.8, 0.8, 0.7] and list(c["trial"]) == [0, 1, 2, 3, 4]


def test_effects_and_importance_match_a_hand_calculation(tmp_path):
    t = _loaded(tmp_path)
    e = ev.effects(t, "m").set_index("value")
    assert e.loc[1, "n"] == 2 and e.loc[1, "mean_rel"] == pytest.approx(0.75) and e.loc[1, "best_rel"] == pytest.approx(0.7)
    assert e.loc[2, "n"] == 1 and e.loc[2, "mean_rel"] == pytest.approx(0.9)
    imp = ev.importance(t, "m")
    assert imp.iloc[0]["setting"] == "a" and imp.iloc[0]["eta2"] == pytest.approx(0.75)        # between 0.015 / total 0.02


def test_a_setting_that_changes_nothing_has_no_importance():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"model": "m", "tuned_at": "t", "trial": range(n), "is_default": False,
                       "rel": 0.9 + 0.01 * rng.normal(size=n), "params_d": [{"x": int(v)} for v in rng.integers(0, 3, n)]})
    assert ev.importance(df, "m").iloc[0]["eta2"] < 0.15


def test_stability_counts_how_often_the_same_value_is_chosen():
    log = pd.DataFrame({"model": "m", "tuned_at": ["a", "b", "c", "d"], "params": ["{'k': 3}", "{'k': 3}", "{'k': 5}", "{}"]})
    st = ev.stability(ev.selected_settings(log)).set_index("setting")
    assert st.loc["k", "rounds"] == 3 and st.loc["k", "most_frequent"] == "3" and st.loc["k", "share_most_frequent"] == pytest.approx(2 / 3)
    assert "(defaults)" in st.index


def test_inner_against_outer_uses_the_days_each_setting_was_used_on():
    day = lambda s: dt.date.fromisoformat(s).toordinal()        # noqa: E731
    log = pd.DataFrame({"model": "m", "tuned_at": ["2023-01-01", "2023-07-01"], "score": [8.0, 9.0], "default_score": [10.0, 10.0]})
    days = np.arange(day("2023-01-03"), day("2023-12-31"))
    first = days <= day("2023-07-02")                              # round 1 covers up to the day after round 2's training end
    y = np.full(len(days), 100.0)
    tuned = pd.DataFrame({"y_true": y, "y_pred": y + np.where(first, 2.0, 4.0), "local_date": days})
    default = pd.DataFrame({"y_true": y, "y_pred": y + 5.0, "local_date": days})
    r = ev.inner_vs_outer(log, {"m": tuned}, {"m": default}).set_index("tuned_at")
    assert r.loc["2023-01-01", "outer_tuned"] == pytest.approx(2.0) and r.loc["2023-07-01", "outer_tuned"] == pytest.approx(4.0)
    assert r.loc["2023-01-01", "outer_default"] == pytest.approx(5.0)
    assert r.loc["2023-01-01", "inner_gain"] == pytest.approx(0.2) and r.loc["2023-01-01", "outer_gain"] == pytest.approx(0.6)
    assert r.loc["2023-01-01", "gap"] == pytest.approx(2.0 - 8.0) and r.loc["2023-07-01", "gap"] == pytest.approx(4.0 - 9.0)


def test_pruning_summary_counts():
    s = ev.pruning_summary(trials_table()).iloc[0]
    assert s["trials"] == 5 and s["pruned"] == 1 and s["share_pruned"] == pytest.approx(0.2)


def test_figures_are_written(tmp_path):
    t = _loaded(tmp_path)
    c = ev.best_so_far(t)
    for p in (figs.convergence({"tpe": c, "random": c}, str(tmp_path)), figs.setting_effects(ev.effects(t, "m"), "m", str(tmp_path)),
              figs.importance_bars(ev.importance(t, "m"), str(tmp_path))):
        assert os.path.getsize(p) > 2000


def _run_folder(path, method, tuned=True):
    tc = small_tcfg(BIG, n_iter=6, val=20, blocks=2, method=method)
    cands = tree_candidates(["decision_tree"], tuned=tuned, tuning_cfg=tc)
    res = evaluate("da_price", synthetic_panel(240), FORM, cfg_small(), PERIODS, "dev", cands, str(path),
                   test_window=(dt.date(2022, 5, 1), dt.date(2022, 6, 20)))
    if tuned:
        res["tuning_log"].to_csv(path / "tuning_log.csv", index=False)
        res["tuning_trials"].to_csv(path / "tuning_trials.csv", index=False)
    return res


def test_the_report_tool_runs_end_to_end_and_is_repeatable(tmp_path):
    runs = {k: tmp_path / k for k in ("tpe", "random", "untuned")}
    for k, p in runs.items():
        p.mkdir()
    _run_folder(runs["tpe"], "tpe")
    _run_folder(runs["random"], "random")
    _run_folder(runs["untuned"], "random", tuned=False)
    outs = []
    for i in (1, 2):
        out = tmp_path / f"out{i}"
        cmd = [sys.executable, os.path.join(_REPO, "tools", "make_tuning_evidence.py"), "--tpe", str(runs["tpe"]), "--random",
               str(runs["random"]), "--untuned", str(runs["untuned"]), "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO, stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stderr[-1500:]
        outs.append(out)
    md = (outs[0] / "tuning_evidence.md").read_text(encoding="utf-8")
    assert "## 1. How fast the search converges" in md and "## 4. The gain seen when choosing" in md
    assert os.path.getsize(outs[0] / "figures" / "tuning_convergence.png") > 2000
    tables = sorted(os.listdir(outs[0] / "tables"))
    assert "convergence.csv" in tables and "inner_vs_outer_tpe.csv" in tables
    for t in tables:                                                                    # byte-identical on repeat
        assert (outs[0] / "tables" / t).read_bytes() == (outs[1] / "tables" / t).read_bytes()
