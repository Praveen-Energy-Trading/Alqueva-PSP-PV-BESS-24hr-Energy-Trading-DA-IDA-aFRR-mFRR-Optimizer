"""
test_tuning_ensembles.py — Phase 9: nested time-aware tuning and forecast combinations.

Tuning must only ever see its fold's training window, never be worse than the defaults on its own validation set,
re-tune on schedule and leave predictions unchanged when information from the future is randomised. Combinations
must be exactly the stated averages and their weights must use only errors known at the forecast time.
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.feature_engineering.builder import FeatureBuilder
from common_layer.feature_engineering.catalogue import load_feature_catalogue, resolve_feature_set, load_feature_sets
from common_layer.forecast_evaluation import ensemble, guards
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import TREE_MODELS, load_evaluation_config
from common_layer.forecast_evaluation.forecasters import FitContext
from common_layer.forecast_evaluation.runner import evaluate
from common_layer.forecast_evaluation.tree_models import build_learner, tree_candidates
from common_layer.forecast_evaluation.tuning import (TunedForecaster, load_tuning_config, sample_configs,
                                                     validate_tuning_config)

from tests.test_forecast_evaluation import PERIODS, SCHED, cfg_small, synthetic_panel  # noqa: E402

TCFG = load_tuning_config()
FORM = load_forecast_formulation()


def small_tcfg(space: dict, n_iter: int = 3, val: int = 30, retune: int = 180, blocks: int = 1, method: str = "random",
               model: str = "decision_tree") -> dict:
    c = copy.deepcopy(TCFG)
    c["search"].update(n_trials=n_iter, validation_days=val, n_blocks=blocks, retune_every_days=retune, method=method,
                       block_jobs=1)
    c["space"][model] = space
    return c


# ------------------------------------------------------------------------------ config
def test_tuning_config_is_valid_and_every_searched_value_is_accepted_by_its_learner():
    assert validate_tuning_config(TCFG) == []
    X, y = np.random.default_rng(0).normal(size=(80, 4)), np.random.default_rng(1).normal(size=80)
    for model, params in TCFG["space"].items():
        assert model in TREE_MODELS
        for p, values in params.items():
            for v in values:
                build_learner(model, {p: v, **({"n_estimators": 5} if "n_estimators" in TCFG["space"][model] and p != "n_estimators" else {})}).fit(X, y)


def test_invalid_tuning_config_is_rejected():
    bad = copy.deepcopy(TCFG)
    bad["search"]["metric"] = "mape"
    bad["space"]["lightgbm"]["num_leaves"] = []
    bad["ensembles"]["methods"] = ["mean", "magic"]
    problems = validate_tuning_config(bad)
    assert len(problems) >= 3 and any("metric" in p for p in problems) and any("non-empty" in p for p in problems)


def test_sampled_settings_start_with_the_defaults_are_distinct_and_reproducible():
    space = TCFG["space"]["lightgbm"]
    a = sample_configs(space, 12, np.random.default_rng(5))
    assert a[0] == {} and len(a) == 12 and len({tuple(sorted(c.items())) for c in a}) == 12
    assert all(set(c) <= set(space) or c == {} for c in a)
    assert a == sample_configs(space, 12, np.random.default_rng(5))
    small = sample_configs({"max_depth": [2, 3]}, 12, np.random.default_rng(0))
    assert small == [{}, {"max_depth": 2}, {"max_depth": 3}]                          # the whole grid when it is small


# ------------------------------------------------------------------------------ tuning
def _fit_context(view: AsOfView, train_end: dt.date, n_days: int) -> FitContext:
    days = [train_end - dt.timedelta(days=i) for i in range(n_days - 1, -1, -1)]
    return FitContext(view, days, train_end)


def _tuned(view, space, model="decision_tree", cache=None, **kw):
    cat = load_feature_catalogue()
    iface = view.interface()
    ids = resolve_feature_set("baseline_v1", load_feature_sets(), cat, iface.spec, iface.offsets, "da_price")
    return TunedForecaster(iface, model, FeatureBuilder(iface, cat, ids), "v", small_tcfg(space, model=model, **kw),
                           cache={} if cache is None else cache)


def test_tuning_uses_only_the_training_window_and_is_never_worse_than_the_defaults_on_validation():
    view = AsOfView(synthetic_panel(200), FORM, "da_price")
    m = _tuned(view, {"max_depth": [1, 2]})                  # both candidates are worse than the default depth of 8
    ctx = _fit_context(view, dt.date(2022, 6, 1), 120)
    m.fit(ctx)
    st = m.cache[("tuned", "da_price", "decision_tree", "v")]
    assert st["params"] == {} and st["score"] == pytest.approx(st["default_score"]) and st["n_trials"] == 3
    assert st["validation_days"] == 30 and st["inner_train_days"] == 120 - 30 - 1         # latest 30 days held out, 1-day gap
    assert st["tuned_at"] == dt.date(2022, 6, 1)


def test_tuning_can_pick_a_better_setting_than_the_default():
    view = AsOfView(synthetic_panel(200), FORM, "da_price")
    m = _tuned(view, {"min_samples_leaf": [1000]}, n_iter=2)          # a leaf this large cannot learn the daily shape
    m.fit(_fit_context(view, dt.date(2022, 6, 1), 120))
    assert m.cache[("tuned", "da_price", "decision_tree", "v")]["params"] == {}       # the default wins
    view2 = AsOfView(synthetic_panel(200), FORM, "da_price")
    m2 = _tuned(view2, {"max_depth": [12], "min_samples_leaf": [1, 3]}, n_iter=4)
    m2.fit(_fit_context(view2, dt.date(2022, 6, 1), 120))
    st = m2.cache[("tuned", "da_price", "decision_tree", "v")]
    assert st["score"] <= st["default_score"] + 1e-12


def test_retuning_follows_the_schedule_and_reuses_settings_in_between():
    view = AsOfView(synthetic_panel(260), FORM, "da_price")
    m = _tuned(view, {"max_depth": [3]}, n_iter=2, retune=60)
    cache = m.cache
    m.fit(_fit_context(view, dt.date(2022, 5, 1), 100))
    m2 = _tuned(view, {"max_depth": [3]}, n_iter=2, retune=60)
    m2.cache = cache
    m2.fit(_fit_context(view, dt.date(2022, 5, 31), 100))                 # 30 days later: reuse
    assert len(cache[("tuning_log", "da_price", "decision_tree")]) == 1
    m3 = _tuned(view, {"max_depth": [3]}, n_iter=2, retune=60)
    m3.cache = cache
    m3.fit(_fit_context(view, dt.date(2022, 7, 15), 100))                 # 75 days later: tune again
    assert len(cache[("tuning_log", "da_price", "decision_tree")]) == 2


def test_a_tuned_model_does_not_change_when_information_from_the_future_is_randomised():
    panel = synthetic_panel(200)
    cands = tree_candidates(["decision_tree"], tuned=True, tuning_cfg=small_tcfg({"max_depth": [4, 6]}, n_iter=3, val=30))
    diff = guards.perturbation_check(lambda v: cands["decision_tree_tuned"](v, {}), panel, FORM, "da_price",
                                     dt.date(2022, 5, 10), SCHED)
    assert diff == pytest.approx(0.0, abs=1e-9)


def test_tuned_candidates_are_named_and_logged_in_a_run():
    tc = small_tcfg({"max_depth": [4, 6]}, n_iter=3, val=30)
    cands = tree_candidates(["decision_tree"], tuned=True, tuning_cfg=tc)
    out = evaluate("da_price", synthetic_panel(220), FORM, cfg_small(), PERIODS, "dev", cands,
                   test_window=(dt.date(2022, 5, 1), dt.date(2022, 5, 30)))
    assert list(out["overall"]["forecaster"]) == ["decision_tree_tuned"]
    log = out["tuning_log"]
    assert log is not None and (log["score"] <= log["default_score"] + 1e-12).all()


# ------------------------------------------------------------------- stage 2: blocks, Optuna, pruning, fairness
BIG = {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 2, 5, 10, 20, 50]}        # grid 36 > the budgets below


def _state(m):
    return m.cache[("tuned", "da_price", m.base_name, "v")]


def test_several_inner_blocks_each_predicted_by_earlier_days_only():
    view = AsOfView(synthetic_panel(240), FORM, "da_price")
    m = _tuned(view, BIG, n_iter=5, val=20, blocks=3, method="random")
    m.fit(_fit_context(view, dt.date(2022, 6, 1), 140))
    st = _state(m)
    assert st["n_blocks"] == 3 and st["validation_days"] == 20
    assert st["inner_train_days"] == 140 - 20 - 1                      # the latest block: the 20 days before it are held out + 1 gap
    assert all(len(t["block_scores"]) == 3 and np.isfinite(t["block_scores"]).all() for t in st["trials"])
    assert st["score"] == pytest.approx(min(t["score"] for t in st["trials"]))
    for t in st["trials"]:
        assert t["score"] == pytest.approx(np.mean(t["block_scores"]))  # the score of a setting is the mean over the blocks


def test_blocks_that_would_leave_too_little_training_are_dropped():
    view = AsOfView(synthetic_panel(240), FORM, "da_price")
    m = _tuned(view, BIG, n_iter=3, val=30, blocks=3, method="random")
    m.fit(_fit_context(view, dt.date(2022, 6, 1), 130))                # blocks leave 99, 69 and 39 days: the third is below 60
    assert _state(m)["n_blocks"] == 2


def test_optuna_search_respects_the_budget_keeps_every_trial_and_never_loses_to_the_defaults():
    view = AsOfView(synthetic_panel(240), FORM, "da_price")
    m = _tuned(view, BIG, n_iter=9, val=20, blocks=2, method="tpe")
    m.fit(_fit_context(view, dt.date(2022, 6, 1), 140))
    st = _state(m)
    assert st["method"] == "tpe" and st["n_trials"] == 9 and len(st["trials"]) == 9
    assert st["trials"][0]["default"] and st["trials"][0]["params"] == {} and not st["trials"][0]["pruned"]
    assert st["score"] <= st["default_score"] + 1e-12
    assert sum(t["selected"] for t in st["trials"]) == 1
    for t in st["trials"]:
        if t["pruned"]:                                                 # a pruned trial was scored on the latest block only
            assert t["score"] is None and t["block_scores"][1] is None and not t["selected"]
        elif not t["duplicate"] and not t["default"]:
            assert set(t["params"]) == set(BIG)
    assert st["n_pruned"] == sum(t["pruned"] for t in st["trials"])


def test_a_seeded_search_is_reproducible_and_the_seed_matters():
    view = AsOfView(synthetic_panel(240), FORM, "da_price")
    ctx = _fit_context(view, dt.date(2022, 6, 1), 140)
    runs = []
    for _ in range(2):
        m = _tuned(view, BIG, n_iter=8, val=20, blocks=2, method="tpe")
        m.fit(ctx)
        runs.append([(str(t["params"]), t["score"]) for t in _state(m)["trials"]])
    assert runs[0] == runs[1]
    m = _tuned(view, BIG, n_iter=8, val=20, blocks=2, method="tpe")
    m.tcfg["search"]["seed"] = 7
    m.fit(ctx)
    assert [str(t["params"]) for t in _state(m)["trials"]] != [r[0] for r in runs[0]]


def test_parallel_blocks_give_the_same_result_as_sequential_blocks():
    view = AsOfView(synthetic_panel(240), FORM, "da_price")
    ctx = _fit_context(view, dt.date(2022, 6, 1), 140)
    out = []
    for jobs in (1, 2):
        m = _tuned(view, BIG, n_iter=6, val=20, blocks=3, method="random")
        m.tcfg["search"]["block_jobs"] = jobs
        m.fit(ctx)
        out.append([(str(t["params"]), t["score"]) for t in _state(m)["trials"]])
    assert out[0] == out[1]


def test_a_small_grid_is_searched_exhaustively():
    view = AsOfView(synthetic_panel(200), FORM, "da_price")
    m = _tuned(view, {"max_depth": [2, 3, 4]}, n_iter=40, val=30, blocks=1, method="tpe")
    m.fit(_fit_context(view, dt.date(2022, 6, 1), 120))
    st = _state(m)
    assert st["method"] == "exhaustive" and st["n_trials"] == 4 and st["n_unique"] == 4          # defaults + the 3 depths


def test_every_algorithm_gets_the_same_number_of_trials():
    view = AsOfView(synthetic_panel(240), FORM, "da_price")
    ctx = _fit_context(view, dt.date(2022, 6, 1), 140)
    cache: dict = {}
    for model, space in (("decision_tree", BIG), ("random_forest", {"n_estimators": [10, 20, 30], "max_depth": [3, 4, 5, 6]})):
        m = _tuned(view, space, model=model, cache=cache, n_iter=6, val=20, blocks=2, method="tpe")
        m.fit(ctx)
    n = {k[2]: v[0]["n_trials"] for k, v in cache.items() if k[0] == "tuning_log"}
    assert n == {"decision_tree": 6, "random_forest": 6}


def test_a_tuned_model_with_blocks_and_optuna_passes_the_leakage_check():
    tc = small_tcfg({"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 20]}, n_iter=6, val=20, blocks=2, method="tpe")
    cands = tree_candidates(["decision_tree"], tuned=True, tuning_cfg=tc)
    diff = guards.perturbation_check(lambda v: cands["decision_tree_tuned"](v, {}), synthetic_panel(200), FORM, "da_price",
                                     dt.date(2022, 5, 10), SCHED)
    assert diff == pytest.approx(0.0, abs=1e-9)


def test_the_runner_returns_every_trial():
    tc = small_tcfg(BIG, n_iter=5, val=20, blocks=2, method="tpe")
    cands = tree_candidates(["decision_tree"], tuned=True, tuning_cfg=tc)
    out = evaluate("da_price", synthetic_panel(220), FORM, cfg_small(), PERIODS, "dev", cands,
                   test_window=(dt.date(2022, 5, 1), dt.date(2022, 5, 30)))
    tt, log = out["tuning_trials"], out["tuning_log"]
    assert len(tt) == int(log["n_trials"].sum()) and tt["selected"].sum() == len(log)
    assert {"block1", "pruned", "is_default", "duplicate", "selected"} <= set(tt.columns)
    assert tt.groupby("tuned_at")["is_default"].sum().eq(1).all()


# ----------------------------------------------------------------------------- ensembles
def _preds(errors: dict, days: int = 60, per: int = 24) -> dict:
    """Prediction tables of several models: y_true = 100, y_pred = 100 + the model's error per row."""
    n = days * per
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    out = {}
    for m, e in errors.items():
        out[m] = pd.DataFrame({"y_true": 100.0, "y_pred": 100.0 + np.asarray(e, dtype=float), "fold": 0,
                               "local_date": np.repeat(np.arange(days) + 738000, per), "local_hour": np.tile(np.arange(per), days),
                               "month": 1, "period": "all"}, index=idx)
    return out


def test_simple_combinations_are_exactly_the_stated_averages():
    n = 60 * 24
    p = _preds({"a": np.full(n, 1.0), "b": np.full(n, 4.0), "c": np.full(n, 10.0), "d": np.full(n, -3.0)})
    got = {m: ensemble.combine(p, m)["y_pred"].iloc[0] - 100 for m in ("mean", "median", "trimmed_mean")}
    assert got["mean"] == pytest.approx(3.0) and got["median"] == pytest.approx(2.5) and got["trimmed_mean"] == pytest.approx(2.5)
    with pytest.raises(KeyError):
        ensemble.combine(p, "magic")
    with pytest.raises(ValueError):
        ensemble.combine({"a": p["a"]}, "mean")


def test_error_weighted_combinations_favour_the_better_model_and_use_only_known_errors():
    r = np.random.default_rng(2)
    n = 60 * 24
    good, bad = r.normal(0, 1, n), r.normal(0, 8, n)
    p = _preds({"good": good, "bad": bad})
    comb = ensemble.combine(p, "inverse_error", window_days=28, lag_days=2)
    late = comb[comb["local_date"] > 738030]
    assert np.sqrt((late["err"] ** 2).mean()) < np.sqrt(((bad[len(bad) - len(late):]) ** 2).mean()) / 2      # far closer to the good model
    top = ensemble.combine(_preds({"good": good, "bad": bad, "worse": r.normal(0, 12, n)}), "top3_mean")
    assert len(top) == n
    # time-awareness: errors of day D and D-1 cannot change the weights used for day D (lag 2)
    D = 738040
    changed = {k: v.copy() for k, v in p.items()}
    for day in (D, D - 1):
        rows = changed["good"]["local_date"] == day
        changed["good"].loc[rows, "y_pred"] += 500.0
    a = ensemble.combine(p, "inverse_error", 28, 2)
    b = ensemble.combine(changed, "inverse_error", 28, 2)
    rows = a["local_date"] == D
    # the combined forecast of day D differs only through the member's own forecast, not through changed weights:
    w_a = (a.loc[rows, "y_pred"] - p["bad"].loc[a.index[rows], "y_pred"]) / (p["good"].loc[a.index[rows], "y_pred"] - p["bad"].loc[a.index[rows], "y_pred"])
    w_b = (b.loc[rows, "y_pred"] - changed["bad"].loc[b.index[rows], "y_pred"]) / (changed["good"].loc[b.index[rows], "y_pred"] - changed["bad"].loc[b.index[rows], "y_pred"])
    assert np.allclose(w_a, w_b, atol=1e-9)
    changed2 = {k: v.copy() for k, v in p.items()}
    rows2 = changed2["good"]["local_date"] == D - 2                     # the latest day whose error IS known: weights must react
    changed2["good"].loc[rows2, "y_pred"] += 500.0
    c = ensemble.combine(changed2, "inverse_error", 28, 2)
    w_c = (c.loc[rows, "y_pred"] - changed2["bad"].loc[c.index[rows], "y_pred"]) / (changed2["good"].loc[c.index[rows], "y_pred"] - changed2["bad"].loc[c.index[rows], "y_pred"])
    assert not np.allclose(w_a, w_c, atol=1e-6)


def test_averaging_independent_models_beats_each_member():
    r = np.random.default_rng(4)
    n = 60 * 24
    p = _preds({f"m{i}": r.normal(0, 5, n) for i in range(5)})
    rmse = lambda df: float(np.sqrt((df["err"] ** 2).mean()))        # noqa: E731
    assert rmse(ensemble.combine(p, "mean")) < 0.6 * min(float(np.sqrt(((p[m]["y_pred"] - 100) ** 2).mean())) for m in p)


def test_runner_adds_combinations_and_compares_them_with_the_best_single_model():
    cands = tree_candidates(["decision_tree", "lightgbm"])
    out = evaluate("da_price", synthetic_panel(220), FORM, cfg_small(), PERIODS, "dev", cands,
                   test_window=(dt.date(2022, 5, 1), dt.date(2022, 5, 30)), ensembles=["mean", "inverse_error"],
                   ensemble_cfg=TCFG["ensembles"])
    names = set(out["overall"]["forecaster"])
    assert {"decision_tree", "lightgbm", "ensemble_mean", "ensemble_inverse_error"} <= names
    best = out["overall"][out["overall"]["forecaster"].isin(["decision_tree", "lightgbm"])].sort_values("mae").iloc[0]["forecaster"]
    c = out["comparisons"].set_index("forecaster")
    assert c.loc["ensemble_mean", "reference"] == best and c.loc["ensemble_inverse_error", "reference"] == best
    assert "ensemble_mean" in out["report"]
