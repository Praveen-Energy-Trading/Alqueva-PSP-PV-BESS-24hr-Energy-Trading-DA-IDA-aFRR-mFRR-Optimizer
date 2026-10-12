"""
test_prediction_guard.py — forecasts are limited to the range of the training outcomes (plus a margin), in training, in the
tuning validation and when serving, so a numerically unstable boosted model cannot produce astronomical prices.

Background: in the day-ahead experiments a tuned XGBoost with the pseudo-Huber objective predicted up to 5e21 EUR/MWh for one
month, because boosted models can leave the range of their training outcomes and the target transform (asinh) is inverted by sinh.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.feature_engineering.builder import FeatureBuilder
from common_layer.feature_engineering.catalogue import (feature_set_version, load_feature_catalogue, load_feature_sets,
                                                         resolve_feature_set)
from common_layer.forecast_evaluation import tuning
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.forecast_evaluation.folds import make_folds
from common_layer.forecast_evaluation.forecasters import CLIP_MARGIN, FeatureModel, FitContext, clip_to_training_range
from common_layer.forecast_evaluation.runner import evaluate
from common_layer.forecast_evaluation.tree_models import build_learner, tree_candidates
from common_layer.model_development import champions, serving

from tests.test_forecast_evaluation import PERIODS, SCHED, cfg_small, synthetic_panel
from tests.test_locked_test import cfg_locked, decision
from tests.test_tuning_ensembles import small_tcfg

FORM = load_forecast_formulation()
FS = load_evaluation_config()["models"]["feature_set"]
T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


class Boom:
    """A learner that diverges: it predicts an enormous value in the transformed space, whatever the input."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), 1.0e3)


def test_values_inside_the_range_pass_unchanged_and_others_are_limited_and_counted():
    z, n = clip_to_training_range(np.array([-0.5, 0.0, 2.0, 2.5]), (-1.0, 2.0))
    assert n == 0 and list(z) == [-0.5, 0.0, 2.0, 2.5]                                  # 2.5 is within the margin of 1.0
    z, n = clip_to_training_range(np.array([-9.0, 1.0, 3.5, 1e9, np.inf]), (-1.0, 2.0))
    assert list(z) == [-1.0 - CLIP_MARGIN, 1.0, 2.0 + CLIP_MARGIN, 3.0, 3.0] and n == 4
    assert np.isfinite(np.sinh(z)).all()


def _fit(view, make_learner, day=dt.date(2022, 6, 1)):
    cat, sets = load_feature_catalogue(), load_feature_sets()
    iface = view.interface()
    ids = resolve_feature_set(FS, sets, cat, iface.spec, iface.offsets, "da_price")
    m = FeatureModel(iface, make_learner, FeatureBuilder(iface, cat, ids), feature_set_version(ids, cat))
    fold = make_folds(day, day, view.first_day(), {**SCHED, "retrain_every_days": 1})[0]
    m.fit(FitContext(view, fold.train_days, fold.train_end, 400))
    return m


def test_a_diverging_model_gives_bounded_prices_and_a_failure_count():
    view = AsOfView(synthetic_panel(220), FORM, "da_price")
    m = _fit(view, Boom)
    day = dt.date(2022, 6, 1)
    p = m.predict(view.at(day, 400), day)
    assert np.isfinite(p).all() and m.n_clipped == len(p) == 24
    zmax = m.z_range[1] + CLIP_MARGIN
    assert p.max() == pytest.approx(m.tr.inverse(np.array([zmax]))[0]) and p.max() < 1.0e4


def test_a_well_behaved_model_is_not_changed_by_the_guard():
    view = AsOfView(synthetic_panel(220), FORM, "da_price")
    m = _fit(view, lambda: build_learner("lightgbm"))
    day = dt.date(2022, 6, 1)
    v = view.at(day, 400)
    f = m.builder.build(v, day)[m.cols]
    raw = m.tr.inverse(m.model.predict(np.where(np.isnan(f.to_numpy()), m.fill, f.to_numpy())))
    np.testing.assert_array_equal(m.predict(v, day), raw)
    assert m.n_clipped == 0


def test_the_tuning_validation_scores_a_diverging_setting_as_bad_not_as_infinite(monkeypatch):
    real = tuning.build_learner
    monkeypatch.setattr(tuning, "build_learner", lambda name, params=None: Boom() if (params or {}).get("boom") else real(name, {}))
    view = AsOfView(synthetic_panel(240), FORM, "da_price")
    cat, sets = load_feature_catalogue(), load_feature_sets()
    iface = view.interface()
    ids = resolve_feature_set(FS, sets, cat, iface.spec, iface.offsets, "da_price")
    tf = tuning.TunedForecaster(iface, "decision_tree", FeatureBuilder(iface, cat, ids), "v", small_tcfg({"boom": [0, 1]}, n_iter=5, val=20, blocks=2),
                                cache={})
    day = dt.date(2022, 6, 1)
    fold = make_folds(day, day, view.first_day(), {**SCHED, "retrain_every_days": 1})[0]
    ctx = FitContext(view, fold.train_days, fold.train_end, 400)
    st = tf._search(tf._collect(ctx), ctx)
    boom = [t for t in st["trials"] if t["params"].get("boom") == 1][0]
    assert np.isfinite(boom["score"]) and boom["score"] > 10 * st["default_score"]               # bad, but a number
    assert st["params"] == {} and not boom["selected"]


def test_the_backtest_counts_the_forecasts_that_had_to_be_limited():
    panel = synthetic_panel(220)
    cat, sets = load_feature_catalogue(), load_feature_sets()

    def boom_factory(iface, cache):
        ids = resolve_feature_set(FS, sets, cat, iface.spec, iface.offsets, iface.target_id)
        return FeatureModel(iface, Boom, FeatureBuilder(iface, cat, ids), feature_set_version(ids, cat), cache)
    out = evaluate("da_price", panel, FORM, cfg_small(), PERIODS, "dev", {"boom": boom_factory, **tree_candidates(["lightgbm"])},
                   test_window=(dt.date(2022, 5, 1), dt.date(2022, 5, 20)))
    r = out["results"]
    assert r["boom"].meta["clipped"] == 20 * 24 and r["lightgbm"].meta["clipped"] == 0
    assert np.isfinite(r["boom"].predictions["y_pred"]).all() and r["boom"].predictions["y_pred"].abs().max() < 1.0e4


def test_the_served_champion_has_the_same_guard(tmp_path):
    db = str(tmp_path / "r.db")
    panel, cfg = synthetic_panel(220), cfg_locked()
    champions.promote_initial(decision({"type": "single", "model": "lightgbm"}), FS, "d", db, T0)
    serving.fit_artifact("da_price", panel, FORM, cfg, {}, dt.date(2022, 6, 1), db, now=T0)
    view = AsOfView(panel, FORM, "da_price")
    fc = serving.load_champion("da_price", view.interface(), db)
    day = dt.date(2022, 6, 1)
    v = view.at(day, 400)
    assert np.isfinite(fc.predict(v, day)).all() and fc.n_clipped == 0
    fc.artifact["members"]["lightgbm"]["model"] = Boom()                                          # the model breaks after fitting
    p = fc.predict(v, day)
    assert np.isfinite(p).all() and p.max() < 1.0e4 and fc.n_clipped == 24
