"""
test_model_registry.py — stage 1 of the model-development section: every run is recorded and traceable.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation.runner import evaluate
from common_layer.forecast_evaluation.tree_models import tree_candidates
from common_layer.model_development import registry

from tests.test_forecast_evaluation import PERIODS, cfg_small, synthetic_panel

FORM = load_forecast_formulation()
BASE = dict(kind="evaluation", target="da_price", stage="dev", window=("2023-01-01", "2023-01-31"),
            feature_set="baseline_v1", schedule={"retrain_every_days": 30}, config_digest="c" * 8, data_digest="d" * 8)


def rows(mae=10.0):
    return [{"forecaster": "lightgbm", "n": 100, "sse": 5000.0, "mae": mae, "rmse": 12.0, "coverage": 1.0},
            {"forecaster": "lightgbm_tuned", "n": 100, "sse": 4800.0, "mae": mae - 1, "rmse": 11.5, "coverage": 1.0},
            {"forecaster": "ensemble_mean", "n": 100, "sse": 4700.0, "mae": mae - 1.2, "rmse": 11.4, "coverage": 1.0}]


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "reg" / "registry.db")


def test_a_run_is_recorded_with_its_scores_comparisons_and_tuning_choices(db):
    r = registry.register(models=rows(), comparisons=[{"forecaster": "lightgbm_tuned", "reference": "lightgbm", "n_days": 31,
                          "mean_diff": -1.0, "statistic": -2.5, "p_value": 0.01}],
                          tuning_choices=[{"model": "lightgbm", "tuned_at": dt.date(2022, 12, 30), "score": 9.0, "default_score": 10.0,
                                           "n_trials": 12, "inner_train_days": 639, "validation_days": 90, "params": "{'a': 1}"}],
                          code_revision="abc1234", db_path=db, **BASE)
    assert r["attempt"] == 1 and r["reproduced"] is None and len(r["run_key"]) == 12
    m = registry.models_of(r["run_key"], db_path=db).set_index("forecaster")
    assert m.loc["lightgbm_tuned", "tuned"] == 1 and m.loc["lightgbm", "tuned"] == 0
    assert m.loc["ensemble_mean", "role"] == "ensemble" and m.loc["lightgbm", "role"] == "single"
    assert list(registry.models_of(r["run_key"], db_path=db)["forecaster"])[0] == "ensemble_mean"      # sorted by MAE
    c = registry.comparisons_of(r["run_key"], db_path=db)
    assert c.iloc[0]["p_value"] == pytest.approx(0.01)
    t = registry.tuning_choices_of(r["run_key"], db_path=db)
    assert t.iloc[0]["tuned_at"] == "2022-12-30" and t.iloc[0]["params_json"] == "{'a': 1}"
    runs = registry.list_runs(db_path=db)
    assert len(runs) == 1 and runs.iloc[0]["code_revision"] == "abc1234" and runs.iloc[0]["attempts"] == 1


def test_repeating_an_experiment_is_checked_for_reproducibility(db):
    a = registry.register(models=rows(), code_revision="r1", db_path=db, **BASE)
    b = registry.register(models=rows(), code_revision="r2", db_path=db, **BASE)          # same experiment, later commit
    assert b["run_key"] == a["run_key"] and b["attempt"] == 2 and b["reproduced"] is True
    assert registry.reproducibility(db_path=db).iloc[0]["reproduced"]
    c = registry.register(models=rows(mae=10.5), db_path=db, **BASE)                      # different scores for the same key
    assert c["attempt"] == 3 and c["reproduced"] is False
    assert not registry.reproducibility(db_path=db).iloc[0]["reproduced"]


def test_a_different_experiment_gets_a_different_key(db):
    k = registry.register(models=rows(), db_path=db, **BASE)["run_key"]
    for change in ({"target": "ida1_price"}, {"stage": "final"}, {"data_digest": "other"}, {"feature_set": "full_v1"},
                   {"schedule": {"retrain_every_days": 7}}, {"tuning_digest": "t"}, {"window": ("2023-01-01", "2023-02-28")}):
        assert registry.register(models=rows(), db_path=db, **{**BASE, **change})["run_key"] != k
    assert registry.register(models=rows()[:2], db_path=db, **BASE)["run_key"] != k       # a different set of forecasters


def test_bad_input_is_rejected(db):
    with pytest.raises(ValueError):
        registry.register(models=[], db_path=db, **BASE)
    with pytest.raises(ValueError):
        registry.register(models=[{"forecaster": "x", "sse": 1.0}], db_path=db, **BASE)
    with pytest.raises(KeyError):
        registry.models_of("nope", db_path=db)


def test_a_real_evaluation_run_is_registered_and_a_repeat_reproduces_it(db):
    cands = tree_candidates(["decision_tree", "lightgbm"])
    kw = dict(test_window=(dt.date(2022, 5, 1), dt.date(2022, 5, 30)), ensembles=["mean"])
    res = evaluate("da_price", synthetic_panel(220), FORM, cfg_small(), PERIODS, "dev", cands, **kw)
    r1 = registry.register_evaluation(res, source="test", db_path=db)
    res2 = evaluate("da_price", synthetic_panel(220), FORM, cfg_small(), PERIODS, "dev", tree_candidates(["decision_tree", "lightgbm"]), **kw)
    r2 = registry.register_evaluation(res2, source="test", db_path=db)
    assert r2["run_key"] == r1["run_key"] and r2["reproduced"] is True
    m = registry.models_of(r1["run_key"], db_path=db)
    assert set(m["forecaster"]) == {"decision_tree", "lightgbm", "ensemble_mean"}
    assert m.set_index("forecaster").loc["lightgbm", "mae"] == pytest.approx(res["overall"].set_index("forecaster").loc["lightgbm", "mae"])
    assert len(registry.comparisons_of(r1["run_key"], db_path=db)) == len(res["comparisons"])


def test_every_tuning_trial_of_a_run_is_stored(db):
    import copy as _copy
    from tests.test_tuning_ensembles import BIG, small_tcfg
    tc = small_tcfg(BIG, n_iter=5, val=20, blocks=2, method="tpe")
    res = evaluate("da_price", synthetic_panel(220), FORM, cfg_small(), PERIODS, "dev",
                   tree_candidates(["decision_tree"], tuned=True, tuning_cfg=tc), test_window=(dt.date(2022, 5, 1), dt.date(2022, 5, 30)))
    r = registry.register_evaluation(res, tuning_cfg=tc, db_path=db)
    t = registry.tuning_trials_of(r["run_key"], db_path=db)
    assert len(t) == len(res["tuning_trials"]) and t["selected"].sum() == len(res["tuning_log"])
    assert json.loads(t.iloc[0]["block_scores"]) and t.iloc[0]["is_default"] == 1


def test_an_earlier_run_folder_can_be_backfilled(db, tmp_path):
    d = tmp_path / "old_run"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"target": "da_price", "stage": "dev", "test_start": "2023-01-01", "test_end": "2025-10-11",
                                            "feature_set": "baseline_v1", "retrain_every_days": 30, "gap_days": 1,
                                            "train_window_days": 730, "config_digest": "cfg", "data_digest": "dat", "code": "552d357"}))
    pd.DataFrame(rows()[:2]).to_csv(d / "metrics_overall.csv", index=False)
    pd.DataFrame([{"model": "lightgbm", "tuned_at": "2022-12-30", "score": 9.0, "default_score": 10.0, "n_trials": 12,
                   "inner_train_days": 639, "validation_days": 90, "params": "{}"}]).to_csv(d / "tuning_log.csv", index=False)
    r = registry.register_run_dir(str(d), tuning_cfg={"search": {"n_iter": 12}}, db_path=db)
    runs = registry.list_runs(db_path=db)
    assert runs.iloc[0]["source"] == "backfill" and runs.iloc[0]["code_revision"] == "552d357"
    assert len(registry.tuning_choices_of(r["run_key"], db_path=db)) == 1
    assert len(registry.models_of(r["run_key"], db_path=db)) == 2


def test_the_default_location_can_be_set_by_environment(tmp_path, monkeypatch):
    p = str(tmp_path / "x" / "r.db")
    monkeypatch.setenv("MODEL_REGISTRY_DB", p)
    registry.register(models=rows(), **BASE)
    assert os.path.exists(p)
