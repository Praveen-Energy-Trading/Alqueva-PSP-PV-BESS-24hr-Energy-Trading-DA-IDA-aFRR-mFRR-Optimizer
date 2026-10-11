"""
test_champions.py — stage 7: the champion registry, the promotion rule, and serving the fitted champion.
"""
from __future__ import annotations

import copy
import datetime as dt
import os
import pickle
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.forecast_evaluation.folds import make_folds
from common_layer.forecast_evaluation.forecasters import FitContext
from common_layer.forecast_evaluation.tree_models import tree_candidates
from common_layer.model_development import champions, registry, serving

from tests.test_algorithm_comparison import D0, make
from tests.test_forecast_evaluation import SCHED, synthetic_panel
from tests.test_locked_test import cfg_locked, decision
from tests.test_tuning_ensembles import small_tcfg

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FORM = load_forecast_formulation()
CMP = load_evaluation_config()["comparison"]
POL = champions.load_champion_policy()
FS = load_evaluation_config()["models"]["feature_set"]
T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "reg" / "registry.db")


def table(err, days=150, seed=0):
    return make({"x": err}, days=days, seed=seed)["x"]


# ---------------------------------------------------------------------------------------------- the registry
def test_the_policy_in_the_repository_is_valid_and_bad_ones_are_rejected(tmp_path):
    assert POL["window_days"] == 90 and POL["min_improvement"] == 0.02
    bad = tmp_path / "p.yaml"
    bad.write_text("promotion: {window_days: 5, alpha: 2, min_improvement: -1, min_days_between_promotions: 1, loss: mape}")
    with pytest.raises(ValueError):
        champions.load_champion_policy(str(bad))


def test_the_first_champion_comes_from_a_decision_and_there_is_only_ever_one_active(db):
    assert champions.active("da_price", db) is None
    r = champions.promote_initial(decision({"type": "ensemble", "members": ["xgboost", "lightgbm"], "method": "mean"}, tuned=True), FS, "d.json", db, T0)
    a = champions.active("da_price", db)
    assert r["version"] == 1 and a["version"] == 1 and a["spec"]["members"] == ["xgboost", "lightgbm"] and a["spec"]["tuned"]
    assert a["spec"]["type"] == "ensemble" and a["spec"]["method"] == "mean" and a["promoted_utc"] == T0.isoformat()
    with pytest.raises(ValueError, match="already has an active champion"):
        champions.promote_initial(decision(), FS, "d.json", db)
    with sqlite3.connect(db) as con:                                                    # the database itself refuses a second active
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO champions VALUES ('da_price', 9, 'active', '{}', NULL, NULL, '', 'x', NULL, NULL)")
    bad = decision()
    bad.update(status="insufficient_evidence", champion=None)
    with pytest.raises(ValueError, match="chosen champion"):
        champions.promote_initial(bad, FS, "d.json", db)
    assert list(champions.events("da_price", db)["event"]) == ["initial_promotion"]


# ---------------------------------------------------------------------------------------- the promotion rule
def test_a_clearly_better_challenger_is_promoted_after_the_window():
    champ, chal = table(5.0, 150, 1), table(3.0, 150, 2)
    ev = champions.evaluate_challenge(champ, chal, POL, CMP, dt.date.fromordinal(D0 + 149), 120)
    assert ev["promote"] and ev["n_days"] == 90 and ev["improvement"] > 0.2 and ev["p_value"] < 0.001
    assert ev["policy_digest"] == registry.digest_of(POL)


def test_each_part_of_the_rule_can_block_a_promotion():
    as_of = dt.date.fromordinal(D0 + 149)
    same = champions.evaluate_challenge(table(5.0, 150, 1), table(5.0, 150, 2), POL, CMP, as_of, 120)
    assert not same["promote"] and "not significantly better" in same["reasons"][0]
    worse = champions.evaluate_challenge(table(3.0, 150, 1), table(5.0, 150, 2), POL, CMP, as_of, 120)
    assert not worse["promote"] and worse["improvement"] < 0
    cool = champions.evaluate_challenge(table(5.0, 150, 1), table(3.0, 150, 2), POL, CMP, as_of, 10)
    assert not cool["promote"] and any("last promotion was 10 days ago" in r for r in cool["reasons"])
    short = champions.evaluate_challenge(table(5.0, 60, 1), table(3.0, 60, 2), POL, CMP, dt.date.fromordinal(D0 + 59), 120)
    assert not short["promote"] and "not enough evidence" in short["reasons"][0]


def test_a_significant_but_tiny_improvement_does_not_cause_churn():
    rng = np.random.default_rng(4)
    e = rng.normal(0, 5.0, 150 * 24)
    ev = champions.evaluate_challenge(table(e), table(0.985 * e), POL, CMP, dt.date.fromordinal(D0 + 149), 120)
    assert ev["p_value"] < 0.05 and 0 < ev["improvement"] < 0.02 and not ev["promote"]
    assert any("below the required 2%" in r for r in ev["reasons"]) and len(ev["reasons"]) == 1


def test_only_days_up_to_the_as_of_date_are_used():
    rng = np.random.default_rng(5)
    n = 200 * 24
    chal_err = np.where(np.arange(n) // 24 < 100, rng.normal(0, 5.0, n), rng.normal(0, 0.5, n))    # excellent only after day 100
    champ = table(rng.normal(0, 5.0, n), 200)
    early = champions.evaluate_challenge(champ, table(chal_err, 200), POL, CMP, dt.date.fromordinal(D0 + 99), 120)
    assert not early["promote"] and early["n_days"] == 90
    changed = table(np.where(np.arange(n) // 24 < 100, chal_err, 1000.0), 200)                       # garbage after as_of: same answer
    again = champions.evaluate_challenge(champ, changed, POL, CMP, dt.date.fromordinal(D0 + 99), 120)
    assert again["challenger_loss"] == early["challenger_loss"]


# ----------------------------------------------------------------------- challenges, promotions and rollbacks
def test_a_promotion_retires_the_old_champion_and_a_rejection_changes_nothing(db):
    champions.promote_initial(decision(), FS, "d1", db, T0)
    as_of = dt.date.fromordinal(D0 + 149)
    good = champions.evaluate_challenge(table(5.0, 150, 1), table(3.0, 150, 2), POL, CMP, as_of, 120)
    new = decision({"type": "single", "model": "xgboost"})
    r = champions.apply_challenge("da_price", new, FS, good, "d2", db, T0 + dt.timedelta(days=120))
    assert r["promoted"] and r["version"] == 2 and r["replaced"] == 1
    assert champions.active("da_price", db)["spec"]["members"] == ["xgboost"]
    v = champions.versions("da_price", db)
    assert list(v["status"]) == ["retired", "active"] and v["retired_utc"].iloc[0] is not None
    bad = champions.evaluate_challenge(table(5.0, 150, 1), table(5.0, 150, 3), POL, CMP, as_of, 120)
    r2 = champions.apply_challenge("da_price", decision(), FS, bad, "d3", db)
    assert not r2["promoted"] and champions.active("da_price", db)["version"] == 2
    ev = champions.events("da_price", db)
    assert list(ev["event"]) == ["initial_promotion", "promotion", "challenge_rejected"]
    assert "p_value" in ev["detail_json"].iloc[2]                                                      # the numbers are in the log
    with pytest.raises(ValueError, match="no active champion"):
        champions.apply_challenge("ida1_price", new, FS, good, "x", db)


def test_the_cooldown_counts_from_the_promotion_time(db):
    champions.promote_initial(decision(), FS, "d1", db, T0)
    assert champions.days_since_last_promotion("da_price", (T0 + dt.timedelta(days=45)).date(), db) == 45.0
    assert champions.days_since_last_promotion("ida1_price", T0.date(), db) is None


def test_a_rollback_needs_a_reason_and_restores_an_earlier_version(db):
    champions.promote_initial(decision(), FS, "d1", db, T0)
    ev = champions.evaluate_challenge(table(5.0, 150, 1), table(3.0, 150, 2), POL, CMP, dt.date.fromordinal(D0 + 149), 120)
    champions.apply_challenge("da_price", decision({"type": "single", "model": "xgboost"}), FS, ev, "d2", db)
    with pytest.raises(ValueError, match="reason"):
        champions.rollback("da_price", 1, "  ", db)
    with pytest.raises(ValueError, match="already active"):
        champions.rollback("da_price", 2, "why", db)
    with pytest.raises(KeyError):
        champions.rollback("da_price", 7, "why", db)
    r = champions.rollback("da_price", 1, "the new model drifted", db)
    assert r == {"version": 1, "from_version": 2} and champions.active("da_price", db)["spec"]["members"] == ["lightgbm"]
    assert champions.events("da_price", db)["event"].iloc[-1] == "rollback"


# ------------------------------------------------------------------------------------------------- serving
def _setup(db, dec, target="da_price", tcfg=None):
    panel, cfg = synthetic_panel(220), cfg_locked()
    champions.promote_initial(dec, FS, "d", db, T0)
    day = dt.date(2022, 6, 1)
    out = serving.fit_artifact(target, panel, FORM, cfg, tcfg or {"ensembles": {}}, day, db, now=T0)
    view = AsOfView(panel, FORM, target)
    return panel, cfg, view, day, out


def _reference_fit(view, spec_members, day, cache=None, tuned=False, tcfg=None):
    fold = make_folds(day, day, view.first_day(), {**SCHED, "min_train_days": 60, "retrain_every_days": 1})[0]
    ctx = FitContext(view, fold.train_days, fold.train_end, 400)
    cands = tree_candidates(spec_members, FS, tuned, tcfg)
    out = {}
    for n, f in cands.items():
        fc = f(view.interface(), {})
        fc.fit(ctx)
        out[n] = fc
    return out, ctx


def test_the_served_champion_gives_the_same_forecast_as_the_model_fitted_in_the_evaluation(db):
    panel, cfg, view, day, out = _setup(db, decision())
    ref, _ = _reference_fit(view, ["lightgbm"], day)
    srv = serving.load_champion("da_price", view.interface(), db)
    for D in (day, day + dt.timedelta(days=3), day + dt.timedelta(days=9)):
        v = view.at(D, 400)
        np.testing.assert_allclose(srv.predict(v, D), ref["lightgbm"].predict(v, D), rtol=0, atol=1e-9)
    assert srv.days_since_fit(day + dt.timedelta(days=9)) == 11 and out["train_end"] == day - dt.timedelta(days=2)


def test_an_averaged_champion_is_served_as_the_average_of_its_members(db):
    panel, cfg, view, day, _ = _setup(db, decision({"type": "ensemble", "members": ["decision_tree", "lightgbm"], "method": "mean"}))
    ref, _ = _reference_fit(view, ["decision_tree", "lightgbm"], day)
    srv = serving.load_champion("da_price", view.interface(), db)
    v = view.at(day, 400)
    np.testing.assert_allclose(srv.predict(v, day), (ref["decision_tree"].predict(v, day) + ref["lightgbm"].predict(v, day)) / 2, atol=1e-9)


def test_a_tuned_champion_is_served_with_the_settings_it_was_tuned_to(db):
    tc = small_tcfg({"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 20]}, n_iter=6, val=20, blocks=2, method="tpe")
    panel, cfg, view, day, out = _setup(db, decision({"type": "single", "model": "decision_tree"}, tuned=True), tcfg=tc)
    srv = serving.load_champion("da_price", view.interface(), db)
    ref, _ = _reference_fit(view, ["decision_tree"], day, tuned=True, tcfg=tc)
    v = view.at(day, 400)
    np.testing.assert_allclose(srv.predict(v, day), ref["decision_tree_tuned"].predict(v, day), atol=1e-9)
    assert srv.artifact["members"]["decision_tree"]["params"] == ref["decision_tree_tuned"].params


def test_a_spread_target_is_served_as_a_price(db):
    panel, cfg = synthetic_panel(220), cfg_locked()
    dec = decision({"type": "single", "model": "decision_tree"})
    dec["target"] = "ida1_price"
    champions.promote_initial(dec, FS, "d", db, T0)
    day = dt.date(2022, 6, 1)
    serving.fit_artifact("ida1_price", panel, FORM, cfg, {}, day, db, now=T0)
    view = AsOfView(panel, FORM, "ida1_price")
    srv = serving.load_champion("ida1_price", view.interface(), db)
    v = view.at(day, 400)
    rows = v[v["local_date"] == day.toordinal()]
    np.testing.assert_allclose(srv.forecast_prices(v, day).to_numpy(), srv.predict(v, day) + rows[view.spec["base"]].to_numpy(), atol=1e-9)


def test_anything_that_does_not_match_the_registry_is_refused(db, monkeypatch):
    panel, cfg, view, day, out = _setup(db, decision())
    iface = view.interface()
    serving.load_champion("da_price", iface, db)                                                       # fine as registered
    monkeypatch.setattr(serving, "feature_set_version", lambda ids, cat: "changed")                    # the definitions moved on
    with pytest.raises(serving.ChampionError, match="feature definitions changed"):
        serving.load_champion("da_price", iface, db)
    monkeypatch.undo()
    with open(out["path"], "ab") as fh:                                                                # tampering with the file
        fh.write(b"x")
    with pytest.raises(serving.ChampionError, match="does not match the hash"):
        serving.load_champion("da_price", iface, db)
    os.remove(out["path"])
    with pytest.raises(serving.ChampionError, match="missing"):
        serving.load_champion("da_price", iface, db)


def test_a_new_version_cannot_be_served_before_it_is_fitted(db):
    panel, cfg, view, day, _ = _setup(db, decision())
    ev = champions.evaluate_challenge(table(5.0, 150, 1), table(3.0, 150, 2), POL, CMP, dt.date.fromordinal(D0 + 149), 120)
    champions.apply_challenge("da_price", decision({"type": "single", "model": "xgboost"}), FS, ev, "d2", db)
    with pytest.raises(serving.ChampionError, match="no fitted artifact"):
        serving.load_champion("da_price", view.interface(), db)
    with pytest.raises(serving.ChampionError, match="no active champion"):
        serving.load_champion("ida1_price", view.interface(), db)


def test_the_command_line_registers_and_shows_a_champion(tmp_path):
    import json
    f = tmp_path / "d.json"
    f.write_text(json.dumps(decision({"type": "single", "model": "lightgbm"}), default=str), encoding="utf-8")
    env = {**os.environ, "MODEL_REGISTRY_DB": str(tmp_path / "r.db")}

    def run(*args):
        r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "champion.py"), *args], capture_output=True, text=True, cwd=_REPO,
                           env=env, stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stderr[-1500:]
        return r.stdout
    assert "version 1 is the active champion" in run("init", "--decision", str(f))
    assert "version 1 since" in run("show", "--target", "da_price")
    assert "initial_promotion" in run("history", "--target", "da_price")
    assert champions.active("da_price", str(tmp_path / "r.db"))["version"] == 1
