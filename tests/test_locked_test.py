"""
test_locked_test.py — stage 6: the checks before, and the report after, the one look at the locked period.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import subprocess
import sys

import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation import guards
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.forecast_evaluation.tuning import load_tuning_config
from common_layer.model_development import locked_test as lt, selection

from tests.test_forecast_evaluation import PERIODS, SCHED, cfg_small, synthetic_panel

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FORM = load_forecast_formulation()
POLICY = selection.load_policy()
TCFG = load_tuning_config()
LOCK = (dt.date(2022, 7, 1), dt.date(2022, 7, 30))


def cfg_locked(window=LOCK) -> dict:
    c = cfg_small()
    c["locked_test"] = {"start": window[0], "end": window[1], "data_digest": None}
    c["models"]["feature_set"] = load_evaluation_config()["models"]["feature_set"]
    return c


def decision(champion=None, tuned=False) -> dict:
    return {"target": "da_price", "status": "chosen", "champion": champion or {"type": "single", "model": "lightgbm"},
            "policy": POLICY, "policy_digest": selection.policy_digest(POLICY), "champion_dev_loss": 5.0,
            "source": {"run": "x/y", "suffix": "_tuned" if tuned else "", "tuned": tuned,
                       "schedule": {k: SCHED[k] for k in ("retrain_every_days", "gap_days", "train_window_days")}}}


@pytest.fixture
def roots(tmp_path):
    (tmp_path / "config").mkdir()
    return str(tmp_path)


def frozen(panel, cfg, root):
    digest, _, _ = lt.truth_digest(AsOfView(panel, FORM, "da_price"), cfg)
    guards.write_lock("da_price", digest, "w", root)


def pre(dec, panel, cfg, root, **kw):
    return lt.preflight(dec, panel, FORM, cfg, log_root=root, lock_root=root, code_revision=kw.pop("code_revision", "abc1234"), **kw)


def failed(result):
    return {c["check"] for c in result["checks"] if not c["ok"]}


def test_every_check_passes_when_everything_is_in_order(roots):
    panel, cfg = synthetic_panel(220), cfg_locked()
    frozen(panel, cfg, roots)
    r = pre(decision(), panel, cfg, roots)
    assert r["ok"] and failed(r) == set() and len(r["checks"]) == 7


def test_each_precondition_is_enforced(roots):
    panel, cfg = synthetic_panel(220), cfg_locked()
    assert "outcomes_frozen" in failed(pre(decision(), panel, cfg, roots))                       # not frozen yet
    frozen(panel, cfg, roots)
    other = copy.deepcopy(decision())
    other["policy_digest"] = "0" * 64
    assert failed(pre(other, panel, cfg, roots)) == {"policy_unchanged"}
    sched = copy.deepcopy(decision())
    sched["source"]["schedule"]["train_window_days"] = 730
    assert failed(pre(sched, panel, cfg, roots)) == {"schedule_matches_development"}
    nosched = copy.deepcopy(decision())
    nosched["source"]["schedule"] = {}
    assert failed(pre(nosched, panel, cfg, roots)) == {"schedule_matches_development"}
    assert failed(pre(decision(), panel, cfg, roots, code_revision="abc+dirty")) == {"code_committed"}
    assert pre(decision(), panel, cfg, roots, code_revision="abc+dirty", allow_dirty=True)["ok"]
    none = copy.deepcopy(decision())
    none.update(status="insufficient_evidence", champion=None)
    assert failed(pre(none, panel, cfg, roots)) == {"decision_chosen"}


def test_a_changed_outcome_series_is_caught(roots):
    panel, cfg = synthetic_panel(220), cfg_locked()
    frozen(panel, cfg, roots)
    revised = panel.copy()
    revised.iloc[-300:, revised.columns.get_loc("da_pt")] += 1.0                                # data of the period was revised
    r = pre(decision(), revised, cfg, roots)
    assert "outcomes_frozen" in failed(r) and any("changed since" in c["detail"] for c in r["checks"])


def test_missing_outcomes_and_an_earlier_look_are_caught(roots):
    panel = synthetic_panel(220)                                                                  # data end 2022-08-08
    late = cfg_locked((dt.date(2022, 8, 1), dt.date(2022, 8, 30)))
    assert "outcomes_present" in failed(pre(decision(), panel, late, roots))
    cfg = cfg_locked()
    frozen(panel, cfg, roots)
    guards.register_final_run(cfg, "da_price", "lightgbm", root=roots)
    assert failed(pre(decision(), panel, cfg, roots)) == {"not_scored_before"}


def test_the_forecasters_to_be_scored_are_known_in_advance():
    assert lt.forecaster_names(decision()) == ["lightgbm", "decision_tree"]
    assert lt.forecaster_names(decision(), with_reference=False) == ["lightgbm"]
    assert lt.forecaster_names(decision({"type": "single", "model": "decision_tree"})) == ["decision_tree"]
    ens = decision({"type": "ensemble", "members": ["xgboost", "lightgbm"], "method": "mean"}, tuned=True)
    assert lt.forecaster_names(ens) == ["xgboost_tuned", "lightgbm_tuned", "ensemble_mean", "decision_tree_tuned"]
    assert lt.champion_name(ens) == "ensemble_mean" and lt.champion_name(decision(tuned=True)) == "lightgbm_tuned"


def test_the_single_look_scores_the_champion_and_the_reference_and_cannot_be_repeated(roots):
    panel, cfg = synthetic_panel(220), cfg_locked()
    frozen(panel, cfg, roots)
    rep = lt.run_final(decision(), panel, FORM, cfg, PERIODS, TCFG, None, roots)
    s = rep["locked_scores"]
    assert rep["champion_forecaster"] == "lightgbm" and rep["lock_status"] == "verified" and s["n"] > 500
    assert rep["reference"] == "decision_tree" and set(rep["dm_vs_reference"]) >= {"statistic", "p_value", "n_days"}
    assert rep["loss_ratio_locked_over_dev"] == pytest.approx(rep["locked_loss"] / 5.0)
    assert rep["reference_scores"]["n"] == s["n"]                                                 # the same hours
    assert guards.final_runs(cfg, "da_price", "lightgbm", roots) == 1 and guards.final_runs(cfg, "da_price", "decision_tree", roots) == 1
    with pytest.raises(guards.LockedPeriodError, match="already been scored"):
        lt.run_final(decision(), panel, FORM, cfg, PERIODS, TCFG, None, roots)
    assert "_results" not in lt.to_json(rep) and json.loads(json.dumps(lt.to_json(rep)))
    md = lt.render(rep, decision())
    assert "Champion: **lightgbm**" in md and "Diebold-Mariano" in md


def test_an_averaged_champion_is_scored_as_the_average_of_its_members(roots):
    panel, cfg = synthetic_panel(220), cfg_locked()
    frozen(panel, cfg, roots)
    dec = decision({"type": "ensemble", "members": ["decision_tree", "lightgbm"], "method": "mean"})
    rep = lt.run_final(dec, panel, FORM, cfg, PERIODS, TCFG, None, roots)
    assert rep["champion_forecaster"] == "ensemble_mean" and rep["reference"] == "decision_tree"
    assert set(rep["members"]) == {"decision_tree", "lightgbm"}
    for n in ("decision_tree", "lightgbm", "ensemble_mean"):
        assert guards.final_runs(cfg, "da_price", n, roots) == 1
    assert "the mean of decision_tree, lightgbm" in lt.render(rep, dec)


def test_when_the_champion_is_the_reference_there_is_no_self_comparison(roots):
    panel, cfg = synthetic_panel(220), cfg_locked()
    frozen(panel, cfg, roots)
    rep = lt.run_final(decision({"type": "single", "model": "decision_tree"}), panel, FORM, cfg, PERIODS, TCFG, None, roots)
    assert "dm_vs_reference" not in rep and rep["champion_forecaster"] == "decision_tree"


def test_the_tool_does_nothing_to_the_locked_period_unless_confirmed(tmp_path):
    dec = decision()
    dec.update(status="insufficient_evidence", champion=None)
    f = tmp_path / "d.json"
    f.write_text(json.dumps(dec), encoding="utf-8")
    log = os.path.join(_REPO, load_evaluation_config()["guards"]["access_log"])
    before = os.path.getsize(log) if os.path.exists(log) else None
    r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "run_locked_test.py"), "--target", "da_price", "--decision", str(f),
                        "--confirm-locked"], capture_output=True, text=True, cwd=_REPO, stdin=subprocess.DEVNULL)
    assert r.returncode == 1 and "NOT READY" in r.stdout and "[FAIL] decision_chosen" in r.stdout
    assert (os.path.getsize(log) if os.path.exists(log) else None) == before          # no access was logged
