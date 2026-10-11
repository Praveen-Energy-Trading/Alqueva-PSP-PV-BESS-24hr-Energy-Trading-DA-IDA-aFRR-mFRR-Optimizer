"""
test_rollout.py — stage 9: what each price target's data allows, and the per-target configuration.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.feature_engineering.catalogue import load_feature_catalogue, load_feature_sets, resolve_feature_set
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.model_development import locked_test as lt, rollout, selection

from tests.test_forecast_evaluation import synthetic_panel
from tests.test_locked_test import decision, frozen

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FORM = load_forecast_formulation()
BASE = load_evaluation_config()
POLICY = selection.load_policy()
R = rollout.load_rollout()


def base_for(panel_days: int = 300) -> dict:
    """The real configuration with its windows moved into the synthetic panel (2022-01-01 onwards)."""
    c = copy.deepcopy(BASE)
    c["development"].update(first_origin=dt.date(2021, 12, 1), test_start=dt.date(2022, 2, 1), test_end=dt.date(2022, 8, 31))
    c["locked_test"].update(start=dt.date(2022, 9, 1), end=dt.date(2022, 10, 31))
    c["schedule"]["min_train_days"] = 60                                  # the synthetic panel is short
    return c


def test_every_price_target_has_a_treatment_and_a_feature_set_that_exists():
    assert set(rollout.PRICE_TARGETS) <= set(R["targets"])
    assert {t: v["treatment"] for t, v in R["targets"].items()} == {
        "da_price": "standard", "afrr_cap_price": "standard", "ida1_price": "limited_evidence", "ida2_price": "limited_evidence",
        "ida3_price": "limited_evidence", "xbid_price": "limited_evidence", "mfrr_price": "deferred"}
    sets = load_feature_sets()
    assert all(v["feature_set"] in sets for v in R["targets"].values())


def test_each_feature_set_is_available_for_its_own_target():
    """The targets that cannot use yesterday's price get a feature set that ends earlier; the standard set would be refused."""
    from common_layer.data_layer.clean_store import CleanStore
    from common_layer.market_analysis import panel as pnl
    import pandas as pd
    cat, sets = load_feature_catalogue(), load_feature_sets()
    panel = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
                         columns=["da_pt", "da_es", "ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt", "afrr_up", "afrr_dn", "mfrr_ap", "imb_short", "imb_long"],
                         dtype=float)
    for t, v in R["targets"].items():
        i = AsOfView(panel, FORM, t).interface()
        ids = resolve_feature_set(v["feature_set"], sets, cat, i.spec, i.offsets, t)
        assert len(ids) >= 7
    i = AsOfView(panel, FORM, "xbid_price").interface()
    with pytest.raises(ValueError, match="y_lag_1d is not available"):
        resolve_feature_set("baseline_v1", sets, cat, i.spec, i.offsets, "xbid_price")


def test_the_rollout_file_is_validated(tmp_path):
    import yaml
    bad = copy.deepcopy(R)
    bad["targets"]["ida1_price"]["treatment"] = "magic"
    bad["targets"]["mfrr_price"].pop("reason")
    bad["targets"].pop("da_price")
    f = tmp_path / "r.yaml"
    f.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError) as e:
        rollout.load_rollout(str(f))
    msg = str(e.value)
    assert "unknown treatment" in msg and "needs a reason" in msg and "without a treatment" in msg


def test_a_standard_target_keeps_the_base_rules_and_only_takes_its_feature_set():
    eff = rollout.effective("da_price", BASE, POLICY, R)
    assert eff["treatment"] == "standard" and not eff["deferred"] and not eff["blocked"]
    assert eff["eval_cfg"]["schedule"] == BASE["schedule"] and eff["policy"] == POLICY
    xb = rollout.effective("xbid_price", BASE, POLICY, R)
    assert xb["eval_cfg"]["models"]["feature_set"] == "baseline_e2_v1" and BASE["models"]["feature_set"] == "baseline_v1"   # base untouched


def test_a_limited_evidence_target_gets_a_shorter_history_rule_and_a_computed_first_test_day():
    panel = synthetic_panel(300)
    view = AsOfView(panel, FORM, "ida1_price")
    eff = rollout.effective("ida1_price", base_for(), POLICY, R, view)
    s = eff["eval_cfg"]["schedule"]
    assert s["min_train_days"] == 180 and s["train_window_days"] is None and eff["policy"]["min_days"] == 250
    assert eff["first_truth"] == "2022-01-01"
    assert eff["dev_start"] == str(dt.date(2022, 1, 1) + dt.timedelta(days=180 + 1 + 1))                  # first day with 180 training days
    assert eff["eval_cfg"]["development"]["test_start"] == dt.date(2022, 1, 1) + dt.timedelta(days=182)
    assert eff["blocked"] and "fewer than the 250" in eff["reasons"][0]                                      # too few days in this small sample


def test_the_first_test_day_is_never_earlier_than_the_configured_start():
    panel = synthetic_panel(300)
    late_start = base_for()
    late_start["development"]["test_start"] = dt.date(2022, 8, 1)
    eff = rollout.effective("ida1_price", late_start, POLICY, R, AsOfView(panel, FORM, "ida1_price"))
    assert eff["dev_start"] == "2022-08-01"                                 # the data would allow 2022-07-02; the configured day is later


def test_a_deferred_target_is_refused_with_its_reason_and_costs_no_data_access():
    eff = rollout.effective("mfrr_price", BASE, POLICY, R)
    assert eff["deferred"] and "2024-11-27" in eff["reasons"][0] and eff["treatment"] == "deferred"
    with_data = rollout.effective("mfrr_price", base_for(), POLICY, R, AsOfView(synthetic_panel(300), FORM, "mfrr_price"))
    assert with_data["deferred"] and not with_data["blocked"]


def test_the_locked_availability_floor_follows_how_often_outcomes_exist_during_development():
    panel = synthetic_panel(300)
    full = rollout.effective("da_price", base_for(), POLICY, R, AsOfView(panel, FORM, "da_price"))
    assert full["dev_outcome_share"] == pytest.approx(1.0) and full["min_truth_share"] == pytest.approx(0.97)
    gappy = panel.copy()
    local_day = gappy.index.tz_convert("Europe/Madrid").date
    rng = np.random.default_rng(1)
    bad_days = set(rng.choice(sorted(set(local_day)), size=40, replace=False))
    gappy.loc[[d in bad_days for d in local_day], "da_pt"] = np.nan
    eff = rollout.effective("da_price", base_for(), POLICY, R, AsOfView(gappy, FORM, "da_price"))
    assert 0.8 < eff["dev_outcome_share"] < 0.95 and eff["min_truth_share"] == pytest.approx(eff["dev_outcome_share"] - 0.03)


def test_the_locked_preflight_uses_the_floor_and_the_policy_of_the_target(tmp_path):
    (tmp_path / "config").mkdir()
    root = str(tmp_path)
    panel = synthetic_panel(300)
    cfg = base_for()
    cfg["locked_test"].update(start=dt.date(2022, 7, 1), end=dt.date(2022, 7, 30))
    cfg["models"]["feature_set"] = BASE["models"]["feature_set"]
    frozen(panel, cfg, root)
    pol = {**POLICY, "min_days": 250}                                                                      # a limited-evidence policy
    dec = decision()
    dec["policy"], dec["policy_digest"] = pol, selection.policy_digest(pol)
    sch = cfg["schedule"]
    dec["source"]["schedule"] = {k: sch[k] for k in ("retrain_every_days", "gap_days", "train_window_days")}
    kw = dict(log_root=root, lock_root=root, code_revision="abc")
    ok = lt.preflight(dec, panel, FORM, cfg, policy=pol, **kw)
    assert ok["ok"], [c for c in ok["checks"] if not c["ok"]]
    base_policy_check = lt.preflight(dec, panel, FORM, cfg, **kw)                                          # judged by the default policy
    assert {c["check"] for c in base_policy_check["checks"] if not c["ok"]} == {"policy_unchanged"}
    thin = copy.deepcopy(cfg)
    thin["locked_test"].update(start=dt.date(2022, 10, 1), end=dt.date(2022, 11, 30))                      # the data ends on 2022-10-27
    frozen(panel, thin, root)
    short = lt.preflight(dec, panel, FORM, thin, policy=pol, min_truth_share=0.9, **kw)
    assert "outcomes_present" in {c["check"] for c in short["checks"] if not c["ok"]}


def test_the_decision_tools_refuse_a_deferred_target_and_the_status_table_lists_every_target():
    r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "evaluate_forecaster.py"), "--target", "mfrr_price", "--no-register"],
                       capture_output=True, text=True, cwd=_REPO, stdin=subprocess.DEVNULL)
    assert r.returncode == 2 and "cannot be evaluated (deferred)" in r.stdout and "2024-11-27" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "select_champion.py"), "--run", ".", "--target", "mfrr_price"],
                       capture_output=True, text=True, cwd=_REPO, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "mfrr_price is deferred" in (r.stderr + r.stdout)
    r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "run_rollout.py"), "--status"], capture_output=True, text=True, cwd=_REPO,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and all(t in r.stdout for t in rollout.PRICE_TARGETS) and "deferred" in r.stdout
    assert "limited_evidence" in r.stdout and "baseline_e2_v1" in r.stdout
