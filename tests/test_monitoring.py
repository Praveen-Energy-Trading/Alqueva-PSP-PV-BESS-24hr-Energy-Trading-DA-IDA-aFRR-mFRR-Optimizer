"""
test_monitoring.py — stage 8: the forecast log, the degradation flags, the refit/re-tuning schedule, checked on constructed
errors with known answers and on a served champion whose market changes.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation import metrics
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.forecast_evaluation.tuning import load_tuning_config
from common_layer.model_development import champions, monitoring, serving

from tests.test_algorithm_comparison import D0, make
from tests.test_forecast_evaluation import synthetic_panel
from tests.test_locked_test import cfg_locked, decision
from tests.test_tuning_ensembles import small_tcfg

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FORM = load_forecast_formulation()
ECFG = load_evaluation_config()
CFG = monitoring.load_monitoring_config(tuning_cfg=load_tuning_config(), eval_cfg=ECFG)
FS = ECFG["models"]["feature_set"]
T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
BASE = {"dev_loss": 4.0, "loss_threshold": 4.6, "bias_threshold": 1.0, "window_days": 28, "quantile": 0.95}


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "reg" / "registry.db")


def log(err, days=None, seed=0, drop=()):
    """A scored log: truth 100, forecast 100 + error; `err` is a sd (or array per hour); `drop` lists day indexes with no forecast."""
    days = days or 40
    df = make({"x": err}, days=days, seed=seed)["x"]
    df["logged_utc"] = "2000-01-01T00:00:00+00:00"
    df = metrics.add_error_columns(df)
    return df[~df["local_date"].isin([D0 + d for d in drop])]


def asof(days=40):
    return dt.date.fromordinal(D0 + days - 1)


def run_assess(champ, ref=None, days=40, cfg=CFG, base=BASE):
    return monitoring.assess(champ, ref, base, cfg, asof(days), [D0 + d for d in range(days)], ECFG["comparison"])


# ---------------------------------------------------------------------------------------------------- config
def test_the_monitoring_config_is_valid_and_consistent_with_the_evaluation_and_tuning_configs():
    assert CFG["tracking"]["window_days"] == 28 and CFG["schedule"] == {"refit_every_days": 30, "retune_every_days": 180}
    tc = copy.deepcopy(load_tuning_config())
    tc["search"]["retune_every_days"] = 90
    with pytest.raises(ValueError, match="retune_every_days"):
        monitoring.load_monitoring_config(tuning_cfg=tc)
    ec = copy.deepcopy(ECFG)
    ec["schedule"]["retrain_every_days"] = 7
    with pytest.raises(ValueError, match="refit_every_days"):
        monitoring.load_monitoring_config(eval_cfg=ec)


# ---------------------------------------------------------------------------------------------------- the log
def test_the_log_is_append_only_and_joins_outcomes(db):
    panel = synthetic_panel(60)
    view = AsOfView(panel, FORM, "da_price")

    class Fixed:
        def forecast_prices(self, v, D):
            rows = v[v["local_date"] == D.toordinal()]
            return pd.Series(np.full(len(rows), 50.0), index=rows.index)
    D = dt.date(2022, 2, 10)
    assert monitoring.log_day("da_price", "champion", Fixed(), view, D, 1, "2022-02-08", db, T0) == 24
    with pytest.raises(monitoring.LogError, match="append-only"):
        monitoring.log_day("da_price", "champion", Fixed(), view, D, 1, "2022-02-08", db, T0)
    got = monitoring.read_log("da_price", "champion", view, db)
    assert len(got) == 24 and (got["y_pred"] == 50.0).all() and got["y_true"].notna().all()
    assert (got["err"] == got["y_pred"] - got["y_true"]).all()
    assert monitoring.read_log("da_price", "reference", view, db).empty                       # another forecaster: its own log
    assert monitoring.late_days(got) == 1                                                    # logged 2026, delivered 2022: after the day


def test_a_forecast_logged_before_the_delivery_day_is_not_late(db):
    view = AsOfView(synthetic_panel(60), FORM, "da_price")

    class Fixed:
        def forecast_prices(self, v, D):
            rows = v[v["local_date"] == D.toordinal()]
            return pd.Series(np.full(len(rows), 50.0), index=rows.index)
    D = dt.date(2022, 2, 10)
    monitoring.log_day("da_price", "champion", Fixed(), view, D, 1, "x", db, dt.datetime(2022, 2, 9, 12, tzinfo=dt.timezone.utc))
    assert monitoring.late_days(monitoring.read_log("da_price", "champion", view, db)) == 0


# ---------------------------------------------------------------------------------------------- thresholds
def test_calibration_gives_thresholds_above_the_normal_level():
    dev = log(5.0, days=200, seed=1)
    b = monitoring.calibrate(dev, CFG)
    assert b["n_windows"] == 200 - 28 + 1 and 3.6 < b["dev_loss"] < 4.4
    assert b["loss_threshold"] > b["dev_loss"] and 0 < b["bias_threshold"] < 0.6
    with pytest.raises(ValueError, match="too few"):
        monitoring.calibrate(log(5.0, days=40), CFG)


# ------------------------------------------------------------------------------------------------- flags
def test_normal_errors_give_status_ok():
    r = run_assess(log(5.0, seed=2))
    assert r["status"] == "ok" and not any(v for v in r["flags"].values() if v is not None) and r["loss_ratio_to_dev"] < 1.2


def test_each_flag_and_the_status_rule():
    n = 40 * 24
    rng = np.random.default_rng(3)
    wide = run_assess(log(rng.normal(0, 8.0, n), seed=3))                                      # larger errors only
    assert wide["flags"]["level"] and not wide["flags"]["bias"] and wide["status"] == "warning"
    shifted = run_assess(log(rng.normal(2.0, 5.0, n)))                                         # bias only
    assert shifted["flags"]["bias"] and not shifted["flags"]["level"] and shifted["status"] == "warning"
    both = run_assess(log(rng.normal(3.0, 8.0, n)))                                            # larger errors and bias
    assert both["flags"]["level"] and both["flags"]["bias"] and both["status"] == "degraded"
    missing = run_assess(log(5.0, seed=4, drop=(35, 36, 37, 38)))                              # 4 of the last 28 days unforecast
    assert missing["flags"]["coverage"] and missing["coverage"] == pytest.approx(24 / 28) and missing["status"] == "warning"
    thin = run_assess(log(5.0, days=40, seed=5, drop=range(20, 40)))
    assert thin["status"] == "insufficient_data" and "need 21" in thin["notes"][0]


def test_only_the_window_ending_at_the_as_of_date_counts():
    n = 40 * 24
    err = np.where(np.arange(n) // 24 >= 30, 15.0, 0.0) * np.random.default_rng(6).normal(0, 1, n) + np.random.default_rng(7).normal(0, 5.0, n)
    early = monitoring.assess(log(err, seed=0), None, BASE, CFG, dt.date.fromordinal(D0 + 29), [D0 + d for d in range(40)], ECFG["comparison"])
    assert early["status"] == "ok"                                                             # the bad days come after the as-of date


def test_the_reference_comparison_flags_a_champion_that_no_longer_earns_its_complexity():
    days = 120
    champ, ref_better = log(5.0, days=days, seed=1), log(3.0, days=days, seed=2)
    r = run_assess(champ, ref_better, days)
    assert r["flags"]["relative"] and r["status"] == "degraded" and r["relative"]["p_value"] < 0.001
    same = run_assess(champ, log(5.0, days=days, seed=3), days)
    assert same["flags"]["relative"] is False and same["status"] == "ok"
    worse = run_assess(log(3.0, days=days, seed=1), log(5.0, days=days, seed=2), days)          # the reference is worse: no flag
    assert worse["flags"]["relative"] is False
    short = run_assess(log(5.0, days=40, seed=1), log(3.0, days=40, seed=2), 40)
    assert short["flags"]["relative"] is None and any("relative check skipped" in n for n in short["notes"])


# ------------------------------------------------------------------------------------------------ schedule
def _promoted(db, tuned=False):
    champions.promote_initial(decision({"type": "single", "model": "lightgbm"}, tuned=tuned), FS, "d", db, T0)
    return champions.active("da_price", db)["version"]


def _artifact(db, first_day, tuned_at=None, version=1):
    champions.register_artifact("da_price", version, first_day - dt.timedelta(days=2), first_day, "p", "h", "s", ["lightgbm"], db, T0,
                                "champion", tuned_at)


def test_the_plan_follows_the_schedule(db):
    _promoted(db, tuned=True)
    d0 = dt.date(2026, 1, 1)
    assert monitoring.plan("da_price", d0, CFG, ECFG, db)["action"] == "fit"                    # nothing fitted yet
    _artifact(db, d0, tuned_at=d0 - dt.timedelta(days=2))
    ok = monitoring.plan("da_price", d0 + dt.timedelta(days=29), CFG, ECFG, db)
    assert ok["action"] == "none" and ok["days_since_refit"] == 29 and ok["next_refit_day"] == "2026-01-31"
    due = monitoring.plan("da_price", d0 + dt.timedelta(days=30), CFG, ECFG, db)
    assert due["action"] == "refit" and due["refit_due"] and not due["retune_due"]
    old = monitoring.plan("da_price", d0 + dt.timedelta(days=185), CFG, ECFG, db)             # settings 180+ days old at the next fit
    assert old["action"] == "refit_and_retune" and old["days_since_tuning"] >= 180


def test_an_untuned_champion_is_never_retuned(db):
    _promoted(db, tuned=False)
    d0 = dt.date(2026, 1, 1)
    _artifact(db, d0)
    p = monitoring.plan("da_price", d0 + dt.timedelta(days=400), CFG, ECFG, db)
    assert p["action"] == "refit" and not p["retune_due"] and "days_since_tuning" not in p


def test_recommendations_follow_the_status_and_the_schedule():
    st = {"status": "degraded", "flags": {"level": True, "bias": True, "coverage": False, "relative": None}}
    rec = monitoring.recommendations(st, {"action": "refit_and_retune"})
    assert any("refit and re-tune" in r for r in rec) and any("review the champion" in r for r in rec)
    assert monitoring.recommendations({"status": "ok", "flags": {}}, {"action": "none"}) == ["no action needed"]
    cov = monitoring.recommendations({"status": "warning", "flags": {"coverage": True, "level": False}}, {"action": "none"})
    assert any("daily forecast job" in r for r in cov)


# ------------------------------------------------------------- refits reuse tuned settings; the reference
def _tuned_setup(db, retune_days):
    tc = small_tcfg({"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 20]}, n_iter=6, val=20, blocks=2, method="tpe", retune=retune_days)
    panel, cfg = synthetic_panel(260), cfg_locked()
    champions.promote_initial(decision({"type": "single", "model": "decision_tree"}, tuned=True), FS, "d", db, T0)
    return panel, cfg, tc


def test_a_refit_reuses_the_tuned_settings_until_they_are_old_enough(db):
    panel, cfg, tc = _tuned_setup(db, retune_days=180)
    a = serving.fit_artifact("da_price", panel, FORM, cfg, tc, dt.date(2022, 5, 1), db, now=T0)
    b = serving.fit_artifact("da_price", panel, FORM, cfg, tc, dt.date(2022, 5, 31), db, now=T0)             # 30 days later
    assert b["tuned_at"] == a["tuned_at"] == dt.date(2022, 4, 29)                                          # settings reused
    forced = serving.fit_artifact("da_price", panel, FORM, cfg, tc, dt.date(2022, 6, 20), db, now=T0, force_retune=True)
    assert forced["tuned_at"] == dt.date(2022, 6, 18)                                                      # searched again
    assert champions.latest_artifact("da_price", 1, db)["tuned_at"] == "2022-06-18"


def test_settings_are_searched_again_once_the_retuning_interval_has_passed(db):
    panel, cfg, tc = _tuned_setup(db, retune_days=25)
    a = serving.fit_artifact("da_price", panel, FORM, cfg, tc, dt.date(2022, 5, 1), db, now=T0)
    b = serving.fit_artifact("da_price", panel, FORM, cfg, tc, dt.date(2022, 5, 31), db, now=T0)             # 30 >= 25 days
    assert b["tuned_at"] > a["tuned_at"]


def test_the_reference_is_fitted_and_served_alongside_the_champion(db):
    panel, cfg = synthetic_panel(220), cfg_locked()
    champions.promote_initial(decision({"type": "single", "model": "lightgbm"}), FS, "d", db, T0)
    day = dt.date(2022, 6, 1)
    serving.fit_artifact("da_price", panel, FORM, cfg, {}, day, db, now=T0)
    r = serving.fit_artifact("da_price", panel, FORM, cfg, {}, day, db, now=T0, role="reference")
    assert r["role"] == "reference" and r["members"] == ["decision_tree"]
    view = AsOfView(panel, FORM, "da_price")
    ch, rf = serving.load_champion("da_price", view.interface(), db), serving.load_champion("da_price", view.interface(), db, role="reference")
    v = view.at(day, 400)
    assert ch.name == "champion_v1" and rf.name == "reference_v1" and not np.allclose(ch.predict(v, day), rf.predict(v, day))
    with pytest.raises(ValueError):
        serving.role_spec({}, "challenger")


# --------------------------------------------- a served champion whose market changes is flagged as degraded
def test_a_champion_is_flagged_when_the_market_it_was_trained_on_changes(db, tmp_path):
    panel = synthetic_panel(330, seed=2)
    shift_day = pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=250)
    after = panel.index >= shift_day
    r = np.random.default_rng(9)
    for col in ("da_pt", "da_es"):
        panel.loc[after, col] = panel.loc[after, col] + 40.0 + r.normal(0, 15.0, after.sum())      # a regime change at day 250
    cfg = cfg_locked()
    champions.promote_initial(decision({"type": "single", "model": "lightgbm"}), FS, "d", db, T0)
    first = dt.date(2022, 4, 11)                                                                    # day 100: trained before the change
    serving.fit_artifact("da_price", panel, FORM, cfg, {}, first, db, now=T0)
    view = AsOfView(panel, FORM, "da_price")
    ch = serving.load_champion("da_price", view.interface(), db)
    for i in range(100, 330):
        D = dt.date(2022, 1, 1) + dt.timedelta(days=i)
        monitoring.log_day("da_price", "champion", ch, view, D, 1, ch.artifact["train_end"], db,
                           dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=i - 1))     # logged the day before delivery
    small = copy.deepcopy(CFG)
    small["tracking"].update(window_days=7, min_days=5, quantile=0.95)
    clog = monitoring.read_log("da_price", "champion", view, db)
    normal = clog[clog["local_date"] < (dt.date(2022, 1, 1) + dt.timedelta(days=250)).toordinal()]
    base = monitoring.calibrate(normal, small)
    monitoring.save_baseline("da_price", 1, base, db, T0)
    assert monitoring.load_baseline("da_price", 1, db)["loss_threshold"] == base["loss_threshold"]
    before = monitoring.build_report("da_price", dt.date(2022, 1, 1) + dt.timedelta(days=245), view, small, ECFG, db)
    after_ = monitoring.build_report("da_price", dt.date(2022, 1, 1) + dt.timedelta(days=300), view, small, ECFG, db)
    assert before["status"]["status"] in ("ok", "warning") and before["status"]["loss_ratio_to_dev"] < 1.5
    assert after_["status"]["status"] == "degraded" and after_["status"]["flags"]["level"] and after_["status"]["loss_ratio_to_dev"] > 2
    assert any("review the champion" in x for x in after_["recommendations"])
    md = monitoring.render(after_, "fig.png")
    assert "**Status: degraded**" in md and "Recommendations" in md
    png = monitoring.plot_rolling(after_, base, str(tmp_path / "r.png"))
    assert os.path.getsize(png) > 2000 and json.loads(json.dumps(monitoring.to_json(after_), default=str))
    ev = champions.evaluate_challenge(make({"x": 5.0}, days=150, seed=1)["x"], make({"x": 3.0}, days=150, seed=2)["x"], champions.load_champion_policy(),
                                      ECFG["comparison"], dt.date.fromordinal(D0 + 149), 120)
    champions.apply_challenge("da_price", decision({"type": "single", "model": "xgboost"}), FS, ev, "d2", db)
    with pytest.raises(ValueError, match="no monitoring baseline for da_price version 2"):                 # a baseline belongs to one version
        monitoring.build_report("da_price", dt.date(2022, 8, 1), view, small, ECFG, db)


def test_the_command_line_plans_for_a_registered_champion(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps(decision({"type": "single", "model": "lightgbm"}), default=str), encoding="utf-8")
    env = {**os.environ, "MODEL_REGISTRY_DB": str(tmp_path / "r.db")}

    def run(script, *args):
        r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", script), *args], capture_output=True, text=True, cwd=_REPO, env=env,
                           stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stderr[-1500:]
        return r.stdout
    run("champion.py", "init", "--decision", str(f))
    out = json.loads(run("monitor_champion.py", "plan", "--target", "da_price", "--as-of", "2026-11-15"))
    assert out["action"] == "fit" and out["version"] == 1
