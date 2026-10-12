"""
test_ml_dashboard.py — the ML Lifecycle pages: the data layer on small fixtures, the data pack transforms, and every page opened headless
with Streamlit's AppTest (no exception, key content present). The pages also run on a checkout without runtime/ or without the pack.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "dashboard"))
sys.path.insert(0, str(REPO))

import ml_data  # noqa: E402
import ml_pack  # noqa: E402

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = sorted(p.name for p in (REPO / "dashboard" / "pages").glob("ml_*.py"))


# ------------------------------------------------------------------------------------------------------------ data layer
def fake_root(tmp_path: Path) -> Path:
    r = tmp_path
    (r / "config").mkdir()
    (r / "config" / "rollout.yaml").write_text("targets:\n  da_price: {treatment: standard, feature_set: baseline_v1}\n"
                                               "  ida1_price: {treatment: standard, feature_set: baseline_v1}\n"
                                               "  mfrr_price: {treatment: deferred, feature_set: x, reason: too short}\n")
    sel, lk = r / "model_development" / "selection", r / "model_development" / "locked_test"
    sel.mkdir(parents=True)
    lk.mkdir()
    (sel / "da_price.json").write_text(json.dumps({"status": "chosen", "champion": {"type": "single", "model": "lightgbm"}, "n_days": 1015,
                                                   "champion_dev_loss": 17.4, "policy": {"reference": "decision_tree"},
                                                   "ranking": [{"model": "lightgbm", "mean_loss": 17.4}, {"model": "decision_tree", "mean_loss": 19.6}]}))
    (lk / "da_price.json").write_text(json.dumps({"locked_scores": {"mae": 18.0}, "loss_ratio_locked_over_dev": 1.03}))
    bmd = r / "model_development" / "benchmark"
    bmd.mkdir()
    (bmd / "da_price.json").write_text(json.dumps({"development": {"gain_of_champion_over": {"reference": 0.11, "naive_seasonal": 0.20, "naive_day": 0.09}},
                                                   "locked": {"gain_of_champion_over": {"reference": 0.06, "naive_seasonal": 0.18, "naive_day": 0.05}}}))
    return r


def test_the_results_table_reads_decisions_and_locked_files(tmp_path):
    root = fake_root(tmp_path)
    df = ml_data.results_table(root).set_index("target")
    assert df.loc["da_price", "champion"] == "LightGBM" and df.loc["da_price", "dev_days"] == 1015
    assert df.loc["da_price", "locked_mae"] == 18.0 and df.loc["da_price", "gain_vs_reference"] == pytest.approx(1 - 17.4 / 19.6)
    assert pd.isna(df.loc["ida1_price", "champion"]) and df.loc["mfrr_price", "treatment"] == "deferred"
    assert df.loc["mfrr_price", "reason"] == "too short" and list(df.index) == list(ml_data.TARGETS)


def test_the_gain_over_naive_is_the_gain_over_the_strongest_naive_forecast(tmp_path):
    root = fake_root(tmp_path)
    df = ml_data.results_table(root).set_index("target")
    assert df.loc["da_price", "gain_vs_naive"] == pytest.approx(0.09) and df.loc["da_price", "locked_gain_vs_naive"] == pytest.approx(0.05)
    assert pd.isna(df.loc["ida1_price", "gain_vs_naive"])                       # no benchmark file for that target
    assert ml_data.strongest_naive_gain({"locked": {"gain_of_champion_over": {"reference": 0.5, "naive_day": None}}}, "locked") is None
    assert ml_data.strongest_naive_gain(None, "development") is None


def test_missing_files_give_empty_results_not_errors(tmp_path):
    assert ml_data.pack("trials", tmp_path).empty and ml_data.pack_json("facts", tmp_path) is None
    assert ml_data.benchmarks(tmp_path) == {} and ml_data.decisions(tmp_path) == {} and ml_data.locked_results(tmp_path) == {} and ml_data.config("nothing", tmp_path) == {}
    assert ml_data.figure_path("x.png", tmp_path) is None and all(df.empty for df in ml_data.comparison_tables("da_price", tmp_path).values())
    st = ml_data.stage_status(tmp_path)
    assert len(st) == 10 and {s["state"] for s in st} <= {"done", "partial", "not run"}


def test_the_stage_states_follow_what_exists(tmp_path):
    root = fake_root(tmp_path)
    s = {x["stage"]: x for x in ml_data.stage_status(root)}
    assert s["Locked test"]["state"] == "partial" and "1 of 2" in s["Locked test"]["fact"]
    assert s["Selection policy"]["state"] == "partial" and s["Champion registry"]["state"] == "not run"


def test_champion_text_and_labels():
    assert ml_data.champion_text({"type": "ensemble", "method": "mean", "members": ["xgboost", "lightgbm"]}) == "mean of XGBoost, LightGBM"
    assert ml_data.champion_text(None) == "none" and ml_data.model_label("lightgbm_tuned") == "LightGBM"


# ----------------------------------------------------------------------------------------------------------- data pack
def test_load_status_never_lets_a_skip_override_a_load():
    recs = [{"source": "omie_ida1", "delivery_date": "2025-10-25", "status": "loaded"},
            {"source": "omie_ida1", "delivery_date": "2025-10-25", "status": "skipped"},
            {"source": "omie_ida1", "delivery_date": "2025-10-27", "status": "empty_publication"},
            {"source": "omie_ida1", "delivery_date": "2025-10-28", "status": "not_published"},
            {"source": "omie_ida1", "delivery_date": "2025-10-28", "status": "loaded"},                    # fetched again later
            {"source": "omie_da", "delivery_date": "2025-10-25", "status": "skipped"}]
    df = ml_pack.load_status_monthly(recs).set_index(["source", "status"])["days"]
    assert df[("omie_ida1", "loaded")] == 2 and df[("omie_ida1", "empty_publication")] == 1 and ("omie_ida1", "not_published") not in df.index
    assert df[("omie_da", "loaded")] == 1


def test_manifest_summary_counts_files_days_and_later_versions():
    recs = [{"event": "stored", "source": "a", "key": "2025-01-01", "path": "a/x__v001__1.txt", "size": 10},
            {"event": "stored", "source": "a", "key": "2025-01-01", "path": "a/x__v002__2.txt", "size": 12},
            {"event": "stored", "source": "a", "key": "2025-01-02", "path": "a/y__v001__3.txt", "size": 5},
            {"event": "skipped", "source": "a", "key": "2025-01-03", "path": "", "size": 0}]
    r = ml_pack.manifest_summary(recs).iloc[0]
    assert (r["files"], r["days"], r["versions_above_1"], r["bytes"], r["first_key"], r["last_key"]) == (3, 2, 1, 27, "2025-01-01", "2025-01-02")


def test_daily_loss_is_mean_absolute_signed_and_squared_error_per_day():
    import datetime as dt
    d0 = dt.date(2025, 1, 1).toordinal()
    p = pd.DataFrame({"y_true": [10.0, 10.0, 20.0, 20.0], "y_pred": [12.0, 8.0, 21.0, 21.0], "local_date": [d0, d0, d0 + 1, d0 + 1]})
    r = ml_pack.daily_loss(p, "da_price", "lightgbm").set_index("date")
    assert r.loc["2025-01-01", "mae"] == 2.0 and r.loc["2025-01-01", "bias"] == 0.0 and r.loc["2025-01-01", "mse"] == 4.0
    assert r.loc["2025-01-02", "mae"] == 1.0 and r.loc["2025-01-02", "bias"] == 1.0 and set(r["model"]) == {"lightgbm"}


def test_the_test_counter_finds_test_functions():
    assert ml_pack.count_test_functions(str(REPO / "tests")) > 300


# ------------------------------------------------------------------------------------------------------------- pages
@pytest.mark.parametrize("page", PAGES)
def test_every_ml_page_opens_without_an_exception(page):
    at = AppTest.from_file(str(REPO / "dashboard" / "pages" / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1 and len(at.expander) >= 1                    # a title and the "What to say" panel


@pytest.mark.parametrize("page", PAGES)
def test_every_ml_page_also_opens_in_interview_mode(page):
    at = AppTest.from_file(str(REPO / "dashboard" / "pages" / page), default_timeout=60)
    at.session_state["ml_interview_mode"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_the_overview_numbers_equal_the_decision_files():
    at = AppTest.from_file(str(REPO / "dashboard" / "pages" / "ml_0_overview.py"), default_timeout=60).run()
    text = " ".join(m.value for m in at.markdown)
    dec = ml_data.decisions()
    assert f"{len(dec)} of 6" in text                                          # targets with a champion
    assert any("lifecycle" in s.value.lower() for s in at.subheader)


# --------------------------------------------------------------------------------------- data layer: replay, trials, folds
def _daily(model_mae, days=120, model="champion", target="da_price"):
    import datetime as dt
    d0 = dt.date(2025, 1, 1)
    return pd.DataFrame({"target": target, "model": model, "date": [(d0 + dt.timedelta(days=i)).isoformat() for i in range(days)], "mae": model_mae})


def _locked(err, days=100, target="da_price"):
    idx = pd.date_range("2025-10-12", periods=days * 24, freq="h", tz="Europe/Madrid").tz_convert("UTC")
    return pd.DataFrame({"target": target, "ts_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "y_true": 100.0, "champion": 100.0 + err, "reference": 100.0})


def test_the_monitoring_replay_flags_only_when_the_rolling_error_exceeds_the_development_threshold():
    dev = _daily(10.0)                                                       # a steady development error of 10
    ok = ml_data.monitoring_replay(_locked(5.0), dev, "da_price", 28, 0.95)  # locked errors of 5 (hourly sign-alternating errors would be the same)
    assert ok["level"] == pytest.approx(10.0) and ok["threshold"] == pytest.approx(10.0) and ok["flagged_days"] == 0 and ok["first_flag"] is None
    worse = ml_data.monitoring_replay(_locked(12.0), dev, "da_price", 28, 0.95)
    assert worse["flagged_days"] == 100 - 27 and worse["first_flag"] == pd.Timestamp("2025-11-08")        # first full window ends on the 28th day
    sim = ml_data.monitoring_replay(_locked(5.0), dev, "da_price", 28, 0.95, shock_day="2025-12-01", factor=3.0)
    assert sim["flagged_days"] > 0 and sim["first_flag"] >= pd.Timestamp("2025-12-01")                       # nothing flags before the simulated change
    assert ml_data.monitoring_replay(_locked(5.0), _daily(10.0, days=20), "da_price") is None                 # too few development days
    assert ml_data.monitoring_replay(_locked(5.0), dev, "ida1_price") is None


def test_prepared_trials_scores_every_trial_against_the_defaults_of_its_round():
    t = pd.DataFrame({"target": "da_price", "search": "tpe", "model": "lightgbm", "tuned_at": "2023-01-01", "trial": [0, 1, 2],
                      "params": ["{}", "{'a': 1}", "{'a': 2}"], "score": [10.0, 8.0, 12.0], "pruned": False, "is_default": [True, False, False]})
    r = ml_data.prepared_trials(t, "da_price", "tpe")
    assert list(r["rel"]) == [1.0, 0.8, 1.2] and r["params_d"].iloc[1] == {"a": 1} and r["params_d"].iloc[0] == {}
    assert ml_data.prepared_trials(t, "ida1_price").empty
    assert list(ml_data.best_so_far(r)["best_rel"].round(2)) == [1.0, 0.8, 0.8]


def test_the_fold_helper_uses_the_frameworks_folds():
    import datetime as dt
    sched = {"retrain_every_days": 30, "gap_days": 1, "min_train_days": 365, "train_window_days": 730}
    f = ml_data.folds_for(dt.date(2020, 1, 1), dt.date(2023, 1, 1), dt.date(2023, 3, 31), sched)
    assert len(f) == 3 and f["test_start"].iloc[0] == dt.date(2023, 1, 1) and f["train_end"].iloc[0] == dt.date(2022, 12, 30)
    assert (f["train_days"] <= 730).all() and f["test_days"].tolist()[:2] == [30, 30]
    with pytest.raises(ValueError):
        ml_data.folds_for(dt.date(2022, 12, 1), dt.date(2023, 1, 1), dt.date(2023, 3, 31), sched)             # fewer than 365 training days


def test_the_worked_test_agrees_with_the_framework_and_needs_enough_days():
    import numpy as np
    rng = np.random.default_rng(0)
    a, b = _daily(8.0 + rng.normal(0, 0.5, 120), model="champion"), _daily(10.0 + rng.normal(0, 0.5, 120), model="reference")
    r = ml_data.dm_example(pd.concat([a, b]), "da_price")
    assert r["n_days"] == 120 and r["statistic"] < -10 and r["p_value"] < 1e-6 and r["mean_diff"] == pytest.approx(-2.0, abs=0.2)
    assert ml_data.dm_example(pd.concat([a.head(10), b.head(10)]), "da_price") is None


def test_the_navigation_lists_every_ml_page_and_each_file_exists():
    nav = (REPO / "dashboard" / "app.py").read_text(encoding="utf-8")
    import ml_ui
    for stem, _ in ml_ui.PAGES:
        assert f"pages/{stem}.py" in nav and (REPO / "dashboard" / "pages" / f"{stem}.py").exists()
    assert 'st.Page("pages/9_ml_forecasting.py"' not in nav                                      # the old page is out of the navigation
    assert sorted(f"{s}.py" for s, _ in ml_ui.PAGES) == PAGES


def test_every_ml_page_has_a_talk_track_and_a_limits_strip():
    for page in PAGES:
        src = (REPO / "dashboard" / "pages" / page).read_text(encoding="utf-8")
        assert "ml_ui.talk_track(" in src and "ml_ui.limits(" in src and "ml_ui.header(" in src, page


def test_the_pages_still_open_when_the_pack_and_runtime_are_missing(tmp_path, monkeypatch):
    """A checkout without the data pack: every page says 'not yet run' instead of failing."""
    monkeypatch.setattr(ml_data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ml_data, "pack", lambda name, root=tmp_path: pd.DataFrame())
    monkeypatch.setattr(ml_data, "pack_json", lambda name, root=tmp_path: None)
    for page in ("ml_0_overview.py", "ml_1_data.py", "ml_6_tuning.py", "ml_9_run.py"):
        at = AppTest.from_file(str(REPO / "dashboard" / "pages" / page), default_timeout=60).run()
        assert not at.exception, (page, [e.value for e in at.exception])
