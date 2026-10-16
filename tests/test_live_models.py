"""
test_live_models.py — stage 10: the live pipeline gets its forecasts from the registry champion only when a target is switched on,
and always falls back to the legacy forecaster otherwise.
"""
from __future__ import annotations

import datetime as dt
import importlib
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.model_development import champions, live, old_new, serving

from tests.test_algorithm_comparison import D0, make
from tests.test_forecast_evaluation import synthetic_panel
from tests.test_locked_test import cfg_locked, decision

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FORM = load_forecast_formulation()
FS = load_evaluation_config()["models"]["feature_set"]
T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
PANEL = synthetic_panel(220)
PROVIDER = lambda: PANEL                                                      # noqa: E731


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "reg" / "registry.db")


def register(db, target="da_price", model="lightgbm"):
    dec = decision({"type": "single", "model": model})
    dec["target"] = target
    champions.promote_initial(dec, FS, "d", db, T0)
    serving.fit_artifact(target, PANEL, FORM, cfg_locked(), {}, dt.date(2022, 6, 1), db, now=T0)


# ------------------------------------------------------------------------------------------------ the modes
def test_only_the_targets_where_the_champion_beat_the_legacy_model_are_switched_on():
    """model_development/old_vs_new: the champion is significantly better for IDA1-3 and aFRR; day-ahead (a tie) and XBID (not significant) stay legacy."""
    assert live.load_modes() == {t: ("legacy" if t in ("da_price", "xbid_price") else "auto") for t in live.SWITCHABLE}


def test_the_mode_file_is_validated_and_the_environment_can_override_it(tmp_path, monkeypatch):
    f = tmp_path / "m.yaml"
    f.write_text("targets: {da_price: {mode: magic}, ida1_price: {mode: legacy}}")
    with pytest.raises(ValueError):
        live.load_modes(str(f))
    monkeypatch.setenv("LIVE_MODELS_MODE", "registry")
    assert set(live.load_modes().values()) == {"registry"}
    monkeypatch.setenv("LIVE_MODELS_MODE", "nonsense")
    with pytest.raises(ValueError):
        live.load_modes()


def test_legacy_mode_does_not_touch_the_registry_or_the_data(db):
    called = []
    out = live.champion_prices("da_price", [1, 2], "2022-06-10", None, db, "legacy", lambda: called.append(1))
    assert out is None and not called


def test_auto_falls_back_with_a_warning_and_registry_mode_refuses(db):
    with pytest.warns(RuntimeWarning, match="no active champion"):
        assert live.champion_prices("da_price", [1], "2022-06-10", None, db, "auto", PROVIDER) is None
    with pytest.raises(serving.ChampionError):
        live.champion_prices("da_price", [1], "2022-06-10", None, db, "registry", PROVIDER)


def test_a_broken_mode_file_never_stops_the_pipeline(monkeypatch):
    monkeypatch.setattr(live, "load_modes", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad file")))
    with pytest.warns(RuntimeWarning, match="unusable"):
        assert live.champion_prices("da_price", [1], "2022-06-10") is None


# --------------------------------------------------------------------------------------- the champion forecast
def test_the_champion_forecast_for_a_future_day_is_made_from_what_is_known_and_matches_the_served_model(db):
    register(db)
    D = dt.date(2022, 8, 20)                                                      # the panel ends 2022-08-08: the day has no rows yet
    got = live.champion_prices("da_price", list(range(1, 25)), str(D), None, db, "auto", PROVIDER, FORM)
    assert sorted(got) == list(range(1, 25)) and np.isfinite(list(got.values())).all()
    view = AsOfView(live.with_delivery_day(PANEL, D, None, None), FORM, "da_price")
    fc = serving.load_champion("da_price", view.interface(), db)
    expected = fc.forecast_prices(view.at(D, 400), D).to_numpy()
    np.testing.assert_allclose([got[h] for h in range(1, 25)], np.round(expected, 2), atol=0)
    assert len(live.with_delivery_day(PANEL, D, None, None)) == len(PANEL) + 24 and len(PANEL) == 220 * 24


def test_only_the_requested_hours_are_returned_and_the_floor_applies(db, monkeypatch):
    register(db)

    class Stub:
        def forecast_prices(self, v, D):
            rows = v[v["local_date"] == D.toordinal()]
            return pd.Series(np.full(len(rows), -1000.0), index=rows.index)
    monkeypatch.setattr(serving, "load_champion", lambda *a, **k: Stub())
    got = live.champion_prices("da_price", [3, 7, 30], "2022-08-20", None, db, "auto", PROVIDER, FORM)
    assert got == {3: -600.0, 7: -600.0}                                          # hour 30 does not exist: left out, as the legacy code does


def test_a_day_with_23_hours_gives_23_forecasts(db, monkeypatch):
    register(db)

    class Stub:
        def forecast_prices(self, v, D):
            rows = v[v["local_date"] == D.toordinal()]
            return pd.Series(np.arange(len(rows), dtype=float), index=rows.index)
    monkeypatch.setattr(serving, "load_champion", lambda *a, **k: Stub())
    got = live.champion_prices("da_price", list(range(1, 25)), "2022-03-27", None, db, "auto", PROVIDER, FORM)   # clocks go forward
    assert sorted(got) == list(range(1, 24))


def test_a_spread_target_needs_the_base_price_and_returns_a_price(db):
    register(db, "ida1_price", "decision_tree")
    with pytest.warns(RuntimeWarning, match="base price"):
        assert live.champion_prices("ida1_price", [1, 2], "2022-08-20", None, db, "auto", PROVIDER, FORM) is None
    base = {h: 50.0 + h for h in range(1, 25)}
    got = live.champion_prices("ida1_price", list(range(1, 25)), "2022-08-20", base, db, "auto", PROVIDER, FORM)
    D = dt.date(2022, 8, 20)
    view = AsOfView(live.with_delivery_day(PANEL, D, "da_pt", base), FORM, "ida1_price")
    fc = serving.load_champion("ida1_price", view.interface(), db)
    spread = fc.predict(view.at(D, 400), D)
    np.testing.assert_allclose([got[h] for h in range(1, 25)], np.round(spread + np.array([base[h] for h in range(1, 25)]), 2), atol=0)


def test_a_champion_that_does_not_match_the_registry_is_not_used(db):
    register(db)
    reg = champions.latest_artifact("da_price", 1, db)
    with open(reg["path"], "ab") as fh:
        fh.write(b"x")
    with pytest.warns(RuntimeWarning, match="hash"):
        assert live.champion_prices("da_price", [1], "2022-08-20", None, db, "auto", PROVIDER, FORM) is None
    with pytest.raises(serving.ChampionError, match="hash"):
        live.champion_prices("da_price", [1], "2022-08-20", None, db, "registry", PROVIDER, FORM)


# ---------------------------------------------------------------------------------------- the legacy wiring
LEGACY = {"da_price": ("phase_2_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster", "forecast_da_prices"),
          "ida1_price": ("phase_3a_ida1_intraday_auction_1.ida1_price_forecasting.ida1_price_forecaster", "forecast_ida1_prices"),
          "ida2_price": ("phase_3b_ida2_intraday_auction_2.ida2_price_forecasting.ida2_price_forecaster", "forecast_ida2_prices"),
          "ida3_price": ("phase_3c_ida3_intraday_auction_3.ida3_price_forecasting.ida3_price_forecaster", "forecast_ida3_prices"),
          "xbid_price": ("phase_3d_xbid_continuous_intraday.xbid_price_forecasting.xbid_price_forecaster", "forecast_xbid_prices")}


@pytest.mark.parametrize("target", sorted(LEGACY))
def test_each_legacy_forecaster_uses_the_champion_when_there_is_one_and_the_legacy_model_otherwise(target, monkeypatch):
    mod, fn = LEGACY[target]
    m = importlib.import_module(mod)
    f = getattr(m, fn)
    base = {} if target == "da_price" else {1: 50.0, 2: 51.0}
    args = ([1, 2], "2022-08-20") + ((base,) if base else ())
    seen = {}

    def fake(tgt, hours, date, base_prices=None, *a, **k):
        seen.update(target=tgt, hours=hours, date=date, base=base_prices)
        return {1: 11.0, 2: 12.0}
    monkeypatch.setattr(live, "champion_prices", fake)
    monkeypatch.setattr(m, "_model_forecast", lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy model must not run")))
    assert f(*args) == {1: 11.0, 2: 12.0}
    assert seen["target"] == target and seen["hours"] == [1, 2] and seen["date"] == "2022-08-20" and seen["base"] == (base or None)
    monkeypatch.setattr(live, "champion_prices", lambda *a, **k: None)
    monkeypatch.setattr(m, "_model_forecast", lambda *a, **k: {1: 99.0})
    assert f(*args) == {1: 99.0}


def test_the_legacy_forecasters_are_unchanged_while_every_target_is_legacy(monkeypatch):
    """Default configuration: the champion code is asked, says no, and the legacy path runs exactly as before."""
    m = importlib.import_module(LEGACY["da_price"][0])
    monkeypatch.setattr(m, "_model_forecast", lambda h, d: {1: 42.0})
    assert m.forecast_da_prices([1], "2022-08-20") == {1: 42.0}


# ------------------------------------------------------------------------------------------ the aFRR sides
AFRR_MOD = "phase_4a_afrr_automatic_frequency_reserve.afrr_price_forecasting.afrr_price_forecaster"


def test_a_level_target_that_uses_the_day_ahead_price_of_the_day_gets_it_from_the_caller(db, monkeypatch):
    register(db, "afrr_cap_price", "decision_tree")
    D = dt.date(2022, 8, 20)
    hours = list(range(1, 25))
    da = {h: 40.0 + 2 * h for h in hours}
    got = live.champion_prices("afrr_cap_price", hours, str(D), da, db, "auto", PROVIDER, FORM)
    view = AsOfView(live.with_delivery_day(PANEL, D, "da_pt", da), FORM, "afrr_cap_price")
    fc = serving.load_champion("afrr_cap_price", view.interface(), db)
    np.testing.assert_allclose([got[h] for h in hours], np.round(fc.predict(view.at(D, 400), D), 2), atol=0)
    assert live.champion_prices("afrr_cap_price", hours, str(D), None, db, "auto", PROVIDER, FORM) is not None      # without it: missing input, still a forecast
    seen = {}

    class Stub:
        def forecast_prices(self, v, day):
            rows = v[v["local_date"] == day.toordinal()]
            seen["da_pt"] = rows["da_pt"].to_numpy()
            return pd.Series(np.full(len(rows), 1.0), index=rows.index)
    monkeypatch.setattr(serving, "load_champion", lambda *a, **k: Stub())
    live.champion_prices("afrr_cap_price", hours, str(D), da, db, "auto", PROVIDER, FORM)
    np.testing.assert_allclose(seen["da_pt"], [da[h] for h in hours])           # the day-ahead price of the day reaches the champion's input
    live.champion_prices("afrr_cap_price", hours, str(D), None, db, "auto", PROVIDER, FORM)
    assert np.isnan(seen["da_pt"]).all()                                        # and without it the input is missing


def test_each_aFRR_side_uses_its_own_champion_and_the_other_side_keeps_the_legacy_model(monkeypatch):
    m = importlib.import_module(AFRR_MOD)
    seen = []

    def fake(tgt, hours, date, base_prices=None, *a, **k):
        seen.append((tgt, base_prices))
        return {1: 11.0, 2: 999.0} if tgt == "afrr_cap_price" else None            # up: champion (one value above the cap); down: legacy
    monkeypatch.setattr(live, "champion_prices", fake)
    monkeypatch.setattr(m, "_model_forecast", lambda *a, **k: ({1: 1.0, 2: 2.0}, {1: 3.0, 2: 4.0}))
    up, dn = m.forecast_afrr_cap_prices([1, 2], "2022-08-20", 250.0, {1: 50.0, 2: 60.0})
    assert up == {1: 11.0, 2: 250.0} and dn == {1: 3.0, 2: 4.0}                  # the champion value is limited to the cap; down is legacy
    assert seen == [("afrr_cap_price", {1: 50.0, 2: 60.0}), ("afrr_dn_price", {1: 50.0, 2: 60.0})]


def test_both_aFRR_sides_from_champions_do_not_run_the_legacy_model(monkeypatch):
    m = importlib.import_module(AFRR_MOD)
    monkeypatch.setattr(live, "champion_prices", lambda tgt, hours, *a, **k: {h: (10.0 if tgt == "afrr_cap_price" else -5.0) for h in hours})
    monkeypatch.setattr(m, "_model_forecast", lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy model must not run")))
    assert m.forecast_afrr_cap_prices([1, 2], "2022-08-20", 250.0) == ({1: 10.0, 2: 10.0}, {1: 0.0, 2: 0.0})        # prices cannot be negative


def test_the_legacy_aFRR_forecaster_is_unchanged_when_nothing_is_switched_and_a_short_champion_is_not_used(monkeypatch):
    m = importlib.import_module(AFRR_MOD)
    monkeypatch.setenv("LIVE_MODELS_MODE", "legacy")                                                       # the legacy path, whatever the file says
    monkeypatch.setattr(m, "_model_forecast", lambda h, d, cap: ({1: 7.0}, {1: 8.0}))
    assert m.forecast_afrr_cap_prices([1], "2022-08-20", 250.0) == ({1: 7.0}, {1: 8.0})                    # default: both legacy
    monkeypatch.setattr(live, "champion_prices", lambda tgt, hours, *a, **k: {1: 5.0})                      # misses hour 2
    with pytest.warns(RuntimeWarning, match="some hours"):
        assert m.forecast_afrr_cap_prices([1, 2], "2022-08-20", 250.0) == ({1: 7.0}, {1: 8.0})


# ----------------------------------------------------------------------------------- old against new
def test_old_against_new_scores_both_on_the_same_hours_and_tests_the_difference():
    days = 90
    new = make({"x": 3.0}, days=days, seed=1)["x"]
    old_err = np.random.default_rng(2).normal(0, 6.0, days * 24)
    old = {D0 + d: {h + 1: 100.0 + old_err[d * 24 + h] for h in range(24)} for d in range(days)}
    r = old_new.compare(new, old)
    assert r["days"] == days and r["hours"] == days * 24 and r["new"]["mae"] < r["old"]["mae"] and r["mae_change"] < -0.3
    assert r["dm"]["statistic"] < -5 and r["dm"]["p_value"] < 0.001
    text = old_new.render("da_price", r, "run")
    assert "the champion is significantly better" in text and "legacy" in text
    del old[D0 + 5], old[D0 + 6]                                                  # days the legacy forecaster did not produce
    r2 = old_new.compare(new, old)
    assert r2["days"] == days - 2 and r2["new"]["sse"] < r["new"]["sse"]


def test_old_against_new_lines_the_legacy_hours_up_by_hour_of_day_for_a_target_that_covers_only_some_hours():
    days = 30
    new = make({"x": 3.0}, days=days, seed=1)["x"]
    new = new[new["local_hour"] >= 12]                                            # like IDA3: hours 12-23 only
    old = {D0 + d: {h + 1: (1000.0 if h == 12 else 100.0) for h in range(24)} for d in range(days)}         # a marked value at local hour 12 (hour 13 of the day) only
    table = old_new.old_table(new, old)
    assert len(table) == days * 12
    assert (table.loc[table["local_hour"] == 12, "y_pred"] == 1000.0).all() and (table.loc[table["local_hour"] > 12, "y_pred"] == 100.0).all()


def test_old_against_new_reports_a_legacy_forecaster_that_is_better():
    days = 90
    new = make({"x": 6.0}, days=days, seed=1)["x"]
    old_err = np.random.default_rng(2).normal(0, 3.0, days * 24)
    old = {D0 + d: {h + 1: 100.0 + old_err[d * 24 + h] for h in range(24)} for d in range(days)}
    r = old_new.compare(new, old)
    assert r["mae_change"] > 0.5 and "the legacy forecaster is significantly better" in old_new.render("da_price", r, "run")


def test_the_comparison_tool_is_available():
    r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "compare_old_new.py"), "--help"], capture_output=True, text=True,
                       cwd=_REPO, stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and "--days" in r.stdout
