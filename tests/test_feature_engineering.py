"""
test_feature_engineering.py — Phase 7: calendar, the feature catalogue and its derived availability, the builder,
point-in-time correctness, the store and the stability selection.

Synthetic panels with known values; the catalogue is checked against the real formulation and data inventory.
"""
from __future__ import annotations

import copy
import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from common_layer.configuration.forecast_catalogue import load_forecast_catalogue
from common_layer.configuration.forecast_data_inventory import load_data_inventory
from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.feature_engineering import builder as fb
from common_layer.feature_engineering import calendar as cal
from common_layer.feature_engineering import selection
from common_layer.feature_engineering.catalogue import (availability, feature_availability, feature_set_version,
                                                        load_feature_catalogue, load_feature_sets, requirements,
                                                        resolve_feature_set, validate_feature_catalogue)
from common_layer.feature_engineering.store import FeatureStore
from common_layer.forecast_evaluation.asof import AsOfView, allowed_offsets, target_spec
from common_layer.forecast_formulation.transforms import make_transform

CAT, FORM = load_feature_catalogue(), load_forecast_formulation()
FC, INV, SETS = load_forecast_catalogue(), load_data_inventory(), load_feature_sets()
D0 = dt.date(2022, 1, 1)


def synthetic_panel(days: int = 120, seed: int = 1) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(D0), periods=24 * days, freq="h", tz="Europe/Madrid").tz_convert("UTC")
    local = idx.tz_convert("Europe/Madrid")
    hour, dow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    base = 60 + 25 * np.sin(2 * np.pi * hour / 24) - 10 * (dow >= 5) + r.normal(0, 3, len(idx))
    p = pd.DataFrame(index=idx)
    p["da_pt"], p["da_es"] = base, base + (r.random(len(idx)) < 0.1) * 5.0
    for c, sd in (("ida1_pt", 4), ("ida2_pt", 4), ("ida3_pt", 4), ("xbid_pt", 5)):
        p[c] = base + r.normal(0, sd, len(idx))
    p["afrr_up"] = 10 + r.normal(0, 1, len(idx))
    p["afrr_dn"] = p["afrr_up"]
    p["mfrr_ap"] = base + r.normal(0, 8, len(idx))
    p["imb_short"], p["imb_long"] = base + r.normal(0, 8, len(idx)), base - 5 + r.normal(0, 8, len(idx))
    for c in ("da_vol_total", "da_vol_pt_buy", "da_vol_pt_sell", "da_flow_pt_to_es", "da_flow_es_to_pt"):
        p[c] = 1000 + r.normal(0, 50, len(idx))
    return p


# ------------------------------------------------------------------------------ calendar
def test_easter_and_portuguese_spanish_holidays():
    assert [cal.easter(y) for y in (2024, 2025, 2026)] == [dt.date(2024, 3, 31), dt.date(2025, 4, 20), dt.date(2026, 4, 5)]
    assert cal.is_holiday(dt.date(2024, 4, 25), "pt") and cal.is_holiday(dt.date(2024, 3, 29), "pt")        # 25 April, Good Friday
    assert cal.is_holiday(dt.date(2024, 5, 30), "pt")                                                       # Corpus Christi
    assert not cal.is_holiday(dt.date(2014, 10, 5), "pt") and cal.is_holiday(dt.date(2016, 10, 5), "pt")  # suspended 2013-2015
    assert cal.is_holiday(dt.date(2025, 10, 12), "es") and not cal.is_holiday(dt.date(2025, 10, 12), "pt")
    assert cal.is_holiday(dt.date(2025, 4, 17), "es") and not cal.is_holiday(dt.date(2025, 4, 17), "pt")   # Holy Thursday


def test_bridge_days_are_working_days_next_to_a_holiday_and_the_weekend():
    assert cal.is_bridge(dt.date(2024, 4, 26), "pt")        # Friday after Thursday 25 April
    assert cal.is_bridge(dt.date(2023, 4, 24), "pt")        # Monday before Tuesday 25 April
    assert cal.is_bridge(dt.date(2025, 5, 2), "es")         # Friday after Thursday 1 May
    assert not cal.is_bridge(dt.date(2024, 4, 25), "pt")    # the holiday itself
    assert not cal.is_bridge(dt.date(2024, 4, 24), "pt")    # Wednesday


def test_day_features_mark_daylight_saving_days_and_market_rules():
    spring, autumn = cal.day_features(dt.date(2025, 3, 30)), cal.day_features(dt.date(2025, 10, 26))
    assert spring["dst_day_type"] == -1 and autumn["dst_day_type"] == 1 and cal.day_features(dt.date(2025, 6, 10))["dst_day_type"] == 0
    assert cal.day_features(dt.date(2025, 9, 30))["da_resolution_min"] == 60 and cal.day_features(dt.date(2025, 10, 1))["da_resolution_min"] == 15
    assert cal.day_features(dt.date(2026, 5, 28))["price_floor_eur_mwh"] == -500 and cal.day_features(dt.date(2026, 5, 29))["price_floor_eur_mwh"] == -600
    h = cal.hour_features([0, 6, 12])
    assert h["hour_sin"][1] == pytest.approx(1.0) and h["hour_cos"][2] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- catalogue
def test_the_catalogue_is_valid_against_the_formulation_and_inventory():
    assert validate_feature_catalogue(CAT, FORM, FC, INV) == []


def test_availability_is_derived_from_what_each_feature_reads():
    t = feature_availability(CAT, FORM, FC)
    ok = lambda f, tgt: bool(t[(t.feature == f) & (t.target == tgt)]["available"].iloc[0])      # noqa: E731
    assert ok("y_lag_1d", "da_price") and not ok("y_lag_1d", "xbid_price") and ok("y_lag_2d", "xbid_price")   # xbid published later
    assert not ok("y_lag_1d", "mfrr_price") and ok("y_lag_7d", "mfrr_price")
    assert not ok("base_value", "da_price") and ok("base_value", "ida1_price")                  # base price only for spread targets
    assert not ok("dar_vol_total_d0", "da_price") and ok("dar_vol_total_lag1d", "da_price") and ok("dar_vol_total_d0", "ida1_price")
    assert ok("ida1_spread_d0", "ida2_price") and not ok("ida1_spread_d0", "ida1_price") and not ok("ida2_spread_d0", "ida2_price")
    assert not t[t.group == "fundamental"]["available"].any()                                    # not collected
    assert t[t.target.isin(["pv_weather", "reservoir_inflow"]) & t.available]["group"].isin(["calendar", "market_rule"]).all()
    assert (t[t.available & t.margin_min.notna()]["margin_min"] >= 0).all()


def test_the_validator_rejects_dead_and_malformed_features():
    bad = copy.deepcopy(CAT)
    bad["features"].append({"id": "future_ida3", "group": "x", "kind": "series_value",
                            "params": {"col": "ida3_pt", "day_offset": 0}, "version": 1, "description": "reads the IDA3 result of day D"})
    assert any("future_ida3" in p and "dead feature" in p for p in validate_feature_catalogue(bad, FORM, FC, INV))
    bad = copy.deepcopy(CAT)
    bad["features"].append({"id": "oops", "group": "x", "kind": "modelled_lag", "params": {}, "version": 1, "description": "d"})
    assert any("oops" in p and "needs params" in p for p in validate_feature_catalogue(bad, FORM, FC, INV))
    bad = copy.deepcopy(CAT)
    next(f for f in bad["features"] if f["id"] == "fund_gas_ttf")["blocked_on"] = "no_such_source"
    assert any("fund_gas_ttf" in p for p in validate_feature_catalogue(bad, FORM, FC, INV))


def test_feature_sets_resolve_per_target_and_versions_follow_definitions(tmp_path):
    spec, off = target_spec(FORM, "da_price")["modelled_quantity"], allowed_offsets(FORM, "da_price")
    ida = target_spec(FORM, "ida1_price")["modelled_quantity"], allowed_offsets(FORM, "ida1_price")
    base_da = resolve_feature_set("baseline_v1", SETS, CAT, spec, off, "da_price")
    base_ida = resolve_feature_set("baseline_v1", SETS, CAT, *ida, "ida1_price")
    assert "base_value" not in base_da and "base_value" in base_ida                              # base features exist only for spreads
    full = resolve_feature_set("full_v1", SETS, CAT, spec, off, "da_price")
    assert len(full) > 40 and all(availability(next(f for f in CAT["features"] if f["id"] == i), spec, off)[0] for i in full)
    with pytest.raises(ValueError, match="no stability selection"):
        resolve_feature_set("selected_v1", SETS, CAT, spec, off, "da_price", results_path=str(tmp_path / "none.yaml"))
    res = tmp_path / "r.yaml"
    res.write_text(yaml.safe_dump({"targets": {"da_price": {"selected": ["y_lag_1d", "cal_dow"]}}}), encoding="utf-8")
    assert resolve_feature_set("selected_v1", SETS, CAT, spec, off, "da_price", results_path=str(res)) == ["y_lag_1d", "cal_dow"]
    with pytest.raises(ValueError, match="not available"):
        resolve_feature_set("selected_v1", {"selected_v1": {"features": ["dar_vol_total_d0"]}}, CAT, spec, off, "da_price")   # reads day D
    v = feature_set_version(["y_lag_1d", "cal_dow"], CAT)
    assert v == feature_set_version(["y_lag_1d", "cal_dow"], CAT) and v != feature_set_version(["cal_dow", "y_lag_1d"], CAT)
    changed = copy.deepcopy(CAT)
    next(f for f in changed["features"] if f["id"] == "y_lag_1d")["version"] = 2
    assert feature_set_version(["y_lag_1d", "cal_dow"], changed) != v                                                      # a new definition is a new version


# ---------------------------------------------------------------------------------- builder
def _builder(target: str, ids):
    view = AsOfView(synthetic_panel(), FORM, target)
    return view, fb.FeatureBuilder(view.interface(), CAT, ids)


def test_builder_values_match_the_panel_for_lags_rolls_and_calendar():
    view, b = _builder("da_price", ["cal_hour", "cal_dow", "cal_is_weekend", "y_lag_1d", "y_lag_7d", "y_roll_mean_7d_hour",
                                    "y_roll_mean_7d_all", "y_yday_max", "dar_flow_pt_to_es_lag1d"])
    D = dt.date(2022, 3, 9)
    X = b.build(view.at(D), D)
    panel = view._truth
    local = panel.index.tz_convert("Europe/Madrid")
    da = pd.Series(panel["da_pt"].to_numpy(), index=pd.MultiIndex.from_arrays([local.date, local.hour]))
    h = 5
    row = X[X["cal_hour"] == h].iloc[0]
    assert row["y_lag_1d"] == pytest.approx(da[(D - dt.timedelta(days=1), h)])
    assert row["y_lag_7d"] == pytest.approx(da[(D - dt.timedelta(days=7), h)])
    assert row["y_roll_mean_7d_hour"] == pytest.approx(np.mean([da[(D - dt.timedelta(days=k), h)] for k in range(1, 8)]))
    assert row["y_roll_mean_7d_all"] == pytest.approx(np.mean([da[(D - dt.timedelta(days=k), x)] for k in range(1, 8) for x in range(24)]))
    assert row["y_yday_max"] == pytest.approx(max(da[(D - dt.timedelta(days=1), x)] for x in range(24)))
    assert row["cal_dow"] == D.weekday() and len(X) == 24 and X["dar_flow_pt_to_es_lag1d"].notna().all()


def test_spread_target_features_use_the_known_base_price_and_earlier_auctions():
    view, b = _builder("ida2_price", ["base_value", "base_dev", "base_shape", "base_rank", "base_lag_1d", "base_change_1d",
                                      "ida1_spread_d0", "ida1_spread_lag1d", "dar_vol_total_d0"])
    D = dt.date(2022, 3, 9)
    X = b.build(view.at(D), D)
    t = view.truth_day(D)
    assert np.allclose(X["base_value"], t["da_pt"]) and np.allclose(X["base_dev"], t["da_pt"] - t["da_pt"].mean())
    assert np.allclose(X["ida1_spread_d0"], t["ida1_pt"] - t["da_pt"])                                  # the IDA1 result of D is known at IDA2
    mid, eve = t["da_pt"].to_numpy()[[11, 12, 13, 14, 15]].mean(), t["da_pt"].to_numpy()[[19, 20, 21]].mean()
    assert np.allclose(X["base_shape"], mid - eve) and X["base_rank"].between(0, 1).all()
    assert np.allclose(X["base_change_1d"], X["base_value"] - X["base_lag_1d"])


def test_building_an_unavailable_feature_is_refused():
    view = AsOfView(synthetic_panel(), FORM, "ida1_price")
    with pytest.raises(fb.FeatureUnavailable, match="ida1_spread_d0"):
        fb.FeatureBuilder(view.interface(), CAT, ["ida1_spread_d0"])                  # IDA1 cannot use its own result
    view = AsOfView(synthetic_panel(), FORM, "da_price")
    with pytest.raises(fb.FeatureUnavailable, match="base_value"):
        fb.FeatureBuilder(view.interface(), CAT, ["base_value"])
    with pytest.raises(fb.FeatureUnavailable, match="fund_load_forecast"):
        fb.FeatureBuilder(view.interface(), CAT, ["fund_load_forecast"])


# ------------------------------------------------------------- point-in-time correctness
@pytest.mark.parametrize("target", ["da_price", "ida1_price", "ida3_price", "xbid_price", "afrr_cap_price", "afrr_dn_price", "mfrr_price"])
def test_every_available_feature_is_point_in_time_correct(target):
    panel = synthetic_panel(150)
    view = AsOfView(panel, FORM, target)
    iface = view.interface()
    ids = resolve_feature_set("full_v1", SETS, CAT, iface.spec, iface.offsets, target)
    days = [dt.date(2022, 3, 12), dt.date(2022, 3, 27), dt.date(2022, 4, 20)]               # includes the spring DST day
    assert fb.point_in_time_check(view, fb.FeatureBuilder(iface, CAT, ids), days).empty


def test_the_point_in_time_check_catches_a_feature_that_reads_ahead(monkeypatch):
    panel = synthetic_panel(100)
    view = AsOfView(panel, FORM, "da_price")
    b = fb.FeatureBuilder(view.interface(), CAT, ["dar_vol_total_lag1d"])
    good = fb._KIND["series_value"]

    def reads_ahead(ctx, p, cache):                        # bug: reads one day later than declared
        return good(ctx, {**p, "day_offset": int(p["day_offset"]) + 1}, cache)
    monkeypatch.setitem(fb._KIND, "series_value", reads_ahead)
    bad = fb.point_in_time_check(view, b, [dt.date(2022, 3, 12)])
    assert list(bad["feature"]) == ["dar_vol_total_lag1d"]


# ------------------------------------------------------------------------------------ store
def test_store_returns_what_the_builder_builds_and_rebuilds_when_the_definition_changes(tmp_path):
    view = AsOfView(synthetic_panel(100), FORM, "da_price")
    ids = ["cal_hour", "y_lag_1d", "y_roll_mean_7d_hour"]
    b = fb.FeatureBuilder(view.interface(), CAT, ids)
    store = FeatureStore(str(tmp_path / "f.db"))
    days = [dt.date(2022, 3, 10) + dt.timedelta(days=i) for i in range(3)]
    n = store.materialize(view, b, CAT, "demo", days)
    assert n == 72
    got = store.read("da_price", "demo", days[0], days[-1])
    ref = pd.concat([b.build(view.at(D), D) for D in days])
    pd.testing.assert_frame_equal(got.drop(columns="delivery_date"), ref, check_freq=False, check_names=False,
                                  check_index_type=False)
    store.materialize(view, b, CAT, "demo", days)                                           # idempotent
    assert len(store.read("da_price", "demo")) == 72
    v1 = store.version("da_price", "demo")
    b2 = fb.FeatureBuilder(view.interface(), CAT, ids + ["y_lag_2d"])
    store.materialize(view, b2, CAT, "demo", days[:1])
    assert store.version("da_price", "demo") != v1 and len(store.read("da_price", "demo")) == 24   # new definition: old rows dropped
    with pytest.raises(KeyError):
        store.read("da_price", "missing")


# -------------------------------------------------------------------------------- selection
def test_stability_selects_features_important_in_most_periods_only():
    def imp(a, b, c):
        return pd.DataFrame({"feature": ["steady", "once", "noise"], "importance": [a, b, c]})
    tabs = {"p1": imp(10, 8, 0.0), "p2": imp(9, 0, 0.01), "p3": imp(11, 0, -0.5), "p4": imp(10, 0, 0.0)}
    s = selection.stability(tabs, min_share=0.05, min_period_fraction=0.75)
    assert list(s.index[s["selected"]]) == ["steady"] and s.loc["once", "periods_passed"] == 1
    assert -1.0 <= s.attrs["mean_rank_correlation"] <= 1.0


def test_permutation_importance_finds_the_features_the_target_depends_on():
    panel = synthetic_panel(160)
    view = AsOfView(panel, FORM, "da_price")
    # no lag features here: "same hour last week" is a near-perfect proxy for hour and weekday, and correlated features
    # share the credit, so it would (rightly) take the importance away from the hour
    ids = ["cal_hour", "cal_dow", "cal_doy_sin", "rule_price_floor"]
    b = fb.FeatureBuilder(view.interface(), CAT, ids)
    tr = make_transform("none")
    days = [dt.date(2022, 2, 1) + dt.timedelta(days=i) for i in range(100)]
    table = selection.training_table(view, b, days, tr)
    from lightgbm import LGBMRegressor
    imp = selection.permutation_importance(table, lambda: LGBMRegressor(n_estimators=100, verbose=-1, random_state=0, n_jobs=2),
                                           ids, 0.7, 3, 0, tr).set_index("feature")["importance"]
    assert imp["cal_hour"] == imp.max()                                                      # the planted daily shape
    assert imp["rule_price_floor"] == pytest.approx(0.0, abs=0.2)                           # constant: no information
