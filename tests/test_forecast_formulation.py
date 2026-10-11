"""
test_forecast_formulation.py — Phase 5: the formulation is complete, consistent with the catalogue and the
market configuration, and every input of every target is published before its as-of time.

The "tested information set" tests also break the configuration on purpose and require the checks to fail,
so a check that cannot fail would be noticed.
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from common_layer.configuration.forecast_catalogue import load_forecast_catalogue
from common_layer.configuration.forecast_formulation import (SAMPLE_DAYS, availability_table, clock_time,
                                                             load_forecast_formulation,
                                                             validate_forecast_formulation)
from common_layer.forecast_formulation import coherence, evidence
from common_layer.forecast_formulation.targets import build_target, to_level
from common_layer.forecast_formulation.transforms import Asinh, Identity, make_transform, retransform_mean
from common_layer.market_analysis.panel import SERIES

import os

FORM = load_forecast_formulation()
CAT = load_forecast_catalogue()
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MARKET = yaml.safe_load(open(os.path.join(_ROOT, "config", "market.yaml"), encoding="utf-8"))


def problems(form=None):
    return validate_forecast_formulation(form or FORM, CAT, MARKET, list(SERIES))


# ------------------------------------------------------------- the written decisions
def test_formulation_is_valid_against_catalogue_market_and_panel():
    assert problems() == []


def test_every_catalogue_target_has_a_complete_formulation():
    ids = {t["id"] for t in FORM["targets"]}
    assert ids == {t["id"] for t in CAT["targets"]}
    for t in FORM["targets"]:
        assert t["structure"]["multi_step"] == "direct"                    # origin fixed, no recursion on predicted prices
        assert t["output"]["target"] in ("quantile", "scenarios") and t["forbidden"]


def test_day_ahead_and_intraday_outputs_follow_what_the_optimiser_consumes():
    by = {t["id"]: t for t in FORM["targets"]}
    assert by["da_price"]["output"]["target"] == "scenarios"                # the stochastic MILP needs joint scenarios
    assert by["ida1_price"]["modelled_quantity"]["kind"] == "spread"
    assert by["afrr_cap_price"]["modelled_quantity"]["kind"] == "level"    # EUR/MW/h: no spread to a EUR/MWh price


# --------------------------------------------------- the information set is tested
def test_every_input_is_published_before_its_as_of_time_on_all_sample_days():
    tab = availability_table(FORM, CAT)
    assert len(tab) > 100 and set(tab["delivery_day"]) == {d.isoformat() for d in SAMPLE_DAYS}
    assert tab["ok"].all(), tab[~tab["ok"]]
    # an input that can be late must come with a policy
    risky = tab[~tab["ok_late"]]
    assert risky["has_late_policy"].all(), risky[~risky["has_late_policy"]]


def test_the_day_ahead_file_really_is_a_late_risk_for_the_gates_that_need_it():
    tab = availability_table(FORM, CAT)
    r = tab[(tab["series"] == "da_result") & tab["target"].isin(["ida1_price", "afrr_cap_price", "mfrr_price"])]
    assert (r["margin_min"] > 0).all() and (r["margin_late_min"] < 0).all()


def test_a_wrong_publication_time_is_caught():
    bad = copy.deepcopy(FORM)
    bad["series_availability"]["ida2_result"]["published"] = "d-1 23:30"            # now after the XBID W2 window
    bad["targets"][[t["id"] for t in bad["targets"]].index("xbid_price")]["inputs"][2]["from_window"] = "W2"
    assert any("ida2_result" in p and "after the as-of time" in p for p in problems(bad))


def test_an_input_from_too_recent_a_day_is_caught():
    bad = copy.deepcopy(FORM)
    da = next(t for t in bad["targets"] if t["id"] == "da_price")
    next(i for i in da["inputs"] if i["series"] == "imbalance_price")["latest_day_offset"] = -2   # D-2 is out at D-1 12:00
    assert any("imbalance_price" in p for p in problems(bad))


def test_a_late_input_without_a_policy_is_caught():
    bad = copy.deepcopy(FORM)
    next(t for t in bad["targets"] if t["id"] == "ida1_price")["late_policy"] = {}
    assert any("late_policy" in p and "da_result" in p for p in problems(bad))


def test_mixing_units_in_a_spread_is_caught():
    bad = copy.deepcopy(FORM)
    mq = next(t for t in bad["targets"] if t["id"] == "afrr_cap_price")["modelled_quantity"]
    mq.update(kind="spread", base="da_pt")
    assert any("mixes units" in p for p in problems(bad))


def test_the_intraday_same_auction_leakage_must_stay_forbidden():
    bad = copy.deepcopy(FORM)
    next(t for t in bad["targets"] if t["id"] == "ida2_price")["forbidden"] = ["the IDA2 result of day D"]
    assert any("finding F4" in p for p in problems(bad))


def test_market_times_must_match_market_yaml():
    bad = copy.deepcopy(FORM)
    bad["as_of_times"]["IDA1"] = "D-1 14:30"
    assert any("IDA1" in p and "market.yaml" in p for p in problems(bad))


def test_clock_rules_are_parsed_on_daylight_saving_days():
    d = dt.date(2026, 10, 25)                                                     # 25-hour day
    assert clock_time("D-1 11:00", d) == dt.datetime(2026, 10, 24, 11, 0, tzinfo=clock_time("D 00:00", d).tzinfo)
    assert clock_time("live", d) is None
    with pytest.raises(ValueError):
        clock_time("tomorrow noon", d)


# --------------------------------------------------------------- transforms and targets
def test_asinh_is_odd_monotone_and_exactly_invertible_through_negative_prices():
    t = Asinh(20.0)
    x = np.array([-500.0, -50.0, -0.5, 0.0, 0.5, 50.0, 4000.0])
    z = t.forward(x)
    assert np.all(np.diff(z) > 0) and np.allclose(t.forward(-x), -z) and np.allclose(t.inverse(z), x)
    assert t.forward(1.0) == pytest.approx(1.0 / 20.0, rel=1e-3)               # linear near zero
    assert t.forward(4000.0) < 6.0                                              # a 4000 EUR/MWh price is compressed
    assert isinstance(make_transform("none"), Identity) and make_transform("asinh", 5).scale == 5
    with pytest.raises(ValueError):
        make_transform("asinh")


def test_back_transformed_mean_differs_from_the_median_forecast_and_smearing_corrects_it():
    t = Asinh(20.0)
    rng = np.random.default_rng(3)
    resid = rng.normal(0, 0.6, 5000)
    z_hat = np.array([np.arcsinh(150 / 20.0)])
    naive = t.inverse(z_hat)[0]
    mean_fc = retransform_mean(t, z_hat, resid)[0]
    assert mean_fc > naive * 1.05                                               # convexity: the mean price is higher
    true_mean = t.inverse(z_hat[0] + resid).mean()
    assert mean_fc == pytest.approx(true_mean)


def test_spread_target_roundtrip_and_level_target_use_the_config_block():
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    r = np.random.default_rng(1)
    panel = pd.DataFrame({"da_pt": 60 + r.normal(0, 10, 48)}, index=idx)
    panel["ida1_pt"] = panel["da_pt"] + r.normal(0, 3, 48)
    spec = next(t for t in FORM["targets"] if t["id"] == "ida1_price")["modelled_quantity"]
    f = build_target(panel, spec)
    assert np.allclose(f["y_raw"], panel["ida1_pt"] - panel["da_pt"])
    back = to_level(spec, f["y_model"].to_numpy(), f["base"].to_numpy())
    assert np.allclose(back, panel["ida1_pt"])
    spec_c = {**spec, "challenger": {"name": "asinh", "scale": 5.0}}
    fc = build_target(panel, spec_c, challenger=True)
    assert np.allclose(to_level(spec_c, fc["y_model"].to_numpy(), fc["base"].to_numpy(), challenger=True), panel["ida1_pt"])
    da_spec = next(t for t in FORM["targets"] if t["id"] == "da_price")["modelled_quantity"]
    fd = build_target(panel, da_spec)
    assert np.allclose(to_level(da_spec, fd["y_model"].to_numpy(), None), panel["da_pt"])
    with pytest.raises(ValueError):
        build_target(panel, next(t for t in FORM["targets"] if t["id"] == "pv_weather")["modelled_quantity"])


# ------------------------------------------------------------ coherence of hourly and quarters
def test_reconcile_keeps_the_shape_and_makes_quarters_average_to_the_hour():
    rng = np.random.default_rng(5)
    q = rng.normal(50, 15, 24 * 4)
    h = rng.normal(50, 15, 24)
    r = coherence.reconcile(q, h)
    assert np.allclose(coherence.hourly_from_quarters(r), h)
    assert np.allclose(np.diff(r.reshape(-1, 4), axis=1), np.diff(q.reshape(-1, 4), axis=1))     # within-hour shape unchanged
    assert np.allclose(coherence.reconcile(r, h), r)                                            # idempotent
    with pytest.raises(ValueError):
        coherence.reconcile(q[:-1], h)


def test_within_hour_shape_recovers_a_planted_ramp_and_prices_the_flat_split():
    idx = pd.date_range("2025-10-01", periods=24 * 4 * 30, freq="15min", tz="UTC")
    ramp = np.tile([-15.0, -5.0, 5.0, 15.0], 24 * 30)
    d = pd.DataFrame({"price": 60 + ramp}, index=idx)
    d["local_hour"] = d.index.tz_convert("Europe/Madrid").hour
    d["quarter"] = d.index.minute // 15
    out = coherence.within_hour_shape(d)
    assert np.allclose(out["shape"].to_numpy(), np.tile([-15, -5, 5, 15], (24, 1)))
    assert out["flat_split_error"].loc["all", "mae"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- evidence
def test_asinh_scale_search_finds_the_scale_that_removes_the_level_dependence():
    """Planted: volatility proportional to the level above a floor. A suitable scale must flatten the elasticity."""
    r = np.random.default_rng(8)
    idx = pd.date_range("2022-01-01", periods=24 * 30 * 40, freq="h", tz="UTC")
    level = np.repeat(np.linspace(30, 300, 40), 24 * 30)
    p = pd.Series(level * (1 + 0.15 * r.normal(size=len(idx))) + 5 * r.normal(size=len(idx)), index=idx)
    t = evidence.heteroscedasticity_by_scale(p, [None, 20, 100, 300]).set_index("scale")
    raw = t.loc[t.index.isna()].iloc[0]["elasticity"]
    assert raw > 0.7 and abs(t.loc[20, "elasticity"]) < 0.35 and abs(t.loc[20, "elasticity"]) < raw


def test_day_shift_share_and_factors_on_planted_structure():
    r = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=24 * 400, freq="h", tz="UTC")
    local = idx.tz_convert("Europe/Madrid")
    dates = pd.Series(local.date)
    shift_by_day = pd.Series(r.normal(0, 10, dates.nunique()), index=dates.unique())
    tilt_by_day = pd.Series(r.normal(0, 4, dates.nunique()), index=dates.unique())
    shift = dates.map(shift_by_day).to_numpy()
    tilt = dates.map(tilt_by_day).to_numpy() * (local.hour.to_numpy() - 11.5) / 11.5
    resid = pd.Series(shift + tilt + r.normal(0, 1, len(idx)), index=idx)
    s = evidence.day_shift_share(resid)
    assert s["common_shift_share"] == pytest.approx(100 / (100 + 16 / 3 + 1), abs=0.05)
    f = evidence.daily_shape_factors(resid)
    assert f.loc[0, "variance_share"] > 0.8 and f.loc[1, "cumulative"] > 0.9


def test_spread_vs_level_reports_how_much_smaller_the_spread_is():
    r = np.random.default_rng(4)
    idx = pd.date_range("2024-01-01", periods=2000, freq="h", tz="UTC")
    base = pd.Series(60 + 30 * np.sin(np.arange(2000) / 20) + r.normal(0, 5, 2000), index=idx)
    level = base + r.normal(0, 4, 2000)
    s = evidence.spread_vs_level(level, base)
    assert s["std_ratio"] < 0.2 and s["r2_level_on_base"] > 0.9 and s["std_spread"] == pytest.approx(4, abs=0.4)


def test_publication_lags_separate_original_files_from_late_reissues():
    issued = pd.Series(["2026-03-09T12:30:00Z", "2026-06-01T10:00:00Z"])            # D-1 13:30 local, and a re-issue months later
    lags = evidence.publication_lags(issued, pd.Series(["2026-03-10", "2026-03-10"]))
    s = evidence.summarise_publication(lags)
    assert s["original_files"] == 1 and s["median_issue_time_of_day"] == pytest.approx(13.5)
    assert s["median_hours_vs_day_start"] == pytest.approx(-10.5)
