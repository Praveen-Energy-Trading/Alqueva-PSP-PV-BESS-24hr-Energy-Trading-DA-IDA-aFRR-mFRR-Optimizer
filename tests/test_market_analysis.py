"""
test_market_analysis.py — Phase 4 analysis functions on synthetic series with a known answer.

Every statistic is checked against something that can be computed by hand or planted in the data:
an AR(1) process has a known autocorrelation and effective sample size, a planted level shift must be found
where it was planted, a Pareto sample has a known tail index, and so on. Seeded, no network, no clean zone
except for the panel test.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from common_layer.data_layer import calendar as cal
from common_layer.data_layer.clean_store import CleanStore
from common_layer.market_analysis import (breaks, dependence, extremes, panel, relationships, sample_size,
                                          seasonality)
from common_layer.market_analysis.config import events, load_analysis_config, period_of

CFG = load_analysis_config()
RNG = np.random.default_rng(12345)


def hourly_index(n: int, start: str = "2023-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def ar1(n: int, phi: float, sigma: float = 1.0, seed: int = 1) -> np.ndarray:
    r = np.random.default_rng(seed)
    e = r.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------- config
def test_config_periods_are_contiguous_and_events_include_the_crisis():
    assert period_of(dt.date(2023, 6, 1), CFG) == "iberian_exception"
    assert period_of(dt.date(2025, 10, 1), CFG) == "da_15min"
    ids = [e["id"] for e in events(CFG)]
    assert "gas_crisis_start" in ids and "sdac_15min" in ids and "afrr_platform_start" not in ids   # deadlines excluded


# ----------------------------------------------------------------- panel
def test_quarters_become_hourly_means_only_when_all_four_are_present():
    t = pd.date_range("2025-10-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"start_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "resolution_min": 15,
                       "p": [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, np.nan, 4.0]})
    h = panel.to_hourly(df, "p")
    assert h.iloc[0] == 25.0 and np.isnan(h.iloc[1])                 # second hour has a missing quarter


def test_hourly_rows_pass_through_and_calendar_uses_market_time():
    t = pd.date_range("2025-10-25 21:00", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame({"start_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "resolution_min": 60, "p": range(6)})
    h = panel.to_hourly(df, "p")
    assert list(h.values) == [0, 1, 2, 3, 4, 5]
    cal_df = panel.add_calendar(pd.DataFrame({"p": range(6)}, index=pd.date_range("2025-10-25 21:00", periods=6, freq="h", tz="UTC")))
    # 2025-10-26 is the 25-hour day: local hour 2 occurs twice (CEST then CET)
    assert (cal_df["local_hour"] == 2).sum() == 2 and cal_df["local_date"].iloc[0] == dt.date(2025, 10, 25)


def test_build_panel_reads_the_clean_zone_and_separates_ida_sessions(tmp_path):
    store = CleanStore(str(tmp_path / "c.db"))
    day = dt.date(2025, 6, 10)

    def frame(vals, session=None):
        n = len(vals)
        df = pd.DataFrame({"delivery_date": day.isoformat(), "period_index": range(1, n + 1), "resolution_min": 60,
                           "start_utc": [cal.period_start_utc(day, i, 60).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(1, n + 1)],
                           "price_pt_eur_mwh": vals})
        if session is not None:
            df["session"] = session
        return df

    da = frame([float(i) for i in range(24)])
    for c in ("buy_pt_mw", "sell_pt_mw"):
        da[c] = 1.0
    da["issued_utc"], da["flags"] = "", ""
    store.replace_day("da_price", da, raw_sha256="x" * 64, raw_version=1, parser_version="1", ingest_id="t")
    for s, off in ((1, 100.0), (2, 200.0)):
        store.replace_day("ida_price", frame([off + i for i in range(24)], s), raw_sha256="x" * 64, raw_version=1,
                          parser_version="1", ingest_id="t")
    p = panel.build_panel(store, names=["da_pt", "ida1_pt", "ida2_pt", "ida3_pt"])
    assert p["da_pt"].notna().sum() == 24 and p["ida1_pt"].iloc[0] == 100.0 and p["ida2_pt"].iloc[0] == 200.0
    assert p["ida3_pt"].isna().all()


# ----------------------------------------------------------- seasonality
def test_variance_shares_sum_to_one_and_find_the_planted_hour_profile():
    idx = hourly_index(24 * 365 * 2)
    df = panel.add_calendar(pd.DataFrame(index=idx))
    level = 50 + 30 * (df["year"] - 2023) + 5 * np.sin(2 * np.pi * df["month"] / 12)
    shape = 20 * np.sin(2 * np.pi * df["local_hour"] / 24)
    df["p"] = level + shape + RNG.normal(0, 3, len(df))
    t = seasonality.variance_shares(df, "p").set_index("component")
    assert t["share_of_total"].sum() == pytest.approx(1.0, abs=1e-9)
    assert t.loc["hour of day", "share_of_within_month"] > 0.9          # the daily shape dominates what is left
    assert t.loc["day of week", "share_of_total"] < 0.01                # no weekday effect was planted


def test_weekend_effect_and_annual_share_recover_planted_structure():
    idx = hourly_index(24 * 365 * 3)
    df = panel.add_calendar(pd.DataFrame(index=idx))
    df["p"] = 60 - 12 * df["is_weekend"] + 10 * np.cos(2 * np.pi * df["month"] / 12) + RNG.normal(0, 2, len(df))
    w = seasonality.weekend_effect(df, "p")
    assert w["difference"].mean() == pytest.approx(-12, abs=0.7)
    assert seasonality.annual_share(df, "p") > 0.5


# ------------------------------------------------------------ dependence
def test_acf_of_an_ar1_process_matches_phi_and_deseasonalising_keeps_it():
    x = pd.Series(ar1(20000, 0.8), index=hourly_index(20000))
    a = dependence.acf_series(x, 5)
    assert a[1] == pytest.approx(0.8, abs=0.03) and a[3] == pytest.approx(0.8 ** 3, abs=0.04)


def test_stationarity_verdicts_on_known_processes():
    ok = dependence.stationarity(pd.Series(ar1(1500, 0.5)), "ar1")
    walk = dependence.stationarity(pd.Series(np.cumsum(np.random.default_rng(3).normal(size=1500))), "walk")
    assert ok["verdict"] == "stationary" and walk["verdict"] != "stationary"


def test_arch_test_detects_volatility_clustering_but_not_white_noise():
    r = np.random.default_rng(5)
    e, x = r.normal(size=4000), np.zeros(4000)
    for i in range(1, 4000):
        x[i] = e[i] * np.sqrt(0.2 + 0.8 * x[i - 1] ** 2)          # ARCH(1) with strong clustering
    assert dependence.arch_lm(pd.Series(x))["p_value"] < 0.001
    assert dependence.arch_lm(pd.Series(r.normal(size=4000)))["p_value"] > 0.01


# ---------------------------------------------------------------- breaks
def planted_shift(n=1000, at=500, jump=40.0, sigma=8.0, seed=7):
    r = np.random.default_rng(seed)
    y = 50 + r.normal(0, sigma, n)
    y[at:] += jump
    return pd.Series(y, index=pd.date_range("2021-01-01", periods=n, freq="D"))


def test_binary_segmentation_finds_the_planted_shift():
    s = planted_shift()
    found = breaks.binary_segmentation(s.to_numpy(), 3, 30)
    assert abs(found[0][0] - 500) <= 5 and found[0][1] > 0.6          # one split removes most of the variance


def test_event_test_detects_a_shift_at_the_planted_date_and_not_elsewhere():
    s = planted_shift()
    real = breaks.event_test(s, s.index[500].date(), 60, 30)
    null = breaks.event_test(s, s.index[250].date(), 60, 30)
    assert real["shift"] == pytest.approx(40, abs=5) and real["p_level"] < 1e-6
    assert null["p_level"] > 0.01
    assert np.isnan(breaks.event_test(s, s.index[5].date(), 60, 30)["shift"])      # too little data before


def test_data_driven_breaks_are_matched_to_the_nearest_event():
    s = planted_shift()
    ev = [{"id": "planted", "date": s.index[505].date(), "status": "t"}, {"id": "other", "date": dt.date(2021, 2, 1), "status": "t"}]
    t = breaks.data_driven_breaks(s, ev, 2, 30, 30)
    assert t.iloc[0]["nearest_event"] == "planted" and bool(t.iloc[0]["matches_event"])


# -------------------------------------------------------------- extremes
def test_spike_mask_flags_a_planted_spike_using_only_earlier_data():
    idx = hourly_index(24 * 40)
    s = pd.Series(50 + 10 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + np.random.default_rng(2).normal(0, 1, len(idx)), index=idx)
    s.iloc[24 * 30 + 5] += 400
    up, dn = extremes.spike_mask(s)
    assert up.iloc[24 * 30 + 5] and up.sum() == 1 and dn.sum() == 0
    s2 = s.copy()
    s2.iloc[24 * 35:] = 9999                                           # change the future
    up2, _ = extremes.spike_mask(s2)
    assert up2.iloc[: 24 * 35].equals(up.iloc[: 24 * 35])               # earlier flags unchanged


def test_hill_estimator_recovers_a_pareto_tail_index():
    x = (np.random.default_rng(9).pareto(3.0, 200000) + 1.0) * 10.0
    assert extremes.hill_tail(x, 0.05)["alpha"] == pytest.approx(3.0, abs=0.35)


def test_negative_share_and_cannibalisation_on_a_duck_shaped_day():
    idx = hourly_index(24 * 100)
    df = panel.add_calendar(pd.DataFrame(index=idx))
    df["p"] = np.where(df["local_hour"].isin([11, 12, 13, 14, 15]), -5.0, np.where(df["local_hour"].isin([19, 20, 21]), 120.0, 60.0))
    neg = extremes.negative_by_hour(df, "p")
    assert neg.loc[12].iloc[0] == 1.0 and neg.loc[3].iloc[0] == 0.0
    c = extremes.cannibalisation(df, "p", [11, 12, 13, 14, 15]).iloc[0]
    assert c["midday_ratio"] < 0 and c["evening_minus_midday"] == pytest.approx(125.0)


def test_heteroscedasticity_elasticity_is_one_when_volatility_scales_with_level():
    r = np.random.default_rng(4)
    levels = np.repeat(np.linspace(20, 200, 40), 30)
    x = levels * (1 + 0.1 * r.normal(size=len(levels)))                # std proportional to level
    el = extremes.heteroscedasticity(pd.Series(x, index=pd.date_range("2020-01-01", periods=len(x), freq="D")))
    assert el["elasticity"] == pytest.approx(1.0, abs=0.2)


# --------------------------------------------------------- relationships
def test_correlation_spread_and_overlap_rules():
    n = 5000
    base = pd.Series(ar1(n, 0.9, 5.0, seed=11) + 60, index=hourly_index(n))
    other = base + 5.0 + np.random.default_rng(6).normal(0, 2, n)
    p = pd.DataFrame({"da": base, "x": other, "y": np.nan})
    p.loc[p.index[:100], "y"] = 1.0
    c = relationships.aligned_correlation(p, ["da", "x", "y"], 500)
    assert c["corr"].loc["da", "x"] > 0.95 and np.isnan(c["corr"].loc["da", "y"])   # y overlaps only 100 hours
    periods = [{"id": "all", "start": dt.date(2023, 1, 1), "end": dt.date(2023, 12, 31)}]
    sp = relationships.spread_stats(p, "da", "x", periods).iloc[0]
    assert sp["mean_spread"] == pytest.approx(5.0, abs=0.2) and sp["slope_on_base"] == pytest.approx(1.0, abs=0.02)


# ----------------------------------------------------------- sample size
def test_effective_sample_size_of_ar1_matches_theory_and_white_noise_keeps_its_n():
    # a single draw scatters by about 13 %, so compare the mean of 10 independent series with the theory n(1-phi)/(1+phi)
    n, phi = 4000, 0.8
    ess = np.mean([sample_size.effective_sample_size(pd.Series(ar1(n, phi, seed=s)))["ess"] for s in range(10)])
    assert ess == pytest.approx(n * (1 - phi) / (1 + phi), rel=0.15)
    wn = sample_size.effective_sample_size(pd.Series(np.random.default_rng(8).normal(size=n)))
    assert wn["ess"] == pytest.approx(n, rel=0.15)


def test_representativeness_separates_a_different_regime_from_a_similar_one():
    r = np.random.default_rng(10)
    idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    vals = np.where(idx.year == 2022, 200 + r.normal(0, 40, len(idx)), 50 + r.normal(0, 10, len(idx)))
    out = sample_size.representativeness(pd.Series(vals, index=idx), [], 12, 0.2).set_index("group")
    assert out.loc["2022", "ks_distance"] > 0.9 and not bool(out.loc["2022", "similar"])
    assert bool(out.loc["2021", "similar"])


# ------------------------------------------------ placebo years, ESS of changes, overlap share
def test_placebo_years_separate_a_seasonal_step_from_a_real_event():
    idx = pd.date_range("2020-01-01", "2026-12-31", freq="D")
    r = np.random.default_rng(31)
    y = 50 + r.normal(0, 4, len(idx))
    y[(idx.month >= 6)] += 30.0                                   # a seasonal step every 1 June
    y[(idx.year == 2022) & (idx.month >= 6)] += 60.0              # plus an extra shift in 2022 only
    s = pd.Series(y, index=idx)
    ev = [{"id": "e2022", "date": dt.date(2022, 6, 1), "status": "t"}, {"id": "e2024", "date": dt.date(2024, 6, 1), "status": "t"}]
    t = breaks.event_tests(s, ev, 60, 30).set_index("event")
    assert t.loc["e2022", "shift"] == pytest.approx(90, abs=6) and t.loc["e2022", "z_vs_placebo"] > 4
    assert t.loc["e2024", "shift"] == pytest.approx(30, abs=6) and abs(t.loc["e2024", "z_vs_placebo"]) < 2
    assert t.loc["e2022", "n_placebo"] == 6


def test_ess_of_changes_is_larger_than_ess_of_a_random_walk_level_and_overlap_is_reported():
    walk = pd.Series(np.cumsum(np.random.default_rng(2).normal(size=1500)), index=pd.date_range("2022-01-01", periods=1500, freq="D"))
    per = [{"id": "all", "label": "all", "start": dt.date(2022, 1, 1), "end": dt.date(2030, 1, 1)}]
    t = sample_size.ess_by_period(walk, per).iloc[0]
    assert t["effective_days_changes"] > 10 * t["effective_days"]       # increments of a random walk are white noise
    idx = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    rep = sample_size.representativeness(pd.Series(np.random.default_rng(1).normal(50, 10, len(idx)), index=idx), [], 12, 0.2)
    last = rep[rep["group"] == "2025"].iloc[0]
    assert last["share_inside_reference"] > 0.95 and rep[rep["group"] == "2024"].iloc[0]["share_inside_reference"] < 0.05
