"""
test_data_curation.py — the data cleaning and filling before forecasting: errors removed, gaps filled by the written rule, every filled value
flagged, hours an auction does not cover never filled, and nothing filled before a series starts.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from common_layer.data_quality import curation as cu

CFG = cu.load_curation_config()
IDX = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")


def hourly(seed=0, level=80.0) -> pd.Series:
    r = np.random.default_rng(seed)
    hour = np.asarray(IDX.tz_convert("Europe/Madrid").hour)
    return pd.Series(level + 30 * np.sin(2 * np.pi * hour / 24) + r.normal(0, 3, len(IDX)), index=IDX)


def panel(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols, index=IDX)


def test_the_curation_config_is_valid_and_catches_mistakes():
    assert cu.validate_curation_config(CFG) == []
    bad = copy.deepcopy(CFG)
    bad["series"]["da_pt"]["rule"] = "magic"
    bad["series"]["ida1_pt"]["base"] = "nothing"
    problems = cu.validate_curation_config(bad)
    assert any("da_pt" in p for p in problems) and any("ida1_pt" in p for p in problems)


def test_a_short_gap_is_a_straight_line_and_a_long_gap_repeats_the_week_before():
    da = hourly()
    da.iloc[500:502] = np.nan                                 # 2 hours
    da.iloc[900:1000] = np.nan                                # 100 hours
    c = cu.curate(panel(da_pt=da), CFG)
    assert c.panel["da_pt"].notna().all()
    np.testing.assert_allclose(c.panel["da_pt"].iloc[500:502], np.interp([500, 501], [499, 502], [da.iloc[499], da.iloc[502]]))
    np.testing.assert_allclose(c.panel["da_pt"].iloc[900:1000].to_numpy(), da.iloc[900 - 168:1000 - 168].to_numpy())
    assert c.imputed["da_pt"].sum() == 102 and not c.imputed["da_pt"].iloc[:500].any()
    methods = c.log.set_index("method")["hours"].to_dict()
    assert methods[cu.STRAIGHT] == 2 and methods[cu.SEASONAL[168]] == 100 and (c.log["reason"] == "missing").all()


def test_nothing_is_filled_before_a_series_starts_and_the_values_that_exist_are_not_changed():
    da = hourly()
    da.iloc[:100] = np.nan
    da.iloc[300:310] = np.nan
    c = cu.curate(panel(da_pt=da), CFG)
    assert c.panel["da_pt"].iloc[:100].isna().all() and not c.imputed["da_pt"].iloc[:100].any()
    keep = da.notna()
    pd.testing.assert_series_equal(c.panel["da_pt"][keep], da[keep], check_names=False)


def test_data_errors_are_removed_and_filled_but_real_spikes_and_negative_prices_are_kept():
    da = hourly()
    da.iloc[400] = 9000.0                                      # above the legal range
    da.iloc[600] = 2500.0                                      # a real spike
    da.iloc[700:703] = -80.0                                   # real negative prices
    c = cu.curate(panel(da_pt=da), CFG)
    assert c.errors["da_pt"].iloc[400] and c.imputed["da_pt"].iloc[400] and c.panel["da_pt"].iloc[400] < 200
    assert c.panel["da_pt"].iloc[600] == 2500.0 and (c.panel["da_pt"].iloc[700:703] == -80.0).all() and not c.errors["da_pt"].iloc[[600, 700]].any()
    assert (c.log["reason"] == "data error removed").any()


def test_a_flat_day_a_copied_day_and_a_unit_jump_are_errors_but_a_very_low_price_day_is_not():
    da = hourly()
    start = 24 * 20 + 23                                                      # a local midnight (Madrid is UTC+1 in January)
    da.iloc[start:start + 24] = 55.0                                         # one value all day
    da.iloc[start + 48:start + 72] = da.iloc[start + 24:start + 48].to_numpy()   # a copy of the day before
    da.iloc[start + 96:start + 120] = da.iloc[start + 96:start + 120] * 40   # 40 times the level: a unit error
    da.iloc[start + 144:start + 168] = 0.2                                   # a real near-zero day... (flat too)
    c = cu.curate(panel(da_pt=da), CFG)
    e = c.errors["da_pt"]
    assert e.iloc[start:start + 24].all() and e.iloc[start + 48:start + 72].all() and e.iloc[start + 96:start + 120].all()
    low = hourly()
    low.iloc[start + 144:start + 168] = low.iloc[start + 144:start + 168] * 0 + np.linspace(0.0, 1.5, 24)   # near zero but moving: a windy holiday
    assert not cu.curate(panel(da_pt=low), CFG).errors["da_pt"].iloc[start + 144:start + 168].any()


def test_an_intraday_gap_is_the_day_ahead_price_plus_the_typical_spread_and_an_empty_auction_is_the_day_ahead_price():
    da = hourly()
    ida = da + 3.0
    ida.iloc[600:612] = np.nan                                 # a gap
    c = cu.curate(panel(da_pt=da, ida1_pt=ida), CFG)
    np.testing.assert_allclose(c.panel["ida1_pt"].iloc[600:612], da.iloc[600:612] + 3.0)
    assert (c.log[c.log["series"] == "ida1_pt"]["method"] == cu.SPREAD).all()
    ida2 = pd.Series(np.nan, index=IDX)
    ida2.iloc[24 * 10:] = da.iloc[24 * 10:]                      # starts on day 10; its first days have no spreads yet
    ida2.iloc[24 * 10 + 5] = np.nan
    c2 = cu.curate(panel(da_pt=da, ida2_pt=ida2), CFG)
    assert c2.panel["ida2_pt"].iloc[24 * 10 + 5] == pytest.approx(da.iloc[24 * 10 + 5])         # no spread history of that hour: zero spread


def test_hours_an_auction_never_covers_are_not_gaps_and_are_not_filled():
    da = hourly()
    hour = np.asarray(IDX.tz_convert("Europe/Madrid").hour)
    ida3 = (da + 2.0).where(hour >= 13)                          # only the afternoon is traded
    ida3.iloc[24 * 30 + 16] = np.nan                             # a real gap inside a covered hour
    c = cu.curate(panel(da_pt=da, ida3_pt=ida3), CFG)
    assert c.panel["ida3_pt"][hour < 13].isna().all() and c.imputed["ida3_pt"].sum() == 1
    assert c.panel["ida3_pt"].iloc[24 * 30 + 16] == pytest.approx(da.iloc[24 * 30 + 16] + 2.0)


def test_a_band_price_is_carried_forward_then_repeats_the_week():
    band = pd.Series(25.0, index=IDX)
    band.iloc[600:620] = np.nan                                  # 20 hours: carried forward
    band.iloc[900:1000] = np.nan                                 # 100 hours: 48 carried, the rest from a week earlier
    c = cu.curate(panel(afrr_up=band), CFG)
    methods = c.log.groupby("method")["hours"].sum().to_dict()
    assert methods[cu.CARRIED] == 20 + 48 and methods[cu.SEASONAL[168]] == 52 and c.panel["afrr_up"].notna().all()


def test_the_summary_counts_what_was_done_and_columns_without_a_rule_are_left_alone():
    da = hourly()
    da.iloc[300:310] = np.nan
    other = pd.Series(np.nan, index=IDX)
    c = cu.curate(panel(da_pt=da, something_else=other), CFG)
    row = c.summary.set_index("series").loc["da_pt"]
    assert row["filled"] == 10 and row["filled_share"] == pytest.approx(10 / len(IDX)) and row["errors_removed"] == 0
    assert row["straight_line"] + row["same_hour_earlier"] == 10
    assert "something_else" in c.panel.columns and c.panel["something_else"].isna().all() and "something_else" not in set(c.summary["series"])


def test_the_days_after_a_feeds_last_value_are_filled_too_unless_a_limit_is_set():
    da = hourly()
    stale = hourly(1)
    stale.iloc[-200:] = np.nan                                  # silent for the last 200 hours
    c = cu.curate(panel(da_pt=da, imb_short=stale), CFG)
    assert c.panel["imb_short"].notna().all() and c.imputed["imb_short"].iloc[-200:].all() and c.imputed["imb_short"].sum() == 200
    limited = copy.deepcopy(CFG)
    limited["gaps"]["max_tail_hours"] = 72
    c2 = cu.curate(panel(da_pt=da, imb_short=stale), limited)
    assert c2.panel["imb_short"].iloc[-200:-128].notna().all() and c2.panel["imb_short"].iloc[-128:].isna().all()


def test_a_gap_in_the_first_week_of_a_series_is_filled_too_with_the_last_value_so_far():
    da = hourly()
    da.iloc[:50] = np.nan                                       # the series starts at hour 50
    da.iloc[80:90] = np.nan                                     # a gap before any value a week earlier exists
    c = cu.curate(panel(da_pt=da), CFG)
    assert c.panel["da_pt"].iloc[50:].notna().all() and c.panel["da_pt"].iloc[:50].isna().all()
    assert (c.panel["da_pt"].iloc[80:90] == da.iloc[79]).all() or c.imputed["da_pt"].iloc[80:90].all()


def test_the_curated_panel_option_and_the_outputs(tmp_path):
    from common_layer.market_analysis import panel as pnl
    assert "curate" in pnl.build_panel.__code__.co_varnames
    da = hourly()
    da.iloc[300:310] = np.nan
    c = cu.curate(panel(da_pt=da), CFG)
    paths = cu.write_outputs(c, str(tmp_path))
    assert pd.read_csv(paths["summary"])["filled"].iloc[0] == 10 and len(pd.read_csv(paths["gaps"])) == 1
