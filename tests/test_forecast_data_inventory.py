"""
test_forecast_data_inventory.py — Phase 1 exit criterion: the data inventory is complete and
consistent with the forecast catalogue.
"""
from __future__ import annotations

import copy

import pandas as pd

from common_layer.configuration.forecast_catalogue import load_forecast_catalogue
from common_layer.configuration.forecast_data_inventory import (
    load_data_inventory, validate_data_inventory,
)


def test_inventory_is_valid_against_the_catalogue():
    assert validate_data_inventory(load_data_inventory(), load_forecast_catalogue()) == []


def test_every_source_has_owner_timestamp_rule_licence_and_quality_note():
    for s in load_data_inventory()["sources"]:
        assert s["owner"] and s["publication"] and s["licence"]["note"] and s["quality_note"], s["id"]


def test_sources_in_use_without_a_live_feed_are_flagged_as_findings():
    inv = load_data_inventory()
    no_feed = [s for s in inv["sources"] if s["status"] == "in_use" and not s["live_feed"]]
    assert {s["id"] for s in no_feed} == {"pv_weather_history", "inflow_history"}
    assert all("D1" in s["findings"] for s in no_feed)


def test_gaps_and_candidates_are_not_used_as_loaders():
    for s in load_data_inventory()["sources"]:
        if s["status"] in ("candidate", "gap"):
            assert s["loader"] in (None, "none"), s["id"]


def test_problems_are_reported():
    inv = copy.deepcopy(load_data_inventory())
    cat = load_forecast_catalogue()
    inv["sources"][0].pop("owner")
    inv["sources"][1]["licence"]["status"] = "maybe"
    inv["sources"][2]["used_by"] = ["no_such_forecast"]
    inv["sources"][3]["status"] = "weird"
    inv["sources"][4]["id"] = inv["sources"][5]["id"]
    problems = " | ".join(validate_data_inventory(inv, cat))
    for expected in ("missing 'owner'", "licence.status", "unknown forecast no_such_forecast",
                     "status must be one of", "duplicate id"):
        assert expected in problems


def test_uncovered_forecast_is_reported():
    inv = copy.deepcopy(load_data_inventory())
    for s in inv["sources"]:
        s["used_by"] = [t for t in s["used_by"] if t != "mfrr_price"]
    assert any("mfrr_price is covered by no data source" in p
               for p in validate_data_inventory(inv, load_forecast_catalogue()))


def test_dates_stated_in_the_inventory_match_the_data_files():
    """The end of the real PV weather and inflow data (2026-06-30) is the factual basis of finding D1."""
    base = "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/"
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pv = pd.read_excel(os.path.join(root, base, "pv_training_data_from_2015.xlsx"))
    pv["Date"] = pd.to_datetime(pv["Date"])
    assert pv.loc[pv.source != "SYNTHETIC", "Date"].max() == pd.Timestamp("2026-06-30")
    inf = pd.read_excel(os.path.join(root, base, "inflow_training_data_from_2015.xlsx"))
    inf["Date"] = pd.to_datetime(inf["Date"])
    assert inf.loc[~inf.source.str.contains("Climatology"), "Date"].max() == pd.Timestamp("2026-06-30")
