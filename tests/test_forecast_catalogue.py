"""
test_forecast_catalogue.py — the forecast catalogue (config/forecast_catalogue.yaml) is complete
and consistent with the gates defined in config/market.yaml.
"""
from __future__ import annotations

import copy
import os

import yaml

from common_layer.configuration.forecast_catalogue import (
    load_forecast_catalogue, validate_forecast_catalogue,
)

_MARKET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "market.yaml")


def _market():
    with open(_MARKET, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_catalogue_is_valid_against_the_market_gates():
    assert validate_forecast_catalogue(load_forecast_catalogue(), _market()) == []


def test_every_forecaster_of_the_pipeline_is_in_the_catalogue():
    ids = {t["id"] for t in load_forecast_catalogue()["targets"]}
    assert {"da_price", "pv_weather", "reservoir_inflow", "ida1_price", "ida2_price", "ida3_price",
            "xbid_price", "afrr_cap_price", "mfrr_price"} <= ids


def test_intraday_gates_name_the_results_that_are_known_at_their_as_of_time():
    by_id = {t["id"]: t for t in load_forecast_catalogue()["targets"]}
    assert any("DA result" in x for x in by_id["ida1_price"]["as_of"]["allowed"])
    assert any("IDA1 result" in x for x in by_id["ida2_price"]["as_of"]["allowed"])
    assert any("IDA2" in x for x in by_id["ida3_price"]["as_of"]["allowed"])


def test_missing_fields_and_unknown_references_are_reported():
    cat = copy.deepcopy(load_forecast_catalogue())
    cat["targets"][0].pop("decision")
    cat["targets"][1]["as_of"]["gate"] = "NOPE"
    cat["targets"][2]["findings"] = ["F999"]
    cat["targets"][3]["as_of"]["forbidden"] = []
    cat["targets"][4]["id"] = cat["targets"][5]["id"]
    problems = " | ".join(validate_forecast_catalogue(cat, _market()))
    for expected in ("missing 'decision'", "unknown as_of gate", "unknown finding F999",
                     "forbidden is empty", "duplicate id"):
        assert expected in problems


def test_gate_missing_from_market_yaml_is_reported():
    cat = load_forecast_catalogue()
    market = _market()
    market["gates"].pop("IDA2")
    assert any("IDA2 not defined" in p for p in validate_forecast_catalogue(cat, market))


def test_findings_have_a_severity_and_a_phase():
    for fid, f in load_forecast_catalogue()["findings"].items():
        assert f["severity"] in ("high", "medium", "low", "info"), fid
        assert isinstance(f["fix_in_phase"], int), fid
        assert f["summary"], fid
