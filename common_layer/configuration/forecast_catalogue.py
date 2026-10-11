"""
forecast_catalogue.py — load and validate config/forecast_catalogue.yaml.

The catalogue (Phase 0 of the forecasting plan) fixes, for every forecast, the decision it
feeds and its AS-OF rule: which information may be used at the moment the decision is taken.
Later phases (evaluation framework, leakage tests) read it, so the rule is written once.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "forecast_catalogue.yaml")

# Gate names a forecast can be tied to. DA, IDA1-3 and XBID are energy gates (market.yaml
# `gates`); AFRR and MFRR are the reserve-band markets (market.yaml `afrr` / `mfrr`).
ENERGY_GATES = ("DA", "IDA1", "IDA2", "IDA3", "XBID")
RESERVE_GATES = ("AFRR", "MFRR")
REQUIRED_TARGET_KEYS = ("id", "description", "decision", "consumers", "resolution", "horizon",
                        "as_of", "output", "current", "fallback", "priority", "findings")
REQUIRED_AS_OF_KEYS = ("gate", "rule", "allowed", "forbidden")


def load_forecast_catalogue(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_forecast_catalogue(cat: dict, market_yaml: Optional[dict] = None) -> List[str]:
    """Return a list of problems ([] = valid)."""
    problems: List[str] = []
    targets = cat.get("targets") or []
    findings: Dict[str, dict] = cat.get("findings") or {}
    if not targets:
        return ["catalogue has no targets"]

    seen = set()
    for t in targets:
        tid = t.get("id", "<missing id>")
        if tid in seen:
            problems.append(f"{tid}: duplicate id")
        seen.add(tid)
        for k in REQUIRED_TARGET_KEYS:
            if k not in t:
                problems.append(f"{tid}: missing '{k}'")
        a = t.get("as_of") or {}
        for k in REQUIRED_AS_OF_KEYS:
            if k not in a:
                problems.append(f"{tid}: as_of missing '{k}'")
        gate = a.get("gate")
        if gate not in ENERGY_GATES + RESERVE_GATES:
            problems.append(f"{tid}: unknown as_of gate {gate!r}")
        elif market_yaml is not None:
            if gate in ("DA", "IDA1", "IDA2", "IDA3") and gate not in (market_yaml.get("gates") or {}):
                problems.append(f"{tid}: gate {gate} not defined in market.yaml gates")
            if gate == "XBID" and "XBID" not in (market_yaml.get("gates") or {}):
                problems.append(f"{tid}: gate XBID not defined in market.yaml gates")
            if gate == "AFRR" and "afrr" not in market_yaml:
                problems.append(f"{tid}: no 'afrr' section in market.yaml")
            if gate == "MFRR" and "mfrr" not in market_yaml:
                problems.append(f"{tid}: no 'mfrr' section in market.yaml")
        if not isinstance(t.get("priority"), int):
            problems.append(f"{tid}: priority must be an integer")
        for f in t.get("findings") or []:
            if f not in findings:
                problems.append(f"{tid}: refers to unknown finding {f}")
        if not (a.get("forbidden") or []):
            problems.append(f"{tid}: as_of.forbidden is empty (state what must not be used)")
    return problems
