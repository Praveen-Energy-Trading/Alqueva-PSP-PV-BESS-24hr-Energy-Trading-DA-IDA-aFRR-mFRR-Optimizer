"""
regimes.py — dated market-rule changes (config/market_regimes.yaml).

`regime_at(day)` returns the descriptive attributes valid on a delivery day and the events
already in force, so a row of data can be stamped with the rules it was generated under.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional

import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "market_regimes.yaml")


def load_regimes(path: Optional[str] = None) -> List[dict]:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        events = yaml.safe_load(fh)["events"]
    for e in events:
        e["date"] = dt.date.fromisoformat(e["date"])
        if e.get("end"):
            e["end"] = dt.date.fromisoformat(e["end"])
    return sorted(events, key=lambda e: e["date"])


def regime_at(day: dt.date, events: Optional[List[dict]] = None) -> Dict[str, object]:
    """Attributes in force on `day`.

    Keys: da_resolution_min (60 or 15), ida_sessions (6 or 3), price_floor_eur_mwh,
    iberian_exception (bool), events_in_force (ids of rule changes and context events already
    started; deadlines are listed separately under deadlines_passed)."""
    events = events if events is not None else load_regimes()
    ids = {e["id"]: e for e in events}
    in_force = [e["id"] for e in events if e["kind"] != "deadline" and e["date"] <= day]
    passed = [e["id"] for e in events if e["kind"] == "deadline" and e["date"] <= day]
    exc = ids.get("iberian_exception")
    return {
        "da_resolution_min": 15 if day >= ids["sdac_15min"]["date"] else 60,
        "ida_sessions": 3 if day >= ids["ida_three_sessions"]["date"] else 6,
        "price_floor_eur_mwh": -600.0 if day >= ids["price_floor_600"]["date"] else -500.0,
        "iberian_exception": bool(exc and exc["date"] <= day <= (exc.get("end") or day)),
        "events_in_force": in_force,
        "deadlines_passed": passed,
    }
