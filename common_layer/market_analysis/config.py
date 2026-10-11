"""Loads config/market_analysis.yaml and joins its events with config/market_regimes.yaml."""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional

import yaml

from common_layer.data_layer import regimes

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "market_analysis.yaml")


def load_analysis_config(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for p in cfg["periods"]:
        p["start"], p["end"] = dt.date.fromisoformat(str(p["start"])), dt.date.fromisoformat(str(p["end"]))
    for e in cfg.get("extra_events", []):
        e["date"] = dt.date.fromisoformat(str(e["date"]))
    problems = validate_analysis_config(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    return cfg


def validate_analysis_config(cfg: dict) -> List[str]:
    out: List[str] = []
    ps = cfg["periods"]
    for a, b in zip(ps, ps[1:]):
        if b["start"] != a["end"] + dt.timedelta(days=1):
            out.append(f"periods {a['id']} and {b['id']} are not contiguous")
    if len({p["id"] for p in ps}) != len(ps):
        out.append("duplicate period ids")
    for k in ("break_tests", "dependence", "extremes", "relationships", "sample_size"):
        if k not in cfg:
            out.append(f"section {k} missing")
    return out


def events(cfg: dict) -> List[dict]:
    """Rule changes and context events (not deadlines) plus the extra events, sorted by date."""
    out: List[dict] = [{"id": e["id"], "date": e["date"], "status": e["status"], "description": e["description"]}
                       for e in regimes.load_regimes() if e["kind"] != "deadline"]
    out += [{"id": e["id"], "date": e["date"], "status": e.get("status", "to_verify"),
             "description": e["description"]} for e in cfg.get("extra_events", [])]
    return sorted(out, key=lambda e: e["date"])


def period_of(day: dt.date, cfg: dict) -> Optional[str]:
    for p in cfg["periods"]:
        if p["start"] <= day <= p["end"]:
            return p["id"]
    return None
