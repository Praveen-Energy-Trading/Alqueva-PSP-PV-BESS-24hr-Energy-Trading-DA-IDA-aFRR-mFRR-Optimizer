"""
forecast_data_inventory.py — load and validate config/forecast_data_inventory.yaml.

Phase 1 exit criterion: every input has an owner, a timestamp rule (publication), a licence note
and a quality note, and every forecast of the catalogue is covered by at least one source.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "forecast_data_inventory.yaml")

STATUSES = ("in_use", "candidate", "gap")
LICENCE_STATUSES = ("verified", "unverified")
REQUIRED_KEYS = ("id", "status", "provider", "content", "access", "resolution", "publication",
                 "history", "live_feed", "loader", "stored_in", "licence", "cost", "vintage",
                 "used_by", "owner", "quality_note", "findings")


def load_data_inventory(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_data_inventory(inv: dict, catalogue: Optional[dict] = None) -> List[str]:
    """Return a list of problems ([] = valid)."""
    problems: List[str] = []
    sources = inv.get("sources") or []
    findings: Dict[str, dict] = inv.get("findings") or {}
    if not sources:
        return ["inventory has no sources"]

    target_ids = {t["id"] for t in (catalogue or {}).get("targets", [])} if catalogue else None
    seen, covered = set(), set()
    for s in sources:
        sid = s.get("id", "<missing id>")
        if sid in seen:
            problems.append(f"{sid}: duplicate id")
        seen.add(sid)
        for k in REQUIRED_KEYS:
            if k not in s:
                problems.append(f"{sid}: missing '{k}'")
        if s.get("status") not in STATUSES:
            problems.append(f"{sid}: status must be one of {STATUSES}")
        lic = s.get("licence") or {}
        if lic.get("status") not in LICENCE_STATUSES:
            problems.append(f"{sid}: licence.status must be one of {LICENCE_STATUSES}")
        if not lic.get("note"):
            problems.append(f"{sid}: licence.note is empty")
        for k in ("owner", "publication", "quality_note"):
            if not s.get(k):
                problems.append(f"{sid}: '{k}' is empty")
        if not isinstance(s.get("live_feed"), bool):
            problems.append(f"{sid}: live_feed must be true or false")
        if s.get("status") == "in_use" and s.get("live_feed") and s.get("loader") in (None, "none"):
            problems.append(f"{sid}: in_use with a live feed needs a loader")
        for f in s.get("findings") or []:
            if f not in findings:
                problems.append(f"{sid}: refers to unknown finding {f}")
        for t in s.get("used_by") or []:
            covered.add(t)
            if target_ids is not None and t not in target_ids:
                problems.append(f"{sid}: used_by refers to unknown forecast {t}")

    if target_ids is not None:
        for t in sorted(target_ids - covered):
            problems.append(f"forecast {t} is covered by no data source")
    return problems
