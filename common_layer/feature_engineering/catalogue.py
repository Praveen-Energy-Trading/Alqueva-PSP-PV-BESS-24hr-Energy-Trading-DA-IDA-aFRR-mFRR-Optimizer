"""
catalogue.py — load and validate the feature catalogue, derive what each feature needs, and decide for each
target whether the feature is available at that target's as-of time.

`requirements(feature, spec)` returns {panel column: latest delivery-day offset the feature reads}, derived from the
feature's kind and params only (a feature that reads yesterday needs offset -1). A feature is AVAILABLE for a
target if, for every column it reads, the target's information set allows that offset or a later one. Availability
is therefore a computed property, and `availability_table` also reports when the latest input is published and the
margin to the as-of time. Feature sets are versioned by a hash of their members' definitions.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from common_layer.configuration.forecast_formulation import _as_of_for, clock_time
from common_layer.forecast_evaluation.asof import PANEL_TO_INPUT, allowed_offsets, target_spec
from common_layer.market_analysis.panel import SERIES

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CATALOGUE_PATH = os.path.join(_REPO, "config", "feature_catalogue.yaml")
SETS_PATH = os.path.join(_REPO, "config", "feature_sets.yaml")
RESULTS_PATH = os.path.join(_REPO, "config", "feature_selection_results.yaml")

KINDS = {
    "calendar_hour": {"field"}, "calendar_day": {"field"}, "modelled_lag": {"lag_days"},
    "modelled_roll": {"window_days", "stat", "scope"}, "modelled_yday": {"stat"}, "base_now": {"field"},
    "base_lag": {"lag_days"}, "base_change": {"lag_days"}, "series_value": {"col", "day_offset"},
    "series_spread": {"col", "base_col", "day_offset"}, "series_day_stat": {"col", "day_offset", "stat"},
    "fundamental": set(),
}
OPTIONAL = {"modelled_roll": {"end_lag_days"}, "modelled_yday": {"lag_days"}}
STATS = {"mean", "std", "min", "max"}
BASE_FIELDS = {"value", "dev", "day_mean", "day_max", "day_min", "rank", "shape"}
SAMPLE_DAY = dt.date(2026, 3, 10)


def load_feature_catalogue(path: Optional[str] = None) -> dict:
    with open(path or CATALOGUE_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_feature_sets(path: Optional[str] = None) -> dict:
    with open(path or SETS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["feature_sets"]


def load_selection_results(path: Optional[str] = None) -> dict:
    """The generated stability-selection results (empty when the selection has not been run)."""
    p = path or RESULTS_PATH
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def by_id(cat: dict) -> Dict[str, dict]:
    return {f["id"]: f for f in cat["features"]}


# ------------------------------------------------------------------------------ what a feature reads
def requirements(f: dict, spec: dict) -> Optional[Dict[str, int]]:
    """{panel column: latest delivery-day offset read}, or None when the feature cannot be built for this target
    (a base-price feature on a level target, a fundamental that is not collected, a target without a price series)."""
    k, p = f["kind"], f.get("params", {})
    series, base = spec.get("series"), spec.get("base")
    if k in ("calendar_hour", "calendar_day"):
        return {}
    if k == "fundamental":
        return None
    if k in ("modelled_lag", "modelled_roll", "modelled_yday"):
        if series is None:
            return None
        off = -int(p["lag_days"]) if k == "modelled_lag" else (
            -int(p.get("end_lag_days", 1)) if k == "modelled_roll" else -int(p.get("lag_days", 1)))
        return {series: off, **({base: off} if base else {})}
    if k in ("base_now", "base_change"):
        return {base: 0} if base else None
    if k == "base_lag":
        return {base: -int(p["lag_days"])} if base else None
    if k in ("series_value", "series_day_stat"):
        return {p["col"]: int(p["day_offset"])}
    if k == "series_spread":
        return {p["col"]: int(p["day_offset"]), p["base_col"]: int(p["day_offset"])}
    raise ValueError(f"unknown feature kind {k!r}")


def availability(f: dict, spec: dict, offsets: Dict[str, int]) -> Tuple[bool, str]:
    if f.get("status") == "not_collected":
        return False, f"data not collected (needs {f.get('blocked_on')})"
    req = requirements(f, spec)
    if req is None:
        return False, "does not apply to this target"
    for col, off in req.items():
        if col not in offsets:
            return False, f"{col} is not in the target's information set"
        if off > offsets[col]:
            return False, f"reads {col} up to D{off:+d} but the target may use it only up to D{offsets[col]:+d}"
    return True, ""


def feature_availability(cat: dict, formulation: dict, forecast_catalogue: dict,
                         day: dt.date = SAMPLE_DAY) -> pd.DataFrame:
    """One row per (feature, target): available or why not; for available features, when the latest input is
    published and the margin to the target's as-of time (minutes, earliest XBID window)."""
    gate_of = {t["id"]: t["as_of"]["gate"] for t in forecast_catalogue["targets"]}
    rows = []
    for t in formulation["targets"]:
        tid = t["id"]
        spec, offsets = t["modelled_quantity"], allowed_offsets(formulation, tid)
        as_of = clock_time(_as_of_for(formulation, gate_of[tid], None), day)
        for f in cat["features"]:
            ok, why = availability(f, spec, offsets)
            row = {"feature": f["id"], "group": f["group"], "target": tid, "available": ok, "reason": why,
                   "latest_input_published": "", "margin_min": None}
            if ok:
                pubs = []
                for col, off in (requirements(f, spec) or {}).items():
                    rule = formulation["series_availability"][PANEL_TO_INPUT[col]]["published"]
                    pub = clock_time(rule, day + dt.timedelta(days=off))
                    if pub is not None:
                        pubs.append(pub)
                if pubs:
                    last = max(pubs)
                    row["latest_input_published"] = last.strftime("%Y-%m-%d %H:%M")
                    row["margin_min"] = (as_of - last).total_seconds() / 60.0
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ validation
def validate_feature_catalogue(cat: dict, formulation: dict, forecast_catalogue: dict,
                               inventory: Optional[dict] = None) -> List[str]:
    problems: List[str] = []
    seen = set()
    inv_ids = {s["id"] for s in (inventory or {}).get("sources", [])} if inventory else None
    for f in cat["features"]:
        fid = f.get("id", "<no id>")
        if fid in seen:
            problems.append(f"{fid}: duplicate id")
        seen.add(fid)
        k = f.get("kind")
        if k not in KINDS:
            problems.append(f"{fid}: unknown kind {k!r}")
            continue
        p = f.get("params", {})
        if not KINDS[k] <= set(p) <= KINDS[k] | OPTIONAL.get(k, set()):
            problems.append(f"{fid}: kind {k} needs params {sorted(KINDS[k])} (optional {sorted(OPTIONAL.get(k, []))}), got {sorted(p)}")
        if not isinstance(f.get("version"), int) or not f.get("description"):
            problems.append(f"{fid}: version (integer) and description are required")
        if "stat" in p and p["stat"] not in STATS:
            problems.append(f"{fid}: stat must be one of {sorted(STATS)}")
        if k == "base_now" and p.get("field") not in BASE_FIELDS:
            problems.append(f"{fid}: base_now field must be one of {sorted(BASE_FIELDS)}")
        for c in ("col", "base_col"):
            if c in p and p[c] not in SERIES:
                problems.append(f"{fid}: {c} {p[c]!r} is not a panel series")
        if k == "fundamental":
            if f.get("status") != "not_collected" or not f.get("blocked_on"):
                problems.append(f"{fid}: a fundamental must be status not_collected with blocked_on")
            elif inv_ids is not None and f["blocked_on"] not in inv_ids:
                problems.append(f"{fid}: blocked_on {f['blocked_on']!r} is not a source in the data inventory")
    if problems:
        return problems
    tab = feature_availability(cat, formulation, forecast_catalogue)
    for f in cat["features"]:
        if f["kind"] != "fundamental" and not tab[(tab["feature"] == f["id"]) & tab["available"]].shape[0]:
            problems.append(f"{f['id']}: available for no target (dead feature)")
    late = tab[tab["available"] & tab["margin_min"].notna() & (tab["margin_min"] < 0)]
    for _, r in late.iterrows():
        problems.append(f"{r['feature']} for {r['target']}: latest input published {r['latest_input_published']}, after the as-of time")
    return problems


# --------------------------------------------------------------------------------- feature sets
def resolve_feature_set(name: str, sets: dict, cat: dict, spec: dict, offsets: Dict[str, int],
                        target_id: Optional[str] = None, results_path: Optional[str] = None) -> List[str]:
    """The ordered feature ids of a set for one target. 'all_available' keeps every active available feature;
    'from_selection' reads the stability-selection result stored for the target; an explicit list must be fully
    available (an unavailable member is an error, not silently dropped)."""
    if name not in sets:
        raise KeyError(f"unknown feature set {name!r}; available: {list(sets)}")
    feats, defs = sets[name]["features"], by_id(cat)
    if feats == "from_selection":
        entry = load_selection_results(results_path).get("targets", {}).get(target_id)
        if not entry:
            raise ValueError(f"feature set {name}: no stability selection stored for {target_id}; "
                             f"run tools/run_feature_selection.py --target {target_id}")
        feats = entry["selected"]
    if feats == "all_available":
        return [f["id"] for f in cat["features"] if availability(f, spec, offsets)[0]]
    out = []
    for fid in feats:
        if fid not in defs:
            raise KeyError(f"feature set {name}: unknown feature {fid!r}")
        ok, why = availability(defs[fid], spec, offsets)
        if not ok:
            if defs[fid]["kind"] in ("base_now", "base_lag", "base_change") and "does not apply" in why:
                continue                                  # base features simply do not exist for level targets
            raise ValueError(f"feature set {name}: {fid} is not available for this target: {why}")
        out.append(fid)
    return out


def feature_set_version(ids: List[str], cat: dict) -> str:
    """Twelve-character hash of the definitions (id, kind, params, version) of the features, in order."""
    defs = by_id(cat)
    payload = [(i, defs[i]["kind"], defs[i].get("params", {}), defs[i]["version"]) for i in ids]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
