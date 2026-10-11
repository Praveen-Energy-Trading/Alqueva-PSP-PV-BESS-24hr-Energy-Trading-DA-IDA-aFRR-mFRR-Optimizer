"""
forecast_formulation.py — load and validate config/forecast_formulation.yaml.

Phase 5 exit criterion: "the information set per target is written down and tested". The test is
`availability_table`: for every input of every target it computes, on sample delivery days (an ordinary
weekday, a Saturday and both daylight-saving days), when the data for the latest day the input may use is
published and compares it with the as-of time of the forecast.

    margin_min        as-of time minus typical publication time (must be >= 0)
    margin_late_min   the same with the late-tail publication time (a negative value needs a late_policy)

Also checked: targets match the forecast catalogue, as-of times match config/market.yaml, spreads only
between series of the same unit, transforms are valid, and the previous-hour-of-the-same-auction leakage
(finding F4) is explicitly forbidden for every intraday target.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "forecast_formulation.yaml")
TZ = ZoneInfo("Europe/Madrid")
SAMPLE_DAYS = [dt.date(2026, 3, 10), dt.date(2026, 3, 14), dt.date(2026, 3, 29), dt.date(2026, 10, 25)]

KINDS = ("level", "spread")
TRANSFORMS = ("none", "asinh")
OUTPUTS = ("point", "quantile", "scenarios")
STATUSES = ("available", "not_collected")
EVIDENCE = ("empirical", "rulebook", "assumed", "none")
_CLOCK = re.compile(r"^[Dd]([+-]\d+)?\s+(\d{2}):(\d{2})$")
INTRADAY_TARGETS = {"ida1_price": "same auction", "ida2_price": "same auction", "ida3_price": "same auction",
                    "xbid_price": "same window"}


def load_forecast_formulation(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def clock_time(rule: str, day: dt.date) -> Optional[dt.datetime]:
    """'D-1 11:00' / 'd+1 04:30' relative to `day`, as a market-time datetime. 'live' -> None (always known)."""
    if rule == "live":
        return None
    m = _CLOCK.match(rule.strip())
    if not m:
        raise ValueError(f"cannot read time rule {rule!r}")
    off, hh, mm = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    d = day + dt.timedelta(days=off)
    return dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=TZ)


def _as_of_for(form: dict, gate: str, window: Optional[str]) -> str:
    v = form["as_of_times"][gate]
    return v[window or "W1"] if isinstance(v, dict) else v


def availability_table(form: dict, catalogue: dict, days: Optional[List[dt.date]] = None) -> pd.DataFrame:
    """One row per (target, input, sample day) with the margins described in the module docstring."""
    gate_of = {t["id"]: t["as_of"]["gate"] for t in catalogue["targets"]}
    avail = form["series_availability"]
    rows = []
    for t in form["targets"]:
        gate = gate_of[t["id"]]
        policy = t.get("late_policy") or {}
        for inp in t["inputs"]:
            window = inp.get("from_window")
            as_of_rule = _as_of_for(form, gate, window)
            rule = avail[inp["series"]]
            for D in days or SAMPLE_DAYS:
                as_of = clock_time(as_of_rule, D)
                d = D + dt.timedelta(days=int(inp["latest_day_offset"]))
                pub = clock_time(rule["published"], d)
                late = clock_time(rule["late"], d) if rule.get("late") else pub
                m = None if pub is None else (as_of - pub).total_seconds() / 60.0
                ml = None if late is None else (as_of - late).total_seconds() / 60.0
                rows.append({"target": t["id"], "series": inp["series"], "status": inp["status"], "window": window or "",
                             "delivery_day": D.isoformat(), "input_day": d.isoformat(), "as_of": as_of.strftime("%Y-%m-%d %H:%M"),
                             "published": "live" if pub is None else pub.strftime("%Y-%m-%d %H:%M"),
                             "margin_min": m, "margin_late_min": ml, "evidence": rule["evidence"],
                             "ok": m is None or m >= 0, "ok_late": ml is None or ml >= 0,
                             "has_late_policy": inp["series"] in policy})
    return pd.DataFrame(rows)


def validate_forecast_formulation(form: dict, catalogue: dict, market_yaml: Optional[dict] = None,
                                  panel_series: Optional[List[str]] = None) -> List[str]:
    problems: List[str] = []
    cat_targets = {t["id"]: t for t in catalogue["targets"]}
    form_ids = [t["id"] for t in form.get("targets", [])]
    if set(form_ids) != set(cat_targets) or len(form_ids) != len(set(form_ids)):
        problems.append(f"targets differ from the catalogue: missing {sorted(set(cat_targets) - set(form_ids))}, "
                        f"extra {sorted(set(form_ids) - set(cat_targets))}")
        return problems
    avail = form.get("series_availability") or {}
    units = form.get("series_units") or {}

    for name, rule in avail.items():
        if rule.get("evidence") not in EVIDENCE:
            problems.append(f"series {name}: evidence must be one of {EVIDENCE}")
        for k in ("published", "late"):
            if rule.get(k):
                try:
                    clock_time(rule[k], dt.date(2026, 1, 1))
                except ValueError as exc:
                    problems.append(f"series {name}: {exc}")

    # as-of times agree with the market configuration
    if market_yaml is not None:
        gates = market_yaml.get("gates", {})
        for g in ("DA", "IDA1", "IDA2", "IDA3"):
            want = gates.get(g, {}).get("pipeline_trigger")
            if form["as_of_times"].get(g) != want:
                problems.append(f"as_of_times.{g} {form['as_of_times'].get(g)!r} differs from market.yaml {want!r}")
        wins = {w["id"]: w["trigger"] for w in gates.get("XBID", {}).get("check_windows", [])}
        if form["as_of_times"].get("XBID") != wins:
            problems.append("as_of_times.XBID windows differ from market.yaml check_windows")

    for t in form["targets"]:
        tid = t["id"]
        if cat_targets[tid]["as_of"]["gate"] not in form["as_of_times"]:
            problems.append(f"{tid}: no as-of time for gate {cat_targets[tid]['as_of']['gate']}")
        mq = t.get("modelled_quantity") or {}
        if mq.get("kind") not in KINDS:
            problems.append(f"{tid}: modelled_quantity.kind must be one of {KINDS}")
        for which in ("transform", "challenger"):
            tr = mq.get(which) or {}
            if tr.get("name") not in TRANSFORMS:
                problems.append(f"{tid}: {which}.name must be one of {TRANSFORMS}")
            if tr.get("name") == "asinh" and not (tr.get("scale") or 0) > 0:
                problems.append(f"{tid}: asinh needs a positive scale")
        series = mq.get("series")
        if series is not None and panel_series is not None and series not in panel_series:
            problems.append(f"{tid}: series {series} is not a panel series")
        if mq.get("kind") == "spread":
            base = mq.get("base")
            if not base or not series:
                problems.append(f"{tid}: a spread needs series and base")
            elif units.get(series) != units.get(base) or units.get(series) is None:
                problems.append(f"{tid}: spread between {series} [{units.get(series)}] and {base} [{units.get(base)}] mixes units")
        out = t.get("output") or {}
        for k in ("now", "target"):
            if out.get(k) not in OUTPUTS:
                problems.append(f"{tid}: output.{k} must be one of {OUTPUTS}")
        if not (t.get("structure") or {}).get("multi_step"):
            problems.append(f"{tid}: structure.multi_step missing")
        for inp in t["inputs"]:
            if inp.get("series") not in avail:
                problems.append(f"{tid}: input {inp.get('series')} has no availability rule")
            if inp.get("status") not in STATUSES:
                problems.append(f"{tid}: input {inp.get('series')} status must be one of {STATUSES}")
            if not isinstance(inp.get("latest_day_offset"), int) or inp["latest_day_offset"] > 0:
                problems.append(f"{tid}: input {inp.get('series')} latest_day_offset must be an integer <= 0")
        if tid in INTRADAY_TARGETS and not any(INTRADAY_TARGETS[tid] in f for f in t.get("forbidden", [])):
            problems.append(f"{tid}: must forbid the previous period of the {INTRADAY_TARGETS[tid]} (finding F4)")
        if not t.get("forbidden"):
            problems.append(f"{tid}: forbidden list is empty")

    if problems:
        return problems
    tab = availability_table(form, catalogue)
    for _, r in tab[~tab["ok"]].drop_duplicates(["target", "series", "window"]).iterrows():
        problems.append(f"{r['target']}: {r['series']} for {r['input_day']} is published {r['published']}, "
                        f"after the as-of time {r['as_of']} (window {r['window'] or '-'})")
    late_bad = tab[tab["ok"] & ~tab["ok_late"] & ~tab["has_late_policy"]].drop_duplicates(["target", "series", "window"])
    for _, r in late_bad.iterrows():
        problems.append(f"{r['target']}: {r['series']} can be late (95 % by the late time) but the target has no late_policy for it")
    return problems
