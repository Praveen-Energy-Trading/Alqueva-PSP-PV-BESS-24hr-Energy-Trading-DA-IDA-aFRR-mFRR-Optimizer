"""
scoring.py — daily data-quality score and the pass/fail gate.

Score per (dataset, session, delivery day): 100 minus the penalties of its warnings (config
`penalties`), floored at 0. A day with no rows scores 0. Info rows (spikes, negative prices,
prices at the limit) never reduce the score.

The gate is green only if, for every dataset (per session for the intraday auctions):
    * the share of missing days is within its limit,
    * the mean score of the present days and the share of days below 70 are within limits,
    * no feed is stale,
    * the raw-zone checksums verify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd


def daily_scores(universe: pd.DataFrame, issues: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """`universe`: one row per expected (dataset, session, delivery_date) with a boolean `present`."""
    pen = cfg["penalties"]
    warn = issues[issues["severity"] == "warning"] if len(issues) else issues
    pts = {}
    if len(warn):
        w = warn.assign(p=warn["check"].map(lambda c: float(pen.get(c, pen["default"]))))
        pts = w.groupby(["dataset", "session", "delivery_date"]).agg(p=("p", "sum"), n=("check", "count"))
    out = universe.copy()
    key = list(zip(out["dataset"], out["session"], out["delivery_date"]))
    penalty = [float(pts.loc[k, "p"]) if k in pts.index else 0.0 for k in key] if len(warn) else [0.0] * len(out)
    nwarn = [int(pts.loc[k, "n"]) if k in pts.index else 0 for k in key] if len(warn) else [0] * len(out)
    out["score"] = [max(0.0, 100.0 - p) if pres else 0.0 for p, pres in zip(penalty, out["present"])]
    out["n_warnings"] = nwarn
    return out


@dataclass
class GateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluate_gate(daily: pd.DataFrame, cfg: dict, freshness: Iterable, raw_problems: List[str]) -> GateResult:
    g = cfg["gate"]
    reasons: List[str] = []
    rows = []
    for (ds, sess), d in daily.groupby(["dataset", "session"]):
        limit = float(g["max_missing_day_share"].get(ds, g["max_missing_day_share"]["default"]))
        n, miss = len(d), int((~d["present"]).sum())
        present = d[d["present"]]
        mean_score = float(present["score"].mean()) if len(present) else 0.0
        below70 = float((present["score"] < 70).mean()) if len(present) else 1.0
        name = f"{ds}" + (f" session {sess}" if sess else "")
        ok_missing = (miss / n) <= limit if n else False
        ok_mean = mean_score >= float(g["min_mean_score"])
        ok_below = below70 <= float(g["max_share_days_below_70"])
        if not ok_missing:
            reasons.append(f"{name}: {miss}/{n} days missing ({miss / n:.1%} > {limit:.1%})")
        if not ok_mean:
            reasons.append(f"{name}: mean score {mean_score:.1f} < {g['min_mean_score']}")
        if not ok_below:
            reasons.append(f"{name}: {below70:.1%} of days below score 70 (> {g['max_share_days_below_70']:.1%})")
        rows.append({"dataset": ds, "session": sess, "days": n, "missing": miss,
                     "missing_share": miss / n if n else 1.0, "mean_score": mean_score,
                     "share_below_70": below70, "ok": ok_missing and ok_mean and ok_below})
    for f in freshness:
        reasons.append(f"stale feed: {f.message}")
    if g.get("require_raw_checksums", True) and raw_problems:
        reasons.append(f"raw zone: {len(raw_problems)} checksum problems, first: {raw_problems[0]}")
    return GateResult(not reasons, reasons, pd.DataFrame(rows))
