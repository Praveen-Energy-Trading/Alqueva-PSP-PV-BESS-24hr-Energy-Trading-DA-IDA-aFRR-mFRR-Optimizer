"""
runner.py — run all quality checks on the clean zone, store the results and enforce the gate.

    run = run_quality(clean, raw, log)              # compute
    store_results(run)                               # dq.db, gate.json, Markdown report
    require_green()                                  # call before any model run; raises if red or stale
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import pandas as pd

from common_layer.data_layer.clean_store import CleanStore
from common_layer.data_layer.contracts import DATASETS, freshness_issues
from common_layer.data_layer.ingest import LoadLog
from common_layer.data_layer.raw_store import RawStore
from common_layer.data_quality import checks, report, scoring
from common_layer.data_quality.config import load_quality_config

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_QUALITY_DIR = os.path.join(_REPO, "runtime", "data", "quality")
DEFAULT_REPORT_DIR = os.path.join(_REPO, "runtime", "reports")


class DataQualityGateError(RuntimeError):
    """The data-quality gate is red, missing or out of date: no model run is allowed."""


@dataclass
class QualityRun:
    issues: pd.DataFrame
    daily: pd.DataFrame
    gate: scoring.GateResult
    generated_utc: str
    start: dt.date
    end: dt.date
    meta: Dict[str, object] = field(default_factory=dict)


def _last_status(log: LoadLog) -> Dict[tuple, str]:
    out: Dict[tuple, str] = {}
    for r in log.read():
        if r.get("status") not in ("skipped", None):
            out[(r["source"], r["delivery_date"])] = r["status"]
    return out


def run_quality(clean: CleanStore, raw: RawStore, log: LoadLog, *, start: Optional[dt.date] = None,
                end: Optional[dt.date] = None, today: Optional[dt.date] = None,
                cfg: Optional[dict] = None, datasets: Optional[Iterable[str]] = None) -> QualityRun:
    cfg = cfg or load_quality_config()
    today = today or dt.datetime.now(dt.timezone.utc).date()
    end = end or (today - dt.timedelta(days=1))
    status = _last_status(log)
    wanted = list(datasets) if datasets else list(DATASETS)
    frames = {ds: clean.read(ds, lineage=True) for ds in wanted}

    issue_parts, universe_rows, latest = [], [], {}
    for ds, df in frames.items():
        dcfg = cfg["datasets"][ds]
        eff_start = max(start or dt.date.fromisoformat(dcfg["expected_start"]),
                        dt.date.fromisoformat(dcfg["expected_start"]))
        srcs = dcfg["sources"]
        pairs = [(i + 1, s) for i, s in enumerate(srcs)] if ds == "ida_price" else [(0, srcs[0])]
        latest[ds] = clean.latest_day(ds)
        issue_parts.append(checks.check_gaps(ds, df, pairs, eff_start, end, status))
        if not df.empty:
            issue_parts += [checks.check_contract_flags(ds, df), checks.check_flat_runs(ds, df, cfg), checks.check_duplicate_day(ds, df, cfg),
                            checks.check_dst_copy(ds, df, cfg), checks.check_value_flags(ds, df, cfg),
                            checks.check_level_jump(ds, df, cfg)]
        have = set(zip(checks._with_session(df)["session"].astype(int), df["delivery_date"])) if not df.empty else set()
        day = eff_start
        while day <= end:
            for sess, _ in pairs:
                universe_rows.append({"dataset": ds, "session": sess, "delivery_date": day.isoformat(),
                                      "present": (sess, day.isoformat()) in have})
            day += dt.timedelta(days=1)
    issue_parts.append(checks.check_cross_source(frames, cfg))

    issues = pd.concat([p for p in issue_parts if len(p)], ignore_index=True) if any(len(p) for p in issue_parts) \
        else pd.DataFrame(columns=checks.COLUMNS)
    # the window for flag-based checks may be wider than the checked window: keep only days inside it
    universe = pd.DataFrame(universe_rows)
    daily = scoring.daily_scores(universe, issues, cfg)
    extra = int(cfg["gate"].get("max_stale_days_extra", 0))
    fresh = [f for f in freshness_issues(latest, today - dt.timedelta(days=extra))
             if any(f.message.startswith(d + ":") for d in wanted)]
    raw_problems = raw.verify_all()
    gate = scoring.evaluate_gate(daily, cfg, fresh, raw_problems)
    return QualityRun(issues, daily, gate, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      universe["delivery_date"].min() and dt.date.fromisoformat(universe["delivery_date"].min()),
                      end, {"today": today.isoformat(), "latest_days": {k: str(v) for k, v in latest.items()}})


def store_results(run: QualityRun, quality_dir: Optional[str] = None, report_dir: Optional[str] = None) -> Dict[str, str]:
    quality_dir = quality_dir or DEFAULT_QUALITY_DIR
    report_dir = report_dir or DEFAULT_REPORT_DIR
    os.makedirs(quality_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    db = os.path.join(quality_dir, "dq.db")
    with sqlite3.connect(db) as c:
        run.issues.to_sql("dq_issues", c, if_exists="replace", index=False)
        run.daily.assign(present=run.daily["present"].astype(int)).to_sql("dq_daily", c, if_exists="replace", index=False)
    gate_path = os.path.join(quality_dir, "gate.json")
    with open(gate_path, "w", encoding="utf-8") as fh:
        json.dump({"passed": run.gate.passed, "reasons": run.gate.reasons, "generated_utc": run.generated_utc,
                   "window": [str(run.start), str(run.end)], **run.meta}, fh, indent=2)
    md_path = os.path.join(report_dir, f"data_quality_{run.generated_utc[:10]}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report.render_markdown(run))
    return {"db": db, "gate": gate_path, "report": md_path}


def require_green(quality_dir: Optional[str] = None, max_age_hours: float = 26.0,
                  now: Optional[dt.datetime] = None) -> dict:
    """Call before any model run. Raises DataQualityGateError unless a green, recent gate result exists."""
    path = os.path.join(quality_dir or DEFAULT_QUALITY_DIR, "gate.json")
    if not os.path.exists(path):
        raise DataQualityGateError("no data-quality result found; run tools/run_data_quality.py first")
    with open(path, encoding="utf-8") as fh:
        g = json.load(fh)
    when = dt.datetime.strptime(g["generated_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    age_h = ((now or dt.datetime.now(dt.timezone.utc)) - when).total_seconds() / 3600.0
    if age_h > max_age_hours:
        raise DataQualityGateError(f"data-quality result is {age_h:.0f} h old (limit {max_age_hours:.0f} h); rerun it")
    if not g["passed"]:
        raise DataQualityGateError("data-quality gate is RED: " + "; ".join(g["reasons"][:5]))
    return g
