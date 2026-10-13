"""
run_data_pipeline.py — Phase 1: the daily data pipeline.

    python phase_1_data_pipeline/run_data_pipeline.py            # real run (network, ~1 request/s)
    python phase_1_data_pipeline/run_data_pipeline.py --demo     # offline, one real DST day, temp folders
                                                                 # (--workdir DIR keeps them in DIR)
    python phase_1_data_pipeline/run_data_pipeline.py --dry-run  # show what would be requested

One run does, in order:
    1. INGEST   for every source, ask for the days from (today - lookback) to the newest day the source
                can have; store new files in the raw zone (immutable, checksummed, versioned)
    2. BUILD    validate each day against its data contract and write the clean zone (SQLite)
    3. CHECK    quality checks, daily score, Markdown report
    4. CURATE   clean the data (remove data errors) and fill every missing hour by the written rule of its series, flagging
                what was filled (config/data_curation.yaml); the models read the curated data
    5. GATE     green / red; red means no model run (see config/data_pipeline.yaml: gate_mode). Missing days are repaired
                by step 4 and do not turn the gate red (config/data_quality.yaml: gate.mode repair)

Exit code: 0 gate green, 1 gate red, 2 configuration problem or another run in progress.
`run_data_phase` is the entry point used by run_production.py (phase key "data").
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import sys
import tempfile
import time
from typing import Dict, Optional

import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from common_layer.data_layer import fetchers, ingest  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.data_layer.raw_store import RawStore  # noqa: E402
from common_layer.data_quality import curation, runner  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.data_quality.config import load_quality_config  # noqa: E402
from common_layer.utilities.timezone_utils import MARKET_TZ  # noqa: E402

CONFIG_PATH = os.path.join(_REPO, "config", "data_pipeline.yaml")
LOCK_PATH = os.path.join(_REPO, "runtime", "data", "pipeline.lock")
STALE_LOCK_HOURS = 6.0


class PipelineBusy(RuntimeError):
    """Another data-pipeline run holds the lock."""


def load_pipeline_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    unknown = set(cfg["sources"]) - set(ingest.SOURCES)
    if unknown:
        raise ValueError(f"unknown sources in data_pipeline.yaml: {sorted(unknown)}")
    if cfg.get("gate_mode") not in ("warn", "block"):
        raise ValueError("gate_mode must be 'warn' or 'block'")
    return cfg


@contextlib.contextmanager
def pipeline_lock(path: str = LOCK_PATH):
    """One run at a time: two runs would write the same SQLite file and raw manifest."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 3600.0 > STALE_LOCK_HOURS:
        os.remove(path)                                   # left behind by a crashed run
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise PipelineBusy(f"another data-pipeline run is in progress ({path})") from None
    try:
        os.write(fd, f"pid={os.getpid()} started={dt.datetime.now(dt.timezone.utc).isoformat()}".encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def market_today() -> dt.date:
    return dt.datetime.now(MARKET_TZ).date()


def run_daily(cfg: Optional[dict] = None, *, today: Optional[dt.date] = None, raw: Optional[RawStore] = None,
              clean: Optional[CleanStore] = None, log: Optional[ingest.LoadLog] = None,
              fetcher: Optional[fetchers.Fetcher] = None, quality_dir: Optional[str] = None,
              report_dir: Optional[str] = None, quality_cfg: Optional[dict] = None,
              quality_end: Optional[dt.date] = None, dry_run: bool = False, lock: bool = True) -> dict:
    cfg = cfg or load_pipeline_config()
    today = today or market_today()
    raw, clean, log = raw or RawStore(), clean or CleanStore(), log or ingest.LoadLog()
    fetcher = fetcher or fetchers.Fetcher(min_interval_s=float(cfg["request_interval_s"]))
    start = today - dt.timedelta(days=int(cfg["lookback_days"]))
    recheck_from = today - dt.timedelta(days=int(cfg["recheck_days"]))

    plan = {src: (start, today + dt.timedelta(days=int(spec["last_day_offset"])))
            for src, spec in cfg["sources"].items()}
    if dry_run:
        return {"status": "DRY_RUN", "plan": {s: (str(a), str(b)) for s, (a, b) in plan.items()}}

    with (pipeline_lock() if lock else contextlib.nullcontext()):
        iid = ingest.new_ingest_id()
        counts: Dict[str, Dict[str, int]] = {}
        for src, (a, b) in plan.items():
            day = a
            while day <= b:
                out = ingest.ingest_day(src, day, raw=raw, clean=clean, fetcher=fetcher, log=log,
                                        ingest_id=iid, refresh=day >= recheck_from)
                counts.setdefault(src, {})
                counts[src][out["status"]] = counts[src].get(out["status"], 0) + 1
                day += dt.timedelta(days=1)

        since = cfg.get("quality_since")
        qrun = runner.run_quality(clean, raw, log, start=dt.date.fromisoformat(str(since)) if since else None,
                                  end=quality_end or (today - dt.timedelta(days=1)), today=today, cfg=quality_cfg)
        paths = runner.store_results(qrun, quality_dir, report_dir)
        cur_info: Dict[str, object] = {}
        try:                                                                  # curation never stops the pipeline: it is a repair step
            cur = curation.curate(pnl.build_panel(clean))
            cur_info = {"filled_hours": int(cur.summary["filled"].sum()), "errors_removed": int(cur.summary["errors_removed"].sum()),
                        "files": curation.write_outputs(cur, quality_dir or runner.DEFAULT_QUALITY_DIR)}
        except Exception as exc:                                              # an empty or tiny clean zone (the demo) has nothing to curate
            cur_info = {"skipped": str(exc)}

    return {"status": "GREEN" if qrun.gate.passed else "RED", "gate_passed": qrun.gate.passed, "repairs": qrun.gate.repairs, "curation": cur_info,
            "reasons": qrun.gate.reasons, "ingest": counts, "report": paths["report"], "gate_file": paths["gate"],
            "today": today.isoformat(), "ingest_id": iid}


def summarise(res: dict) -> str:
    loaded = sum(v.get("loaded", 0) for v in res["ingest"].values())
    other = {}
    for v in res["ingest"].values():
        for k, n in v.items():
            if k != "loaded":
                other[k] = other.get(k, 0) + n
    filled = res.get("curation", {}).get("filled_hours")
    return (f"gate {res['status']} | loaded {loaded} | " + ", ".join(f"{k} {n}" for k, n in sorted(other.items()))
            + (f" | curation filled {filled} hours" if filled is not None else ""))


def run_data_phase(date: str, app_cfg=None, use_synthetic: bool = False) -> dict:
    """Entry point for run_production.py (phase key 'data'). `date` is the delivery date being traded;
    the data pipeline always works relative to today's market date."""
    if use_synthetic:
        return {"status": "SKIPPED", "reason": "synthetic run: no live data pipeline"}
    cfg = load_pipeline_config()
    try:
        res = run_daily(cfg)
    except PipelineBusy as exc:
        return {"status": "DATA_WARN", "ref": str(exc)}
    print(f"  Data pipeline: {summarise(res)}")
    print(f"  Quality report: {res['report']}")
    for r in res["reasons"][:10]:
        print(f"   - {r}")
    if res["gate_passed"]:
        return {"status": "OK", "ref": summarise(res)}
    if cfg["gate_mode"] == "block":
        return {"status": "FAILED", "reason": "data-quality gate RED: " + "; ".join(res["reasons"][:3])}
    return {"status": "DATA_WARN", "ref": summarise(res)}


def run_demo(workdir: Optional[str] = None) -> dict:
    """Whole pipeline on the saved real DST day (2025-10-26), offline, in temporary folders."""
    from phase_1_data_pipeline.demo import DEMO_DAY, DemoHttp
    work = workdir or tempfile.mkdtemp(prefix="data_pipeline_demo_")
    cfg = {"lookback_days": 0, "recheck_days": 0, "request_interval_s": 0.0, "gate_mode": "warn",
           "quality_since": None, "sources": {s: {"last_day_offset": 0} for s in ingest.SOURCES}}
    qcfg = load_quality_config()
    for d in qcfg["datasets"].values():
        d["expected_start"] = DEMO_DAY.isoformat()
    http = DemoHttp()
    res = run_daily(
        cfg, today=DEMO_DAY, raw=RawStore(os.path.join(work, "raw")),
        clean=CleanStore(os.path.join(work, "clean", "market_clean.db")),
        log=ingest.LoadLog(os.path.join(work, "load_log.jsonl")),
        fetcher=fetchers.Fetcher(http, min_interval_s=0.0, backoff_s=0.0, sleep=lambda s: None),
        quality_dir=os.path.join(work, "quality"), report_dir=os.path.join(work, "reports"),
        quality_cfg=qcfg, quality_end=DEMO_DAY, lock=False)
    res["workdir"] = work
    res["requests"] = len(http.calls)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="offline run on one real saved day (temp folders)")
    ap.add_argument("--dry-run", action="store_true", help="print the request plan and stop")
    ap.add_argument("--workdir", help="with --demo: folder for the demo's zones and report (default: a temp folder)")
    a = ap.parse_args()
    try:
        if a.demo:
            res = run_demo(a.workdir)
            print(f"DEMO on the 25-hour day 2025-10-26: {res['requests']} requests served from saved files")
        else:
            res = run_daily(dry_run=a.dry_run)
            if a.dry_run:
                for src, (s, e) in res["plan"].items():
                    print(f"{src:<14} {s} .. {e}")
                return 0
    except PipelineBusy as exc:
        print(exc, file=sys.stderr)
        return 2
    print(summarise(res))
    for r in res["reasons"][:15]:
        print(" -", r)
    print("report:", res["report"])
    return 0 if res["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
