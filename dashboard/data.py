"""
data.py — single data-access layer for the dashboard.

Every page imports from here instead of touching runtime/reports,
runtime/logs, or runtime/audit directly. Centralising it means the
mtime-based caching strategy, date-selection logic, and file-path
conventions are defined exactly once, in one place, and stay consistent
across every page of the multipage app.

Nothing here runs the pipeline or writes anything — read-only, mirrors
whatever run_production.py has already produced on disk.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

REPO_ROOT   = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "runtime" / "reports"
LOGS_DIR    = REPO_ROOT / "runtime" / "logs"
AUDIT_DIR   = REPO_ROOT / "runtime" / "audit"
FIGURES_DIR = REPO_ROOT / "figures" / "output"
CONFIG_DIR  = REPO_ROOT / "config"

from common_layer.database import PositionStore, ReserveStore, DeliveryStore, ActivationStore, ComponentStore  # noqa: E402
from common_layer.utilities import date_utils as du  # noqa: E402
from common_layer.configuration.config_loader import load_config  # noqa: E402
from common_layer.optimisation_model.fcr_activation import simulate_fcr_response  # noqa: E402
from common_layer.optimisation_model.reserve_activation import simulate_ace_series  # noqa: E402
from common_layer.optimisation_model.agc_mechanism_demo import simulate_agc_dispatch  # noqa: E402


@st.cache_data
def load_plant_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "plant.yaml").read_text(encoding="utf-8"))


@st.cache_data
def load_market_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "market.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Date selection — which delivery date a fresh dashboard tab should land on
# ---------------------------------------------------------------------------

def list_report_dates() -> list[str]:
    # Not cached: cheap directory listing, and callers need it to notice a
    # brand-new daily_report_*.xlsx as soon as the pipeline finishes writing it.
    dates = []
    for f in REPORTS_DIR.glob("daily_report_*.xlsx"):
        m = re.match(r"daily_report_(\d{4}-\d{2}-\d{2})\.xlsx", f.name)
        if m:
            dates.append(m.group(1))
    return sorted(dates, reverse=True)


def list_log_dates() -> list[str]:
    return sorted(
        {m.group(1) for f in LOGS_DIR.glob("pipeline_*.log")
         if (m := re.match(r"pipeline_(\d{4}-\d{2}-\d{2})\.log", f.name))},
        reverse=True,
    )


def most_recent_activity_date(report_dates: list[str], log_dates: list[str]) -> list[str]:
    """Union of report/log dates, ordered by which was touched on disk most
    recently — not by calendar date. A delivery date actively running right
    now must outrank an old leftover report just because its date string
    happens to sort higher. This is what every page's date picker uses, so a
    fresh tab lands on whatever's actually live."""
    def _mtime(d: str) -> float:
        candidates = [
            REPORTS_DIR / f"daily_report_{d}.xlsx",
            LOGS_DIR / f"pipeline_{d}.log",
        ]
        return max((p.stat().st_mtime for p in candidates if p.exists()), default=0.0)

    all_dates = set(report_dates) | set(log_dates)
    return sorted(all_dates, key=_mtime, reverse=True)


def available_dates() -> tuple[list[str], list[str]]:
    """Returns (available_dates, report_ready_dates) — the second is a
    subset of the first, used to label dates that are still mid-run."""
    report_dates = list_report_dates()
    log_dates = list_log_dates()
    return most_recent_activity_date(report_dates, log_dates), report_dates


# ---------------------------------------------------------------------------
# Daily report (Excel)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_daily_report_cached(path_str: str, mtime: float) -> dict[str, pd.DataFrame]:
    # mtime is part of the cache key so a re-run pipeline (new mtime, same
    # filename) invalidates the cache automatically instead of showing stale data.
    path = Path(path_str)
    sheets = pd.read_excel(path, sheet_name=None, header=[1, 2])
    dispatch = sheets["Dispatch_Hourly"]
    dispatch.columns = [c[1] for c in dispatch.columns]

    isp   = pd.read_excel(path, sheet_name="ISP_Activation", header=1)
    gates = pd.read_excel(path, sheet_name="Gate_Decisions", header=1)
    kpis  = pd.read_excel(path, sheet_name="Summary_KPIs", header=1)
    return {"dispatch": dispatch, "isp": isp, "gates": gates, "kpis": kpis}


def load_daily_report(delivery_date: str) -> dict[str, pd.DataFrame] | None:
    path = REPORTS_DIR / f"daily_report_{delivery_date}.xlsx"
    if not path.exists():
        return None
    return _load_daily_report_cached(str(path), path.stat().st_mtime)


def kpi_value(kpis: pd.DataFrame, metric_substr: str) -> float | None:
    row = kpis[kpis["Metric"].astype(str).str.contains(metric_substr, case=False, na=False)]
    if row.empty:
        return None
    return float(row.iloc[0]["Value"])


# ---------------------------------------------------------------------------
# Backtest report (Excel) — multi-day aggregate, separate file family
# ---------------------------------------------------------------------------

def list_backtest_reports() -> list[Path]:
    return sorted(REPORTS_DIR.glob("backtest_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)


def parse_label_value_sheet(raw: pd.DataFrame) -> list[tuple[str, object, bool]]:
    """Summary/Risk sheets are 2-column (label, value) with a title row,
    blank separator rows, and '--- Section Name ---' group headers — not a
    normal table, so pandas' default header inference doesn't apply. Returns
    a flat list of (label, value, is_section_header) tuples, blank/title
    rows already dropped."""
    out: list[tuple[str, object, bool]] = []
    for _, row in raw.iterrows():
        label, value = row.iloc[0], row.iloc[1] if len(row) > 1 else None
        if pd.isna(label):
            continue
        label = str(label).strip()
        if pd.isna(value):
            out.append((label, None, True))  # title or "--- Section ---" row
        else:
            out.append((label, value, False))
    return out


@st.cache_data
def _load_backtest_report_cached(path_str: str, mtime: float) -> dict:
    path = Path(path_str)
    all_sheets = pd.read_excel(path, sheet_name=None, header=0)
    result: dict = dict(all_sheets)
    # Summary/Risk are label-value sheets, not tables — re-read without
    # header inference and parse into a flat structure the pages can render
    # directly. Risk is optional (only written when backtest risk metrics
    # were computed), so it may be absent entirely.
    for sheet in ("Summary", "Risk"):
        if sheet in result:
            raw = pd.read_excel(path, sheet_name=sheet, header=None)
            result[sheet] = parse_label_value_sheet(raw)
    return result


def load_backtest_report(path: Path) -> dict:
    return _load_backtest_report_cached(str(path), path.stat().st_mtime)


# ---------------------------------------------------------------------------
# Run status (JSON) — the health banner's data source
# ---------------------------------------------------------------------------

@st.cache_data
def _load_run_status_cached(path_str: str, mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_run_status(delivery_date: str) -> dict | None:
    path = LOGS_DIR / f"run_status_{delivery_date}.json"
    if not path.exists():
        return None
    return _load_run_status_cached(str(path), path.stat().st_mtime)


def run_phase_state(delivery_date: str) -> str:
    """One of 'running', 'idle_running', 'stopped', 'done', 'none'.

    run_status_<date>.json gets a RUNNING sentinel (with pid) written at
    the START of a run and is overwritten with the full result at the end
    — see run_production.py. This lets the dashboard tell "a run is
    genuinely in progress right now" apart from "no new record, so we're
    still looking at whatever the PREVIOUS run for this date left behind".
    'idle_running' means the sentinel says RUNNING, the log hasn't moved in
    the freshness window, AND the process is still actually alive —
    consistent with sitting on an interactive Approve/Reject or ENTER
    prompt. 'stopped' means the sentinel says RUNNING but the recorded pid
    is no longer alive — a Ctrl+C (or crash) killed it before it could
    write the final DONE status, and without the pid check this would
    otherwise show as "paused, waiting on you" forever, even though
    nothing is listening.
    """
    status = load_run_status(delivery_date)
    if status is None:
        return "none"
    if status.get("status") == "RUNNING":
        if is_pipeline_active(delivery_date):
            return "running"
        return "idle_running" if _is_pid_alive(status.get("pid")) else "stopped"
    return "done"


def no_report_message(delivery_date: str) -> str:
    """A daily Excel report only appears once the analytics phase runs — but
    a run can also finish (with a failure) before ever reaching it. Pages
    that gate on report_ready use this so they don't tell the user
    "still running" about a run that's actually already finished and failed,
    or vice versa show a stale FAILED banner while a new run is genuinely
    still in progress (including paused on interactive input)."""
    state = run_phase_state(delivery_date)
    if state == "running":
        return (f"🟢 Pipeline is actively running for **{delivery_date}** right "
                f"now — no report yet. Check Run & Monitor for live progress.")
    if state == "idle_running":
        status = load_run_status(delivery_date)
        if status and status.get("mode") == "trader":
            return (f"🟡 A run is in progress for **{delivery_date}** but nothing's "
                    f"been written to the log in a while — it's most likely paused "
                    f"on an Approve/Reject or ENTER prompt waiting on you in trader "
                    f"mode. Check the terminal it was started from, or Console Log.")
        return (f"🟡 A run is in progress for **{delivery_date}** but nothing's "
                f"been written to the log in a while — most likely still computing "
                f"(e.g. a slow model fit) rather than stuck. Check Console Log for "
                f"the last line printed.")
    if state == "stopped":
        return (f"⚫ A run for **{delivery_date}** was started but the process is no "
                f"longer running — most likely Ctrl+C or a crash while it was paused "
                f"waiting for input. No report yet; start a fresh run when ready.")
    status = load_run_status(delivery_date)
    if state == "none" or status is None:
        return (f"🟡 Pipeline hasn't been run yet for **{delivery_date}** — no "
                f"report and no run record.")
    n_fail = sum(1 for r in status["results"] if r["status"] == "FAIL")
    if n_fail:
        failed = [r["key"] for r in status["results"] if r["status"] == "FAIL"]
        return (f"🔴 Pipeline for **{delivery_date}** finished but FAILED "
                f"(phase(s): {', '.join(failed)}) before reaching the analytics "
                f"phase that writes the report — see Run & Monitor for details.")
    return (f"🟡 Pipeline for **{delivery_date}** finished but no report was "
            f"written — check Run & Monitor / Console Log for what happened.")


# ---------------------------------------------------------------------------
# Audit trail (JSONL) — decision rationale source
# ---------------------------------------------------------------------------

@st.cache_data
def _load_all_audit_events_cached(dir_listing_key: str) -> list[dict]:
    # The audit file is named by EXECUTION date (when the phase ran), not by
    # delivery date — e.g. a DA run executed today logs delivery_date=tomorrow
    # inside the record, but the *filename* is today's date. IDA3/XBID phases
    # even run ON the delivery date itself, landing in yet another file. So
    # there's no way to know in advance which file(s) hold a given delivery
    # date's events — we read every audit file and filter by the record's own
    # "delivery_date" field instead of trusting the filename.
    events: list[dict] = []
    for path in sorted(AUDIT_DIR.glob("audit_*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def load_audit_events(delivery_date: str, event_prefix: str | None = None) -> list[dict]:
    # Cache key = sorted (filename, mtime) pairs, so any audit file changing
    # (new event appended, or a new day's file appearing) invalidates it.
    dir_listing_key = "|".join(f"{p.name}:{p.stat().st_mtime}" for p in sorted(AUDIT_DIR.glob("audit_*.jsonl")))
    events = _load_all_audit_events_cached(dir_listing_key)
    events = [e for e in events if e.get("delivery_date") == delivery_date]
    if event_prefix:
        events = [e for e in events if str(e.get("event", "")).startswith(event_prefix)]
    return sorted(events, key=lambda e: e.get("timestamp_cet", ""))


# ---------------------------------------------------------------------------
# Console log
# ---------------------------------------------------------------------------

@st.cache_data
def _load_log_cached(path_str: str, mtime: float) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="replace")


def load_log(delivery_date: str) -> str:
    path = LOGS_DIR / f"pipeline_{delivery_date}.log"
    if not path.exists():
        return ""
    return _load_log_cached(str(path), path.stat().st_mtime)


ACTIVE_WINDOW_S = 20.0


def is_pipeline_active(delivery_date: str) -> bool:
    """True if pipeline_<date>.log was written to within the last
    ACTIVE_WINDOW_S seconds — source-agnostic (VS Code, terminal, Spyder,
    or the in-app runner all write the same file via the same _Tee, so this
    catches a live run regardless of where it was started). Not process
    detection — just "is something actively appending to this file right
    now" — which is exactly what "mirror the terminal" needs: the dashboard
    should feel live whenever the log is actually moving, not only when it
    happens to know about its own subprocess.
    """
    path = LOGS_DIR / f"pipeline_{delivery_date}.log"
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ACTIVE_WINDOW_S


def _is_pid_alive(pid: int) -> bool:
    """Cross-platform "is this process still running". Used to tell a
    genuinely-paused run (waiting minutes on an Approve/Reject or ENTER
    prompt — normal, log freshness alone can't distinguish this from dead)
    apart from one killed by Ctrl+C, which otherwise leaves run_status.json
    stuck at RUNNING forever with no way to know the process is gone."""
    if not pid:
        return True  # older run_status.json (pre-pid) — can't check, assume alive
    try:
        if os.name == "nt":
            import subprocess
            # CREATE_NO_WINDOW -- without it, every subprocess.run() of a
            # console tool from a windowed/GUI-launched Python (which
            # Streamlit is, when started via a desktop shortcut or as a
            # background process) briefly flashes a new console window.
            # This function runs on every dashboard rerun while any date is
            # RUNNING (every 2-5s during live monitoring), so that flash
            # became a rapidly flickering Anaconda Prompt window.
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return str(pid) in out.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return True  # can't determine — don't falsely report a live run as dead


# ---------------------------------------------------------------------------
# Live gate ticket — the most recent gate decision anywhere in the pipeline
# (DA/aFRR/mFRR/IDA1-3/XBID), with its hourly numbers straight from
# PositionStore/ReserveStore (SQLite, written the moment a gate submits —
# not the Excel report, which only exists once analytics has run). This is
# the one Overview element that's genuinely live mid-pipeline: it updates
# gate-by-gate as the run progresses, well before any report exists.
# ---------------------------------------------------------------------------

_GATE_STORE_KEY = {
    "DA": "DA", "IDA1": "IDA1", "IDA2": "IDA2", "IDA3": "IDA3", "XBID": "XBID",
    "AFRR": "aFRR", "MFRR": "mFRR",
}
_GATE_LABEL = {
    "DA": "Day-Ahead (DA)", "IDA1": "IDA1", "IDA2": "IDA2", "IDA3": "IDA3",
    "XBID": "XBID", "AFRR": "aFRR capacity", "MFRR": "mFRR capacity",
}
# Folder/module naming convention (phase_3a_afrr, phase_3b_mfrr) -- distinct
# from run_production.py's own _PHASES table (which numbers these 2 and 3).
# Shown on the reserve-capacity cards because that's the scheme the user
# asked these cards to be labeled with.
_RESERVE_PHASE_LABEL = {"AFRR": "Phase 3A", "MFRR": "Phase 3B"}
# event suffix -> (pill text, status class). Only terminal decision events —
# *_START/*_PASSED/*_POSITION_SAVED are progress markers, not decisions.
_DECISION_SUFFIX = {
    "SUBMITTED":       ("Submitted", "good"),
    "ORDERS_PLACED":   ("Submitted", "good"),
    "REJECTED":        ("Rejected", "neutral"),
    "NO_CHANGE":       ("Held", "neutral"),
    "NO_ORDER":        ("Held", "neutral"),
    "SKIPPED":         ("Skipped", "neutral"),
    "RISK_BLOCKED":    ("Blocked", "critical"),
    "BIDCHECK_FAILED": ("Blocked", "critical"),
    "SCHEMA_FAILED":   ("Failed", "critical"),
    "SOLVE_FAILED":    ("Failed", "critical"),
    "CHECK_FAILED":    ("Blocked", "critical"),
}
# Different gates log revenue under different names, and DA specifically
# logs TWO figures together (energy_revenue_eur AND objective_eur, both on
# *_SOLVED, not *_SUBMITTED) — collect every field present, not just the
# first, so DA's ticket shows both instead of silently dropping one.
_REVENUE_FIELDS = ["energy_revenue_eur", "capacity_revenue_eur", "objective_eur", "improvement_eur"]
_REVENUE_LABELS = {
    "energy_revenue_eur": "Energy revenue",
    "capacity_revenue_eur": "Capacity revenue",
    "objective_eur": "Objective (net)",
    "improvement_eur": "Improvement vs committed",
}


def _collect_decisions(delivery_date: str) -> tuple[list, list]:
    """Returns (events, decisions) — decisions is [(gate, suffix, event), ...]
    sorted oldest-first, for every terminal decision event found."""
    events = load_audit_events(delivery_date)
    decisions = []
    for e in events:
        ev = str(e.get("event", ""))
        for gate in _GATE_STORE_KEY:
            prefix = gate + "_"
            if ev.startswith(prefix) and ev[len(prefix):] in _DECISION_SUFFIX:
                decisions.append((gate, ev[len(prefix):], e))
                break
    decisions.sort(key=lambda t: t[2].get("timestamp_cet", ""))
    return events, decisions


def _da_anchor(delivery_date: str) -> str | None:
    """This run's latest DA decision timestamp, or None if DA hasn't decided
    yet. DA always runs first in any single pipeline execution, so any
    other phase's event timestamped before this cannot belong to the
    current run — see all_gate_tickets' docstring for the full rationale.
    Shared by every staleness check in this module (gate tickets, XBID
    windows, and the phase 4A/4B/4C delivery cards)."""
    _, decisions = _collect_decisions(delivery_date)
    da_timestamps = [e.get("timestamp_cet", "") for g, s, e in decisions if g == "DA"]
    return max(da_timestamps) if da_timestamps else None


def _collect_xbid_windows(events: list, anchor: str | None) -> list[dict] | None:
    """XBID isn't a single decision like DA/IDA — it re-checks several
    times a day (one audit event per check-window: W1, W2, ...), each with
    its own Held/Submitted call. Keeping only the chronologically latest
    event (as all_gate_tickets does for every other gate) would silently
    hide every earlier window's decision. Returns one entry per window,
    latest event per window id, anchor-filtered the same way as the main
    gate list (drops windows from an earlier, since-superseded run) —
    or None if there's nothing to show (non-XBID gate, or no window field
    on any matching event, e.g. an older run before this field existed)."""
    prefix = "XBID_"
    latest_by_window: dict[str, dict] = {}
    for e in events:
        ev = str(e.get("event", ""))
        if not ev.startswith(prefix):
            continue
        suffix = ev[len(prefix):]
        if suffix not in _DECISION_SUFFIX:
            continue
        ts = e.get("timestamp_cet", "")
        if anchor is not None and ts < anchor:
            continue
        w = e.get("window")
        if not w:
            continue
        if w not in latest_by_window or ts > latest_by_window[w].get("timestamp_cet", ""):
            latest_by_window[w] = e
    if not latest_by_window:
        return None

    def _win_num(w: str) -> int:
        m = re.match(r"W(\d+)", w)
        return int(m.group(1)) if m else 0

    windows = []
    for w in sorted(latest_by_window, key=_win_num):
        e = latest_by_window[w]
        label, status_class = _DECISION_SUFFIX[str(e.get("event", ""))[len(prefix):]]
        windows.append({
            "window": w, "decision": label, "status_class": status_class,
            "timestamp": e.get("timestamp_cet", "")[:19], "ref": e.get("ref"),
            "rebid_info": {
                "improvement_eur": e.get("improvement_eur"),
                "threshold_eur": e.get("dynamic_threshold_eur"),
                "tradable_hours": sorted(e.get("tradable_hours") or []),
                "n_total_hours": e.get("n_total_hours") or 24,
                "traded_hours": sorted(e.get("traded_hours") or []),
            },
        })
    return windows


def _build_ticket(gate: str, suffix: str, event: dict, events: list, delivery_date: str,
                   anchor: str | None = None) -> dict | None:
    """Builds one gate's ticket dict, or None if its backing store data has
    gone missing (an orphaned decision from an earlier, since-cleared run —
    see latest_gate_ticket's docstring)."""
    label, status_class = _DECISION_SUFFIX[suffix]

    found: dict = {f: event[f] for f in _REVENUE_FIELDS if event.get(f) is not None}
    if not found:
        # DA_SUBMITTED itself carries no revenue field — only DA_SOLVED
        # does (a non-decision progress event, so it's not in `decisions`),
        # and DA_SOLVED carries BOTH energy_revenue_eur and objective_eur
        # together. Search all of this gate's events, newest first, and
        # take every revenue field the first matching event has — not just
        # one — so DA shows both figures.
        gate_prefix = gate + "_"
        same_gate_all = [e for e in events if str(e.get("event", "")).startswith(gate_prefix)]
        for e in reversed(same_gate_all):
            found = {f: e[f] for f in _REVENUE_FIELDS if e.get(f) is not None}
            if found:
                break
    revenue_items = [(_REVENUE_LABELS[f], found[f]) for f in _REVENUE_FIELDS if f in found]

    store_key = _GATE_STORE_KEY[gate]
    hourly: list[dict] = []
    is_reserve = gate in ("AFRR", "MFRR")
    if is_reserve:
        by_hour = ReserveStore().load_reserve(delivery_date, store_key)
        for h in sorted(by_hour):
            v = by_hour[h]
            hourly.append({"hour": h, "up_mw": v["up_mw"], "dn_mw": v["dn_mw"],
                            "cap_up_eur_mw": v["cap_up_eur_mw"], "cap_dn_eur_mw": v["cap_dn_eur_mw"]})
    else:
        by_hour = PositionStore().load_position(delivery_date, store_key)
        if not by_hour and suffix in ("NO_CHANGE", "NO_ORDER"):
            # A held gate never wrote its own row (only a re-bid does) —
            # the committed position didn't change, so show what's
            # actually committed as of this gate instead of an empty chart.
            net = PositionStore().committed_position(delivery_date, as_of_gate=store_key)
            by_hour = {h: {"volume_mwh": v, "price_eur_mwh": None} for h, v in net.items()}
        for h in sorted(by_hour):
            v = by_hour[h]
            hourly.append({"hour": h, "volume_mwh": v["volume_mwh"], "price_eur_mwh": v["price_eur_mwh"]})

    if not hourly:
        # Orphaned decision from an earlier, since-cleared run. Applies to
        # EVERY decision type, not just SUBMITTED — a NO_CHANGE/Held ticket
        # falls back to committed_position() above, and that too goes empty
        # once the underlying DA position has been cleared by a fresh run,
        # so a held gate can be just as orphaned as a submitted one. Within
        # one genuine run, DA always precedes IDA/XBID, so a legitimately
        # current NO_CHANGE ticket can never actually have empty hourly data.
        return None

    # IDA1-3/XBID re-optimisation gates: the meaningful "chart" isn't the
    # full 24h position (which usually looks identical to DA's, since most
    # hours are frozen/unchanged) — it's whether the improvement cleared
    # the dynamic re-bid threshold, and which hours were actually tradable
    # vs traded. Only these gates log these fields (see
    # ida_reoptimiser.py/xbid_optimiser.py).
    is_rebid_gate = gate in ("IDA1", "IDA2", "IDA3", "XBID")
    rebid_info = None
    if is_rebid_gate:
        tradable_hours = event.get("tradable_hours") or []
        rebid_info = {
            "improvement_eur": event.get("improvement_eur"),
            "threshold_eur": event.get("dynamic_threshold_eur"),
            "tradable_hours": sorted(tradable_hours),
            "n_total_hours": event.get("n_total_hours") or 24,
            "traded_hours": sorted(event.get("traded_hours") or []),
        }

    xbid_windows = _collect_xbid_windows(events, anchor) if gate == "XBID" else None

    return {
        "gate": gate, "label": _GATE_LABEL[gate], "decision": label,
        "status_class": status_class, "timestamp": event.get("timestamp_cet", "")[:19],
        "ref": event.get("ref"), "revenue_items": revenue_items, "hourly": hourly, "is_reserve": is_reserve,
        "is_rebid_gate": is_rebid_gate, "rebid_info": rebid_info, "xbid_windows": xbid_windows,
        "phase_label": _RESERVE_PHASE_LABEL.get(gate),
    }


def all_gate_tickets(delivery_date: str) -> list[dict]:
    """One ticket per gate that has reached a decision so far today, oldest
    first — DA, then aFRR, then whatever's next — so a gate's card stays
    visible on Overview after a later gate becomes 'latest', instead of
    only ever showing the single most recent decision. Only the most
    recent decision per gate is kept (a gate deciding twice in one date's
    audit trail across restarts only shows its latest)."""
    events, decisions = _collect_decisions(delivery_date)

    # DA always runs first in any single execution of the pipeline, so any
    # OTHER gate's decision timestamped before the current run's DA
    # decision cannot belong to this run — it's a leftover from an earlier,
    # separate attempt. This matters even though _build_ticket already
    # drops decisions with empty backing data: a stale IDA*_NO_CHANGE
    # event's committed_position() fallback walks GATE_ORDER starting from
    # DA, so once DA has fresh data (as it always does by the time IDA1
    # could plausibly run) that fallback comes back non-empty anyway —
    # borrowing the CURRENT run's DA numbers under an OLD IDA1 decision
    # label, masking exactly the staleness the emptiness check was meant
    # to catch. A stale IDA1 card showing "Held" from hours ago, with
    # today's fresh DA committed position silently underneath it, is
    # worse than showing nothing.
    anchor = _da_anchor(delivery_date)

    seen_gates: set[str] = set()
    tickets: list[dict] = []
    for gate, suffix, event in reversed(decisions):
        if gate in seen_gates:
            continue
        if gate != "DA" and anchor is not None and event.get("timestamp_cet", "") < anchor:
            continue  # older than this run's DA decision -- from an earlier run
        ticket = _build_ticket(gate, suffix, event, events, delivery_date, anchor=anchor)
        if ticket is None:
            continue
        seen_gates.add(gate)
        tickets.append(ticket)
    tickets.sort(key=lambda t: t["timestamp"])
    return tickets


# ---------------------------------------------------------------------------
# Delivery cards — phases 4A (RT dispatch), 4B (aFRR activation), 4C (mFRR
# activation). Not "decisions" like the gates above (no Submitted/Held
# choice — delivery always happens once a committed position exists), so
# they don't fit _DECISION_SUFFIX/_build_ticket's vocabulary. Read directly
# from DeliveryStore/ActivationStore (the per-ISP tables Phase 4 writes)
# plus each phase's own audit summary event.
# ---------------------------------------------------------------------------

def load_rt_delivery(delivery_date: str) -> dict | None:
    """Phase 4A: RT_DELIVERED audit summary + the per-ISP scheduled/actual
    trace from DeliveryStore. None if the phase hasn't run yet for this
    date, or if the only matching event predates this run's DA decision
    (a leftover from an earlier, since-cleared attempt — see _da_anchor)."""
    events = load_audit_events(delivery_date, "RT_")
    done = next((e for e in reversed(events) if e.get("event") == "RT_DELIVERED"), None)
    if done is None:
        return None
    anchor = _da_anchor(delivery_date)
    if anchor is not None and done.get("timestamp_cet", "") < anchor:
        return None
    rows = DeliveryStore().load(delivery_date)
    if not rows:
        return None
    max_dev = max(abs(r["actual_mw"] - r["scheduled_mw"]) for r in rows)
    return {
        "n_isp": done.get("n_isp", len(rows)),
        "total_abs_deviation_mwh": done.get("total_abs_deviation_mwh"),
        "max_deviation_mw": max_dev,
        "timestamp": done.get("timestamp_cet", "")[:19],
        "rows": sorted(rows, key=lambda r: r["isp"]),
    }


def load_activation_summary(delivery_date: str, product: str) -> dict | None:
    """Phase 4B (aFRR) / 4C (mFRR): {PRODUCT}_ACT_DONE audit summary + per-ISP
    activated energy/prices from ActivationStore, with revenue computed
    directly from up/dn MW * price * eff_isp_h (the same math reserve
    settlement uses) — available the moment activation runs, well before
    settlement (phase 6B) aggregates it. None if not activated yet."""
    prefix = product.upper() + "_ACT_"
    events = load_audit_events(delivery_date, prefix)
    done = next((e for e in reversed(events) if e.get("event") == prefix + "DONE"), None)
    if done is None or done.get("n_isp", 0) == 0:
        return None
    anchor = _da_anchor(delivery_date)
    if anchor is not None and done.get("timestamp_cet", "") < anchor:
        return None
    rows = ActivationStore().load(delivery_date, product)
    if not rows:
        # Same orphan case as load_rt_delivery/gate tickets: the audit
        # event survived (append-only log) but its ActivationStore rows
        # are gone -- cleared by a fresh run that hasn't reached this
        # phase yet. Reading n_isp/up_mwh/dn_mwh straight from the old
        # event here (as an earlier version of this function did) would
        # silently show that old run's numbers under today's date with
        # a correct-looking €0 revenue (computed from the now-empty
        # rows), which is worse than showing nothing.
        return None
    revenue = sum(
        r["up_mw"] * r["up_price_eur_mwh"] * r["eff_isp_h"]
        + r["dn_mw"] * r["dn_price_eur_mwh"] * r["eff_isp_h"]
        for r in rows
    )
    up_mwh = sum(r["up_mw"] * r["eff_isp_h"] for r in rows)
    dn_mwh = sum(r["dn_mw"] * r["eff_isp_h"] for r in rows)

    # Full-day (96-ISP) synthetic Area Control Error trace -- the signal this
    # product's activation actually responds to (see reserve_activation.py's
    # simulate_ace_series docstring). Regenerated fresh here rather than
    # persisted: it's a pure, deterministic function of (product, date), same
    # pattern load_fcr_activation uses for FCR's frequency trace, and it's the
    # SAME function simulate_and_log_activation calls internally, so this is
    # guaranteed to be the exact signal that drove the real activation rows
    # above, not a separately-approximated one.
    ace_by_isp = simulate_ace_series(product, delivery_date)
    resp_by_isp = {r["isp"]: (r["up_mw"] - r["dn_mw"]) for r in rows}
    n_isp_total = 96
    ace_mw = [round(ace_by_isp.get(isp, 0.0), 1) for isp in range(1, n_isp_total + 1)]
    response_mw = [round(resp_by_isp.get(isp, 0.0), 2) for isp in range(1, n_isp_total + 1)]

    # Gauge/chart reference scale: the largest capacity actually offered this
    # product on this date, across both directions -- the natural "100%" for
    # a response gauge, mirroring how the FCR card scales off its reserved
    # headroom.
    offers = ReserveStore().load_reserve(delivery_date, product)
    max_offer_mw = max(
        (max(o.get("up_mw", 0.0), o.get("dn_mw", 0.0)) for o in offers.values()),
        default=0.0,
    )

    return {
        "n_isp": len(rows),
        "up_mwh": up_mwh,
        "dn_mwh": dn_mwh,
        "revenue_eur": revenue,
        "timestamp": done.get("timestamp_cet", "")[:19],
        "rows": rows,
        "ace_mw": ace_mw,
        "response_mw": response_mw,
        "max_offer_mw": max_offer_mw,
    }




_MIN_REAL_ISP_COUNT = 90  # 96 nominal, 92/100 on DST days -- below this, the
# stored components are still hourly-resolution (pre state-continuity-fix
# runs), and repeating them as "ISP data" would fabricate resolution that
# doesn't exist. This widget only renders for dates genuinely re-solved at
# real ISP granularity.


@st.cache_data(ttl=5)
def load_isp_dispatch(delivery_date: str) -> dict | None:
    """Real per-asset dispatch at true ISP (15-min) resolution -- PV, BESS
    discharge, and PSP turbine, straight from ComponentStore's
    pv_schedule/bess_schedule/psp_schedule. Since the DA/IDA/XBID gates now
    solve natively at 96-ISP resolution (see run_da.py), these are 96 real,
    independently-solved values per day, not a repeated hourly number.
    Returns None for dates that predate that fix (still hourly-resolution
    components) -- never fabricates ISP-level detail that isn't real."""
    comp = ComponentStore().load(delivery_date)
    if comp is None:
        return None
    psp_sched = comp.get("psp_schedule") or {}
    bess_sched = comp.get("bess_schedule") or {}
    pv_sched = comp.get("pv_schedule") or {}
    if len(psp_sched) < _MIN_REAL_ISP_COUNT:
        return None

    isps = sorted(psp_sched)
    day = du.parse_date(delivery_date)

    pv_mw = [pv_sched.get(isp, {}).get("used_mw", 0.0) for isp in isps]
    bess_dis_mw = [bess_sched.get(isp, {}).get("discharge_mw", 0.0) for isp in isps]
    psp_turb_mw = [psp_sched[isp]["turbine_mw"] for isp in isps]
    psp_pump_mw = [psp_sched[isp]["pump_mw"] for isp in isps]
    hours = [du.isp_to_hour(isp, day) for isp in isps]

    return {
        "isps": isps,
        "hours": hours,
        "pv_mw": pv_mw,
        "bess_dis_mw": bess_dis_mw,
        "psp_turb_mw": psp_turb_mw,
        "psp_pump_mw": psp_pump_mw,
        "delivery_date": delivery_date,
    }


@st.cache_data(ttl=5)
def load_afrr_dispatch(delivery_date: str, product: str = "aFRR") -> dict | None:
    """Real per-ISP reserved capacity (ReserveStore) vs actually activated MW
    (ActivationStore), split by resource -- BESS (fast responder, up to its
    power rating) vs PSP (ramp covers the rest). PV is never a contributor:
    the activation model has no PV term in its call-sizing logic (weather-
    dependent generation isn't relied on for regulation), so it's correctly
    absent here rather than omitted. None if this product hasn't activated
    yet (no ActivationStore rows) or has no offered capacity for the date."""
    rows = ActivationStore().load(delivery_date, product)
    offers = ReserveStore().load_reserve(delivery_date, product)
    if not rows or not offers:
        return None

    isps = sorted(offers)
    act_by_isp = {r["isp"]: r for r in rows}

    reserved_up_mw = [offers[isp].get("up_mw", 0.0) for isp in isps]
    reserved_dn_mw = [offers[isp].get("dn_mw", 0.0) for isp in isps]
    up_mw = [act_by_isp.get(isp, {}).get("up_mw", 0.0) for isp in isps]
    dn_mw = [act_by_isp.get(isp, {}).get("dn_mw", 0.0) for isp in isps]
    bess_up_mw = [act_by_isp.get(isp, {}).get("bess_up_mw", 0.0) for isp in isps]
    bess_dn_mw = [act_by_isp.get(isp, {}).get("bess_dn_mw", 0.0) for isp in isps]
    psp_up_mw = [max(0.0, u - b) for u, b in zip(up_mw, bess_up_mw)]
    psp_dn_mw = [max(0.0, d - b) for d, b in zip(dn_mw, bess_dn_mw)]

    # Same signal load_activation_summary's ACE panel uses (see that
    # function's docstring) -- a pure, deterministic function of (product,
    # date), so pulling it in here too is the exact signal that drove this
    # dispatch's real up_mw/dn_mw rows, not a separately-approximated one.
    ace_by_isp = simulate_ace_series(product, delivery_date)
    ace_mw = [round(ace_by_isp.get(isp, 0.0), 1) for isp in isps]

    return {
        "isps": isps,
        "product": product,
        "reserved_up_mw": reserved_up_mw,
        "reserved_dn_mw": reserved_dn_mw,
        "bess_up_mw": bess_up_mw,
        "psp_up_mw": psp_up_mw,
        "bess_dn_mw": bess_dn_mw,
        "psp_dn_mw": psp_dn_mw,
        "ace_mw": ace_mw,
        "delivery_date": delivery_date,
    }


@st.cache_data(ttl=5)
def load_reservoir_trajectory(delivery_date: str) -> dict | None:
    """Real solved MILP reservoir trajectory (upper=Alqueva, lower=Pedrogao)
    -- ComponentStore.reservoir_trajectory is written straight from the
    solved model's v_up/v_low/spill/head variables (see
    core_milp_solver.py::extract_results), not a display-side approximation.
    None if this date's DA gate hasn't solved (no ComponentStore record)."""
    comp = ComponentStore().load(delivery_date)
    if comp is None:
        return None
    traj = comp.get("reservoir_trajectory") or {}
    if not traj:
        return None
    hours = sorted(traj)
    cfg = load_config()
    r = cfg.plant.reservoir
    init = comp.get("initial_state") or {}
    upper_initial = init.get("upper_reservoir_hm3", r.upper_initial_hm3)
    lower_initial = init.get("lower_reservoir_hm3", r.lower_initial_hm3)
    upper_hm3 = [traj[h]["upper_hm3"] for h in hours]
    lower_hm3 = [traj[h]["lower_hm3"] for h in hours]
    return {
        "hours": hours,
        "upper_hm3": upper_hm3,
        "lower_hm3": lower_hm3,
        "spill_m3h": [traj[h]["spill_m3h"] for h in hours],
        "head_m": [traj[h]["head_m"] for h in hours],
        "upper_min_hm3": r.upper_min_hm3,
        "upper_usable_hm3": r.upper_usable_hm3,
        "upper_capacity_hm3": r.upper_capacity_hm3,
        "lower_min_hm3": r.lower_min_hm3,
        "lower_capacity_hm3": r.lower_capacity_hm3,
        "upper_initial_hm3": upper_initial,
        "lower_initial_hm3": lower_initial,
        "terminal_ok": upper_hm3[-1] >= upper_initial - 1e-6,
    }


@st.cache_data(ttl=5)
def load_pv_routing(delivery_date: str) -> dict | None:
    """Real per-hour PV allocation from the solved MILP -- ComponentStore's
    pv_schedule is written straight from pv_used/pv_to_bess/pv_curt/pv_av
    (core_milp_solver.py), the exact three-way split the real pv_balance
    constraint enforces (pv_used + pv_to_bess + pv_curt == pv_available,
    core_milp_builder.py's PR-10). Curtailment cause (BESS full vs no grid
    room) is derived here by cross-checking soc_mwh against the BESS's real
    e_max_mwh bound -- not a new rule, just reading the same bound the
    solver's soc_hi constraint enforces."""
    comp = ComponentStore().load(delivery_date)
    if comp is None:
        return None
    pv_sched = comp.get("pv_schedule") or {}
    bess_sched = comp.get("bess_schedule") or {}
    if not pv_sched:
        return None
    cfg = load_config()
    e_max = cfg.plant.bess.e_max_mwh
    hours = sorted(pv_sched)

    used = [pv_sched[h]["used_mw"] for h in hours]
    to_bess = [pv_sched[h]["to_bess_mw"] for h in hours]
    curtailed = [pv_sched[h]["curtailed_mw"] for h in hours]
    available = [pv_sched[h]["available_mw"] for h in hours]

    causes = []
    for h in hours:
        c = pv_sched[h]["curtailed_mw"]
        if c <= 1e-6:
            causes.append(None)
        else:
            soc = bess_sched.get(h, {}).get("soc_mwh", 0.0)
            causes.append("BESS full" if soc >= e_max - 1e-3 else "no grid room")

    total_avail = sum(available)
    total_curt = sum(curtailed)
    econ = cfg.plant.economics
    curt_cost_eur = total_curt * econ.pv_curtailment_penalty_eur_mwh

    return {
        "hours": hours,
        "used_mw": used,
        "to_bess_mw": to_bess,
        "curtailed_mw": curtailed,
        "available_mw": available,
        "causes": causes,
        "total_available_mwh": total_avail,
        "total_curtailed_mwh": total_curt,
        "curtailed_pct": (total_curt / total_avail * 100) if total_avail > 1e-9 else 0.0,
        "curtailment_cost_eur": curt_cost_eur,
    }


@st.cache_data(ttl=5)
def load_multi_asset_dispatch(delivery_date: str) -> dict | None:
    """Per-hour DA dispatch decomposed by physical asset -- PV (to grid),
    BESS (discharge/charge), PSP (turbine/pump) -- from ComponentStore's
    pv_schedule/bess_schedule/psp_schedule, the real solved-model split, not
    a derived approximation. Reserve headroom (Gen/Pump_headroom_MW,
    aFRR/mFRR up/dn MW) is plant-level only in the real pipeline -- there is
    no per-asset attribution of reserve capacity anywhere in
    reserve_offer_builder.py/reserve_activation.py (fat_deliverable_mw sums
    psp_ramp_cap + bess_cap into one combined ceiling) -- shown here honestly
    as a combined overlay, not fabricated into a PSP-vs-BESS split."""
    comp = ComponentStore().load(delivery_date)
    if comp is None:
        return None
    psp_sched = comp.get("psp_schedule") or {}
    bess_sched = comp.get("bess_schedule") or {}
    pv_sched = comp.get("pv_schedule") or {}
    if not psp_sched:
        return None
    hours = sorted(psp_sched)

    report = load_daily_report(delivery_date)
    df = report["dispatch"] if report is not None else None
    has_headroom = df is not None and {"Gen_headroom_MW", "Pump_headroom_MW"}.issubset(df.columns)
    gen_headroom = df.set_index("Hour")["Gen_headroom_MW"].to_dict() if has_headroom else {}
    pump_headroom = df.set_index("Hour")["Pump_headroom_MW"].to_dict() if has_headroom else {}

    pv_mw = [pv_sched.get(h, {}).get("used_mw", 0.0) for h in hours]
    bess_dis = [bess_sched.get(h, {}).get("discharge_mw", 0.0) for h in hours]
    bess_chg = [bess_sched.get(h, {}).get("total_charge_mw", 0.0) for h in hours]
    psp_turb = [psp_sched[h]["turbine_mw"] for h in hours]
    psp_pump = [psp_sched[h]["pump_mw"] for h in hours]
    net_mw = [pv_mw[i] + bess_dis[i] + psp_turb[i] - bess_chg[i] - psp_pump[i] for i in range(len(hours))]

    return {
        "hours": hours,
        "pv_mw": pv_mw,
        "bess_dis_mw": bess_dis,
        "bess_chg_mw": bess_chg,
        "psp_turb_mw": psp_turb,
        "psp_pump_mw": psp_pump,
        "net_mw": net_mw,
        "gen_headroom_mw": [float(gen_headroom.get(h, 0.0)) for h in hours],
        "pump_headroom_mw": [float(pump_headroom.get(h, 0.0)) for h in hours],
        "has_headroom": has_headroom,
    }


@st.cache_data(ttl=5)
def load_water_balance(delivery_date: str) -> dict | None:
    """Per-hour hydrological water balance for the upper reservoir (Alqueva)
    -- inflow_m3h (natural river inflow, real forecast input) + pump-return
    flow (q_pump_total_m3h, water lifted into the upper reservoir) - turbine
    draw (q_turb_total_m3h) - spill (spill_m3h) should equal the observed
    hour-to-hour change in upper_hm3 (converted via 1 hm3 = 1,000,000 m3).
    This is the real identity the solved model's reservoir continuity
    constraint enforces -- shown here as a decomposition, not a new rule."""
    comp = ComponentStore().load(delivery_date)
    if comp is None:
        return None
    inflow = comp.get("inflow_m3h") or {}
    psp_sched = comp.get("psp_schedule") or {}
    traj = comp.get("reservoir_trajectory") or {}
    init = comp.get("initial_state") or {}
    if not inflow or not traj:
        return None
    hours = sorted(inflow)

    cfg = load_config()
    upper_initial_hm3 = init.get("upper_reservoir_hm3", cfg.plant.reservoir.upper_initial_hm3)
    prev_hm3 = upper_initial_hm3

    rows = []
    for h in hours:
        inflow_m3h = inflow[h]
        pump_m3h = psp_sched.get(h, {}).get("q_pump_total_m3h", 0.0)
        turb_m3h = psp_sched.get(h, {}).get("q_turb_total_m3h", 0.0)
        spill_m3h = traj[h]["spill_m3h"]
        upper_hm3 = traj[h]["upper_hm3"]
        delta_hm3 = upper_hm3 - prev_hm3
        net_flow_hm3 = (inflow_m3h + pump_m3h - turb_m3h - spill_m3h) / 1e6
        rows.append({
            "hour": h,
            "inflow_m3h": inflow_m3h,
            "pump_m3h": pump_m3h,
            "turb_m3h": turb_m3h,
            "spill_m3h": spill_m3h,
            "delta_hm3": delta_hm3,
            "net_flow_hm3": net_flow_hm3,
            "upper_hm3": upper_hm3,
        })
        prev_hm3 = upper_hm3

    return {
        "hours": hours,
        "rows": rows,
        "total_inflow_m3h": sum(r["inflow_m3h"] for r in rows),
        "total_spill_m3h": sum(r["spill_m3h"] for r in rows),
    }


@st.cache_data(ttl=5)
def load_bess_soc_price(delivery_date: str) -> dict | None:
    """BESS state-of-charge trajectory vs DA clearing price, the real
    storage-arbitrage decision the solver actually made -- soc_mwh,
    charge_mw/discharge_mw from ComponentStore.bess_schedule, DA_price_EUR_MWh
    from the same Dispatch_Hourly sheet every other price series here reads."""
    comp = ComponentStore().load(delivery_date)
    if comp is None:
        return None
    bess_sched = comp.get("bess_schedule") or {}
    if not bess_sched:
        return None
    report = load_daily_report(delivery_date)
    if report is None:
        return None
    df = report["dispatch"]
    if "DA_price_EUR_MWh" not in df.columns:
        return None
    price_by_hour = df.set_index("Hour")["DA_price_EUR_MWh"].to_dict()

    hours = sorted(bess_sched)
    cfg = load_config()
    e_min, e_max = cfg.plant.bess.e_min_mwh, cfg.plant.bess.e_max_mwh

    soc_mwh = [bess_sched[h]["soc_mwh"] for h in hours]
    chg_mw = [bess_sched[h]["total_charge_mw"] for h in hours]
    dis_mw = [bess_sched[h]["discharge_mw"] for h in hours]
    price = [float(price_by_hour.get(h, 0.0)) for h in hours]

    return {
        "hours": hours,
        "soc_mwh": soc_mwh,
        "soc_pct": [(s - e_min) / max(e_max - e_min, 1e-9) * 100 for s in soc_mwh],
        "chg_mw": chg_mw,
        "dis_mw": dis_mw,
        "price_eur_mwh": price,
        "e_min_mwh": e_min,
        "e_max_mwh": e_max,
    }


@st.cache_data(ttl=5)
def load_bess_charge_source(delivery_date: str) -> dict | None:
    """Real per-hour split of what BESS charging power actually came from --
    grid (bess_schedule.charge_mw, the p_chg solver variable) vs PV
    (bess_schedule.pv_to_bess_mw, the pv_to_bess solver variable) -- the
    exact two terms the real total_charge = p_chg + pv_to_bess constraint
    sums (core_milp_builder.py's bess_chg constraint). Only hours where the
    plant actually charged are meaningful here; idle/discharge hours have
    zero on both."""
    comp = ComponentStore().load(delivery_date)
    if comp is None:
        return None
    bess_sched = comp.get("bess_schedule") or {}
    if not bess_sched:
        return None
    hours = sorted(bess_sched)

    grid_mw = [bess_sched[h]["charge_mw"] for h in hours]
    pv_mw = [bess_sched[h]["pv_to_bess_mw"] for h in hours]
    total_mw = [grid_mw[i] + pv_mw[i] for i in range(len(hours))]

    total_grid_mwh = sum(grid_mw)
    total_pv_mwh = sum(pv_mw)
    total_mwh = total_grid_mwh + total_pv_mwh

    cfg = load_config()
    bess = cfg.plant.bess
    round_trip_eff = bess.eta_charge * bess.eta_discharge

    return {
        "hours": hours,
        "grid_mw": grid_mw,
        "pv_mw": pv_mw,
        "total_mw": total_mw,
        "total_grid_mwh": total_grid_mwh,
        "total_pv_mwh": total_pv_mwh,
        "pv_share_pct": (total_pv_mwh / total_mwh * 100) if total_mwh > 1e-9 else 0.0,
        "grid_share_pct": (total_grid_mwh / total_mwh * 100) if total_mwh > 1e-9 else 0.0,
        "any_charging": total_mwh > 1e-9,
        "degradation_cost_eur_mwh": bess.degradation_cost_eur_mwh,
        "round_trip_eff_pct": round_trip_eff * 100,
    }


@st.cache_data(ttl=5)
def load_gate_position_evolution(delivery_date: str) -> dict | None:
    """Net committed position (MW) AND effective price (EUR/MWh) as of each
    gate's close, true ISP resolution -- shows how each re-bid (IDA1-3/XBID)
    moved the position (and what it traded at) away from the original DA
    commitment, the same net-MW-vs-price pairing the Dispatch profile widget
    uses for the final position, but per gate. gates_mw uses
    PositionStore.committed_position per gate cutoff (same store
    dispatch_sheet_builder.py's committed_final sources from). gates_price
    chains each gate's own stored price forward ISP-by-ISP -- an ISP a gate
    didn't touch keeps whatever the previous gate traded it at, mirroring
    dispatch_sheet_builder.py's eff_after_idaN chains used for revenue.

    DA is the fixed reference -- its own chart is always the plain absolute
    position. Every later gate is shown as its DELTA against DA specifically
    (gates_mw_delta/gates_price_delta), not against the gate right before it,
    so each button always answers "how far has this moved from the original
    DA commitment by now" against one constant baseline. A held gate's delta
    is exactly zero everywhere -- an honest, visibly flat line, not a
    disguised repeat of the previous chart."""
    from common_layer.database.position_store import GATE_ORDER

    store = PositionStore()
    da_pos = store.load_position(delivery_date, "DA")
    if len(da_pos) < _MIN_REAL_ISP_COUNT:
        return None
    day = du.parse_date(delivery_date)
    isps = sorted(da_pos)
    hours = [du.isp_to_hour(isp, day) for isp in isps]

    gates_mw = {}
    for gate in GATE_ORDER:
        net = store.committed_position(delivery_date, as_of_gate=gate)
        gates_mw[gate] = [net.get(isp, 0.0) for isp in isps]

    gates_price = {}
    eff_price = {isp: da_pos.get(isp, {}).get("price_eur_mwh", 0.0) for isp in isps}
    gates_price["DA"] = [eff_price[isp] for isp in isps]
    for gate in GATE_ORDER[1:]:
        raw = store.load_position(delivery_date, gate)
        for isp in isps:
            if isp in raw:
                eff_price[isp] = raw[isp]["price_eur_mwh"]
        gates_price[gate] = [eff_price[isp] for isp in isps]

    gates_mw_delta = {"DA": [0.0] * len(isps)}
    gates_price_delta = {"DA": [0.0] * len(isps)}
    diverged_isps = {"DA": 0}
    net_mw_delta = {"DA": 0.0}
    for gate in GATE_ORDER[1:]:
        mw_d = [gates_mw[gate][i] - gates_mw["DA"][i] for i in range(len(isps))]
        price_d = [gates_price[gate][i] - gates_price["DA"][i] for i in range(len(isps))]
        gates_mw_delta[gate] = mw_d
        gates_price_delta[gate] = price_d
        diverged_isps[gate] = sum(1 for v in mw_d if abs(v) > 1e-6)
        net_mw_delta[gate] = sum(mw_d)

    return {
        "isps": isps,
        "hours": hours,
        "gate_order": GATE_ORDER,
        "gates_mw": gates_mw,
        "gates_price": gates_price,
        "gates_mw_delta": gates_mw_delta,
        "gates_price_delta": gates_price_delta,
        "diverged_isps": diverged_isps,
        "net_mw_delta": net_mw_delta,
    }


@st.cache_data(ttl=5)
def load_da_vs_activation(delivery_date: str) -> dict | None:
    """Two independent obligations at true 96-ISP (15-min) resolution, NOT a
    delivery reconciliation: the real per-ISP DA-committed net position
    (PositionStore.committed_position, ISP-keyed since the DA/IDA/XBID gates
    solve natively at ISP resolution) versus the real per-ISP aFRR/mFRR
    activation that ISP (ActivationStore, driven by system-wide Area Control
    Error -- see reserve_activation.py's simulate_ace_series docstring).
    Alqueva's own DA delivery accuracy has nothing to do with whether the
    control area's ACE calls it for balancing; these are two separate
    obligations that happen to land on the same plant at the same moment,
    shown honestly as such -- not "did we deliver our promise". Raw
    up_mw - dn_mw is used directly (not the eff_isp_h ramp-corrected energy
    factor settlement uses) since this compares two instantaneous MW
    quantities at native ISP grain, not an hourly-aggregated energy total.
    Combined aFRR+mFRR, plant-level only (no per-asset attribution exists in
    the real activation pipeline, same caveat as the Multi-asset dispatch
    widget). None for dates that predate ISP-native resolution."""
    committed = PositionStore().committed_position(delivery_date)
    if len(committed) < _MIN_REAL_ISP_COUNT:
        return None
    day = du.parse_date(delivery_date)
    isps = sorted(committed)
    da_net = [committed[isp] for isp in isps]

    activation_mw = {isp: 0.0 for isp in isps}
    any_activation = False
    for product in ("aFRR", "mFRR"):
        rows = ActivationStore().load(delivery_date, product)
        for r in rows:
            isp = r["isp"]
            if isp not in activation_mw:
                continue
            delta = r["up_mw"] - r["dn_mw"]
            if abs(delta) > 1e-9:
                any_activation = True
            activation_mw[isp] += delta

    activation_delta_mw = [activation_mw[isp] for isp in isps]
    delivered_mw = [da_net[i] + activation_delta_mw[i] for i in range(len(isps))]
    hours = [du.isp_to_hour(isp, day) for isp in isps]

    return {
        "isps": isps,
        "hours": hours,
        "da_net_mw": da_net,
        "activation_delta_mw": activation_delta_mw,
        "delivered_mw": delivered_mw,
        "any_activation": any_activation,
    }


@st.cache_data(ttl=5)
def load_agc_mechanism_demo(delivery_date: str, product: str) -> dict | None:
    """Illustrative, standalone AGC merit-order dispatch demo (see
    agc_mechanism_demo.py) -- NOT settlement data, purely explains the
    mechanism behind the real activation numbers shown elsewhere. Gated on
    the same activation summary existing so it doesn't appear before real
    delivery data does."""
    if load_activation_summary(delivery_date, product) is None:
        return None
    cfg = load_config()
    ticks = simulate_agc_dispatch(product, delivery_date, cfg)
    if not ticks:
        return None
    provider_names = [p.name for p in ticks[0].providers]
    dispatched_by_provider = {
        name: [round(next(p.dispatched_mw for p in t.providers if p.name == name), 2) for t in ticks]
        for name in provider_names
    }
    capacity_by_provider = {
        name: [round(next(p.capacity_mw for p in t.providers if p.name == name), 1) for t in ticks]
        for name in provider_names
    }
    price_by_provider = {
        name: [round(next(p.price_eur_mw for p in t.providers if p.name == name), 1) for t in ticks]
        for name in provider_names
    }
    return {
        "provider_names": provider_names,
        "ace_mw": [round(t.ace_mw, 1) for t in ticks],
        "required_mw": [round(t.required_mw, 1) for t in ticks],
        "direction": [t.direction for t in ticks],
        "dispatched_by_provider": dispatched_by_provider,
        "capacity_by_provider": capacity_by_provider,
        "price_by_provider": price_by_provider,
        "alqueva_dispatched_mw": [round(t.alqueva_dispatched_mw, 2) for t in ticks],
    }


@st.cache_data(ttl=5)
def load_fcr_activation(delivery_date: str) -> dict | None:
    """FCR droop-response compliance simulation. Unlike every other delivery
    card, this is NOT read from a pipeline run's audit trail or database --
    fcr_activation.py is explicitly a standalone module, not wired into
    run_production.py's phase list (FCR in Portugal is a mandatory,
    non-remunerated grid-code obligation, not a market-procured/settled
    product -- see that module's docstring). It's a pure, deterministic
    function of (delivery_date, config) though -- same seeded RNG every
    call -- so computing it here on demand is equivalent to it having "run",
    not a fabrication. Gated on a real RT delivery record existing for this
    date (same signal the other delivery cards use) so it doesn't appear
    for a date nothing has actually run for.
    """
    if load_rt_delivery(delivery_date) is None:
        return None
    cfg = load_config()
    summary = simulate_fcr_response(delivery_date, cfg)
    if summary.reserved_headroom_mw <= 0:
        return None
    rows = [{"isp": r.isp, "hour": r.hour, "energy_mwh": r.energy_mwh,
             "response_mw": r.response_mw, "direction": r.direction} for r in summary.rows]
    # summary.n_isp_in_deadband/n_isp_responding are misleadingly named in
    # fcr_activation.py -- despite the "n_isp_" prefix they're actually
    # counts of 30-second TICKS (2,880/day), not ISPs (96/day), since FCR's
    # native resolution is the tick, not the ISP. Recompute a genuine
    # ISP-level count here (an ISP counts as "responding" if any of its
    # ticks had a non-deadband response) so the card's "X/96 ISPs" badge
    # compares like with like instead of ticks against ISPs.
    n_isp_responding = sum(1 for r in rows if r["direction"] != "none")
    # Tick-grain arrays (2,880/day, FCR's real 30s resolution) for the
    # overlaid frequency+response replay chart -- rounded to keep the
    # embedded-in-HTML payload reasonable (a few hundred KB, not MB).
    tick_freq_mhz = [round(t.freq_deviation_mhz, 1) for t in summary.ticks]
    tick_response_mw = [round(t.response_mw, 2) for t in summary.ticks]
    return {
        "reserved_headroom_mw": summary.reserved_headroom_mw,
        "n_ticks_in_deadband": summary.n_isp_in_deadband,
        "n_ticks_responding": summary.n_isp_responding,
        "n_isp_responding": n_isp_responding,
        "up_mwh": summary.up_mwh,
        "dn_mwh": summary.dn_mwh,
        "max_response_mw": summary.max_response_mw,
        "rows": rows,
        "tick_freq_mhz": tick_freq_mhz,
        "tick_response_mw": tick_response_mw,
        "delivery_date": delivery_date,
    }


def latest_gate_ticket(delivery_date: str) -> dict | None:
    """The most recent DA/aFRR/mFRR/IDA1-3/XBID decision for this delivery
    date, with its hourly bid/offer table. None if nothing has decided yet.

    Skips a decision whose store data has gone missing — run_production.py
    clears PositionStore/ReserveStore for this date at the start of a
    genuine fresh run, but audit_*.jsonl is append-only forever, so a
    SUBMITTED event from a Ctrl+C-killed earlier attempt can still be the
    chronologically "latest" audit record even after its position/reserve
    rows are gone. Showing that would be a ticket with an empty chart for a
    decision that no longer actually holds — walk backward to the next one
    that still has real data instead.
    """
    tickets = all_gate_tickets(delivery_date)
    return tickets[-1] if tickets else None


# ---------------------------------------------------------------------------
# ML Forecasting overview -- reads the *_selected_model.json bake-off
# results every forecaster already writes (never recomputed here) plus the
# row count of the training data workbook each one trains from, straight
# off disk. Real numbers, same ones run_production.py's own forecasters
# used to pick a model this run -- nothing here is illustrative.
# ---------------------------------------------------------------------------

_MODEL_ROSTER = [
    # (group, target label, json path, training data path or None, unit)
    ("Day-Ahead (DA)", "DA price (hourly)",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/da_selected_model.json",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/da_training_data_isp_2025_2026.xlsx",
     "EUR/MWh"),
    ("Day-Ahead (DA)", "DA price (96-ISP)",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/da_selected_model_isp.json",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/da_training_data_isp_2025_2026.xlsx",
     "EUR/MWh"),
    ("Day-Ahead (DA)", "PV output — solar irradiance (GHI)",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/pv_selected_model.json:GHI",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/pv_training_data_from_2015.xlsx", "W/m²"),
    ("Day-Ahead (DA)", "PV output — ambient temperature",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/pv_selected_model.json:T_amb",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/pv_training_data_from_2015.xlsx", "°C"),
    ("Day-Ahead (DA)", "Reservoir inflow",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/inflow_selected_model.json",
     "phase_1_da_day_ahead_bidding/da_price_pv_inflow_forecasting/inflow_training_data_from_2015.xlsx", "m³/h"),
    ("IDA1", "IDA1 price",
     "phase_2a_ida1_intraday_auction_1/ida1_price_forecasting/ida1_selected_model.json",
     "phase_2a_ida1_intraday_auction_1/ida1_price_forecasting/ida1_training_data_2024_2025.xlsx",
     "EUR/MWh"),
    ("IDA2", "IDA2 price",
     "phase_2b_ida2_intraday_auction_2/ida2_price_forecasting/ida2_selected_model.json",
     "phase_2b_ida2_intraday_auction_2/ida2_price_forecasting/ida2_training_data_2024_2025.xlsx",
     "EUR/MWh"),
    ("IDA3", "IDA3 price",
     "phase_2c_ida3_intraday_auction_3/ida3_price_forecasting/ida3_selected_model.json",
     "phase_2c_ida3_intraday_auction_3/ida3_price_forecasting/ida3_training_data_2024_2025.xlsx",
     "EUR/MWh"),
    ("XBID", "XBID price",
     "phase_2d_xbid_continuous_intraday/xbid_price_forecasting/xbid_selected_model.json",
     "phase_2d_xbid_continuous_intraday/xbid_price_forecasting/xbid_training_data_2024_2025.xlsx",
     "EUR/MWh"),
    ("aFRR", "aFRR up-regulation price",
     "phase_3a_afrr_automatic_frequency_reserve/afrr_price_forecasting/afrr_up_selected_model.json",
     "phase_3a_afrr_automatic_frequency_reserve/afrr_price_forecasting/afrr_training_data_2019_2025.xlsx",
     "EUR/MW"),
    ("aFRR", "aFRR down-regulation price",
     "phase_3a_afrr_automatic_frequency_reserve/afrr_price_forecasting/afrr_dn_selected_model.json",
     "phase_3a_afrr_automatic_frequency_reserve/afrr_price_forecasting/afrr_training_data_2019_2025.xlsx",
     "EUR/MW"),
    ("mFRR", "mFRR up-regulation price",
     "phase_3b_mfrr_manual_frequency_reserve/mfrr_price_forecasting/mfrr_up_selected_model.json",
     "phase_3b_mfrr_manual_frequency_reserve/mfrr_price_forecasting/mfrr_training_data_2024_2025.xlsx",
     "EUR/MW"),
    ("mFRR", "mFRR down-regulation price",
     "phase_3b_mfrr_manual_frequency_reserve/mfrr_price_forecasting/mfrr_dn_selected_model.json",
     "phase_3b_mfrr_manual_frequency_reserve/mfrr_price_forecasting/mfrr_training_data_2024_2025.xlsx",
     "EUR/MW"),
]

_MODEL_BLURBS = {
    "LightGBM":     "Fast gradient-boosted trees — usually the quickest to train.",
    "XGBoost":      "Gradient-boosted trees with strong regularisation — a steady all-rounder.",
    "RandomForest": "Many independent decision trees averaged together — robust to noisy days.",
    "CatBoost":     "Gradient-boosted trees tuned to resist overfitting on smaller datasets.",
}


@st.cache_data(ttl=60)
def load_ml_models_overview() -> dict:
    """One row per forecasting target: which of the 4 candidate models won
    its bake-off, by how much, and how much history it trained on. Reads
    everything straight from disk -- the *_selected_model.json files
    written by each forecaster's own _auto_select_model() (see
    da_price_forecaster.py etc.) and the training workbook row counts.
    A target whose json is missing (forecaster never run yet) is simply
    omitted, not faked."""
    rows = []
    for group, target, json_rel, train_rel, unit in _MODEL_ROSTER:
        json_path = json_rel.split(":")
        path = REPO_ROOT / json_path[0]
        sub_key = json_path[1] if len(json_path) > 1 else None
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if sub_key:
            payload = payload.get(sub_key)
            if payload is None:
                continue

        cv_mae = payload.get("cv_mae", {})
        selected = payload.get("selected")
        if not selected or not cv_mae:
            continue
        ranked = sorted(cv_mae.items(), key=lambda kv: kv[1])
        best_name, best_mae = ranked[0]
        runner_up_gap_pct = None
        if len(ranked) > 1 and ranked[1][1] > 0:
            runner_up_gap_pct = (ranked[1][1] - best_mae) / ranked[1][1] * 100

        # openpyxl's read-only mode just walks the worksheet's dimensions
        # instead of parsing every cell -- pd.read_excel on the largest of
        # these workbooks (66k+ rows) took several seconds each; max_row
        # alone takes milliseconds, and a row count is all this needs.
        n_rows = None
        if train_rel:
            train_path = REPO_ROOT / train_rel
            if train_path.exists():
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(train_path, read_only=True)
                    n_rows = wb.active.max_row - 1  # minus header
                    wb.close()
                except Exception:
                    n_rows = None

        rows.append({
            "group": group,
            "target": target,
            "selected": selected,
            "cv_mae": cv_mae,
            "best_mae": round(best_mae, 4),
            "unit": unit,
            "runner_up_gap_pct": runner_up_gap_pct,
            "n_training_rows": n_rows,
            "data_end_date": payload.get("data_end_date"),
            "updated_on": payload.get("updated_on"),
        })

    win_counts: dict[str, int] = {}
    for r in rows:
        win_counts[r["selected"]] = win_counts.get(r["selected"], 0) + 1

    return {
        "rows": rows,
        "win_counts": win_counts,
        "model_blurbs": _MODEL_BLURBS,
        "n_targets": len(rows),
    }
