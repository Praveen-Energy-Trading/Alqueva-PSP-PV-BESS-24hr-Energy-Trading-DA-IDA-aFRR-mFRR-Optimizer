"""
run_production.py
=================
Alqueva PSP-PV-BESS  24-Hour Energy Trading Pipeline Orchestrator.

Runs all 19 pipeline phases in delivery order for a single date
(XBID counted as 6 separate check-window phases, W1-W6).
Reads config/run.yaml for settings; every option is overridable via CLI.

QUICK START
-----------
Just run (AUTO mode is the default — delivery date = tomorrow in Portugal):

    python run_production.py

COMMON CLI OVERRIDES  (no YAML edit needed)
-------------------------------------------
    # Run for a specific date
    python run_production.py --date 08-07-2026

    # Recovery: skip phases already completed, restart from real-time dispatch
    python run_production.py --date 08-07-2026 --from-phase realtime

    # Backtest: fully automated, synthetic prices, no live API calls
    python run_production.py --date 08-07-2026 --auto --synthetic

    # Run only selected phases (good for debugging individual phases)
    python run_production.py --date 08-07-2026 --only da,afrr,mfrr

    # Validate config and imports without executing anything
    python run_production.py --dry-run

PHASE KEYS (--from-phase / --only)
------------------------------------
  da  ida1  ida2  ida3
  xbid_w1  xbid_w2  xbid_w3  xbid_w4  xbid_w5  xbid_w6
  afrr  mfrr  realtime
  afrr_activation  mfrr_activation
  energy_settlement  reserve_settlement  imbalance_settlement  analytics

EXIT CODES
----------
  0   All enabled phases passed or warned (non-critical).
  1   One or more critical phases failed.
  2   Configuration or import error.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set

import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PORT = 8501


def _ensure_dashboard_running() -> None:
    """Auto-open the Streamlit dashboard (dashboard/app.py) on a normal run.
    Best-effort only: any failure here (streamlit missing, port probe error,
    browser-open error, etc.) is swallowed — the pipeline itself must never
    fail because the dashboard couldn't start. Scoped to this script only;
    phase_6_backtesting_and_validation/run_backtest.py does not import
    run_production.py, so backtests are unaffected.
    """
    try:
        with socket.create_connection(("127.0.0.1", DASHBOARD_PORT), timeout=0.3):
            # Already running (e.g. a previous run left it up) — don't spawn a
            # second instance, but DO surface a tab. Streamlit only auto-opens
            # a browser on its own *first* launch, so without this, re-running
            # the pipeline while a dashboard from an earlier run is still
            # alive would silently do nothing visible.
            import webbrowser
            webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")
            return
    except OSError:
        pass  # nothing listening yet, proceed to launch

    cmd = [sys.executable, "-m", "streamlit", "run",
           os.path.join(REPO_ROOT, "dashboard", "app.py"),
           f"--server.port={DASHBOARD_PORT}", "--server.headless=false"]
    # A diagnostic log, not DEVNULL: some IDE run-configurations (VS Code's
    # "Run Python File" in particular) execute the pipeline inside a
    # restrictive Windows Job Object, and DETACHED_PROCESS there can fail
    # silently at CreateProcess if the job doesn't grant breakaway rights —
    # previously that failure vanished into DEVNULL with nothing printed,
    # since even our own except-branch below writes to stderr, which some
    # run-configurations don't surface either. Logging to a file makes the
    # failure diagnosable regardless of where stdout/stderr end up.
    diag_path = os.path.join(REPO_ROOT, "runtime", "logs", "_dashboard_launch.log")
    os.makedirs(os.path.dirname(diag_path), exist_ok=True)

    def _spawn(creationflags: int) -> None:
        with open(diag_path, "a", encoding="utf-8") as diag_fh:
            diag_fh.write(f"\n--- launch attempt {time.strftime('%Y-%m-%d %H:%M:%S')} "
                           f"(creationflags={creationflags}) ---\ncmd={cmd}\n")
            subprocess.Popen(
                cmd, cwd=REPO_ROOT,
                stdout=diag_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )

    try:
        if os.name == "nt":
            # CREATE_BREAKAWAY_FROM_JOB: explicitly request escaping whatever
            # job object the parent (IDE run-configuration) may have placed
            # us in, so the dashboard survives the pipeline process exiting.
            try:
                _spawn(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                       | subprocess.CREATE_BREAKAWAY_FROM_JOB)
            except OSError:
                # Breakaway denied (job doesn't allow it, or nested-job
                # restrictions on this Windows version) — fall back to a
                # plain child. It'll die with the parent's job/process tree
                # instead of surviving independently, but at least it starts.
                _spawn(0)
        else:
            _spawn(0)
    except Exception as exc:
        msg = f"  (dashboard auto-launch skipped: {exc})"
        print(msg, file=sys.stderr)
        try:
            with open(diag_path, "a", encoding="utf-8") as diag_fh:
                diag_fh.write(msg + "\n")
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Phase output tee.
# Every print() and every logger ultimately writes through sys.stdout, which
# for the whole run is this single _Tee — so the log file (and therefore the
# dashboard's Console Log tab) always receives everything, in BOTH auto mode
# and interactive Trader mode. Auto mode additionally mutes the console side
# (via _silence) so only the clean orchestrator table appears on screen —
# but muting never affects what reaches the log file.
#
# Previously, Trader-mode phase output was never redirected at all (only
# auto mode used _silence, which fully swapped stdout+logger handlers to
# point at the log file instead of the console). That meant a Trader-mode
# run — e.g. one started with the plain Run button, no --auto
# flag — showed everything live in the console but wrote NOTHING to the log
# file, so the dashboard's Console Log tab had nothing to display even
# though the pipeline was genuinely running. This _Tee fixes that: both
# modes now write to the log file, so the dashboard stays in sync
# regardless of which mode you ran in.
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, real_stdout, log_fh: io.TextIOWrapper):
        self.real_stdout = real_stdout
        self.log_fh = log_fh
        self.mute_console = False

    def write(self, s: str) -> int:
        self.log_fh.write(s)
        if not self.mute_console:
            self.real_stdout.write(s)
        return len(s)

    def flush(self) -> None:
        self.log_fh.flush()
        self.real_stdout.flush()

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def _silence(tee: "_Tee"):
    """Mute only the console side of output for this phase (auto mode) —
    the log file still receives everything via the shared _Tee above."""
    tee.mute_console = True
    try:
        yield
    finally:
        tee.mute_console = False


# ---------------------------------------------------------------------------
# Date parsing — user-facing format is DD-MM-YYYY.
# Internal format passed to all phase functions is YYYY-MM-DD (ISO 8601).
# ---------------------------------------------------------------------------

def _parse_date(raw: str) -> str:
    """Accept DD-MM-YYYY (user input / YAML) and return YYYY-MM-DD.

    Also accepts YYYY-MM-DD transparently so old scripts still work.
    Raises ValueError with a clear message on any other format.
    """
    raw = raw.strip()
    from datetime import datetime
    for fmt_in, fmt_out in (
        ("%d-%m-%Y", "%Y-%m-%d"),   # DD-MM-YYYY  (primary)
        ("%Y-%m-%d", "%Y-%m-%d"),   # YYYY-MM-DD  (fallback / ISO)
    ):
        try:
            return datetime.strptime(raw, fmt_in).strftime(fmt_out)
        except ValueError:
            continue
    raise ValueError(
        f"Unrecognised date '{raw}'. Use DD-MM-YYYY (e.g. 06-07-2026)."
    )


# ---------------------------------------------------------------------------
# Ensure repo root is importable.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Status tokens
# ---------------------------------------------------------------------------
PASS = "PASS"
SKIP = "SKIP"
WARN = "WARN"
FAIL = "FAIL"

# Status codes from individual run_*() functions.
_OK_CODES   = {"SUBMITTED", "OK", "NO_CHANGE"}
_WARN_CODES = {"NO_OFFER", "REJECTED"}

# ---------------------------------------------------------------------------
# Pipeline registry
#   (internal_key, display_label, is_critical)
#
# is_critical=True  → pipeline aborts immediately if this phase FAILs.
# is_critical=False → WARN logged, subsequent phases still run.
# ---------------------------------------------------------------------------
_PHASES: List[tuple] = [
    # REAL REN sequencing, CONFIRMED against REN's own official rulebook:
    # Manual de Procedimentos da Gestao Global do Sistema (MPGGS), ERSE
    # Directive 9/2025, Article 80(3): "A programacao e resolucao de desvios
    # incluem os seguintes processos sucessivos: a) Criacao do Programa
    # Diario Viavel Definitivo (PDVD); b) Mercado de Banda de aFRR;
    # c) Mercado de banda diaria de mFRR ..." — i.e. the DA-derived program
    # (PDVD, REN's grid-security-checked version of the DA market result)
    # is created FIRST, THEN the aFRR band market, THEN the daily mFRR band
    # market. An earlier session pass reordered this to aFRR/mFRR-before-DA
    # based on Germany's regelleistung.net precedent (a valid EU pattern,
    # but NOT what REN's own document specifies for Portugal) — reverted
    # here once the real REN sequence was found and confirmed (Article 81
    # for PDVD's DA-dependency, Article 82-83 for aFRR/mFRR band markets).
    ("da",                  "1      Day-Ahead bidding  (OMIE DA)",        True),
    ("afrr",                "2      aFRR capacity offer  (PICASSO/REN)", True),
    ("mfrr",                "3      mFRR capacity offer  (MARI)",        False),
    ("ida1",                "4A     IDA1 intraday re-optimisation",       True),
    ("ida2",                "4B     IDA2 intraday re-optimisation",       True),
    ("ida3",                "4C     IDA3 intraday re-optimisation",       True),
    ("xbid_w1",             "4D/W1  XBID continuous  (D-1 18:30)",       False),
    ("xbid_w2",             "4D/W2  XBID continuous  (D-1 22:30)",       False),
    ("xbid_w3",             "4D/W3  XBID continuous  (D  03:00)",        False),
    ("xbid_w4",             "4D/W4  XBID continuous  (D  06:00)",        False),
    ("xbid_w5",             "4D/W5  XBID continuous  (D  09:30)",        False),
    ("xbid_w6",             "4D/W6  XBID continuous  (D  12:00)",        False),
    ("realtime",            "5A     RT dispatch simulation  (96 ISPs)",   True),
    ("afrr_activation",     "5B     aFRR activation response",            True),
    ("mfrr_activation",     "5C     mFRR activation response",            False),
    ("energy_settlement",   "6A     Energy settlement  (DA / IDA)",       True),
    ("reserve_settlement",  "6B     Reserve settlement  (aFRR / mFRR)",   True),
    ("imbalance_settlement","6C     Imbalance settlement  (REN balance)", True),
    ("analytics",           "6D     Analytics + KPI report + Excel",      False),
]

# The YAML phases block uses "xbid" to gate all check windows (W1..W6).
_YAML_KEY = {f"xbid_w{i}": "xbid" for i in range(1, 7)}
_ALL_KEYS  = [k for k, _, _ in _PHASES]


# ---------------------------------------------------------------------------
# Phase dispatcher
# ---------------------------------------------------------------------------

def _dispatch(key: str, date: str, cfg, syn: bool, auto: bool) -> tuple[str, str]:
    """Call the phase runner; return (PASS|WARN|FAIL, one-line detail)."""

    def _run(fn, *a, **kw) -> tuple[str, str]:
        r = fn(*a, **kw)
        st = r.get("status", "UNKNOWN") if isinstance(r, dict) else str(r)
        if st in _OK_CODES:
            return PASS, _detail(r)
        if st in _WARN_CODES:
            return WARN, st
        if st == "SKIPPED":
            reason = r.get("reason", "skipped") if isinstance(r, dict) else "skipped"
            return SKIP, reason[:50]
        reason = ""
        if isinstance(r, dict):
            reason = r.get("reason", r.get("violations", ""))
        return FAIL, f"{st}: {reason}"

    if key == "da":
        from phase_1_da_day_ahead_bidding.run_da import run_da
        return _run(run_da, date, use_synthetic=syn, auto_approve=auto)

    if key == "ida1":
        from phase_2a_ida1_intraday_auction_1.ida1_milp_reoptimiser.ida1_reoptimiser import optimise_ida1
        return _run(optimise_ida1, date, cfg, no_pause=auto, use_synthetic=syn)

    if key == "ida2":
        from phase_2b_ida2_intraday_auction_2.ida2_milp_reoptimiser.ida2_reoptimiser import optimise_ida2
        return _run(optimise_ida2, date, cfg, no_pause=auto, use_synthetic=syn)

    if key == "ida3":
        from phase_2c_ida3_intraday_auction_3.ida3_milp_reoptimiser.ida3_reoptimiser import optimise_ida3
        return _run(optimise_ida3, date, cfg, no_pause=auto, use_synthetic=syn)

    if key.startswith("xbid_w"):
        from phase_2d_xbid_continuous_intraday.xbid_milp_optimiser.xbid_optimiser import optimise_xbid
        window = "W" + key[len("xbid_w"):]
        return _run(optimise_xbid, date, cfg, window=window, no_pause=auto, use_synthetic=syn)

    if key == "afrr":
        from phase_3a_afrr_automatic_frequency_reserve.run_afrr import run_afrr
        return _run(run_afrr, date, cfg, no_pause=auto, use_synthetic=syn)

    if key == "mfrr":
        from phase_3b_mfrr_manual_frequency_reserve.run_mfrr import run_mfrr
        return _run(run_mfrr, date, cfg, no_pause=auto, use_synthetic=syn)

    if key == "realtime":
        from phase_4a_isp_real_time_dispatch.run_realtime import run_realtime
        return _run(run_realtime, date, cfg, no_pause=auto)

    if key == "afrr_activation":
        from phase_4b_afrr_activation_response.run_afrr_activation import run_afrr_activation
        return _run(run_afrr_activation, date, cfg, no_pause=auto)

    if key == "mfrr_activation":
        from phase_4c_mfrr_activation_response.run_mfrr_activation import run_mfrr_activation
        return _run(run_mfrr_activation, date, cfg, no_pause=auto)

    if key == "energy_settlement":
        from phase_5a_da_ida_settlement.run_energy_settlement import run_energy_settlement
        return _run(run_energy_settlement, date)

    if key == "reserve_settlement":
        from phase_5b_reserve_settlement.run_reserve_settlement import run_reserve_settlement
        return _run(run_reserve_settlement, date)

    if key == "imbalance_settlement":
        from phase_5c_imbalance_settlement.run_imbalance_settlement import run_imbalance_settlement
        return _run(run_imbalance_settlement, date)

    if key == "analytics":
        from phase_5d_analytics_and_reporting.run_analytics import run_analytics
        return _run(run_analytics, date, cfg, export_excel=True)

    return FAIL, f"Unknown phase key: {key}"


def _detail(r) -> str:
    """Extract a compact one-line financial note from a phase result dict."""
    if not isinstance(r, dict):
        return ""
    parts = []
    for k in ("energy_revenue_eur", "capacity_revenue_eur", "total_pnl_eur",
              "objective_eur", "pnl_change_eur", "activation_eur"):
        v = r.get(k)
        if v is not None:
            parts.append(f"{k.replace('_eur','').replace('_',' ')} {v:+,.0f}")
    return "  |  ".join(parts) if parts else (r.get("ref") or "")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
_W = 82   # total line width

def _header(date: str, mode: str, source: str, dry: bool) -> None:
    print()
    print("=" * _W)
    print("  ALQUEVA PSP-PV-BESS   24-HOUR ENERGY TRADING PIPELINE")
    print("=" * _W)
    print(f"  Delivery date : {date}" + ("  [DRY-RUN — no submissions]" if dry else ""))
    print(f"  Mode          : {mode.upper()}")
    print("=" * _W)
    print(f"  {'PHASE':<44}  {'STATUS':^8}  {'TIME':>6}  NOTE")
    print("  " + "-" * (_W - 2))


def _plant_specs(cfg) -> None:
    p = cfg.plant
    freq = cfg.market.frequency
    turb_tot = p.psp.n_units * p.psp.p_turbine_max_mw
    pump_tot = p.psp.n_units * p.psp.p_pump_max_mw
    fcr_lo = freq.nominal_hz - freq.fcr_full_activation_hz
    fcr_hi = freq.nominal_hz + freq.fcr_full_activation_hz
    print("=" * _W)
    print("  PLANT SPECS  (Alqueva PSP + PV + BESS)")
    print("=" * _W)
    print(f"  Turbines : {p.psp.n_units} x {p.psp.p_turbine_max_mw:.1f} MW   = {turb_tot:.1f} MW total")
    print(f"  Pumps    : {p.psp.n_units} x {p.psp.p_pump_max_mw:.1f} MW   = {pump_tot:.1f} MW total")
    print(f"  PV       : {p.pv.peak_capacity_mw:.1f} MWp")
    print(f"  BESS     : {p.bess.power_mw:.1f} MW / {p.bess.capacity_mwh:.1f} MWh")
    print(f"  FCR      : {p.fcr.mandatory_headroom_mw:.1f} MW mandatory headroom   |   "
          f"Band: {fcr_lo:.3f}-{fcr_hi:.3f} Hz   |   "
          f"Activation: <{freq.fcr_full_activation_time_s:.0f} sec")
    print("  " + "-" * (_W - 2))
    print(f"  Max generation envelope : {p.p_max_generation_mw:.1f} MW   (turbines + PV + BESS)")
    print(f"  Max pump envelope       : {p.p_max_pump_mw:.1f} MW   (pumps + BESS)")
    print("=" * _W)


def _row(label: str, status: str, elapsed: float, detail: str) -> None:
    icons = {PASS: "[ OK  ]", SKIP: "[  -- ]", WARN: "[ !!  ]", FAIL: "[ XX  ]"}
    t     = f"{elapsed:.2f}s" if elapsed > 0.001 else ""
    note  = detail[:34] if detail else ""
    print(f"  {label:<44}  {icons.get(status,'[????]')}  {t:>6}  {note}")


def _footer(results: list, total: float) -> int:
    n = {s: sum(1 for r in results if r["status"] == s)
         for s in (PASS, SKIP, WARN, FAIL)}
    print()
    print("=" * _W)
    outcome = "PIPELINE COMPLETE" if n[FAIL] == 0 else "PIPELINE FAILED"
    print(f"  {outcome}")
    print(f"  {n[PASS]} passed   {n[SKIP]} skipped   "
          f"{n[WARN]} warnings   {n[FAIL]} failed   "
          f"({total:.1f}s total)")
    fails = [r for r in results if r["status"] == FAIL]
    if fails:
        print()
        print("  FAILURES:")
        for r in fails:
            print(f"    [{r['key']}]  {r['detail']}")
    print("=" * _W)
    print()
    return 0 if n[FAIL] == 0 else 1


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _load_yaml(config_dir: Optional[str]) -> dict:
    path = os.path.join(
        config_dir or os.path.join(_ROOT, "config"), "run.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"run.yaml not found: {path}\n"
            "Create config/run.yaml (copy from config/run.yaml.example) "
            "or pass --config <dir>."
        )
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"run.yaml did not parse to a mapping: {path}")
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_production.py",
        description="Alqueva 24-hour energy trading pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--date",       metavar="DD-MM-YYYY",
                   help="Delivery date DD-MM-YYYY (e.g. 06-07-2026) for a MANUAL run, "
                        "or 'auto' for tomorrow in Portugal. Default (no flag) is AUTO.")
    p.add_argument("--config",     metavar="DIR",
                   help="Config directory (default: <repo>/config/)")
    p.add_argument("--from-phase", metavar="KEY", dest="from_phase",
                   help="Start pipeline from this phase (recovery restart)")
    p.add_argument("--only",       metavar="K1,K2,...",
                   help="Run only these phase keys (comma-separated)")
    p.add_argument("--auto",       action="store_true",
                   help="Auto mode: no operator prompts — overrides run.yaml")
    p.add_argument("--synthetic",  action="store_true",
                   help="Synthetic data: no live API calls — overrides run.yaml")
    p.add_argument("--dry-run",    dest="dry_run", action="store_true",
                   help="Validate config and imports; run nothing")
    p.add_argument("--no-dashboard", dest="no_dashboard", action="store_true",
                   help="Don't auto-launch the Streamlit dashboard (it's on by default)")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ns = _build_parser().parse_args()

    if not ns.no_dashboard and not ns.dry_run:
        _ensure_dashboard_running()

    # ── 1. Load YAML ────────────────────────────────────────────────────────
    try:
        yml = _load_yaml(ns.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        return 2

    # ── 2. Resolve delivery date (default AUTO = tomorrow in Portugal) ───────
    #   --date DD-MM-YYYY  -> manual (overrides yaml)
    #   run.yaml: auto     -> tomorrow in Portugal (Europe/Lisbon), no editing
    from common_layer.utilities.date_utils import resolve_delivery_date, portugal_today
    try:
        date, display_date, is_auto = resolve_delivery_date(
            ns.date, yml.get("delivery_date"))
    except ValueError as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        return 2
    if is_auto:
        print(f"  Delivery date: AUTO -> {display_date}  "
              f"(tomorrow; today in Portugal is {portugal_today().strftime('%d-%m-%Y')})")
    else:
        print(f"  Delivery date: {display_date}  (manual)")
    mode   = "auto"      if ns.auto      else str(yml.get("mode",        "auto"))
    source = "synthetic" if ns.synthetic else str(yml.get("data_source", "synthetic"))
    is_auto = (mode   == "auto")
    is_syn  = (source == "synthetic")
    enabled: Dict[str, bool] = yml.get("phases", {})

    # ── 3. --from-phase index ───────────────────────────────────────────────
    from_idx = 0
    if ns.from_phase:
        fk = "xbid_w1" if ns.from_phase.strip() == "xbid" else ns.from_phase.strip()
        if fk not in _ALL_KEYS:
            print(f"\n  ERROR: Unknown phase key '{ns.from_phase}'. "
                  f"Valid: {', '.join(_ALL_KEYS)}", file=sys.stderr)
            return 2
        from_idx = _ALL_KEYS.index(fk)

    # ── 4. --only filter ────────────────────────────────────────────────────
    only_keys: Optional[Set[str]] = None
    if ns.only:
        raw = {k.strip() for k in ns.only.split(",")}
        only_keys = set()
        for k in raw:
            if k == "xbid":
                only_keys.update(f"xbid_w{i}" for i in range(1, 7))
            else:
                only_keys.add(k)
        bad = only_keys - set(_ALL_KEYS)
        if bad:
            print(f"\n  ERROR: Unknown phase key(s): {', '.join(sorted(bad))}",
                  file=sys.stderr)
            return 2

    # ── 5. Load plant/market/solver config ──────────────────────────────────
    try:
        from common_layer.configuration import load_config
        app_cfg = load_config(ns.config)
    except Exception as exc:
        print(f"\n  ERROR loading plant/market config: {exc}", file=sys.stderr)
        return 2

    # ── 6. Execute pipeline ─────────────────────────────────────────────────
    log_path = os.path.join(_ROOT, "runtime", "logs", f"pipeline_{date}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    _header(display_date, mode, source, ns.dry_run)
    _plant_specs(app_cfg)
    if not ns.dry_run:
        print(f"  Log file      : runtime/logs/pipeline_{date}.log")
        print("  " + "-" * (_W - 2))

    critical_map = {k: c for k, _, c in _PHASES}
    results: List[dict] = []
    t_start = time.perf_counter()

    # Mark this date as in-progress *before* the phase loop starts, so the
    # dashboard can tell "actively running" apart from "stale result from a
    # previous run for this same date" — interactive mode can sit paused on
    # an Approve/Reject or ENTER prompt for minutes, far longer than the
    # dashboard's log-freshness window, so without this sentinel it would
    # silently fall back to showing whatever the last completed run left
    # behind. Overwritten with the real results (status omitted -> DONE via
    # the "results" key's presence) once the loop finishes below.
    # pid lets the dashboard tell "genuinely still running" apart from
    # "process was killed (Ctrl+C) while paused at an Approve/Reject or
    # ENTER prompt" — without it, a hard kill leaves this file stuck at
    # RUNNING forever (the DONE write below never runs), and the dashboard
    # has no way to know the difference from a real, still-alive pause.
    run_status_path = os.path.join(_ROOT, "runtime", "logs", f"run_status_{date}.json")
    os.makedirs(os.path.dirname(run_status_path), exist_ok=True)
    with open(run_status_path, "w", encoding="utf-8") as fh:
        json.dump({"date": date, "mode": mode, "status": "RUNNING", "pid": os.getpid(),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

    # A genuine fresh run (not --from-phase/--only, which need prior rows to
    # resume from) clears this date's committed positions/reserve offers
    # first. Without this, a run killed mid-way (Ctrl+C) and restarted would
    # have its later phases layered on top of the killed attempt's leftover
    # SQLite rows -- e.g. an old aFRR_SUBMITTED row surviving even though
    # this run hasn't reached aFRR yet, making the dashboard's live gate
    # ticket show a decision that isn't really this run's.
    if from_idx == 0 and only_keys is None and not ns.dry_run:
        from common_layer.database import PositionStore, ReserveStore, DeliveryStore, ActivationStore
        PositionStore().clear_date(date)
        ReserveStore().clear_date(date)
        # Same leftover-row problem as Position/ReserveStore above, but for
        # the delivery-phase tables (5A RT dispatch, 5B/5C activation) added
        # later -- a killed run's DeliveryStore/ActivationStore rows would
        # otherwise survive and make the dashboard's new delivery cards show
        # "already delivered" for a phase this fresh run hasn't reached yet.
        DeliveryStore().clear_date(date)
        ActivationStore().clear_date(date)
        # daily_report_<date>.xlsx is only written by the LAST phase
        # (analytics) — if an older run for this same date already
        # produced one, it silently survives on disk for the entire
        # duration of this new run, so the dashboard's report_ready check
        # stays True the whole time and Trading Desk/Overview keep showing
        # the previous run's P&L as if it were current. Removing it here
        # makes "no report yet" accurate again from the first phase.
        stale_report = os.path.join(_ROOT, "runtime", "reports", f"daily_report_{date}.xlsx")
        if os.path.exists(stale_report):
            os.remove(stale_report)

    # Truncate rather than append: each run starts a fresh log for its
    # delivery date, so Console Log always shows the current run only, never
    # scrollback stitched together from earlier same-date attempts.
    _log_fh = open(log_path, "w", encoding="utf-8") if not ns.dry_run else None
    _tee: Optional[_Tee] = None
    if _log_fh:
        # Everything printed or logged for the rest of this run goes through
        # _tee, which always writes to the log file (so the dashboard's
        # Console Log tab stays in sync) and, unless muted by _silence
        # below, also echoes to the real console. try/finally below
        # guarantees sys.stdout is restored even on an unhandled exception —
        # critical for any long-lived interactive Python session (a REPL or
        # notebook kernel) where a stuck sys.stdout would break it afterward.
        _tee = _Tee(sys.stdout, _log_fh)
        sys.stdout = _tee

    try:
        for idx, (key, label, critical) in enumerate(_PHASES):
            yaml_key = _YAML_KEY.get(key, key)

            # Determine skip condition
            if only_keys is not None and key not in only_keys:
                r = dict(key=key, status=SKIP, detail="", elapsed=0.0)
                results.append(r)
                _row(label, SKIP, 0.0, "")
                continue
            if not enabled.get(yaml_key, True):
                r = dict(key=key, status=SKIP, detail="disabled in run.yaml", elapsed=0.0)
                results.append(r)
                _row(label, SKIP, 0.0, "disabled in run.yaml")
                continue
            if idx < from_idx:
                r = dict(key=key, status=SKIP, detail="--from-phase", elapsed=0.0)
                results.append(r)
                _row(label, SKIP, 0.0, "--from-phase")
                continue
            if ns.dry_run:
                r = dict(key=key, status=SKIP, detail="dry-run", elapsed=0.0)
                results.append(r)
                _row(label, SKIP, 0.0, "dry-run")
                continue

            # Run the phase.
            # Auto mode: mute the console side — clean orchestrator table only.
            # Trader mode: let output flow so operator sees bid tables and can
            #              type A/R or press ENTER at each gate.
            # Either way, _tee (set up above) writes everything to the log
            # file, so the dashboard sees the same thing regardless of mode.
            t0 = time.perf_counter()
            try:
                if is_auto and _tee:
                    with _silence(_tee):
                        status, detail = _dispatch(key, date, app_cfg, is_syn, is_auto)
                else:
                    phase_label = f"[ PHASE {label.split()[0]} : {label.split(None,1)[1].strip()} ]"
                    total   = 100 - 2         # fit a standard 100-char terminal width
                    pad     = total - len(phase_label)
                    left_n  = pad // 2
                    right_n = pad - left_n
                    left_stars  = ("* " * (left_n  // 2 + 1))[:left_n]
                    right_stars = (" *" * (right_n // 2 + 1))[:right_n]
                    bar = left_stars + phase_label + right_stars
                    print()
                    print("  " + bar)
                    print()
                    status, detail = _dispatch(key, date, app_cfg, is_syn, is_auto)
                    print()
            except Exception as exc:
                status, detail = FAIL, str(exc)
            elapsed = time.perf_counter() - t0

            r = dict(key=key, status=status, detail=detail, elapsed=elapsed)
            results.append(r)
            _row(label, status, elapsed, detail)

            # Critical failure → abort remaining phases
            if status == FAIL and critical_map[key]:
                print(f"\n  ABORT: critical failure in phase [{key}]\n"
                      f"         {detail}")
                for rk, rl, _ in _PHASES[idx + 1:]:
                    a = dict(key=rk, status=SKIP, detail="aborted", elapsed=0.0)
                    results.append(a)
                    _row(rl, SKIP, 0.0, "aborted")
                break
    finally:
        if _tee:
            sys.stdout = _tee.real_stdout
        if _log_fh:
            _log_fh.close()

    # Structured run status for the dashboard's health banner — the console
    # table above is human-readable only; this is the machine-readable twin,
    # written unconditionally (including dry-run) from data already built
    # above, no new computation. Any earlier file for the same date is
    # overwritten, so it always reflects the most recent run for that date.
    with open(run_status_path, "w", encoding="utf-8") as fh:
        json.dump({"date": date, "mode": mode, "status": "DONE", "results": results,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

    rc = _footer(results, time.perf_counter() - t_start)
    if not ns.dry_run:
        print(f"  Full phase output: runtime/logs/pipeline_{date}.log\n")

    # Auto-generate figures after every successful (or partial) run.
    # Figures overwrite the previous set — figures/output/ always reflects
    # the most recent run. Skipped on --dry-run and on critical abort.
    if not ns.dry_run and rc in (0, 1):
        _generate_figures(date)

    return rc


def _generate_figures(date: str) -> None:
    """Run the figure package; warn but never crash the pipeline."""
    try:
        import figures
        figures.generate(date)
    except Exception as exc:
        import warnings
        warnings.warn(f"[Figures] Generation skipped ({exc})", RuntimeWarning)


if __name__ == "__main__":
    sys.exit(main())
