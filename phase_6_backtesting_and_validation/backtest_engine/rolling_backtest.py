"""
rolling_backtest.py — keep a rolling 1-year real-price backtest current.

Called by run_production.py as phase 6E after every pipeline run (and seeded by
run_backtest.py), so the backtest and its Profit-at-Risk always cover the
latest year without re-running it from scratch:

  * New days:     every completed day after the last stored one, up to
                  yesterday (Portugal). The delivery date itself is never
                  backtested -- its real prices do not exist yet. At most
                  MAX_NEW_DAYS_PER_RUN days per call so a long gap (or an
                  empty store) never stalls the trading pipeline; later runs
                  continue where this one stopped. Fill a whole year at once
                  with run_backtest.py, which seeds this store.
  * Late data:    stored days from the last REFRESH_LOOKBACK_DAYS whose price
                  for a gate that did run was still "unavailable" (OMIE
                  publishes IDA3/XBID with a delay) are backtested again.
  * Actual bids:  for each processed day the pipeline actually bid on, the
                  committed positions are also re-valued at real prices
                  (live_bid_resettlement) and exported.

Per-day rows live in runtime/backtest/rolling_backtest.json; the last
WINDOW_DAYS of them are summarised (summarize_rows: realized-P&L Profit-at-Risk)
into runtime/reports/backtest_rolling_365d.xlsx, which the dashboard reads.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from common_layer.utilities import date_utils as du
from common_layer.utilities import get_logger

log = get_logger("phase6.rolling_backtest")

WINDOW_DAYS = 365              # industry-standard 1-year backtest window
REFRESH_LOOKBACK_DAYS = 7      # re-check recent days whose prices arrived late
MAX_NEW_DAYS_PER_RUN = 7       # cap per pipeline run (~70 s per day)
ROLLING_REPORT = "backtest_rolling_365d.xlsx"

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STORE = os.path.join(_REPO, "runtime", "backtest", "rolling_backtest.json")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def load_store(path: str = _STORE) -> Dict[str, dict]:
    """{ 'YYYY-MM-DD': backtest row } -- empty when nothing is stored yet."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_store(store: Dict[str, dict], path: str = _STORE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(store.items())), fh, default=_json_default)
    os.replace(tmp, path)        # atomic: an interrupted run never corrupts the store


def _json_default(o):
    # numpy scalars and similar from the MILP KPIs
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def upsert_rows(rows: List[dict], path: str = _STORE) -> int:
    """Insert or replace backtest rows by date (used by run_backtest.py)."""
    store = load_store(path)
    for r in rows:
        store[r["date"]] = r
    save_store(store, path)
    return len(rows)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

_INTRADAY = ("ida1", "ida2", "ida3", "xbid")


def gates_pending(row: dict) -> bool:
    """True if a gate that ran on this day still lacks its real price."""
    if not row.get("feasible"):
        return False
    if row.get("realised_price_source") == "unavailable":
        return True
    if row.get("realised_reserve_price_source") == "unavailable":
        return True
    return any(row.get(f"{g}_feasible") is True
               and row.get(f"realised_{g}_price_source") == "unavailable"
               for g in _INTRADAY)


def plan_days(store: Dict[str, dict], end: dt.date,
              max_new: int = MAX_NEW_DAYS_PER_RUN) -> Tuple[List[dt.date], List[dt.date]]:
    """Return (new_days, refresh_days) to backtest so the store reaches `end`.

    New days run oldest-first so the series stays contiguous; anything over
    `max_new` is left for the next call.
    """
    stored = sorted(dt.date.fromisoformat(d) for d in store)
    stored = [d for d in stored if d <= end]
    first_new = stored[-1] + dt.timedelta(days=1) if stored else end - dt.timedelta(days=max_new - 1)
    new_days = [first_new + dt.timedelta(days=i) for i in range((end - first_new).days + 1)][:max_new]
    since = end - dt.timedelta(days=REFRESH_LOOKBACK_DAYS)
    refresh = [d for d in stored
               if d > since and d not in new_days and gates_pending(store[d.isoformat()])]
    return new_days, refresh


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

@dataclass
class RollingUpdate:
    new_days: List[str] = field(default_factory=list)
    refreshed_days: List[str] = field(default_factory=list)
    resettled_days: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    window_first: Optional[str] = None
    window_last: Optional[str] = None
    n_window_days: int = 0
    n_days_behind: int = 0        # completed days still missing after this call
    report_path: Optional[str] = None
    mean_realised_pnl_eur: Optional[float] = None
    par_95_eur: Optional[float] = None     # Profit-at-Risk: VaR(95%) of realized daily P&L
    cvar_95_eur: Optional[float] = None


def _resettle(date: str, cfg) -> bool:
    """Re-value the bids the pipeline actually committed for `date` at real
    prices; True if there was anything to re-value."""
    from phase_6_backtesting_and_validation.backtest_engine.live_bid_resettlement import resettle_live_bid
    from phase_6_backtesting_and_validation.backtest_excel_reports.backtest_report_exporter import (
        export_live_resettlement,
    )
    res = resettle_live_bid(date, cfg)
    if not res.gates and not res.reserves:
        return False                # the pipeline never bid for this day
    export_live_resettlement(date, res)
    return True


def update_rolling_backtest(cfg, today: Optional[dt.date] = None,
                           max_new_days: int = MAX_NEW_DAYS_PER_RUN,
                           backtest_fn: Optional[Callable[[str, object], dict]] = None,
                           resettle: bool = True,
                           store_path: str = _STORE,
                           export: bool = True,
                           refresh: bool = True) -> RollingUpdate:
    """Bring the rolling backtest up to yesterday and rewrite its report."""
    from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import (
        backtest_one_day, summarize_rows,
    )

    today = today or du.portugal_today()
    end = today - dt.timedelta(days=1)
    backtest_fn = backtest_fn or backtest_one_day
    out = RollingUpdate()

    store = load_store(store_path)
    new_days, refresh_days = plan_days(store, end, max_new_days)
    if not refresh:
        refresh_days = []

    for day in refresh_days + new_days:
        iso = day.isoformat()
        try:
            store[iso] = backtest_fn(iso, cfg)
            save_store(store, store_path)          # keep progress if interrupted
            (out.refreshed_days if day in refresh_days else out.new_days).append(iso)
        except Exception as exc:                    # one bad day must not stop the rest
            out.errors.append(f"{iso}: {exc}")
            log.warning(f"rolling backtest {iso} failed: {exc}")
            continue
        if resettle:
            try:
                if _resettle(iso, cfg):
                    out.resettled_days.append(iso)
            except Exception as exc:
                out.errors.append(f"{iso} re-settlement: {exc}")
                log.warning(f"live-bid re-settlement {iso} failed: {exc}")

    first = end - dt.timedelta(days=WINDOW_DAYS - 1)
    window = sorted(d for d in store if first <= dt.date.fromisoformat(d) <= end)
    out.n_window_days = len(window)
    last_stored = max((dt.date.fromisoformat(d) for d in store), default=None)
    out.n_days_behind = (end - last_stored).days if last_stored else (end - first).days + 1
    if window:
        out.window_first, out.window_last = window[0], window[-1]
        result = summarize_rows([store[d] for d in window])
        if result.risk is not None:
            out.mean_realised_pnl_eur = result.risk.mean_pnl_eur
            out.par_95_eur = result.risk.var_95_eur
            out.cvar_95_eur = result.risk.cvar_95_eur
        if export:
            from phase_6_backtesting_and_validation.backtest_excel_reports.backtest_report_exporter import (
                export_backtest,
            )
            out.report_path = export_backtest(
                window[0], result, filename=ROLLING_REPORT,
                title=f"Rolling backtest — last {len(window)} days ({window[0]} to {window[-1]})")
    return out


def run_rolling_backtest(cfg) -> dict:
    """Pipeline entry point (phase 6E). Never raises: problems become a WARN."""
    try:
        upd = update_rolling_backtest(cfg)
    except Exception as exc:
        log.warning(f"rolling backtest failed: {exc}")
        return {"status": "BACKTEST_INCOMPLETE", "reason": str(exc)}
    if upd.errors:
        return {"status": "BACKTEST_INCOMPLETE", "reason": "; ".join(upd.errors)[:200]}
    if not upd.new_days and not upd.refreshed_days:
        return {"status": "NO_CHANGE", "ref": f"up to date ({upd.n_window_days}d window)"}
    behind = f", {upd.n_days_behind}d behind" if upd.n_days_behind else ""
    return {"status": "OK",
            "ref": f"+{len(upd.new_days)} new, {len(upd.refreshed_days)} refreshed, "
                   f"{upd.n_window_days}d window{behind}"}
