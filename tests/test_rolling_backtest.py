"""
test_rolling_backtest.py — phase 6E rolling backtest (no solver needed).

The per-day backtest is replaced by a stub, so these tests check only the
rolling logic: which days get (re)run, that the store is updated idempotently,
and that the risk series is realized P&L at real prices, not the forecast
objective.
"""
from __future__ import annotations

import datetime as dt

from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import (
    realised_pnl, summarize_rows,
)
from phase_6_backtesting_and_validation.backtest_engine.rolling_backtest import (
    MAX_NEW_DAYS_PER_RUN, REFRESH_LOOKBACK_DAYS, gates_pending, load_store, plan_days,
    update_rolling_backtest,
)

TODAY = dt.date(2026, 10, 3)
END = TODAY - dt.timedelta(days=1)


def _row(date: str, revenue=100.0, afrr=10.0, mfrr=5.0, objective=999.0, ida3_src="OMIE_LIVE"):
    return {
        "date": date, "feasible": True, "checker_pass": True,
        "objective_eur": objective, "solve_sec": 1.0, "price_mae": 2.0, "price_rmse": 3.0,
        "pv_mae": 0.1, "note": None, "price_actual_source": "OMIE_LIVE", "pv_actual_source": "synthetic",
        "realised_revenue_eur": revenue,
        "realised_price_source": "OMIE_LIVE" if revenue is not None else "unavailable",
        "realised_afrr_capacity_eur": afrr, "realised_mfrr_capacity_eur": mfrr,
        "realised_reserve_price_source": "REN_LIVE",
        "ida1_feasible": True, "realised_ida1_revenue_eur": revenue, "realised_ida1_price_source": "OMIE_LIVE",
        "ida2_feasible": True, "realised_ida2_revenue_eur": revenue, "realised_ida2_price_source": "OMIE_LIVE",
        "ida3_feasible": True, "realised_ida3_revenue_eur": revenue if ida3_src == "OMIE_LIVE" else None,
        "realised_ida3_price_source": ida3_src,
        "xbid_feasible": True, "realised_xbid_revenue_eur": revenue, "realised_xbid_price_source": "OMIE_LIVE",
    }


def _stub(calls):
    def fn(date, cfg):
        calls.append(date)
        return _row(date, revenue=100.0 + len(calls))
    return fn


def test_realised_pnl_sums_real_energy_and_reserve_revenue():
    assert realised_pnl(_row("2026-09-01", revenue=100.0, afrr=10.0, mfrr=None)) == 110.0
    assert realised_pnl(_row("2026-09-01", revenue=None)) is None


def test_risk_series_is_realised_pnl_not_forecast_objective():
    rows = [_row(f"2026-09-{d:02d}", revenue=float(d), afrr=0.0, mfrr=0.0, objective=1e6) for d in range(1, 21)]
    res = summarize_rows(rows)
    assert res.risk is not None
    assert res.risk.mean_pnl_eur == round(sum(range(1, 21)) / 20, 2)   # not 1e6
    assert summarize_rows([_row("2026-09-01", revenue=None)]).risk is None


def test_gates_pending_flags_late_published_gate():
    assert gates_pending(_row("2026-10-01", ida3_src="unavailable"))
    assert not gates_pending(_row("2026-10-01"))


def test_plan_days_empty_store_starts_with_recent_capped_window():
    new, refresh = plan_days({}, END, max_new=MAX_NEW_DAYS_PER_RUN)
    assert len(new) == MAX_NEW_DAYS_PER_RUN and new[-1] == END and refresh == []


def test_plan_days_adds_only_missing_days_and_refreshes_pending():
    store = {d.isoformat(): _row(d.isoformat()) for d in
             (END - dt.timedelta(days=i) for i in range(3, 30))}
    pending = (END - dt.timedelta(days=4)).isoformat()
    store[pending] = _row(pending, ida3_src="unavailable")
    old_pending = (END - dt.timedelta(days=REFRESH_LOOKBACK_DAYS + 5)).isoformat()
    store[old_pending] = _row(old_pending, ida3_src="unavailable")
    new, refresh = plan_days(store, END, max_new=7)
    assert new == [END - dt.timedelta(days=i) for i in (2, 1, 0)]
    assert [d.isoformat() for d in refresh] == [pending]        # too-old pending day is left alone


def test_update_is_incremental_and_idempotent(tmp_path):
    path = str(tmp_path / "rolling.json")
    calls: list = []
    first = update_rolling_backtest(None, today=TODAY, max_new_days=3, backtest_fn=_stub(calls),
                                    resettle=False, store_path=path, export=False)
    assert first.new_days == [(END - dt.timedelta(days=i)).isoformat() for i in (2, 1, 0)]
    assert first.n_window_days == 3 and first.mean_realised_pnl_eur is not None
    second = update_rolling_backtest(None, today=TODAY, backtest_fn=_stub(calls),
                                     resettle=False, store_path=path, export=False)
    assert second.new_days == [] and second.refreshed_days == []
    assert len(calls) == 3 and len(load_store(path)) == 3
    third = update_rolling_backtest(None, today=TODAY + dt.timedelta(days=2), backtest_fn=_stub(calls),
                                    resettle=False, store_path=path, export=False)
    assert third.new_days == [(END + dt.timedelta(days=i)).isoformat() for i in (1, 2)]


def test_failed_day_is_reported_and_retried_next_run(tmp_path):
    path = str(tmp_path / "rolling.json")

    def boom(date, cfg):
        raise RuntimeError("solver down")

    upd = update_rolling_backtest(None, today=TODAY, max_new_days=2, backtest_fn=boom,
                                  resettle=False, store_path=path, export=False)
    assert len(upd.errors) == 2 and upd.new_days == [] and load_store(path) == {}
    calls: list = []
    again = update_rolling_backtest(None, today=TODAY, max_new_days=2, backtest_fn=_stub(calls),
                                    resettle=False, store_path=path, export=False)
    assert len(again.new_days) == 2
