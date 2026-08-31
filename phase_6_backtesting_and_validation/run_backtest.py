"""
run_backtest.py — Phase 6 backtest over a span of delivery days.

Command line:
    python phase_6_backtesting_and_validation/run_backtest.py --start 2026-06-01 --days 7

Or just edit DEFAULT_START / DEFAULT_DAYS below and hit Run (F5) / %runfile with
no arguments — no command-line args needed for a quick local run.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_layer.configuration import load_config
from common_layer.utilities import get_logger, AuditLogger
from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import run_backtest
from phase_6_backtesting_and_validation.backtest_excel_reports.backtest_report_exporter import (
    export_backtest,
)

log = get_logger("phase6.backtest")

# ---------------------------------------------------------------------------
# Edit these two lines to change the backtest window when running with no
# command-line arguments (e.g. VS Code's Run button, F5).
# Command-line --start/--days, if given, always override these.
# ---------------------------------------------------------------------------
DEFAULT_START = "2026-06-01"
DEFAULT_DAYS = 7


def main():
    p = argparse.ArgumentParser(description="Run Phase 6 backtest")
    p.add_argument("--start", default=DEFAULT_START, help="start delivery date YYYY-MM-DD")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--config", default=None)
    p.add_argument("--no-excel", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    audit = AuditLogger()
    res = run_backtest(args.start, args.days, cfg)

    print("\n" + "=" * 64)
    print(f"  BACKTEST  —  {args.start}  x {args.days} days")
    print("=" * 64)
    print(f"  {'Date':<12} {'Feas':>5} {'Chk':>4} {'Objective':>12} "
          f"{'Realised':>12} {'Src':>4} "
          f"{'Solve s':>8} {'PxMAE':>7} {'PVMAE':>7}")
    print("  " + "-" * 64)
    for r in res.rows:
        realised_str = (f"{r['realised_revenue_eur']:>12,.0f}"
                        if r["realised_revenue_eur"] is not None else f"{'--':>12}")
        src_str = "real" if r["realised_price_source"] == "OMIE_LIVE" else "synt"
        print(f"  {r['date']:<12} {str(r['feasible']):>5} {str(r['checker_pass']):>4} "
              f"{r['objective_eur']:>12,.0f} {realised_str} {src_str:>4} "
              f"{r['solve_sec']:>8.3f} "
              f"{r['price_mae']:>7.2f} {r['pv_mae']:>7.3f}")
    print("  " + "-" * 64)
    print(f"  Feasible: {res.n_feasible}/{res.n_days}   "
          f"Checker pass: {res.n_checker_pass}/{res.n_days}   "
          f"Real-price days: {res.n_real_price_days}/{res.n_days}")
    print(f"  Avg objective (forecast-based):  {res.avg_objective_eur:,.0f} EUR   "
          f"Avg solve: {res.avg_solve_sec:.3f} s")
    if res.avg_realised_revenue_eur is not None:
        print(f"  Avg realised revenue (real OMIE price, {res.n_real_price_days} day(s)): "
              f"{res.avg_realised_revenue_eur:,.0f} EUR")
    else:
        print(f"  Avg realised revenue: unavailable (no real OMIE price coverage "
              f"in this date range)")
    print(f"  Avg price MAE: {res.avg_price_mae:.2f} EUR/MWh   "
          f"Avg PV MAE: {res.avg_pv_mae:.4f} MW  (PV actuals always synthetic — "
          f"no real generation-telemetry source exists in this project)")
    print(f"  Real-aFRR-capacity days: {res.n_real_afrr_days}/{res.n_days}   "
          f"Real-mFRR-capacity days: {res.n_real_mfrr_days}/{res.n_days}")
    if res.avg_realised_afrr_capacity_eur is not None:
        print(f"  Avg realised aFRR capacity revenue (real REN price, "
              f"{res.n_real_afrr_days} day(s)): {res.avg_realised_afrr_capacity_eur:,.0f} EUR")
    else:
        print(f"  Avg realised aFRR capacity revenue: unavailable "
              f"(no real REN aFRR price coverage in this date range)")
    if res.avg_realised_mfrr_capacity_eur is not None:
        print(f"  Avg realised mFRR capacity revenue (real REN price, "
              f"{res.n_real_mfrr_days} day(s)): {res.avg_realised_mfrr_capacity_eur:,.0f} EUR")
    else:
        print(f"  Avg realised mFRR capacity revenue: unavailable "
              f"(no real REN mFRR price coverage, or aFRR real price unavailable "
              f"that date — mFRR sizing nets against aFRR's committed MW)")
    for gate_label, n_days_gate, avg_val in [
        ("IDA1", res.n_real_ida1_price_days, res.avg_realised_ida1_revenue_eur),
        ("IDA2", res.n_real_ida2_price_days, res.avg_realised_ida2_revenue_eur),
        ("IDA3", res.n_real_ida3_price_days, res.avg_realised_ida3_revenue_eur),
        ("XBID (W1)", res.n_real_xbid_price_days, res.avg_realised_xbid_revenue_eur),
    ]:
        print(f"  Real-{gate_label}-price days: {n_days_gate}/{res.n_days}")
        if avg_val is not None:
            print(f"  Avg realised {gate_label} revenue (real OMIE price, "
                  f"{n_days_gate} day(s)): {avg_val:,.0f} EUR")
        else:
            print(f"  Avg realised {gate_label} revenue: unavailable "
                  f"(no real archived {gate_label} clearing price coverage in this date "
                  f"range, or the prior gate in the chain was infeasible)")
    print(f"  NOTE: activation/imbalance revenue is OUT OF SCOPE — the activated MW is "
          f"always internally-simulated (no real REN/SCADA telemetry loader exists in "
          f"this project), so it can never be priced with the same real-price x real-"
          f"quantity honesty standard as DA energy or reserve capacity above. IDA3 has "
          f"only 7 real archived dates today (coverage from 2026-08-15) — genuinely thin, "
          f"reported as such rather than padded. XBID is backtested at window W1 only "
          f"(production supports 6 continuous-intraday check windows).")
    print("=" * 64)

    if res.risk is not None:
        rm = res.risk
        print("\n  PORTFOLIO RISK METRICS")
        print("  " + "-" * 62)
        print(f"  {'Mean daily P&L':<38} {rm.mean_pnl_eur:>14,.0f} EUR")
        print(f"  {'Std daily P&L':<38} {rm.std_pnl_eur:>14,.0f} EUR")
        print(f"  {'Min / Max daily P&L':<38} {rm.min_pnl_eur:>14,.0f} / "
              f"{rm.max_pnl_eur:,.0f} EUR")
        print()
        print(f"  {'VaR(95%)  historical [5th pct]':<38} {rm.var_95_eur:>14,.0f} EUR")
        print(f"  {'CVaR(95%) Expected Shortfall':<38} {rm.cvar_95_eur:>14,.0f} EUR")
        print(f"  {'VaR(99%)  historical [1st pct]':<38} {rm.var_99_eur:>14,.0f} EUR")
        print(f"  {'CVaR(99%) Expected Shortfall':<38} {rm.cvar_99_eur:>14,.0f} EUR")
        print()
        print(f"  Monte Carlo bootstrap (n=10,000, alpha=95%)")
        print(f"  {'  VaR(95%)  mean ± std':<38} {rm.var_95_mean:>14,.0f} ± "
              f"{rm.var_95_std:,.0f} EUR")
        print(f"  {'  CVaR(95%) mean ± std':<38} {rm.cvar_95_mean:>14,.0f} ± "
              f"{rm.cvar_95_std:,.0f} EUR")
        print()
        print(f"  {'Sharpe ratio (annualised, rf=0)':<38} {rm.sharpe_ratio:>14.4f}")
        print(f"  {'Max drawdown':<38} {rm.max_drawdown_eur:>14,.0f} EUR")
        print("  " + "-" * 62)

    if not args.no_excel:
        try:
            path = export_backtest(args.start, res)
            print(f"\n  Excel report: {path}")
        except Exception as exc:
            log.error(f"Excel export failed: {exc}")

    audit.log("BACKTEST_DONE", days=res.n_days, feasible=res.n_feasible,
              checker_pass=res.n_checker_pass)
    sys.exit(0 if res.n_feasible == res.n_days == res.n_checker_pass else 1)


if __name__ == "__main__":
    main()
