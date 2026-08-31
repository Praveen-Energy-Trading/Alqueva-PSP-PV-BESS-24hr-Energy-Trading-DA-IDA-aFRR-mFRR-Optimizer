"""
run_risk_comparison.py — Phase 6: does the CVaR-averse DA objective actually
help, on real prices?

Solves the DA gate TWICE per real-price day (expected-value vs CVaR-averse
risk measure) on the IDENTICAL real scenario fan, values both resulting
bids against the REAL settled DA price, and reports the realized-outcome
risk/return tradeoff -- the first evidence-based answer in this project to
whether CVaR-averse bidding helps, not a theoretical claim about the
formulation alone. See backtest_engine.backtest_runner.run_stochastic_risk_comparison
and core_milp_builder.build_core_model_stochastic for the underlying
Rockafellar-Uryasev CVaR formulation.

Command line:
    python phase_6_backtesting_and_validation/run_risk_comparison.py --start 2026-08-21 --days 2
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_layer.configuration import load_config
from common_layer.utilities import get_logger
from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import (
    run_stochastic_risk_comparison,
)
from phase_6_backtesting_and_validation.backtest_excel_reports.backtest_report_exporter import (
    export_risk_comparison,
)

log = get_logger("phase6.risk_comparison")

DEFAULT_START = "2026-08-21"
DEFAULT_DAYS = 2
DEFAULT_CVAR_ALPHA = 0.90
# 3, not 5: at real 96-ISP DA scale a 5-scenario CVaR MILP failed to reach a
# good solution within a 300 s budget on a real date (confirmed by direct
# inspection -- see run_stochastic_risk_comparison's docstring); 3 scenarios
# solves to genuine optimality in ~3 minutes.
DEFAULT_N_SCENARIOS = 3


def main():
    p = argparse.ArgumentParser(description="EV vs CVaR realized-outcome comparison (DA gate)")
    p.add_argument("--start", default=DEFAULT_START, help="start delivery date YYYY-MM-DD")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--cvar-alpha", type=float, default=DEFAULT_CVAR_ALPHA)
    p.add_argument("--n-scenarios", type=int, default=DEFAULT_N_SCENARIOS)
    p.add_argument("--config", default=None)
    p.add_argument("--no-excel", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    res = run_stochastic_risk_comparison(args.start, args.days, cfg,
                                         cvar_alpha=args.cvar_alpha, n_scenarios=args.n_scenarios)

    print("\n" + "=" * 74)
    print(f"  EV vs CVaR (alpha={args.cvar_alpha})  —  {args.start} x {args.days} days")
    print("=" * 74)
    print(f"  {'Date':<12} {'Src':>10} {'Feas EV':>8} {'Realised EV':>13} "
          f"{'Feas CVaR':>10} {'Realised CVaR':>14}")
    print("  " + "-" * 70)
    for r in res.rows:
        if r["price_source"] == "unavailable":
            print(f"  {r['date']:<12} {'unavail':>10} {'--':>8} {'--':>13} {'--':>10} {'--':>14}")
            continue
        ev_str = f"{r['realised_ev_eur']:>13,.0f}" if r["realised_ev_eur"] is not None else f"{'--':>13}"
        cvar_str = f"{r['realised_cvar_eur']:>14,.0f}" if r["realised_cvar_eur"] is not None else f"{'--':>14}"
        print(f"  {r['date']:<12} {'real':>10} {str(r['feasible_ev']):>8} {ev_str} "
              f"{str(r['feasible_cvar']):>10} {cvar_str}")
    print("  " + "-" * 70)
    print(f"  Real-price days used: {res.n_real_days}/{res.n_days}")

    if res.mean_realised_ev_eur is not None:
        print(f"\n  Mean realised revenue, expected-value strategy: {res.mean_realised_ev_eur:>14,.0f} EUR")
        print(f"  Mean realised revenue, CVaR-averse strategy:    {res.mean_realised_cvar_eur:>14,.0f} EUR")
        print(f"  Cost of risk aversion (EV mean - CVaR mean):    {res.cost_of_risk_aversion_eur:>14,.0f} EUR")
        print(f"  Tail improvement (CVaR's realized CVaR95 "
              f"- EV's realized CVaR95): {res.tail_improvement_eur:>14,.0f} EUR")

        print("\n  REALIZED-OUTCOME RISK METRICS")
        print("  " + "-" * 70)
        print(f"  {'Metric':<32} {'EV':>17} {'CVaR':>17}")
        rows = [
            ("Mean daily P&L (EUR)", res.risk_ev.mean_pnl_eur, res.risk_cvar.mean_pnl_eur),
            ("Std daily P&L (EUR)", res.risk_ev.std_pnl_eur, res.risk_cvar.std_pnl_eur),
            ("VaR(95%) (EUR)", res.risk_ev.var_95_eur, res.risk_cvar.var_95_eur),
            ("CVaR(95%) (EUR)", res.risk_ev.cvar_95_eur, res.risk_cvar.cvar_95_eur),
            ("Sharpe ratio (annualised)", res.risk_ev.sharpe_ratio, res.risk_cvar.sharpe_ratio),
            ("Max drawdown (EUR)", res.risk_ev.max_drawdown_eur, res.risk_cvar.max_drawdown_eur),
        ]
        for label, ev_val, cvar_val in rows:
            print(f"  {label:<32} {ev_val:>17,.2f} {cvar_val:>17,.2f}")
    else:
        print("\n  No real-price days available in this range -- comparison unavailable.")
    print("=" * 74)

    if not args.no_excel and res.mean_realised_ev_eur is not None:
        try:
            path = export_risk_comparison(args.start, res)
            print(f"\n  Excel report: {path}")
        except Exception as exc:
            log.error(f"Excel export failed: {exc}")


if __name__ == "__main__":
    main()
