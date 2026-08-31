"""
run_live_resettlement.py — Phase 6: what would the ACTUAL bid we already
committed have been worth against REAL prices, with no re-solve?

Reads the exact historically-committed position (PositionStore) and
reserve capacity (ReserveStore) for a delivery date, values it two ways --
at each gate's own forecast/bid price (the same figure Trading Desk's
Summary_KPIs already shows) and at the real archived settled price -- and
reports both side by side. See
backtest_engine.live_bid_resettlement.resettle_live_bid for the exact
methodology and its disclosed gaps (no real activation price source, no
real imbalance price source).

Command line:
    python phase_6_backtesting_and_validation/run_live_resettlement.py --date 2026-08-21
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_layer.utilities import get_logger
from phase_6_backtesting_and_validation.backtest_engine.live_bid_resettlement import (
    resettle_live_bid,
)
from phase_6_backtesting_and_validation.backtest_excel_reports.backtest_report_exporter import (
    export_live_resettlement,
)

log = get_logger("phase6.live_resettlement")

DEFAULT_DATE = "2026-08-21"


def main():
    p = argparse.ArgumentParser(description="Live-bid real-price re-settlement (no re-solve)")
    p.add_argument("--date", default=DEFAULT_DATE, help="delivery date YYYY-MM-DD")
    p.add_argument("--no-excel", action="store_true")
    args = p.parse_args()

    res = resettle_live_bid(args.date)

    print("\n" + "=" * 78)
    print(f"  Live-bid real-price re-settlement — {args.date}")
    print("  (actual committed position, no re-solve)")
    print("=" * 78)
    print(f"  {'Gate':<8} {'Committed MWh':>14} {'Src':>10} "
          f"{'Forecast EUR':>14} {'Real EUR':>14}")
    print("  " + "-" * 74)
    for g in res.gates:
        real_str = f"{g.realised_revenue_eur:>14,.0f}" if g.realised_revenue_eur is not None else f"{'--':>14}"
        print(f"  {g.gate:<8} {g.committed_mwh:>14,.1f} {g.price_source:>10} "
              f"{g.forecast_revenue_eur:>14,.0f} {real_str}")
    print("  " + "-" * 74)
    print(f"  Gates with real price coverage: {res.n_gates_real}/{res.n_gates_total}")
    print(f"  Total forecast-valued revenue (matches Trading Desk): {res.total_forecast_revenue_eur:>14,.0f} EUR")
    if res.total_realised_revenue_eur is not None:
        delta = res.total_realised_revenue_eur - res.total_forecast_revenue_eur
        pct = (delta / res.total_forecast_revenue_eur * 100.0) if res.total_forecast_revenue_eur else float("nan")
        print(f"  Total real-price-valued revenue:                      {res.total_realised_revenue_eur:>14,.0f} EUR")
        print(f"  Delta (real - forecast):                              {delta:>14,.0f} EUR  ({pct:+.1f}%)")
    else:
        print("  Total real-price-valued revenue: unavailable (partial real-price coverage across gates)")

    if res.reserves:
        print("\n  RESERVE CAPACITY")
        print("  " + "-" * 74)
        for r in res.reserves:
            real_str = (f"{r.realised_capacity_revenue_eur:>14,.0f}"
                        if r.realised_capacity_revenue_eur is not None else f"{'--':>14}")
            print(f"  {r.product:<8} {r.committed_capacity_mwh:>14,.1f} {r.price_source:>10} "
                  f"{r.forecast_capacity_revenue_eur:>14,.0f} {real_str}")
        print("  (activation revenue excluded — no real activation price source exists)")
    print("=" * 78)

    if not args.no_excel:
        try:
            path = export_live_resettlement(args.date, res)
            print(f"\n  Excel report: {path}")
        except Exception as exc:
            log.error(f"Excel export failed: {exc}")


if __name__ == "__main__":
    main()
