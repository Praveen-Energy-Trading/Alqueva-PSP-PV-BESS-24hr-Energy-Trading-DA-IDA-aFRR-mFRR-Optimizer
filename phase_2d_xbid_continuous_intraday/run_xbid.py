"""
run_xbid.py — Phase 2D XBID continuous intraday gate.

Evaluates the still-open hours at a check window and places capped opportunistic
orders when the price beats the spread. Windows are config-driven (see
config/market.yaml gates.XBID.check_windows) — six by default, spread across
D-1 evening through D midday, approximating the real continuous market with
more checkpoints than a two-window demo:
    W1  D-1 18:30   W2  D-1 22:30   W3  D 03:00
    W4  D 06:00     W5  D 09:30     W6  D 12:00
Run DA (and IDAs) first so a committed baseline exists.

    python phase_2d_xbid_continuous_intraday/run_xbid.py --date 2026-06-26 --window W1
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_layer.configuration import load_config
from phase_2d_xbid_continuous_intraday.xbid_milp_optimiser.xbid_optimiser import optimise_xbid
from phase_2d_xbid_continuous_intraday.xbid_bid_formatting.xbid_bid_formatter import (
    format_xbid_orders,
    to_xbid_payload,
    render_table,
)


def main():
    p = argparse.ArgumentParser(description="Run the Phase 2D XBID gate")
    p.add_argument("--date", required=True, help="delivery date YYYY-MM-DD")
    p.add_argument("--window", default="W1", choices=["W1", "W2", "W3", "W4", "W5", "W6"],
                   help="check window (see config/market.yaml gates.XBID.check_windows)")
    p.add_argument("--config", default=None)
    p.add_argument("--no-pause", action="store_true")
    p.add_argument("--real-data", action="store_true", help="use live OMIE training-data backfill")
    args = p.parse_args()

    result = optimise_xbid(args.date, load_config(args.config),
                           window=args.window, no_pause=args.no_pause,
                           use_synthetic=not args.real_data)
    status = result.get("status")
    print(f"\n  XBID {args.window} gate result: {status}")

    if status == "SUBMITTED":
        orders = format_xbid_orders(
            committed=result["committed_net_mw"],
            new_net=result["new_net_mw"],
            xbid_prices=result["xbid_prices"],
            open_hours=result["open_hours"],
            window=args.window,
        )
        print()
        print(render_table(orders, args.window))
        payload = to_xbid_payload(orders, args.date, args.window)
        print(f"\n  XBID payload: {len(payload['orders'])} orders  "
              f"net impact {payload['net_revenue_impact_eur']:+.2f} EUR")
    else:
        for k, v in result.items():
            if k != "status":
                print(f"    {k}: {v}")

    sys.exit(0 if status in ("SUBMITTED", "NO_CHANGE") else 1)


if __name__ == "__main__":
    main()
