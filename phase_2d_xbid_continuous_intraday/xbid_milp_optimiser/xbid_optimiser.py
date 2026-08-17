"""
xbid_optimiser.py — continuous-intraday opportunistic re-optimisation.

XBID differs from the IDA auctions in two ways that matter to the model:
  * trades are small and continuous — each check window may only move a position
    by up to a per-order MW cap (config xbid_max_volume_per_order_mw),
  * only hours that are still open (delivery >1h away) can be traded.

Mechanics: re-solve the shared 24h MILP under XBID proxy prices, with
  * hours outside the open window frozen to the committed net, and
  * a trade-band constraint on each open hour:
        committed[h] - cap  <=  p_net[h]  <=  committed[h] + cap
so the optimiser can only nudge the position within the per-order cap. Orders are
placed only if the average gain beats the bid-ask spread threshold (no churn).
"""
from __future__ import annotations

from typing import Dict, List

import pyomo.environ as pyo

from common_layer.configuration.config_loader import AppConfig
from common_layer.utilities import get_logger, AuditLogger
from common_layer.utilities import date_utils as du
from common_layer.database import PositionStore, ReserveStore, ComponentStore, validate_inputs, SchemaError
from common_layer.optimisation_model.core_milp_builder import build_core_model
from common_layer.optimisation_model.core_milp_solver import (
    solve_core_model, extract_results, SolveError,
)
from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.pv_power_forecaster import (
    forecast_pv_available_isp,
)
from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.reservoir_inflow_forecaster import (
    forecast_inflow,
)
from phase_1_da_day_ahead_bidding.da_bid_formatting.da_bid_checker import (
    check_da_bid, BidCheckError,
)
from phase_1_da_day_ahead_bidding.risk_and_bid_validation.pre_trade_risk_checker import (
    PreTradeRiskChecker,
)
from phase_2d_xbid_continuous_intraday.xbid_price_forecasting.xbid_price_loader import (
    fetch_xbid_prices_isp, tradable_hours_for_window,
)

log = get_logger("phase2.xbid")


def _pause(message: str, no_pause: bool) -> None:
    if no_pause:
        return
    try:
        input(f"\n  {message}  [ENTER to continue] ")
    except (EOFError, KeyboardInterrupt):
        pass


def optimise_xbid(delivery_date: str, cfg: AppConfig, window: str = "W1",
                  no_pause: bool = False, use_synthetic: bool = True) -> dict:
    audit = AuditLogger()
    audit.log("XBID_START", delivery_date=delivery_date, window=window)
    store = PositionStore()
    day = du.parse_date(delivery_date)
    dt = du.isp_duration_min(day) / 60.0
    th = cfg.market.trading_thresholds
    cap = th.xbid_max_volume_per_order_mw

    committed = store.committed_position(delivery_date)   # net across all prior gates
    if not committed:
        msg = "[XBID] no committed baseline; run DA (and IDAs) first."
        log.error(msg)
        return {"status": "NO_BASELINE", "reason": msg}

    all_isps = du.delivery_isps(day)
    all_hours = du.delivery_hours(day)
    windows = {w["id"]: w["trigger"] for w in cfg.market.gate("XBID").check_windows}
    if window not in windows:
        return {"status": "SCHEMA_FAILED", "reason": f"Unknown XBID window '{window}'"}
    open_hours = tradable_hours_for_window(all_isps, delivery_date, windows[window],
                                           period_duration_min=du.isp_duration_min(day))

    inflow_hourly = forecast_inflow(all_hours, delivery_date, cfg.plant.reservoir)
    inflow = {isp: inflow_hourly[h] for h in all_hours for isp in du.hour_to_isps(h, day)}

    inputs = {
        "delivery_date": delivery_date, "hours": all_isps, "dt_h": dt,
        "da_prices": fetch_xbid_prices_isp(all_isps, all_hours, day, delivery_date,
                                           window, use_synthetic),
        "pv_available_mw": forecast_pv_available_isp(all_isps, delivery_date, cfg.plant.pv),
        "inflow_m3h": inflow,
        "initial_state": ComponentStore().load_chained_initial_state(
            delivery_date, cfg.plant.bess.capacity_mwh,
            default_state={
                "upper_reservoir_hm3": cfg.plant.initial_state.upper_reservoir_hm3,
                "lower_reservoir_hm3": cfg.plant.initial_state.lower_reservoir_hm3,
                "bess_soc_frac": cfg.plant.initial_state.bess_soc_frac,
            },
            reservoir_bounds={
                "upper_min_hm3": cfg.plant.reservoir.upper_min_hm3,
                "upper_usable_hm3": cfg.plant.reservoir.upper_usable_hm3,
                "lower_min_hm3": cfg.plant.reservoir.lower_min_hm3,
                "lower_capacity_hm3": cfg.plant.reservoir.lower_capacity_hm3,
            },
        ),
    }
    try:
        validate_inputs(inputs, cfg)
    except SchemaError as e:
        audit.log("XBID_SCHEMA_FAILED", delivery_date=delivery_date, window=window, reason=str(e))
        return {"status": "SCHEMA_FAILED", "reason": str(e)}

    # Reserve capacity already committed at its (earlier) real gate time —
    # same envelope tightening as DA/IDA (see run_da.py).
    rstore = ReserveStore()
    reserved_up = dict(rstore.reserved_up(delivery_date, "aFRR"))
    reserved_dn = dict(rstore.reserved_dn(delivery_date, "aFRR"))
    for h, v in rstore.reserved_up(delivery_date, "mFRR").items():
        reserved_up[h] = reserved_up.get(h, 0.0) + v
    for h, v in rstore.reserved_dn(delivery_date, "mFRR").items():
        reserved_dn[h] = reserved_dn.get(h, 0.0) + v

    # Freeze closed hours to committed; build, then add the per-order trade band.
    fixed_net = {isp: committed.get(isp, 0.0) for isp in all_isps if isp not in open_hours}
    try:
        model, meta = build_core_model(inputs, cfg, fixed_net_position=fixed_net,
                                        reserved_up_mw=reserved_up,
                                        reserved_dn_mw=reserved_dn)
        model.xbid_band_hi = pyo.Constraint(
            open_hours, rule=lambda mm, h: mm.p_net[h] <= committed.get(h, 0.0) + cap)
        model.xbid_band_lo = pyo.Constraint(
            open_hours, rule=lambda mm, h: mm.p_net[h] >= committed.get(h, 0.0) - cap)
        solve_time = solve_core_model(model, cfg, gate="XBID")
    except SolveError as e:
        audit.log("XBID_SOLVE_FAILED", delivery_date=delivery_date, window=window, reason=str(e))
        return {"status": "SOLVE_FAILED", "reason": str(e)}

    results = extract_results(model, meta)
    new_net = results.net_position_mw
    price = inputs["da_prices"]

    deltas = {h: (new_net[h] - committed.get(h, 0.0)) for h in open_hours}
    one_way_vol = 0.5 * sum(abs(d) * dt for d in deltas.values())
    improvement = sum(price[h] * (new_net[h] - committed.get(h, 0.0)) * dt for h in open_hours)
    material = {h: d for h, d in deltas.items() if abs(d * dt) >= th.ida_min_delta_mwh}

    # XBID order test: average gain per repositioned MWh must beat the spread.
    if not material or improvement < th.xbid_min_spread_eur_mwh * max(one_way_vol, 1e-9):
        log.info(f"XBID[{window}]: NO_ORDER — gain {improvement:,.0f} EUR over "
                 f"{one_way_vol:.1f} MWh below spread {th.xbid_min_spread_eur_mwh} EUR/MWh")
        audit.log("XBID_NO_ORDER", delivery_date=delivery_date, window=window, improvement_eur=improvement,
                  one_way_vol_mwh=one_way_vol,
                  dynamic_threshold_eur=th.xbid_min_spread_eur_mwh * max(one_way_vol, 1e-9),
                  tradable_hours=sorted(open_hours), n_total_hours=len(all_isps))
        _print_xbid(window, price, committed, new_net, material, improvement, "NO_ORDER")
        _pause(f"XBID {window}: no opportunity beats the spread — no order.", no_pause)
        return {"status": "NO_CHANGE", "improvement_eur": improvement}

    try:
        check_da_bid(results, inputs, cfg, gate="XBID")
    except BidCheckError as e:
        audit.log("XBID_BIDCHECK_FAILED", delivery_date=delivery_date, window=window, reason=str(e))
        return {"status": "BID_CHECK_FAILED", "reason": str(e)}
    risk = PreTradeRiskChecker(cfg).check(results, dt_h=dt)
    if not risk.passed:
        audit.log("XBID_RISK_BLOCKED", delivery_date=delivery_date, window=window, violations=risk.violations)
        return {"status": "RISK_BLOCKED", "violations": risk.violations}

    _print_xbid(window, price, committed, new_net, material, improvement, "PLACE ORDERS")
    _pause(f"XBID {window}: {len(material)} order(s), +{improvement:,.0f} EUR.", no_pause)

    ref = f"XBID-{delivery_date.replace('-', '')}-{window}"
    position = {h: {"volume_mwh": new_net[h] * dt, "price_eur_mwh": price[h]} for h in open_hours}
    store.save_position(delivery_date, "XBID", position)
    audit.log("XBID_ORDERS_PLACED", delivery_date=delivery_date, window=window, ref=ref,
              n_orders=len(material), improvement_eur=improvement,
              dynamic_threshold_eur=th.xbid_min_spread_eur_mwh * max(one_way_vol, 1e-9),
              tradable_hours=sorted(open_hours), n_total_hours=len(all_isps),
              traded_hours=sorted(material.keys()))
    log.info(f"XBID[{window}] placed {len(material)} order(s) ref {ref}")
    return {"status": "SUBMITTED", "ref": ref, "n_orders": len(material),
            "improvement_eur": improvement, "solve_time_sec": solve_time,
            "committed_net_mw": {h: committed.get(h, 0.0) for h in open_hours},
            "new_net_mw"      : {h: float(new_net[h]) for h in open_hours},
            "xbid_prices"     : {h: price[h] for h in open_hours},
            "open_hours"      : list(open_hours)}


def _print_xbid(window: str, price: Dict[int, float], committed: Dict[int, float],
                new_net: Dict[int, float], material: Dict[int, float],
                improvement: float, decision: str) -> None:
    print("\n" + "=" * 62)
    print(f"  XBID CONTINUOUS  window {window}  —  decision: {decision}")
    print("=" * 62)
    print(f"  Expected gain: {improvement:>10,.2f} EUR   (orders capped per hour)")
    print("  Gate closes: continuous — each hour closes 1h before its own delivery,"
          " not one fixed time")
    if material:
        print(f"\n  {'Hour':<5} {'Price':>7} {'Committed':>11} {'New':>9} {'Order MWh':>11}")
        print("  " + "-" * 48)
        for h in sorted(material):
            print(f"  H{h:02d}  {price[h]:>7.1f} {committed.get(h,0.0):>+11.1f} "
                  f"{new_net[h]:>+9.1f} {material[h]:>+11.2f}")
    print("=" * 62)
