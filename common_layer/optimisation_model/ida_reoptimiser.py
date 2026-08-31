"""
ida_reoptimiser.py — shared intraday re-optimisation engine for IDA1/2/3.

All three intraday auctions do the same thing; they differ only in:
  * which gate's schedule is the baseline (DA -> IDA1 -> IDA2 -> IDA3),
  * the tradable hour window (IDA3 = hours 12-24 ONLY; hours 1-11 are frozen),
  * the updated intraday price curve (new information closer to delivery).

The engine:
  1. loads the committed baseline (the previous gate's net schedule),
  2. builds intraday inputs (updated prices, PV nowcast, inflow),
  3. freezes every hour OUTSIDE the tradable window to the committed net (INV-11),
  4. re-solves the shared 24h MILP under the new prices,
  5. applies the no-churn threshold (PR-14): if the re-optimised schedule does not
     beat holding the committed position by the configured volume AND spread, it
     returns NO_CHANGE and submits nothing,
  6. otherwise runs the Phase 3A physical checker + risk check, pauses for the
     operator (ENTER), submits (stub), and saves the new committed position.

Submitting the whole re-optimised schedule (not patched per hour) keeps the
position a physically consistent optimum; the per-hour deltas are the IDA trades.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from common_layer.configuration.config_loader import AppConfig
from common_layer.utilities import get_logger, AuditLogger
from common_layer.utilities import date_utils as du
from common_layer.utilities.timezone_utils import resolve_gate_time
from common_layer.database import PositionStore, ReserveStore, ComponentStore, validate_inputs, SchemaError
from common_layer.optimisation_model.core_milp_builder import (
    build_core_model, build_core_model_stochastic,
)
from common_layer.optimisation_model.core_milp_solver import (
    solve_core_model, extract_results, SolveError,
    extract_stochastic_results, bridge_stochastic_to_gate_results,
)
from common_layer.optimisation_model.scenario_generator import (
    generate_price_scenarios, default_scenarios_for_gate,
)
import os
from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster import (
    forecast_da_prices, forecast_da_prices_isp,
)
from phase_2a_ida1_intraday_auction_1.ida1_price_forecasting.ida1_price_forecaster import (
    forecast_ida1_prices,
)
from phase_2b_ida2_intraday_auction_2.ida2_price_forecasting.ida2_price_forecaster import (
    forecast_ida2_prices,
)
from phase_2c_ida3_intraday_auction_3.ida3_price_forecasting.ida3_price_forecaster import (
    forecast_ida3_prices,
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

log = get_logger("phase2.ida")

# Which gate's committed schedule each IDA starts from.
_BASELINE_GATE = {"IDA1": "DA", "IDA2": "IDA1", "IDA3": "IDA2"}


def _get_intraday_prices_isp(isps: List[int], hours: List[int], day, delivery_date: str,
                             gate: str) -> Dict[int, float]:
    """Gate-specific intraday price forecast at real ISP resolution.

    Each IDA gate's spread model (ida{1,2,3}_price_forecaster.py) is trained
    on that gate's historical SIDC clearing prices at HOURLY resolution --
    no real 15-min IDA spread history exists to retrain on (same situation
    as the PV forecaster's cloud-correction model, see
    pv_power_forecaster.py's forecast_pv_available_isp). The genuinely
    ISP-resolved DA price (real 15-min OMIE data where available) is kept at
    full resolution; the hourly-predicted spread on top of it is applied
    flat across that hour's 4 ISPs -- honest about which part of the signal
    is real 15-min and which is a coarser correction, not blended into one
    false-precision number."""
    da_hourly = forecast_da_prices(hours, delivery_date)
    if gate == "IDA1":
        ida_hourly = forecast_ida1_prices(hours, delivery_date, da_hourly)
    elif gate == "IDA2":
        ida_hourly = forecast_ida2_prices(hours, delivery_date, da_hourly)
    else:
        ida_hourly = forecast_ida3_prices(hours, delivery_date, da_hourly)

    da_isp = forecast_da_prices_isp(isps, delivery_date)
    out: Dict[int, float] = {}
    for h in hours:
        spread = ida_hourly.get(h, da_hourly.get(h, 55.0)) - da_hourly.get(h, 55.0)
        for isp in du.hour_to_isps(h, day):
            out[isp] = max(-600.0, round(da_isp.get(isp, 55.0) + spread, 2))
    return out


def _build_inputs(gate: str, delivery_date: str, cfg: AppConfig) -> dict:
    day = du.parse_date(delivery_date)
    isps = du.delivery_isps(day)
    hours = du.delivery_hours(day)
    isp_h = du.isp_duration_min(day) / 60.0

    inflow_hourly = forecast_inflow(hours, delivery_date, cfg.plant.reservoir)
    inflow = {isp: inflow_hourly[h] for h in hours for isp in du.hour_to_isps(h, day)}

    return {
        "delivery_date": delivery_date,
        "hours": isps,
        "dt_h": isp_h,
        "da_prices": _get_intraday_prices_isp(isps, hours, day, delivery_date, gate),
        "pv_available_mw": forecast_pv_available_isp(isps, delivery_date, cfg.plant.pv),
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


def _pause(message: str, no_pause: bool) -> None:
    """Operator pause for the demo (ENTER to continue). Skipped if no_pause."""
    if no_pause:
        return
    try:
        input(f"\n  {message}  [ENTER to continue] ")
    except (EOFError, KeyboardInterrupt):
        pass


def _update_gate_training_data(gate: str, delivery_date: str) -> None:
    """Live-fetch real settled clearing prices to backfill this gate's
    training history up to yesterday. Falls back to synthetic per-date on
    any OMIE download failure (never blocks the pipeline)."""
    try:
        if gate == "IDA1":
            from phase_2a_ida1_intraday_auction_1.ida1_price_forecasting.omie_ida1_price_loader import (
                update_training_data,
            )
        elif gate == "IDA2":
            from phase_2b_ida2_intraday_auction_2.ida2_price_forecasting.omie_ida2_price_loader import (
                update_training_data,
            )
        else:
            from phase_2c_ida3_intraday_auction_3.ida3_price_forecasting.omie_ida3_price_loader import (
                update_training_data,
            )
        update_training_data(delivery_date)
    except Exception as exc:
        log.warning(f"[{gate}] training-data live update failed ({exc}); "
                    f"using existing/cached history.")


def reoptimise_ida(gate: str, delivery_date: str, cfg: AppConfig,
                   no_pause: bool = False, use_synthetic: bool = True) -> dict:
    """Run one IDA gate. Returns a status dict.

    use_synthetic=True (default) skips the live OMIE training-data backfill,
    matching run_da's convention — set False to fetch real settled IDA
    clearing prices for any missing training dates up to yesterday.
    """
    if not use_synthetic:
        _update_gate_training_data(gate, delivery_date)

    audit = AuditLogger()
    audit.log(f"{gate}_START", delivery_date=delivery_date)
    gate_cfg = cfg.market.gate(gate)
    store = PositionStore()
    day = du.parse_date(delivery_date)
    dt = du.isp_duration_min(day) / 60.0

    # 1. committed baseline (previous gate's running net). ------------------
    baseline_gate = _BASELINE_GATE[gate]
    committed = store.committed_position(delivery_date, as_of_gate=baseline_gate)
    if not committed:
        msg = (f"[{gate}] no committed baseline from {baseline_gate}; "
               f"run the earlier gate(s) first.")
        log.error(msg)
        return {"status": "NO_BASELINE", "reason": msg}

    # 2. intraday inputs ----------------------------------------------------
    inputs = _build_inputs(gate, delivery_date, cfg)
    hours = inputs["hours"]  # real ISP indices (see _build_inputs)
    try:
        validate_inputs(inputs, cfg)
    except SchemaError as e:
        audit.log(f"{gate}_SCHEMA_FAILED", delivery_date=delivery_date, reason=str(e))
        return {"status": "SCHEMA_FAILED", "reason": str(e)}

    # 3. freeze ISPs outside the tradable window (IDA3 -> freeze hours 1-11,
    # i.e. their real ISPs). gate_cfg.hour_in_product() checks real clock
    # hours (config's delivery_hours), so convert each ISP to its hour first.
    tradable = [isp for isp in hours if gate_cfg.hour_in_product(du.isp_to_hour(isp, day))]
    fixed_net = {isp: committed.get(isp, 0.0) for isp in hours if isp not in tradable}
    log.info(f"{gate}: ISPs {tradable[0]}-{tradable[-1]} tradable "
             f"({len(tradable)} of {len(hours)}, {len(fixed_net)} frozen)")

    # 3b. reserve capacity already committed at its (earlier) real gate time -
    # same envelope tightening as DA (see run_da.py) — intraday re-optimisation
    # must also respect the reserve commitment, it cannot re-sell that MW.
    rstore = ReserveStore()
    reserved_up = dict(rstore.reserved_up(delivery_date, "aFRR"))
    reserved_dn = dict(rstore.reserved_dn(delivery_date, "aFRR"))
    for h, v in rstore.reserved_up(delivery_date, "mFRR").items():
        reserved_up[h] = reserved_up.get(h, 0.0) + v
    for h, v in rstore.reserved_dn(delivery_date, "mFRR").items():
        reserved_dn[h] = reserved_dn.get(h, 0.0) + v

    # 4. re-solve under intraday prices -------------------------------------
    # Opt-in two-stage stochastic mode (same engine as DA, see run_da.py):
    # the tradable-window bid becomes the here-and-now first-stage decision,
    # physical dispatch becomes per-scenario recourse; hours outside the
    # tradable window are frozen identically in both paths (INV-11).
    stochastic_mode = cfg.stochastic.enabled_for(gate)
    try:
        if stochastic_mode:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            scenarios, probabilities = default_scenarios_for_gate(
                gate, inputs["da_prices"], repo_root, n_scenarios=cfg.stochastic.n_scenarios)
            model, meta = build_core_model_stochastic(
                inputs, cfg, scenarios, probabilities,
                fixed_net_position=fixed_net,
                reserved_up_mw=reserved_up, reserved_dn_mw=reserved_dn)
            solve_time = solve_core_model(model, cfg, gate=gate)
            stoch_results = extract_stochastic_results(model, meta)
            results = bridge_stochastic_to_gate_results(stoch_results, meta)
        else:
            model, meta = build_core_model(inputs, cfg, fixed_net_position=fixed_net,
                                            reserved_up_mw=reserved_up,
                                            reserved_dn_mw=reserved_dn)
            solve_time = solve_core_model(model, cfg, gate=gate)
            results = extract_results(model, meta)
    except SolveError as e:
        audit.log(f"{gate}_SOLVE_FAILED", delivery_date=delivery_date, reason=str(e))
        return {"status": "SOLVE_FAILED", "reason": str(e)}
    new_net = results.net_position_mw
    price = inputs["da_prices"]

    # 5. no-churn threshold (PR-14). ----------------------------------------
    # Decision: re-bid only if the re-optimised schedule's expected energy value
    # (under intraday prices, tradable hours) beats holding the committed position
    # by at least a DYNAMIC threshold, and at least one hour moves by more than
    # the volume noise floor. one_way_vol halves the summed |delta| so a pure swap
    # (sell hour A / buy hour B) is counted once, not twice.
    #
    # Dynamic threshold: max(floor, pct% of |DA_position_value| in tradable hours).
    # This auto-scales to the plant's actual market exposure — a large committed
    # position at high prices warrants a higher absolute improvement bar, preventing
    # microstructure noise from triggering unnecessary re-bids.
    th = cfg.market.trading_thresholds
    deltas = {h: (new_net[h] - committed.get(h, 0.0)) for h in tradable}
    one_way_vol = 0.5 * sum(abs(d) * dt for d in deltas.values())
    committed_value = sum(price[h] * committed.get(h, 0.0) * dt for h in tradable)
    new_value = sum(price[h] * new_net[h] * dt for h in tradable)
    improvement = new_value - committed_value

    # Dynamic threshold: 0.15% of the absolute DA position value in tradable hours
    da_value_tradable = sum(abs(price[h] * committed.get(h, 0.0)) for h in tradable)
    dynamic_threshold = (th.ida_min_rebid_pct / 100.0) * da_value_tradable
    material = {h: d for h, d in deltas.items() if abs(d * dt) >= th.ida_min_delta_mwh}

    if not material or improvement < dynamic_threshold:
        log.info(f"{gate}: no re-bid — switching now would gain only "
                 f"{improvement:,.0f} EUR, but we only re-bid if the gain is at "
                 f"least {dynamic_threshold:,.0f} EUR ({th.ida_min_rebid_pct:.2f}% of "
                 f"today's {da_value_tradable:,.0f} EUR position value)")
        audit.log(f"{gate}_NO_CHANGE", delivery_date=delivery_date, one_way_vol_mwh=one_way_vol,
                  improvement_eur=improvement, dynamic_threshold_eur=dynamic_threshold,
                  da_value_tradable_eur=da_value_tradable, tradable_hours=tradable,
                  n_total_hours=len(hours))
        _print_ida_summary(gate, price, committed, new_net, tradable, deltas,
                           improvement, decision="NO_CHANGE",
                           dynamic_threshold=dynamic_threshold,
                           cfg=cfg, delivery_date=delivery_date)
        _pause(f"{gate}: no material improvement — holding committed position.", no_pause)
        return {"status": "NO_CHANGE", "improvement_eur": improvement,
                "one_way_vol_mwh": one_way_vol,
                "dynamic_threshold_eur": dynamic_threshold}

    # 6. Phase 3A physical checker + risk -----------------------------------
    try:
        check_da_bid(results, inputs, cfg, gate=gate)
    except BidCheckError as e:
        audit.log(f"{gate}_BIDCHECK_FAILED", delivery_date=delivery_date, reason=str(e))
        return {"status": "BID_CHECK_FAILED", "reason": str(e)}
    risk = PreTradeRiskChecker(cfg).check(results, dt_h=dt)
    if not risk.passed:
        audit.log(f"{gate}_RISK_BLOCKED", delivery_date=delivery_date, violations=risk.violations)
        return {"status": "RISK_BLOCKED", "violations": risk.violations}

    _print_ida_summary(gate, price, committed, new_net, tradable, deltas,
                       improvement, decision="RE-BID",
                       dynamic_threshold=dynamic_threshold,
                       cfg=cfg, delivery_date=delivery_date)
    _pause(f"{gate}: re-optimised, +{improvement:,.0f} EUR — about to submit.", no_pause)

    # 7. submit (stub) + save new committed position ------------------------
    ref = f"{gate}-{delivery_date.replace('-', '')}-001"
    position = {isp: {"volume_mwh": new_net[isp] * dt, "price_eur_mwh": price[isp]}
                for isp in tradable}
    store.save_position(delivery_date, gate, position)
    audit.log(f"{gate}_SUBMITTED", delivery_date=delivery_date, ref=ref, improvement_eur=improvement,
              n_hours=len(position), dynamic_threshold_eur=dynamic_threshold,
              da_value_tradable_eur=da_value_tradable, tradable_hours=tradable,
              n_total_hours=len(hours), traded_hours=sorted(material.keys()))
    log.info(f"{gate} submitted (stub) ref {ref}; saved {len(position)} hours")
    return {"status": "SUBMITTED", "ref": ref, "improvement_eur": improvement,
            "one_way_vol_mwh": one_way_vol, "solve_time_sec": solve_time,
            "committed_net_mw": dict(committed),
            "new_net_mw": {h: float(new_net[h]) for h in tradable},
            "ida_prices": dict(price),
            "tradable_hours": tradable,
            "dynamic_threshold_eur": dynamic_threshold,
            "da_value_tradable_eur": da_value_tradable}


def _print_ida_summary(gate: str, price: Dict[int, float], committed: Dict[int, float],
                       new_net: Dict[int, float], tradable: List[int],
                       deltas: Dict[int, float], improvement: float, decision: str,
                       dynamic_threshold: Optional[float] = None,
                       cfg: Optional[AppConfig] = None,
                       delivery_date: Optional[str] = None) -> None:
    print("\n" + "=" * 62)
    print(f"  {gate} RE-OPTIMISATION  —  decision: {decision}")
    print("=" * 62)
    if cfg is not None and delivery_date is not None:
        gate_close = resolve_gate_time(cfg.market.gate(gate).gate_close,
                                        du.parse_date(delivery_date)).strftime("%Y-%m-%d %H:%M %Z")
        print(f"  Gate closes (CET): {gate_close}   <-- submit before this")
    print(f"  Tradable hours : {tradable[0]}-{tradable[-1]}")
    print(f"  Expected improvement vs committed: {improvement:>12,.2f} EUR")
    if dynamic_threshold is not None:
        status = "PASS" if improvement >= dynamic_threshold else "FAIL"
        print(f"  Dynamic re-bid threshold       : {dynamic_threshold:>12,.2f} EUR  [{status}]  "
              f"(see table total below)")
    print(f"\n  {'Hour':<5} {'Price':>7} {'Committed':>11} {'New':>9} {'Delta MWh':>11} "
          f"{'|Price x Vol|':>13}")
    print("  " + "-" * 63)
    exposure_sum = 0.0
    for h in tradable:
        d = deltas[h]
        row_exposure = abs(price[h] * committed.get(h, 0.0))
        exposure_sum += row_exposure
        mark = "  <-- trade" if abs(d) >= 0.5 else ""
        print(f"  H{h:02d}  {price[h]:>7.1f} {committed.get(h,0.0):>+11.1f} "
              f"{new_net[h]:>+9.1f} {d:>+11.2f} {row_exposure:>13,.1f}{mark}")
    print("  " + "-" * 63)
    print(f"  {'TOTAL exposure (sum |price x volume|):':<47} {exposure_sum:>13,.1f}")
    print("=" * 62)
