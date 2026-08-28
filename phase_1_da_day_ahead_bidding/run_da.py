"""
run_da.py — Phase 1 Day-Ahead gate orchestrator.

Pipeline (the order is the spec's INV-9 — checks before approval before submit):
    1. load config
    2. assemble inputs (forecast prices / PV / inflow, or OMIE live)
    3. schema-validate inputs
    4. build + solve the shared 24h MILP with CPLEX  (PR-13: stop if unsolved)
    5. extract the schedule
    6. Phase 3A physical bid checker            (stop on any violation)
    7. pre-trade risk checker                   (stop if limits breached)
    8. trader [A]/[R] approval                  (unless --auto-approve)
    9. submit (stub) + save position + audit

Run standalone for the demo:
    python phase_1_da_day_ahead_bidding/run_da.py --date 2026-06-22
    python phase_1_da_day_ahead_bidding/run_da.py --date 2026-06-22 --auto-approve
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Allow running this file directly: put the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_layer.configuration import load_config, AppConfig
from common_layer.utilities import get_logger, AuditLogger
from common_layer.utilities import date_utils as du
from common_layer.utilities.timezone_utils import resolve_gate_time
from common_layer.database import PositionStore, ComponentStore, validate_inputs, SchemaError
from common_layer.optimisation_model import (
    build_core_model, solve_core_model, extract_results, SolveError,
    build_core_model_stochastic, extract_stochastic_results,
    generate_price_scenarios, load_selected_model_mae,
)
from common_layer.optimisation_model.core_milp_solver import GateResults
from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.omie_da_price_loader import (
    update_training_data, update_isp_training_data,
)
from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster import (
    forecast_da_prices_isp,
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
from phase_1_da_day_ahead_bidding.da_bid_formatting.da_bid_formatter import (
    format_da_bids, to_omie_payload, render_table,
)
from phase_1_da_day_ahead_bidding.risk_and_bid_validation.pre_trade_risk_checker import (
    PreTradeRiskChecker,
)
from phase_1_da_day_ahead_bidding.trader_approval.trader_approval_prompt import (
    request_da_approval,
)

log = get_logger("phase1.da")


def _bridge_stochastic_to_gate_results(stoch_results, meta) -> GateResults:
    """Adapt a StochasticGateResults into the existing GateResults shape so
    every downstream consumer (physical bid checker, pre-trade risk checker,
    ComponentStore report save) keeps working completely unchanged.

    The first-stage bid (da_bids / net_position_mw) is exactly the shared,
    scenario-independent decision — no approximation there. For the
    per-hour dispatch trajectory fields (psp_schedule etc.), which only
    exist per-scenario in the stochastic model, this uses the CENTRAL
    scenario (offset k=0, i.e. the original point-forecast scenario) as the
    representative trajectory shown to the physical/risk checkers and saved
    to the report — a defensible choice since it's literally the same
    forecast the deterministic path would have used, not an arbitrary pick.
    """
    H = meta.hours
    central_s = min(
        meta.scenarios,
        key=lambda s: sum(abs(meta.scenario_prices[s][h] - stoch_results.da_bids[h]["expected_price_eur_mwh"])
                           for h in H),
    )
    central = stoch_results.per_scenario_dispatch[central_s]

    psp_schedule = {h: central[h]["psp"] for h in H}
    bess_schedule = {h: central[h]["bess"] for h in H}
    pv_schedule = {h: central[h]["pv"] for h in H}
    reservoir_trajectory = {h: central[h]["reservoir"] for h in H}
    # Efficiency-per-hour isn't tracked per-scenario in the stochastic
    # extractor (omega weights aren't pulled there) -- report zeros rather
    # than fabricate a plausible-looking number; a later iteration can add
    # per-scenario efficiency extraction if a consumer needs it.
    efficiency_per_hour = {h: {"eta_trb_pw": 0.0, "eta_pmp_pw": 0.0} for h in H}

    return GateResults(
        da_bids={h: {"volume_mwh": stoch_results.da_bids[h]["volume_mwh"],
                     "price_eur_mwh": stoch_results.da_bids[h]["expected_price_eur_mwh"]}
                 for h in H},
        net_position_mw=stoch_results.net_position_mw,
        psp_schedule=psp_schedule,
        bess_schedule=bess_schedule,
        pv_schedule=pv_schedule,
        reservoir_trajectory=reservoir_trajectory,
        efficiency_per_hour=efficiency_per_hour,
        energy_revenue_eur=stoch_results.expected_energy_revenue_eur,
        objective_eur=stoch_results.objective_eur,
    )


def _expand_hourly_to_isp(hourly: dict, day, hours: list) -> dict:
    """Repeat each hour's value flat across that hour's real ISPs. Used only
    for inflow: river inflow's physical time constant is hours/days, not
    minutes, so a flat hourly value genuinely is the correct treatment at
    15-min resolution, not an approximation (same reasoning already applied
    to the BESS SOC-vs-price and multi-asset dispatch dashboard widgets)."""
    out = {}
    for h in hours:
        for isp in du.hour_to_isps(h, day):
            out[isp] = hourly[h]
    return out


def _assemble_inputs(delivery_date: str, cfg: AppConfig, use_synthetic: bool) -> tuple[dict, str]:
    """Build the optimisation input bundle at real ISP (15-min MTU) resolution,
    matching MIBEL's actual settlement/commitment granularity since
    2025-10-01. Returns (inputs, price_source)."""
    day   = du.parse_date(delivery_date)
    isps  = du.delivery_isps(day)
    hours = du.delivery_hours(day)
    isp_h = du.isp_duration_min(day) / 60.0

    # Step 1 — fill Excel with prices up to yesterday (both hourly-legacy and
    # real ISP history; the ISP price forecaster trains on the latter).
    if not use_synthetic:
        update_training_data(delivery_date, zone="PT")
        update_isp_training_data(delivery_date, zone="PT")
    else:
        # Fill any historical gap with synthetic prices so lag features are valid.
        from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.omie_da_price_loader import (
            _fill_synthetic_gap,
        )
        _fill_synthetic_gap(delivery_date, zone="PT")

    # Step 2 — ML forecaster trains on updated history, predicts delivery_date
    # at true ISP resolution (real GHI-physics-scaled PV, real quarter-hour
    # price history where available).
    da_prices  = forecast_da_prices_isp(isps, delivery_date)
    price_source = "ML_FORECAST"

    pv = forecast_pv_available_isp(isps, delivery_date, cfg.plant.pv)

    # Inflow forecaster is hourly-native (the underlying dataset is daily-mean
    # sourced -- see reservoir_inflow_forecaster.py); expand flat per-hour,
    # the physically correct treatment, not an approximation.
    inflow_hourly = forecast_inflow(hours, delivery_date, cfg.plant.reservoir)
    inflow = _expand_hourly_to_isp(inflow_hourly, day, hours)

    inputs = {
        "delivery_date": delivery_date,
        "hours": isps,
        "dt_h": isp_h,
        "da_prices": da_prices,
        "pv_available_mw": pv,
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
    return inputs, price_source


def run_da(delivery_date: str, config_dir: Optional[str] = None,
           use_synthetic: bool = True, auto_approve: bool = False) -> dict:
    cfg = load_config(config_dir)
    audit = AuditLogger()
    audit.log("DA_START", delivery_date=delivery_date, synthetic=use_synthetic)
    log.info(f"DA gate for delivery {delivery_date}")

    # 2. inputs --------------------------------------------------------------
    inputs, price_source = _assemble_inputs(delivery_date, cfg, use_synthetic)

    # 3. schema validation ---------------------------------------------------
    try:
        validate_inputs(inputs, cfg)
    except SchemaError as e:
        log.error(str(e))
        audit.log("DA_SCHEMA_FAILED", delivery_date=delivery_date, reason=str(e))
        return {"status": "SCHEMA_FAILED", "reason": str(e)}

    # DA runs FIRST in the real REN sequence (MPGGS Article 80(3): PDVD ->
    # aFRR band -> mFRR band — see run_production.py for the full citation),
    # so no reserve capacity is committed yet at this point. DA solves
    # against the full FCR-reduced envelope; aFRR/mFRR are sized afterward
    # from DA's leftover headroom (see run_afrr.py / afrr_offer_builder.py).

    # 4. solve ---------------------------------------------------------------
    # Opt-in two-stage stochastic mode (config/stochastic.yaml): the DA bid
    # is decided under a price scenario fan instead of a single point
    # forecast. Default OFF -- the deterministic path below is unchanged.
    stochastic_mode = cfg.stochastic.enabled_for("DA")
    try:
        if stochastic_mode:
            mae_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "da_price_pv_inflow_forecasting", "da_selected_model_isp.json")
            mae = load_selected_model_mae(mae_path)
            scenarios, probabilities = generate_price_scenarios(
                inputs["da_prices"], mae, n_scenarios=cfg.stochastic.n_scenarios)
            model, meta = build_core_model_stochastic(inputs, cfg, scenarios, probabilities)
            solve_time = solve_core_model(model, cfg, gate="DA")
            stoch_results = extract_stochastic_results(model, meta)
            results = _bridge_stochastic_to_gate_results(stoch_results, meta)
        else:
            model, meta = build_core_model(inputs, cfg)
            solve_time = solve_core_model(model, cfg, gate="DA")
            results = extract_results(model, meta)
    except SolveError as e:
        log.error(str(e))
        audit.log("DA_SOLVE_FAILED", delivery_date=delivery_date, reason=str(e))
        return {"status": "SOLVE_FAILED", "reason": str(e)}

    log.info(f"Solved in {solve_time:.2f}s | energy revenue "
             f"{results.energy_revenue_eur:,.2f} EUR"
             + (" (stochastic, expected value)" if stochastic_mode else ""))
    audit.log("DA_SOLVED", delivery_date=delivery_date, solve_time=solve_time,
              energy_revenue_eur=results.energy_revenue_eur,
              objective_eur=results.objective_eur, stochastic=stochastic_mode)

    # 6. Phase 3A physical bid checker --------------------------------------
    try:
        check_da_bid(results, inputs, cfg, gate="DA")
    except BidCheckError as e:
        log.error(str(e))
        audit.log("DA_BIDCHECK_FAILED", delivery_date=delivery_date, reason=str(e))
        return {"status": "BID_CHECK_FAILED", "reason": str(e)}
    audit.log("DA_BIDCHECK_PASSED", delivery_date=delivery_date)

    # 7. risk checker --------------------------------------------------------
    risk = PreTradeRiskChecker(cfg).check(results, dt_h=inputs["dt_h"])
    if not risk.passed:
        log.error("RISK CHECK FAILED: " + "; ".join(risk.violations))
        audit.log("DA_RISK_BLOCKED", delivery_date=delivery_date, violations=risk.violations)
        return {"status": "RISK_BLOCKED", "violations": risk.violations}
    audit.log("DA_RISK_PASSED", delivery_date=delivery_date)

    bids = format_da_bids(results)
    gate_close = resolve_gate_time(cfg.market.gate("DA").gate_close,
                                   du.parse_date(delivery_date)).strftime("%Y-%m-%d %H:%M %Z")

    # 8. approval ------------------------------------------------------------
    if auto_approve:
        log.info("Auto-approve mode -- skipping trader prompt")
        print("\n  DA recommendation (auto-approve):")
        print(render_table(bids))
    else:
        approved = request_da_approval(bids, results, price_source, solve_time, gate_close)
        if not approved:
            log.info("Trader REJECTED DA bids — nothing submitted")
            audit.log("DA_REJECTED", delivery_date=delivery_date)
            return {"status": "REJECTED"}

    # 9. submit (stub) + save ------------------------------------------------
    payload = to_omie_payload(bids, delivery_date)
    ref = f"DA-{delivery_date.replace('-', '')}-001"
    audit.log("DA_SUBMITTED", delivery_date=delivery_date, ref=ref, n_hours=len(bids), n_bids=len(payload["bids"]))
    log.info(f"Submitted (stub). OMIE ref {ref}")

    position = {b.hour: {"volume_mwh": b.volume_mwh, "price_eur_mwh": b.price_eur_mwh}
                for b in bids}
    PositionStore().save_position(delivery_date, "DA", position)
    audit.log("DA_POSITION_SAVED", delivery_date=delivery_date, n_hours=len(position))

    # Save rich component data for analytics Excel report
    ComponentStore().save(
        delivery_date=delivery_date,
        psp_schedule=results.psp_schedule,
        bess_schedule=results.bess_schedule,
        pv_schedule=results.pv_schedule,
        reservoir_trajectory=results.reservoir_trajectory,
        efficiency_per_hour=results.efficiency_per_hour,
        inflow_m3h=inputs["inflow_m3h"],
        solver_metrics={
            "solve_time_sec": round(solve_time, 3),
            "objective_eur": round(results.objective_eur, 2),
            "energy_revenue_eur": round(results.energy_revenue_eur, 2),
            "solver": "CPLEX",
        },
        initial_state=inputs.get("initial_state", {}),
    )

    return {
        "status": "SUBMITTED",
        "ref": ref,
        "price_source": price_source,
        "energy_revenue_eur": results.energy_revenue_eur,
        "objective_eur": results.objective_eur,
        "solve_time_sec": solve_time,
    }


def main():
    p = argparse.ArgumentParser(description="Run the Phase 1 Day-Ahead gate")
    p.add_argument("--date", required=True, help="delivery date YYYY-MM-DD")
    p.add_argument("--config", default=None, help="config dir (default: repo config/)")
    p.add_argument("--auto-approve", action="store_true")
    p.add_argument("--real-data", action="store_true", help="use OMIE live prices")
    args = p.parse_args()

    result = run_da(delivery_date=args.date, config_dir=args.config,
                    use_synthetic=not args.real_data, auto_approve=args.auto_approve)

    print("\n  RESULT:", result.get("status"))
    for k, val in result.items():
        if k != "status":
            print(f"    {k}: {val}")
    sys.exit(0 if result.get("status") == "SUBMITTED" else 1)


if __name__ == "__main__":
    main()
