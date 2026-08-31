"""
backtest_runner.py — replay the optimiser over a span of days.

For each delivery day it:
  * assembles the same DA inputs the live pipeline would,
  * probes MILP solve quality (feasible? objective? checker pass?),
  * scores DA price forecast accuracy vs REAL archived OMIE prices where
    available (real_omie_price()), falling back to the synthetic perturbation
    (realised_from_forecast()) only for dates with no real coverage -- PV
    always uses the synthetic series, since no real generation-telemetry
    source exists anywhere in this project. Every row is labelled with which
    kind of "actual" it used (price_actual_source), never silently mixed.
  * re-settles the already-solved, already-committed DA bid
    (gate_results.net_position_mw -- fixed by the forecast-based solve, NOT
    re-optimized) against the REAL price, producing realised_revenue_eur --
    the first genuinely realised-outcome P&L number in this project, distinct
    from objective_eur (which remains the forecast-based solve objective).
    Only computed when real price data exists for that date; otherwise None,
    reported plainly as "unavailable", never approximated.
Returns per-day rows plus aggregate metrics — the evidence that the model solves
cleanly and the forecasts are reasonable across many days, not just one.

After the loop, compute_risk_metrics() derives VaR(95%/99%), CVaR(95%/99%),
Monte Carlo bootstrap confidence intervals, Sharpe ratio, and max drawdown
from the per-day objective P&L series.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field, replace as dc_replace
from typing import List, Optional

import pyomo.environ as pyo

from common_layer.configuration.config_loader import AppConfig
from common_layer.optimisation_model.core_milp_builder import build_core_model, build_core_model_stochastic
from common_layer.optimisation_model.core_milp_solver import (
    solve_core_model, extract_results, extract_stochastic_results, SolveError,
)
from common_layer.optimisation_model.scenario_generator import (
    generate_price_scenarios, load_selected_model_mae,
)
from common_layer.optimisation_model.reserve_offer_builder import check_reserve_offers, ReserveCheckError
from common_layer.optimisation_model import ida_reoptimiser
from common_layer.utilities import date_utils as du
from common_layer.utilities.date_utils import hour_to_isps
from phase_1_da_day_ahead_bidding.run_da import _assemble_inputs
from phase_2d_xbid_continuous_intraday.xbid_milp_optimiser.xbid_optimiser import build_xbid_inputs
from phase_3a_afrr_automatic_frequency_reserve.afrr_reserve_offer_builder.afrr_offer_builder import (
    build_afrr_offers,
)
from phase_3b_mfrr_manual_frequency_reserve.mfrr_reserve_offer_builder.mfrr_offer_builder import (
    build_mfrr_offers,
)
from phase_6_backtesting_and_validation.backtest_engine.historical_data_loader import (
    date_range, realised_from_forecast, real_omie_price, real_ren_capacity_price,
    real_ida_price,
)
from phase_6_backtesting_and_validation.forecast_and_model_validation.price_forecast_validator import (
    error_metrics,
)
from phase_6_backtesting_and_validation.forecast_and_model_validation.pv_forecast_validator import (
    validate_pv,
)
from phase_6_backtesting_and_validation.forecast_and_model_validation.milp_solution_quality_checker import (
    check_solution_quality,
)
from phase_6_backtesting_and_validation.risk_analytics.portfolio_risk_metrics import (
    RiskMetrics, compute_risk_metrics,
)


@dataclass
class BacktestResult:
    rows: List[dict] = field(default_factory=list)
    n_days: int = 0
    n_feasible: int = 0
    n_checker_pass: int = 0
    n_real_price_days: int = 0           # days with genuine OMIE_LIVE coverage
    n_real_afrr_days: int = 0            # days with genuine REN_LIVE aFRR coverage
    n_real_mfrr_days: int = 0            # days with genuine REN_LIVE mFRR coverage (also needs aFRR real)
    avg_objective_eur: float = 0.0
    avg_solve_sec: float = 0.0
    avg_price_mae: float = 0.0
    avg_pv_mae: float = 0.0
    avg_realised_revenue_eur: Optional[float] = None   # real-price days only
    avg_realised_afrr_capacity_eur: Optional[float] = None
    avg_realised_mfrr_capacity_eur: Optional[float] = None
    n_real_ida1_price_days: int = 0      # days with genuine OMIE_LIVE IDA1 coverage (needs DA-feasible too)
    avg_realised_ida1_revenue_eur: Optional[float] = None
    n_real_ida2_price_days: int = 0      # needs IDA1-feasible too (chained baseline)
    avg_realised_ida2_revenue_eur: Optional[float] = None
    n_real_ida3_price_days: int = 0      # needs IDA2-feasible too (chained baseline); only 7 real dates exist
    avg_realised_ida3_revenue_eur: Optional[float] = None
    n_real_xbid_price_days: int = 0      # needs IDA3-feasible too (chained baseline); window W1 only
    avg_realised_xbid_revenue_eur: Optional[float] = None
    risk: Optional[RiskMetrics] = None   # populated after loop


def _run_intraday_gate_backtest(gate: str, date: str, cfg: AppConfig,
                                baseline_net: dict, day_date) -> dict:
    """Re-optimize one intraday gate in-memory (store-free) and value its
    FINAL committed position against the real archived clearing price for
    that gate. `baseline_net` is the PRIOR gate's committed net position
    (DA for IDA1, IDA1 for IDA2, IDA2 for IDA3 -- the real production
    chain, INV-11) -- passed as a plain dict, never touching PositionStore.

    Returns {feasible, net_position_mw (or None), realised_revenue_eur,
    realised_price_source}. Same "forecast decides, real price values"
    pattern as DA; aFRR/mFRR reserve headroom NOT subtracted (see
    run_backtest's docstring) -- reserve capacity is backtested separately.
    """
    inputs = ida_reoptimiser._build_inputs(gate, date, cfg)
    gate_cfg = cfg.market.gate(gate)
    tradable = [isp for isp in inputs["hours"]
                if gate_cfg.hour_in_product(du.isp_to_hour(isp, day_date))]
    fixed_net = {isp: baseline_net.get(isp, 0.0)
                 for isp in inputs["hours"] if isp not in tradable}
    q = check_solution_quality(inputs, cfg, gate=gate, fixed_net_position=fixed_net)

    realised_revenue_eur = None
    realised_price_source = "unavailable"
    net_position_mw = q.gate_results.net_position_mw if q.feasible and q.gate_results is not None else None

    if q.feasible and net_position_mw is not None:
        real_hourly = real_ida_price(gate, date, list(range(1, 25)))
        if real_hourly is not None:
            real_isp = {isp: real_hourly[du.isp_to_hour(isp, day_date)] for isp in inputs["hours"]}
            dt_h = inputs.get("dt_h", 1.0)
            realised_revenue_eur = round(
                sum(real_isp[isp] * net_position_mw[isp] * dt_h for isp in inputs["hours"]), 2)
            realised_price_source = "OMIE_LIVE"

    return {
        "feasible": q.feasible,
        "net_position_mw": net_position_mw,
        "realised_revenue_eur": realised_revenue_eur,
        "realised_price_source": realised_price_source,
    }


# XBID window backtested: W1 only, the first continuous-intraday check
# (D-1 18:30 CET). Production supports 6 check windows (config/market.yaml);
# a full chain across all 6 is a natural, structurally identical follow-up
# once this lands -- scoped narrower here for the same reason IDA3 was
# scoped to its own gate rather than bundled: keep each real-price addition
# independently verifiable.
_XBID_BACKTEST_WINDOW = "W1"


def _run_xbid_backtest(date: str, cfg: AppConfig, baseline_net: dict) -> dict:
    """Re-optimize XBID's W1 continuous-intraday window in-memory
    (store-free) and value its FINAL committed position against the real
    archived XBID proxy price. `baseline_net` is IDA3's committed net (the
    real production chain: DA -> IDA1 -> IDA2 -> IDA3 -> XBID).

    Unlike the IDA gates, XBID adds a per-order trade-band constraint on
    top of build_core_model (xbid_optimiser.optimise_xbid's own mechanism,
    lines 128-131 there) -- not expressible via
    milp_solution_quality_checker.check_solution_quality's plain
    fixed_net_position kwarg, so this calls build_core_model/
    solve_core_model/extract_results directly, mirroring optimise_xbid's
    real solve sequence exactly (minus the store reads/writes/no-churn
    threshold/operator pause/submission, none of which belong in a
    backtest loop).
    """
    try:
        inputs, open_hours = build_xbid_inputs(date, cfg, _XBID_BACKTEST_WINDOW)
    except ValueError:
        return {"feasible": None, "net_position_mw": None,
                "realised_revenue_eur": None, "realised_price_source": "unavailable"}

    all_isps = inputs["hours"]
    cap = cfg.market.trading_thresholds.xbid_max_volume_per_order_mw
    fixed_net = {isp: baseline_net.get(isp, 0.0) for isp in all_isps if isp not in open_hours}

    try:
        model, meta = build_core_model(inputs, cfg, fixed_net_position=fixed_net)
        model.xbid_band_hi = pyo.Constraint(
            open_hours, rule=lambda mm, h: mm.p_net[h] <= baseline_net.get(h, 0.0) + cap)
        model.xbid_band_lo = pyo.Constraint(
            open_hours, rule=lambda mm, h: mm.p_net[h] >= baseline_net.get(h, 0.0) - cap)
        solve_core_model(model, cfg, gate="XBID")
    except SolveError:
        return {"feasible": False, "net_position_mw": None,
                "realised_revenue_eur": None, "realised_price_source": "unavailable"}

    results = extract_results(model, meta)
    net_position_mw = results.net_position_mw

    realised_revenue_eur = None
    realised_price_source = "unavailable"
    real_hourly = real_ida_price("XBID", date, list(range(1, 25)))
    if real_hourly is not None:
        day_date = du.parse_date(date)
        real_isp = {isp: real_hourly[du.isp_to_hour(isp, day_date)] for isp in all_isps}
        dt_h = inputs.get("dt_h", 1.0)
        realised_revenue_eur = round(
            sum(real_isp[isp] * net_position_mw[isp] * dt_h for isp in all_isps), 2)
        realised_price_source = "OMIE_LIVE"

    return {
        "feasible": True,
        "net_position_mw": net_position_mw,
        "realised_revenue_eur": realised_revenue_eur,
        "realised_price_source": realised_price_source,
    }


def run_backtest(start_date: str, n_days: int, cfg: AppConfig) -> BacktestResult:
    res = BacktestResult(n_days=n_days)
    obj_sum = solve_sum = price_mae_sum = pv_mae_sum = 0.0
    realised_revenue_sum = 0.0
    realised_afrr_sum = realised_mfrr_sum = 0.0
    realised_ida1_sum = realised_ida2_sum = realised_ida3_sum = 0.0
    realised_xbid_sum = 0.0

    for date in date_range(start_date, n_days):
        inputs, _ = _assemble_inputs(date, cfg, use_synthetic=True)
        q = check_solution_quality(inputs, cfg)

        hours = list(inputs["da_prices"].keys())
        real_price = real_omie_price(date, hours)
        if real_price is not None:
            price_actual = real_price
            price_actual_source = "OMIE_LIVE"
        else:
            price_actual = realised_from_forecast(inputs["da_prices"], date, 0.10, "px")
            price_actual_source = "synthetic"

        pv_actual = realised_from_forecast(inputs["pv_available_mw"], date, 0.15, "pv")
        pm = error_metrics(inputs["da_prices"], price_actual)
        vm = validate_pv(inputs["pv_available_mw"], pv_actual)

        # Real re-settlement: value the ALREADY-COMMITTED DA bid (fixed by the
        # forecast-based solve, not re-optimized) against the real price.
        # Only when real price data exists -- otherwise plainly "unavailable",
        # never approximated from the synthetic series.
        realised_revenue_eur = None
        if real_price is not None and q.feasible and q.gate_results is not None:
            dt_h = inputs.get("dt_h", 1.0)
            net_pos = q.gate_results.net_position_mw
            realised_revenue_eur = round(
                sum(real_price[h] * net_pos[h] * dt_h for h in hours), 2)

        # Reserve capacity revenue: real REN cap_up/cap_dn priced against the
        # SAME offer sizing the live pipeline would compute from the
        # already-committed DA bid (build_afrr_offers / build_mfrr_offers are
        # pure functions of committed_net + cap prices + config, no store
        # dependency). aFRR computed first; mFRR only if aFRR's real price was
        # also available for this date, since mFRR's sizing nets against
        # aFRR's committed MW -- never a mixed real/synthetic capacity figure.
        # Activation/imbalance revenue stays OUT of scope: the activated MW is
        # always internally-simulated in this project (no real REN/SCADA
        # telemetry loader exists), so a "real activation revenue" number
        # would be real price x synthetic quantity -- not the same honesty
        # standard as capacity revenue, where both price and sizing are real.
        realised_afrr_capacity_eur = None
        realised_mfrr_capacity_eur = None
        realised_reserve_price_source = "unavailable"
        day_date = dt.date.fromisoformat(date)
        afrr_offers = None
        if q.feasible and q.gate_results is not None:
            net_pos = q.gate_results.net_position_mw
            hourly_net = {
                h: sum(net_pos[i] for i in hour_to_isps(h, day_date)) / len(hour_to_isps(h, day_date))
                for h in range(1, 25)
            }
            real_afrr = real_ren_capacity_price(date, list(range(1, 25)), "aFRR")
            if real_afrr is not None:
                cap_up, cap_dn = real_afrr
                try:
                    afrr_offers = build_afrr_offers(hourly_net, cap_up, cap_dn, cfg)
                    check_reserve_offers(afrr_offers, hourly_net, cfg, cfg.market.afrr.fat_min, product="aFRR")
                    realised_afrr_capacity_eur = round(
                        sum(o.up_mw * cap_up[h] + o.dn_mw * cap_dn[h] for h, o in afrr_offers.items()), 2)
                    realised_reserve_price_source = "REN_LIVE"
                except ReserveCheckError:
                    afrr_offers = None  # envelope violation -- don't report a capacity figure

            real_mfrr = real_ren_capacity_price(date, list(range(1, 25)), "mFRR")
            if real_mfrr is not None and afrr_offers is not None:
                cap_up, cap_dn = real_mfrr
                reserved_up = {h: o.up_mw for h, o in afrr_offers.items()}
                reserved_dn = {h: o.dn_mw for h, o in afrr_offers.items()}
                try:
                    mfrr_offers = build_mfrr_offers(hourly_net, cap_up, cap_dn, reserved_up, reserved_dn, cfg)
                    check_reserve_offers(mfrr_offers, hourly_net, cfg, cfg.market.mfrr.fat_min, product="mFRR",
                                          reserved_up=reserved_up, reserved_dn=reserved_dn)
                    realised_mfrr_capacity_eur = round(
                        sum(o.up_mw * cap_up[h] + o.dn_mw * cap_dn[h] for h, o in mfrr_offers.items()), 2)
                except ReserveCheckError:
                    pass  # mFRR envelope violation -- aFRR figure still stands on its own

        # Real-price intraday re-optimization, chained DA -> IDA1 -> IDA2 ->
        # IDA3 (INV-11, the real production baseline chain -- each gate's
        # committed net becomes the next gate's frozen baseline for hours
        # outside its own tradable window). Each gate only runs if the PRIOR
        # gate in the chain was feasible; each is valued independently
        # against its own real archived clearing price -- never a mixed
        # real/synthetic figure, and IDA3 (only 7 real archived dates,
        # coverage starting 2026-08-15) is reported honestly thin, not
        # padded. aFRR/mFRR reserve headroom NOT subtracted at any gate here
        # (reserve-capacity revenue is backtested completely separately
        # above; mixing would either double scope this pass or force
        # synthetic cap prices into an otherwise clean real-price test).
        ida_results = {}
        prior_net = q.gate_results.net_position_mw if q.feasible and q.gate_results is not None else None
        for gate in ("IDA1", "IDA2", "IDA3"):
            if prior_net is None:
                ida_results[gate] = {"feasible": None, "net_position_mw": None,
                                     "realised_revenue_eur": None, "realised_price_source": "unavailable"}
                continue
            r = _run_intraday_gate_backtest(gate, date, cfg, prior_net, day_date)
            ida_results[gate] = r
            prior_net = r["net_position_mw"] if r["feasible"] else None

        # XBID (window W1 only, see _run_xbid_backtest docstring): chained
        # from IDA3's committed net, the real production sequence's final
        # step. Only runs if IDA3 was feasible.
        if prior_net is not None:
            xbid_result = _run_xbid_backtest(date, cfg, prior_net)
        else:
            xbid_result = {"feasible": None, "net_position_mw": None,
                           "realised_revenue_eur": None, "realised_price_source": "unavailable"}

        row = {
            "date": date, "feasible": q.feasible, "checker_pass": q.checker_passed,
            "objective_eur": round(q.objective_eur, 2), "solve_sec": round(q.solve_time_sec, 3),
            "price_mae": round(pm.mae, 2), "price_rmse": round(pm.rmse, 2),
            "pv_mae": round(vm.mae, 4), "note": q.note,
            "price_actual_source": price_actual_source,
            "pv_actual_source": "synthetic",
            "realised_revenue_eur": realised_revenue_eur,
            "realised_price_source": "OMIE_LIVE" if real_price is not None else "unavailable",
            "realised_afrr_capacity_eur": realised_afrr_capacity_eur,
            "realised_mfrr_capacity_eur": realised_mfrr_capacity_eur,
            "realised_reserve_price_source": realised_reserve_price_source,
            "ida1_feasible": ida_results["IDA1"]["feasible"],
            "realised_ida1_revenue_eur": ida_results["IDA1"]["realised_revenue_eur"],
            "realised_ida1_price_source": ida_results["IDA1"]["realised_price_source"],
            "ida2_feasible": ida_results["IDA2"]["feasible"],
            "realised_ida2_revenue_eur": ida_results["IDA2"]["realised_revenue_eur"],
            "realised_ida2_price_source": ida_results["IDA2"]["realised_price_source"],
            "ida3_feasible": ida_results["IDA3"]["feasible"],
            "realised_ida3_revenue_eur": ida_results["IDA3"]["realised_revenue_eur"],
            "realised_ida3_price_source": ida_results["IDA3"]["realised_price_source"],
            "xbid_feasible": xbid_result["feasible"],
            "realised_xbid_revenue_eur": xbid_result["realised_revenue_eur"],
            "realised_xbid_price_source": xbid_result["realised_price_source"],
        }
        if q.feasible:
            row["ops"]  = q.operational
            row["tmp"]  = q.temporal
            row["eco"]  = q.economic_ext
        res.rows.append(row)
        res.n_feasible += int(q.feasible)
        res.n_checker_pass += int(q.checker_passed)
        obj_sum += q.objective_eur
        solve_sum += q.solve_time_sec
        price_mae_sum += pm.mae
        pv_mae_sum += vm.mae
        if realised_revenue_eur is not None:
            res.n_real_price_days += 1
            realised_revenue_sum += realised_revenue_eur
        if realised_afrr_capacity_eur is not None:
            res.n_real_afrr_days += 1
            realised_afrr_sum += realised_afrr_capacity_eur
        if realised_mfrr_capacity_eur is not None:
            res.n_real_mfrr_days += 1
            realised_mfrr_sum += realised_mfrr_capacity_eur
        if ida_results["IDA1"]["realised_revenue_eur"] is not None:
            res.n_real_ida1_price_days += 1
            realised_ida1_sum += ida_results["IDA1"]["realised_revenue_eur"]
        if ida_results["IDA2"]["realised_revenue_eur"] is not None:
            res.n_real_ida2_price_days += 1
            realised_ida2_sum += ida_results["IDA2"]["realised_revenue_eur"]
        if ida_results["IDA3"]["realised_revenue_eur"] is not None:
            res.n_real_ida3_price_days += 1
            realised_ida3_sum += ida_results["IDA3"]["realised_revenue_eur"]
        if xbid_result["realised_revenue_eur"] is not None:
            res.n_real_xbid_price_days += 1
            realised_xbid_sum += xbid_result["realised_revenue_eur"]

    if n_days:
        res.avg_objective_eur = obj_sum / n_days
        res.avg_solve_sec     = solve_sum / n_days
        res.avg_price_mae     = price_mae_sum / n_days
        res.avg_pv_mae        = pv_mae_sum / n_days
    if res.n_real_price_days:
        res.avg_realised_revenue_eur = realised_revenue_sum / res.n_real_price_days
    if res.n_real_afrr_days:
        res.avg_realised_afrr_capacity_eur = realised_afrr_sum / res.n_real_afrr_days
    if res.n_real_mfrr_days:
        res.avg_realised_mfrr_capacity_eur = realised_mfrr_sum / res.n_real_mfrr_days
    if res.n_real_ida1_price_days:
        res.avg_realised_ida1_revenue_eur = realised_ida1_sum / res.n_real_ida1_price_days
    if res.n_real_ida2_price_days:
        res.avg_realised_ida2_revenue_eur = realised_ida2_sum / res.n_real_ida2_price_days
    if res.n_real_ida3_price_days:
        res.avg_realised_ida3_revenue_eur = realised_ida3_sum / res.n_real_ida3_price_days
    if res.n_real_xbid_price_days:
        res.avg_realised_xbid_revenue_eur = realised_xbid_sum / res.n_real_xbid_price_days

    # Risk metrics from the daily objective P&L series (feasible days only).
    pnl_series = [r["objective_eur"] for r in res.rows if r["feasible"]]
    res.risk = compute_risk_metrics(pnl_series)

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic risk-measure comparison: expected-value vs CVaR-averse bidding,
# both solved on the IDENTICAL real scenario fan, both valued against the
# REAL settled DA price -- isolates the one variable that matters (the risk
# measure), and reports the actual realized-outcome tradeoff rather than a
# theoretical claim. See core_milp_builder.build_core_model_stochastic and
# tests/test_cvar_stochastic.py for the underlying CVaR formulation.
# ─────────────────────────────────────────────────────────────────────────────

_DA_SELECTED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "phase_1_da_day_ahead_bidding", "da_price_pv_inflow_forecasting",
    "da_selected_model_isp.json")


@dataclass
class StochasticComparisonResult:
    rows: List[dict] = field(default_factory=list)
    n_days: int = 0
    n_real_days: int = 0          # days with real DA price coverage AND both strategies feasible
    mean_realised_ev_eur: Optional[float] = None
    mean_realised_cvar_eur: Optional[float] = None
    cost_of_risk_aversion_eur: Optional[float] = None   # mean_ev - mean_cvar ("insurance premium")
    tail_improvement_eur: Optional[float] = None        # risk_cvar.cvar_95_eur - risk_ev.cvar_95_eur
    risk_ev: Optional[RiskMetrics] = None
    risk_cvar: Optional[RiskMetrics] = None


_DEFAULT_COMPARISON_OFFSETS = [-1.0, 0.0, 1.0]   # 3-scenario fan, see docstring below


def run_stochastic_risk_comparison(
    start_date: str, n_days: int, cfg: AppConfig,
    cvar_alpha: float = 0.90, n_scenarios: int = 3,
    offsets: Optional[List[float]] = None,
    time_limit_sec: int = 240,
) -> StochasticComparisonResult:
    """Solve the DA gate twice per real-price day -- once with
    risk_measure="expected_value", once with risk_measure="cvar" at
    cvar_alpha -- on the IDENTICAL scenario fan (same real walk-forward-CV
    MAE, same offsets), then value BOTH resulting bids against the REAL
    settled DA price. Only days with real archived DA price coverage are
    used (a day without real coverage cannot honestly show a "realized
    outcome" for either strategy, so it is skipped, not padded).

    n_scenarios/time_limit_sec, and the real solver-difficulty finding
    behind their defaults: the CVaR unit tests (test_cvar_stochastic.py)
    only ever exercised a small 24-hour toy fixture, never the real DA
    gate's actual 96-ISP resolution. At real scale, a 5-scenario CVaR MILP
    (2160+ binaries, plus the eta/shortfall variables) genuinely failed to
    reach a good solution within 300 s on a real date -- confirmed by
    direct inspection: CPLEX terminated "maxTimeLimit" with an "integer
    feasible" incumbent whose objective was ~10x smaller than a healthy
    solve (a near-degenerate near-zero-dispatch result, not a real economic
    optimum). Re-tested at 3 scenarios: CPLEX reaches genuine
    "optimal" (0.5% tolerance) in ~165 s. So this function defaults to a
    3-scenario fan (still genuinely stochastic, still isolates the risk
    measure as the only variable) and a 240 s budget -- not the same
    5-scenario fan the live DA gate's opt-in stochastic path uses, and
    that mismatch is deliberate: this analysis needs a SOLVED-TO-OPTIMALITY
    comparison to mean anything, and 5 scenarios could not deliver one
    within a reasonable budget on real data. `offsets` lets a caller pass
    a custom fan (must have len(offsets) == n_scenarios).

    time_limit_sec overrides the DA gate's normal 120 s production time
    budget (cfg.solver.time_limit_sec["DA"]) for BOTH solves in this call
    only -- this analysis is not the live gate (which keeps its own 120 s
    SLA) and needs the extra budget to reach a genuine solve. Does NOT
    touch `cfg` itself -- only the local copies used for this call.

    Returns realized-outcome risk statistics (VaR/CVaR/Sharpe/drawdown,
    via the same compute_risk_metrics() used elsewhere) for each strategy's
    REALIZED daily revenue series -- the first evidence-based answer in
    this project to "does the CVaR objective actually help," rather than a
    theoretical claim about the formulation alone.
    """
    res = StochasticComparisonResult(n_days=n_days)
    solver_override = dc_replace(
        cfg.solver, time_limit_sec={**cfg.solver.time_limit_sec, "DA": time_limit_sec})
    ev_cfg = dc_replace(cfg, solver=solver_override,
                        stochastic=dc_replace(cfg.stochastic, risk_measure="expected_value"))
    cvar_cfg = dc_replace(cfg, solver=solver_override,
                          stochastic=dc_replace(cfg.stochastic, risk_measure="cvar", cvar_alpha=cvar_alpha))
    mae = load_selected_model_mae(_DA_SELECTED_MODEL_PATH)
    fan_offsets = offsets if offsets is not None else (
        _DEFAULT_COMPARISON_OFFSETS if n_scenarios == 3 else None)

    ev_realised: List[float] = []
    cvar_realised: List[float] = []

    for date in date_range(start_date, n_days):
        inputs, _ = _assemble_inputs(date, cfg, use_synthetic=True)
        hours = list(inputs["da_prices"].keys())
        real_price = real_omie_price(date, hours)
        if real_price is None:
            res.rows.append({"date": date, "realised_ev_eur": None, "realised_cvar_eur": None,
                             "feasible_ev": None, "feasible_cvar": None,
                             "price_source": "unavailable"})
            continue

        scenarios, probabilities = generate_price_scenarios(
            inputs["da_prices"], mae, n_scenarios=n_scenarios, offsets=fan_offsets)
        dt_h = inputs.get("dt_h", 1.0)

        row = {"date": date, "price_source": "OMIE_LIVE"}
        strategy_net: dict = {}
        for label, strat_cfg in (("ev", ev_cfg), ("cvar", cvar_cfg)):
            try:
                model, meta = build_core_model_stochastic(inputs, strat_cfg, scenarios, probabilities)
                solve_core_model(model, strat_cfg, gate="DA")
                stoch_results = extract_stochastic_results(model, meta)
                strategy_net[label] = stoch_results.net_position_mw
                row[f"feasible_{label}"] = True
            except SolveError:
                strategy_net[label] = None
                row[f"feasible_{label}"] = False

        for label in ("ev", "cvar"):
            net = strategy_net[label]
            if net is not None:
                realised = round(sum(real_price[h] * net[h] * dt_h for h in hours), 2)
                row[f"realised_{label}_eur"] = realised
            else:
                row[f"realised_{label}_eur"] = None

        res.rows.append(row)
        if row.get("realised_ev_eur") is not None and row.get("realised_cvar_eur") is not None:
            res.n_real_days += 1
            ev_realised.append(row["realised_ev_eur"])
            cvar_realised.append(row["realised_cvar_eur"])

    if ev_realised:
        res.mean_realised_ev_eur = sum(ev_realised) / len(ev_realised)
        res.mean_realised_cvar_eur = sum(cvar_realised) / len(cvar_realised)
        res.cost_of_risk_aversion_eur = res.mean_realised_ev_eur - res.mean_realised_cvar_eur
        res.risk_ev = compute_risk_metrics(ev_realised)
        res.risk_cvar = compute_risk_metrics(cvar_realised)
        res.tail_improvement_eur = res.risk_cvar.cvar_95_eur - res.risk_ev.cvar_95_eur

    return res
