"""
live_bid_resettlement.py — value the ACTUAL historically-committed bid
against REAL archived prices, with ZERO re-solving.

This is deliberately a third, distinct thing from what already exists:
  - backtest_runner.py's real-price backtest RE-SOLVES each gate fresh
    against TODAY's forecaster state and reports the FULL CUMULATIVE
    position's value -- a different bid entirely from what was actually
    committed on the real day, on a cumulative (not incremental) basis.
  - Trading Desk's Summary_KPIs (dispatch_sheet_builder.py) values the
    REAL committed bid, but at each gate's own FORECAST/bid price at
    commit time (PositionStore's stored price_eur_mwh), not a real
    archived price.

This module values the REAL committed bid (read directly from
PositionStore/ReserveStore, no re-solve) against REAL archived prices
(historical_data_loader.py) -- the true apples-to-apples check against
what Trading Desk already shows for the same date.

Known, disclosed gaps (never silently mixed in):
  - No real ACTIVATION price source exists anywhere in this project
    (historical_data_loader.py only has capacity prices for aFRR/mFRR) --
    activation revenue is reported as "forecast-only, excluded", not
    zeroed or estimated.
  - Imbalance settlement in Summary_KPIs uses the DA stored price x MIBEL
    factors, not a real imbalance price -- excluded from this
    re-settlement for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from common_layer.database.position_store import PositionStore, GATE_ORDER
from common_layer.database.reserve_store import ReserveStore
from common_layer.utilities import date_utils as du

from .historical_data_loader import real_omie_price, real_ida_price, real_ren_capacity_price


@dataclass
class GateResettlement:
    gate: str
    committed_mwh: float                    # sum of |incremental volume| across the day
    price_source: str                       # "OMIE_LIVE" / "unavailable"
    realised_revenue_eur: Optional[float]
    forecast_revenue_eur: float             # settled at each ISP's OWN stored price (cur gate)


@dataclass
class ReserveResettlement:
    product: str                            # "aFRR" / "mFRR"
    committed_capacity_mwh: float
    price_source: str
    realised_capacity_revenue_eur: Optional[float]
    forecast_capacity_revenue_eur: Optional[float]
    activation_excluded: bool = True


@dataclass
class LiveBidResettlement:
    delivery_date: str
    gates: List[GateResettlement] = field(default_factory=list)
    reserves: List[ReserveResettlement] = field(default_factory=list)
    total_realised_revenue_eur: Optional[float] = None
    total_forecast_revenue_eur: float = 0.0
    n_gates_real: int = 0
    n_gates_total: int = 0


def _prior_gate(gate: str) -> Optional[str]:
    idx = GATE_ORDER.index(gate)
    return GATE_ORDER[idx - 1] if idx > 0 else None


def _isp_delta_revenue(cur_pos: Dict[int, dict], prior_pos: Dict[int, dict],
                       real_price_by_isp: Dict[int, float]) -> tuple[float, float, float]:
    """Per-ISP incremental committed volume, settled at REAL price and
    separately at the gate's own stored (forecast/bid) price -- mirrors
    dispatch_sheet_builder.py's sum_delta_revenue exactly, but with a
    second, real-price valuation added alongside.

    Returns (realised_revenue_eur, forecast_revenue_eur, committed_mwh).
    """
    realised = 0.0
    forecast = 0.0
    committed_mwh = 0.0
    for isp, cur in cur_pos.items():
        prior_vol = prior_pos.get(isp, {}).get("volume_mwh", 0.0)
        cur_vol = cur.get("volume_mwh", prior_vol)
        cur_prc = cur.get("price_eur_mwh", 0.0)
        delta_vol = cur_vol - prior_vol
        forecast += cur_prc * delta_vol
        committed_mwh += abs(delta_vol)
        real_p = real_price_by_isp.get(isp)
        if real_p is not None:
            realised += real_p * delta_vol
    return realised, forecast, committed_mwh


def resettle_live_bid(delivery_date: str, cfg=None) -> LiveBidResettlement:
    """Store-free: reads PositionStore/ReserveStore + real archived prices
    only. No MILP build/solve, no store writes."""
    day = du.parse_date(delivery_date)
    isps = du.delivery_isps(day)
    hours = du.delivery_hours(day)

    store = PositionStore()
    result = LiveBidResettlement(delivery_date=delivery_date)

    # Real DA price is natively ISP-keyed.
    real_da = real_omie_price(delivery_date, isps)

    for gate in GATE_ORDER:
        cur_pos = store.load_position(delivery_date, gate)
        if not cur_pos:
            continue  # gate never ran / nothing committed for this date

        prior_gate = _prior_gate(gate)
        prior_pos = store.committed_position(delivery_date, as_of_gate=prior_gate) if prior_gate else {}
        # committed_position returns {isp: net_mw}; sum_delta_revenue-style
        # helper expects {isp: {"volume_mwh": ...}} shape for BOTH sides, so
        # wrap the plain-MW prior dict to match (dt=1h-equivalent volume ==
        # MW figure already, per PositionStore.committed_position's own
        # docstring: it converts volume_mwh back to MW using real dt).
        prior_pos_wrapped = {isp: {"volume_mwh": mw} for isp, mw in prior_pos.items()}

        if gate == "DA":
            real_price_by_isp = real_da or {}
            price_source = "OMIE_LIVE" if real_da is not None else "unavailable"
        else:
            real_hourly = real_ida_price(gate, delivery_date, hours)
            price_source = "OMIE_LIVE" if real_hourly is not None else "unavailable"
            real_price_by_isp = {}
            if real_hourly is not None:
                for h, p in real_hourly.items():
                    for isp in du.hour_to_isps(h, day):
                        real_price_by_isp[isp] = p

        realised, forecast, committed_mwh = _isp_delta_revenue(
            cur_pos, prior_pos_wrapped, real_price_by_isp)

        result.gates.append(GateResettlement(
            gate=gate,
            committed_mwh=committed_mwh,
            price_source=price_source,
            realised_revenue_eur=round(realised, 2) if price_source == "OMIE_LIVE" else None,
            forecast_revenue_eur=round(forecast, 2),
        ))
        result.total_forecast_revenue_eur += forecast
        result.n_gates_total += 1
        if price_source == "OMIE_LIVE":
            result.n_gates_real += 1

    if result.n_gates_total > 0 and result.n_gates_real == result.n_gates_total:
        result.total_realised_revenue_eur = round(
            sum(g.realised_revenue_eur for g in result.gates), 2)
    result.total_forecast_revenue_eur = round(result.total_forecast_revenue_eur, 2)

    # ── Reserve capacity (aFRR/mFRR) ────────────────────────────────────────
    # ReserveStore's row key is labelled "hour" in its schema but, like
    # PositionStore, is written at real ISP (15-min) resolution post
    # transition -- confirmed by direct row-count inspection (96 rows, not
    # 24, on a real ISP-resolution date). Treating each row as covering a
    # full hour would overcount capacity revenue ~4x. Detect which regime
    # this date's rows are actually in from the row count itself, rather
    # than assuming, and value each row over its own dt_h.
    rsvr = ReserveStore()
    n_isps_today = du.isp_per_day(day)
    for product in ("aFRR", "mFRR"):
        offers = rsvr.load_reserve(delivery_date, product)
        if not offers:
            continue
        is_isp_keyed = len(offers) > len(hours) or max(offers.keys(), default=0) > len(hours)
        dt_h = (du.isp_duration_min(day) / 60.0) if is_isp_keyed else 1.0

        real_prices = real_ren_capacity_price(delivery_date, hours, product)
        price_source = "REN_LIVE" if real_prices is not None else "unavailable"
        real_up, real_dn = real_prices if real_prices is not None else ({}, {})

        realised_cap = 0.0
        forecast_cap = 0.0
        committed_mwh = 0.0
        for key, o in offers.items():
            up_mw, dn_mw = o.get("up_mw", 0.0), o.get("dn_mw", 0.0)
            hour = du.isp_to_hour(key, day) if is_isp_keyed else key
            forecast_cap += (up_mw * o.get("cap_up_eur_mw", 0.0)
                            + dn_mw * o.get("cap_dn_eur_mw", 0.0)) * dt_h
            committed_mwh += (up_mw + dn_mw) * dt_h
            if price_source == "REN_LIVE":
                realised_cap += (up_mw * real_up.get(hour, 0.0)
                                + dn_mw * real_dn.get(hour, 0.0)) * dt_h

        result.reserves.append(ReserveResettlement(
            product=product,
            committed_capacity_mwh=round(committed_mwh, 2),
            price_source=price_source,
            realised_capacity_revenue_eur=round(realised_cap, 2) if price_source == "REN_LIVE" else None,
            forecast_capacity_revenue_eur=round(forecast_cap, 2),
            activation_excluded=True,
        ))

    return result
