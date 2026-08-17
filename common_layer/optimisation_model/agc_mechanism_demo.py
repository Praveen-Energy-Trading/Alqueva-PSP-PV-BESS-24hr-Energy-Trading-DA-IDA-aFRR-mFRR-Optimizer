"""agc_mechanism_demo.py — illustrative, standalone simulation of how AGC's
system-wide merit-order dispatch actually works.

NOT part of the trading pipeline and NOT wired into settlement/revenue --
Alqueva's real activation (reserve_activation.py) depends only on this
plant's own committed offer and the ACE signal, exactly as a real plant only
receives its own dispatch setpoint from the TSO. This module exists purely
to demonstrate the mechanism AGC uses to decide WHICH providers get called
and how much of the total requirement each one covers: at every ISP, AGC
ranks all providers cheapest-first (merit order) and dispatches capacity in
that order until the area's required regulation MW is covered.

Alqueva's own capacity/price in the ladder are REAL (pulled from its actual
ReserveStore offer for the date) -- only the other providers in the control
area are synthetic (this repo has no data on Portugal's actual PICASSO/MARI
competitor fleet), clearly labelled as such wherever this is displayed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from common_layer.configuration.config_loader import AppConfig
from common_layer.database import ReserveStore
from common_layer.optimisation_model.reserve_activation import _ACE_PARAMS, simulate_ace_series
from common_layer.utilities import date_utils as du

# Illustrative other providers sharing the control area with Alqueva --
# NOT real competitor data (none is publicly available), a plausible spread
# of capacities/prices for a mid-sized European balancing area.
_SYNTHETIC_PROVIDERS = [
    ("Wind Fleet A", 180.0, 320.0, 8.0, 26.0),
    ("Gas Peaker B", 90.0, 150.0, 14.0, 40.0),
    ("Hydro C", 220.0, 260.0, 6.0, 22.0),
    ("BESS Fleet D", 60.0, 80.0, 10.0, 34.0),
    ("Interconnector E", 300.0, 380.0, 12.0, 30.0),
]


@dataclass
class ProviderDispatch:
    name: str
    capacity_mw: float
    price_eur_mw: float
    dispatched_mw: float
    is_alqueva: bool = False


@dataclass
class AgcTick:
    isp: int
    hour: int
    ace_mw: float
    direction: str          # "up", "down", "none"
    required_mw: float
    providers: List[ProviderDispatch] = field(default_factory=list)
    alqueva_dispatched_mw: float = 0.0


def simulate_agc_dispatch(product: str, delivery_date: str, cfg: AppConfig) -> List[AgcTick]:
    """Per-ISP illustrative AGC merit-order dispatch for the whole delivery day.

    Reuses the exact same ACE series the real activation model responds to
    (simulate_ace_series) so this demo is consistent with what the aFRR/mFRR
    delivery cards show -- same signal, this module just additionally shows
    the multi-provider allocation mechanism behind it.
    """
    p = _ACE_PARAMS.get(product, _ACE_PARAMS["aFRR"])
    ace_by_isp = simulate_ace_series(product, delivery_date)
    offers = ReserveStore().load_reserve(delivery_date, product)
    day = du.parse_date(delivery_date)

    rng = random.Random(f"agc-demo-{product}-{delivery_date}")
    # Small per-date jitter on the illustrative providers so the ladder isn't
    # bit-for-bit identical every day, while staying in a plausible range.
    synthetic_up = [(n, u * rng.uniform(0.9, 1.1), rng.uniform(lo, hi))
                     for n, u, d, lo, hi in _SYNTHETIC_PROVIDERS]
    synthetic_dn = [(n, d * rng.uniform(0.9, 1.1), rng.uniform(lo, hi))
                     for n, u, d, lo, hi in _SYNTHETIC_PROVIDERS]

    total_up_cap = sum(u for _, u, _ in synthetic_up)
    total_dn_cap = sum(d for _, d, _ in synthetic_dn)

    ticks: List[AgcTick] = []
    for h in du.delivery_hours(day):
        offer = offers.get(h, {"up_mw": 0.0, "dn_mw": 0.0, "cap_up_eur_mw": 20.0, "cap_dn_eur_mw": 20.0})
        alqueva_up = (offer.get("up_mw", 0.0), offer.get("cap_up_eur_mw", 20.0))
        alqueva_dn = (offer.get("dn_mw", 0.0), offer.get("cap_dn_eur_mw", 20.0))

        for isp in du.hour_to_isps(h, day):
            ace = ace_by_isp.get(isp, 0.0)
            deadband, full_scale = p["deadband_mw"], p["full_scale_mw"]

            if ace > deadband:
                direction = "up"
                intensity = min((ace - deadband) / (full_scale - deadband), 1.0)
                required_mw = intensity * (total_up_cap + alqueva_up[0])
                ladder = [("Alqueva (this plant)", alqueva_up[0], alqueva_up[1], True)]
                ladder += [(n, u, pr, False) for n, u, pr in synthetic_up]
            elif ace < -deadband:
                direction = "down"
                intensity = min((-ace - deadband) / (full_scale - deadband), 1.0)
                required_mw = intensity * (total_dn_cap + alqueva_dn[0])
                ladder = [("Alqueva (this plant)", alqueva_dn[0], alqueva_dn[1], True)]
                ladder += [(n, d, pr, False) for n, d, pr in synthetic_dn]
            else:
                direction = "none"
                required_mw = 0.0
                ladder = [("Alqueva (this plant)", 0.0, 0.0, True)]
                ladder += [(n, 0.0, 0.0, False) for n, _, _ in synthetic_up]

            ladder.sort(key=lambda row: row[2])  # merit order: cheapest first
            remaining = required_mw
            providers: List[ProviderDispatch] = []
            alqueva_dispatched = 0.0
            for name, cap, price, is_alq in ladder:
                dispatched = min(remaining, cap)
                remaining = max(0.0, remaining - dispatched)
                providers.append(ProviderDispatch(name, cap, price, dispatched, is_alq))
                if is_alq:
                    alqueva_dispatched = dispatched

            ticks.append(AgcTick(
                isp=isp, hour=h, ace_mw=ace, direction=direction,
                required_mw=required_mw, providers=providers,
                alqueva_dispatched_mw=alqueva_dispatched,
            ))

    return ticks
