"""
reserve_activation.py — shared TSO-activation simulation for aFRR / mFRR.

During delivery the TSO calls part of the offered reserve in some ISPs. This
engine, shared by Phase 4B (aFRR) and 4C (mFRR):
  * simulates a synthetic per-ISP Area Control Error (ACE, MW) for the control
    area — the actual physical quantity PICASSO/MARI dispatch responds to —
    via simulate_ace_series() (see its docstring for the modeling detail),
  * reads the committed offer (ReserveStore) for the product,
  * reads the scheduled net (DeliveryStore) per ISP for physical headroom check,
  * decides direction from the ACE signal (ACE > deadband => up-regulation
    needed, ACE < -deadband => down-regulation needed) and scales the
    activated MW continuously with |ACE| between deadband and full-scale —
    proportional to signal magnitude, not a fixed depth fraction,
  * once a direction starts, holds it for >= min_hold_isps consecutive ISPs
    (PICASSO/MARI minimum activation duration) even if ACE dips back through
    the deadband briefly — real regulation calls don't toggle every 15 min,
  * never both up AND down in the same ISP — mutually exclusive by construction,
  * tracks BESS SOC depletion so availability shrinks after sustained activations,
  * enforces physical headroom: scheduled_net ± activated ≤ plant envelope,
  * stores SEPARATE up_price and dn_price per activation row for correct settlement,
  * logs activated MW + energy prices per ISP (ActivationStore) for settlement.

aFRR (continuous AGC) has a tighter ACE deadband than mFRR (discrete TSO
instruction, only called for larger/sustained imbalances) — see _ACE_PARAMS.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from common_layer.configuration.config_loader import AppConfig
from common_layer.utilities import date_utils as du
from common_layer.database import ReserveStore, ActivationStore, DeliveryStore
from common_layer.optimisation_model.reserve_offer_builder import (
    fat_deliverable_mw, fat_deliverable_dn_mw,
)
from common_layer.optimisation_model.activation_ramp_tracker import effective_isp_hours
from phase_1_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster import (
    forecast_da_prices_isp,
)

# Synthetic Area Control Error model parameters (illustrative, seeded per
# product+date for reproducibility — NOT real TSO SCADA data, same honesty
# standard as fcr_activation.py's frequency-deviation model). Mean-reverting
# AR(1) process in MW-of-area-imbalance around a per-day bias, with occasional
# larger transient spikes (generation/load trip events).
_ACE_PARAMS = {
    "aFRR": dict(
        deadband_mw=25.0, full_scale_mw=180.0, std_mw=55.0, ar_phi=0.80,
        daily_bias_std_mw=15.0, spike_prob=0.02, spike_std_mw=180.0,
    ),
    "mFRR": dict(
        deadband_mw=100.0, full_scale_mw=350.0, std_mw=70.0, ar_phi=0.88,
        daily_bias_std_mw=20.0, spike_prob=0.015, spike_std_mw=280.0,
    ),
}

# Per-product activation behaviour (hold time + minimum call size only —
# direction/magnitude now come from the ACE signal, not a fixed probability).
_PROFILE = {
    "aFRR": {
        "min_hold_isps": 2,    # PICASSO minimum activation duration (30 min)
        "min_activate_mw": 1.0,
    },
    "mFRR": {
        "min_hold_isps": 3,    # MARI minimum activation duration (45 min)
        "min_activate_mw": 2.0,
    },
}

_DIR_NONE = 0
_DIR_UP   = 1
_DIR_DN   = -1


def simulate_ace_series(product: str, delivery_date: str) -> Dict[int, float]:
    """Synthetic per-ISP Area Control Error (MW, + = area short/needs
    up-regulation, - = area long/needs down-regulation) for the control area
    Alqueva's aFRR/mFRR bid responds to.

    Illustrative model, NOT real TSO SCADA data: mean-reverting AR(1) process
    (captures real ACE's serial correlation — an imbalance persists across
    consecutive ISPs rather than resetting independently each time) around a
    per-day bias, with occasional larger transient spikes (generation/load
    trip events). Same modeling family as fcr_activation.py's frequency-
    deviation model, in MW-of-area-imbalance units instead of mHz-of-
    frequency units — the physically distinct signal aFRR/mFRR actually
    respond to (PICASSO/MARI dispatch off ACE-driven calls, not raw
    frequency the way FCR's droop does).

    Seeded independently of the settlement RNG (own "ace-{product}-{date}"
    stream) so it can be regenerated identically and cheaply by callers that
    only want the signal for display (e.g. the dashboard), without re-running
    the full activation/settlement pass.
    """
    p = _ACE_PARAMS.get(product, _ACE_PARAMS["aFRR"])
    day = du.parse_date(delivery_date)
    rng = random.Random(f"ace-{product}-{delivery_date}")
    daily_bias = rng.gauss(0.0, p["daily_bias_std_mw"])
    prev = daily_bias
    series: Dict[int, float] = {}
    for h in du.delivery_hours(day):
        for isp in du.hour_to_isps(h, day):
            innov_std = p["std_mw"] * (1.0 - p["ar_phi"] ** 2) ** 0.5
            ace = daily_bias + p["ar_phi"] * (prev - daily_bias) + rng.gauss(0.0, innov_std)
            if rng.random() < p["spike_prob"]:
                ace += rng.gauss(0.0, p["spike_std_mw"])
            series[isp] = ace
            prev = ace
    return series


@dataclass
class ActivationSummary:
    product: str
    n_isp_activated: int
    up_mwh: float
    dn_mwh: float
    rows: List[dict]


def simulate_and_log_activation(product: str, delivery_date: str, cfg: AppConfig,
                                fat_min: float) -> ActivationSummary:
    offers = ReserveStore().load_reserve(delivery_date, product)
    if not offers:
        return ActivationSummary(product, 0, 0.0, 0.0, [])

    # Load scheduled net per ISP — needed to check physical headroom at activation time.
    delivery_rows = DeliveryStore().load(delivery_date)
    scheduled_by_isp: Dict[int, float] = {r["isp"]: r["scheduled_mw"] for r in delivery_rows}

    day = du.parse_date(delivery_date)
    isp_duration_min = du.isp_duration_min(day)
    isp_h = isp_duration_min / 60.0
    # Ramp-corrected effective ISP hours: accounts for linear ramp-up within FAT
    # rather than crediting full MW from t=0. aFRR: 0.2083 h, mFRR: 0.1458 h.
    eff_isp_h = effective_isp_hours(fat_min, isp_duration_min)
    # offers is already ISP-keyed -- ReserveStore holds whatever resolution
    # the DA/IDA/XBID gate that sized it actually ran at (96 real ISPs since
    # run_afrr.py builds offers straight off the ISP-native committed
    # position). No hourly expansion needed; forecast at the same resolution.
    isps = sorted(offers)
    da = forecast_da_prices_isp(isps, delivery_date)

    p_gen_cap = cfg.plant.p_max_generation_mw
    p_pump_cap = cfg.plant.p_max_pump_mw

    # BESS SOC tracking — initialize from plant config; depletes/charges each ISP.
    bess_cap_mwh = cfg.plant.bess.capacity_mwh
    bess_soc_mwh = cfg.plant.bess.initial_soc_frac * bess_cap_mwh
    bess_soc_min = cfg.plant.bess.e_min_mwh
    bess_soc_max = cfg.plant.bess.e_max_mwh
    bess_power_mw = cfg.plant.bess.power_mw

    prof = _PROFILE.get(product, _PROFILE["aFRR"])
    ace_params = _ACE_PARAMS.get(product, _ACE_PARAMS["aFRR"])
    ace_by_isp = simulate_ace_series(product, delivery_date)

    rows: List[dict] = []
    up_mwh = dn_mwh = 0.0

    # Hold state machine: direction is driven by the ACE signal (see
    # simulate_ace_series), but once started is held for >= min_hold_isps
    # consecutive ISPs (PICASSO/MARI minimum activation duration) even if ACE
    # dips back through the deadband briefly — mutually exclusive by
    # construction (sequential if/elif on ACE sign, never both directions).
    current_dir = _DIR_NONE
    hold_remaining = 0
    # Trailing hysteresis: after the guaranteed hold expires, if ACE dips back
    # inside the deadband for just a couple of ISPs, real AGC/mFRR dispatch
    # doesn't instantly release the unit -- it keeps serving the same
    # direction through a brief lull rather than fully standing down and
    # re-dispatching moments later. trailing_left counts down that grace
    # window; it's refreshed on every ISP that actively continues the call.
    _TRAILING_GRACE_ISPS = 2
    trailing_left = 0

    for isp in isps:
        off = offers[isp]
        h = du.isp_to_hour(isp, day)
        # Separate activation prices per direction — up and down settle at different rates.
        # aFRR: ±25% of DA (continuous AGC, tight to DA).
        # mFRR: ±30% of DA (discrete TSO instruction, higher premium).
        # Capped at realistic MIBEL regulatory limits: 200 EUR/MWh up, floor 0.
        if product == "aFRR":
            up_price = round(min(da[isp] * 1.25, 200.0), 2)
            dn_price = round(max(da[isp] * 0.75, 0.0), 2)
        else:  # mFRR
            up_price = round(min(da[isp] * 1.30, 150.0), 2)
            dn_price = round(max(da[isp] * 0.70, 0.0), 2)

        sched = scheduled_by_isp.get(isp, 0.0)

        # BESS available power depends on current SOC — drops to zero at limits.
        bess_up_avail = bess_power_mw if bess_soc_mwh > bess_soc_min + 1e-6 else 0.0
        bess_dn_avail = bess_power_mw if bess_soc_mwh < bess_soc_max - 1e-6 else 0.0

        # Mode-aware FAT cap: in pump mode with short FAT, crossing to generation
        # is not guaranteed; up deliverable limited to ramp-to-zero + BESS.
        fat_up = fat_deliverable_mw(cfg, fat_min, current_net_mw=sched)
        fat_up = fat_up - bess_power_mw + bess_up_avail   # swap BESS term for SOC-aware
        fat_dn = fat_deliverable_dn_mw(cfg, fat_min)
        fat_dn = fat_dn - bess_power_mw + bess_dn_avail   # swap BESS term for SOC-aware

        # Physical headroom from actual scheduled net (not offer size) — prevents
        # activating more than the plant can physically deliver in this ISP.
        headroom_up = max(0.0, p_gen_cap - sched)
        headroom_dn = max(0.0, sched + p_pump_cap)

        ace = ace_by_isp.get(isp, 0.0)
        deadband = ace_params["deadband_mw"]
        full_scale = ace_params["full_scale_mw"]

        # Hold state: once activated, direction is held for min_hold_isps ISPs
        # regardless of ACE dipping back through the deadband — mutually
        # exclusive by construction (sequential if/elif on ACE sign, never
        # both directions). After the guaranteed hold, ACE genuinely
        # re-crossing the SAME direction is a continuation (the need never
        # stopped); ACE inside the deadband is bridged by trailing grace
        # rather than instantly releasing (see _TRAILING_GRACE_ISPS above).
        raw_dir = _DIR_UP if ace > deadband else (_DIR_DN if ace < -deadband else _DIR_NONE)

        if hold_remaining > 0:
            direction = current_dir
            hold_remaining -= 1
            is_hold_continuation = True
            trailing_left = _TRAILING_GRACE_ISPS
        elif raw_dir != _DIR_NONE and raw_dir == current_dir:
            direction = raw_dir
            is_hold_continuation = True
            hold_remaining = prof["min_hold_isps"] - 1
            trailing_left = _TRAILING_GRACE_ISPS
        elif raw_dir != _DIR_NONE:
            # Fresh start (from idle) or a genuine reversal.
            direction = raw_dir
            is_hold_continuation = False
            hold_remaining = prof["min_hold_isps"] - 1
            trailing_left = _TRAILING_GRACE_ISPS
            current_dir = direction
        elif current_dir != _DIR_NONE and trailing_left > 0:
            # ACE dipped inside the deadband, but the call just ran --
            # keep serving the same direction through the grace window.
            direction = current_dir
            is_hold_continuation = True
            trailing_left -= 1
        else:
            direction = _DIR_NONE
            current_dir = _DIR_NONE
            trailing_left = 0

        if direction == _DIR_NONE:
            continue

        # Continuous, signal-proportional response: intensity ramps linearly
        # from 0 at the deadband edge to 1 at full-scale ACE — the same
        # droop-style shape FCR uses, in MW-of-imbalance units instead of
        # mHz. A held-over ISP still tracks its own ACE, not the value that
        # triggered the hold -- EXCEPT a real regulation call, once running,
        # doesn't drop to zero just because ACE dipped for one ISP; it's
        # sustained at least at the minimum call size for the rest of the
        # hold window (that's what "minimum activation duration" means).
        intensity = min(max(abs(ace) - deadband, 0.0) / (full_scale - deadband), 1.0)
        offer_mw = off["up_mw"] if direction == _DIR_UP else off["dn_mw"]
        call_mw = offer_mw * intensity
        if is_hold_continuation and call_mw < prof["min_activate_mw"]:
            call_mw = min(prof["min_activate_mw"], offer_mw)

        if direction == _DIR_UP:
            up = min(call_mw, fat_up, headroom_up, off["up_mw"])
            dn = 0.0
        else:
            up = 0.0
            dn = min(call_mw, fat_dn, headroom_dn, off["dn_mw"])

        # Minimum activation threshold (very small calls are not issued) --
        # only gates a FRESH trigger. A held-over continuation keeps
        # reporting whatever it can physically deliver even if a tight
        # fat/headroom limit pushes it below the threshold for one ISP;
        # real ongoing regulation doesn't stop billing mid-hold.
        if not is_hold_continuation and up < prof["min_activate_mw"] and dn < prof["min_activate_mw"]:
            continue
        if is_hold_continuation and up <= 1e-9 and dn <= 1e-9:
            continue

        # Update BESS SOC after this ISP's activation; clamp to configured limits.
        if up > 0:
            bess_contrib = min(bess_up_avail, up)
            bess_soc_mwh = max(bess_soc_min, bess_soc_mwh - bess_contrib * isp_h)
        elif dn > 0:
            bess_contrib = min(bess_dn_avail, dn)
            bess_soc_mwh = min(bess_soc_max, bess_soc_mwh + bess_contrib * isp_h)

        # Energy credited uses ramp-corrected hours, not face ISP duration.
        up_mwh += up * eff_isp_h
        dn_mwh += dn * eff_isp_h
        rows.append({
            "isp":  isp,
            "hour": h,
            "up_mw":             up,
            "dn_mw":             dn,
            "up_price_eur_mwh":  up_price,
            "dn_price_eur_mwh":  dn_price,
            "eff_isp_h":         eff_isp_h,  # stored for settlement accuracy
        })

    ActivationStore().save(delivery_date, product, rows)
    return ActivationSummary(product, len(rows), up_mwh, dn_mwh, rows)
