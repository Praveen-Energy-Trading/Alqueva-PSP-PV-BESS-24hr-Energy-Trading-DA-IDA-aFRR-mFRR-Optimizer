"""
fcr_activation.py — real-time FCR (primary control) response simulation.

Standalone demonstration module. NOT wired into run_production.py's 15-phase
pipeline (see module docstring note below) — kept separate so the pipeline's
documented 15-phase count stays accurate.

FCR is fundamentally different from aFRR/mFRR and is NOT simulated the same
way (see reserve_activation.py for the TSO-call model used there):

  * aFRR/mFRR: discrete, TSO-triggered calls, simulated via a stochastic
    "does the TSO call this ISP" probability model.
  * FCR: continuous, AUTOMATIC, local droop response to the live grid
    frequency deviation. There is no TSO call to simulate — every FCR-capable
    unit responds proportionally, all the time, whenever frequency is outside
    the deadband.

Regulatory basis (verified against REN and ENTSO-E sources):
  * Portugal: FCR is NOT market-procured. It is a mandatory grid-code
    obligation on large generators, non-remunerated (they bear the wear cost).
    Source: REN FCR Pilot Project materials (Jan 2025) —
    "Atualmente, em Portugal, a FCR não é adquirida em mercado, sendo
    fornecida obrigatoriamente pelos grandes geradores."
  * ENTSO-E Continental Europe synchronous area (governs Portugal):
      - Full Activation Time (FAT):            30 seconds
      - Deadband (no response inside this):    ±10 mHz
      - Full activation frequency deviation:   ±200 mHz (linear droop between
        the deadband and this point)
      - Minimum sustained activation period:   15-30 minutes per incident

Because FCR is non-remunerated (INV-7), this module produces NO settlement
and NO revenue — only a physical/compliance simulation: did the reserved
headroom respond correctly to the frequency signal, and how much energy was
delivered as a droop response.

IMPORTANT — headroom prerequisite: FCR response is proportional to the
RESERVED headroom (`plant.fcr.mandatory_headroom_mw`). This repo's real
config currently sets that to 5.0 MW (an ESTIMATED obligation — see
config/plant.yaml for the reasoning chain; not a disclosed REN figure for
this plant), so a real run of this module against the shipped config
produces non-zero activation proportional to that headroom. For
demonstration purposes, this module also accepts an explicit
`demo_headroom_mw` override so the mechanism can be shown working at any
headroom level — this is clearly NOT the same as changing the production
config, and any caller must know it is illustrative.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from common_layer.configuration.config_loader import AppConfig
from common_layer.utilities import date_utils as du

# ── ENTSO-E Continental Europe FCR parameters (verified, see module docstring) ──
FCR_FAT_MIN = 0.5                 # 30 seconds
FCR_DEADBAND_MHZ = 10.0           # no response inside ±10 mHz
FCR_FULL_ACTIVATION_MHZ = 200.0   # full reserved headroom delivered at ±200 mHz

# Synthetic frequency deviation model parameters (illustrative, seeded per date
# for reproducibility — NOT real REN/ENTSO-E telemetry, same honesty standard
# as reserve_activation.py's stochastic TSO-call model).
#
# Modeled as a mean-reverting AR(1) process plus a small per-day bias, rather
# than i.i.d. noise per tick: real frequency excursions persist across
# consecutive ticks (a generation/load imbalance doesn't reset every 30s) and
# individual days can be systematically skewed rather than always netting to
# a near-exact up/down split.
_FREQ_STD_MHZ = 25.0        # stationary std-dev of the AR(1) process
_FREQ_AR_PHI = 0.85         # tick-to-tick persistence (0=white noise, 1=random walk)
_FREQ_DAILY_BIAS_STD_MHZ = 12.0   # per-day mean offset, drawn once per delivery_date
_FREQ_SPIKE_PROB = 0.02     # probability of a larger transient excursion per ISP
_FREQ_SPIKE_STD_MHZ = 120.0


@dataclass
class FCRActivationTick:
    """One real 30-second simulation tick — FCR's own native resolution,
    matching its real 30s Full Activation Time. This is the primitive;
    FCRActivationRow (ISP-grain) is an honest aggregation of these ticks,
    not a separate simulation."""
    tick: int                # 1-based, global index across the whole day
    isp: int                 # which 15-min ISP this tick belongs to
    hour: int
    time_hhmmss: str         # HH:MM:SS, start of this 30s tick
    freq_deviation_mhz: float
    direction: str           # "up", "down", or "none"
    response_mw: float
    energy_mwh: float        # response_mw * (30s / 3600s)


@dataclass
class FCRActivationRow:
    """ISP-grain (15-min) view, honestly AGGREGATED from real 30s ticks
    (see FCRActivationTick) — kept for compatibility with ISP-indexed
    reporting elsewhere, NOT a separate/coarser simulation."""
    isp: int
    hour: int
    freq_deviation_mhz: float   # mean |deviation| of this ISP's 30s ticks
    direction: str              # "up", "down", "mixed", or "none"
    response_mw: float          # mean |response| of this ISP's 30s ticks
    energy_mwh: float           # true sum of this ISP's 30s tick energies (net, up +, down -)


@dataclass
class FCRActivationSummary:
    reserved_headroom_mw: float
    n_isp_in_deadband: int
    n_isp_responding: int
    up_mwh: float
    dn_mwh: float
    max_response_mw: float
    rows: List[FCRActivationRow]
    ticks: List[FCRActivationTick]   # full 30s-resolution detail


def _simulate_frequency_deviation_mhz(rng: random.Random, prev_mhz: float, daily_bias_mhz: float) -> float:
    """Next tick's synthetic grid frequency deviation (mHz, + = over-frequency).

    Illustrative model, NOT real grid telemetry: a mean-reverting AR(1) process
    around a small per-day bias (so consecutive ticks are correlated, like a
    real imbalance event that persists for minutes rather than resetting every
    30s) with occasional larger transient spikes (loss of generation/load
    events) layered on top. Matches the qualitative shape of real frequency
    traces without claiming to reproduce any specific historical incident.
    """
    innovation_std = _FREQ_STD_MHZ * (1.0 - _FREQ_AR_PHI ** 2) ** 0.5
    base = daily_bias_mhz + _FREQ_AR_PHI * (prev_mhz - daily_bias_mhz) + rng.gauss(0.0, innovation_std)
    if rng.random() < _FREQ_SPIKE_PROB:
        return base + rng.gauss(0.0, _FREQ_SPIKE_STD_MHZ)
    return base


def fcr_droop_response_mw(freq_deviation_mhz: float, reserved_headroom_mw: float) -> float:
    """Linear droop response per ENTSO-E Continental Europe FCR parameters.

    + deviation (over-frequency) -> generators reduce output / increase pumping
      -> DOWN response (negative sign here, i.e. power reduced).
    - deviation (under-frequency) -> generators increase output
      -> UP response (positive sign here, i.e. power increased).

    Zero inside the ±10 mHz deadband. Full |reserved_headroom_mw| delivered
    at |deviation| >= 200 mHz. Linear in between. Sign convention matches
    p_net elsewhere in this codebase: + generation/up, - pumping/down.
    """
    dev = freq_deviation_mhz
    if abs(dev) <= FCR_DEADBAND_MHZ:
        return 0.0
    band = FCR_FULL_ACTIVATION_MHZ - FCR_DEADBAND_MHZ
    excess = min(abs(dev), FCR_FULL_ACTIVATION_MHZ) - FCR_DEADBAND_MHZ
    fraction = excess / band if band > 0 else 0.0
    magnitude = reserved_headroom_mw * fraction
    # Under-frequency (dev < 0) -> upward power response (+).
    # Over-frequency  (dev > 0) -> downward power response (-).
    return magnitude if dev < 0 else -magnitude


FCR_TICK_SECONDS = 30   # matches the real ENTSO-E FCR Full Activation Time exactly


def simulate_fcr_response(
    delivery_date: str,
    cfg: AppConfig,
    demo_headroom_mw: Optional[float] = None,
) -> FCRActivationSummary:
    """Simulate FCR droop response at its real native resolution: one fresh
    frequency-deviation sample every 30 seconds (FCR_TICK_SECONDS), matching
    the real ENTSO-E Full Activation Time — not once per 15-min ISP.

    Earlier versions of this module sampled once per ISP (96 samples/day),
    which conflated FCR (a continuous, automatic, non-remunerated physical
    response) with the ISP-based settlement grain used for the PAID aFRR/mFRR
    products. That was a real modelling gap: FCR's own regulatory FAT is 30s,
    not 15 minutes. Fixed by simulating every 30s tick directly
    (FCRActivationTick, 2,880/day) and honestly aggregating up to ISP grain
    (FCRActivationRow) only for callers that need it — the ISP rows are a
    real summary of real sub-ISP ticks, not a separate coarser simulation.

    Args:
        delivery_date: the date being simulated.
        cfg: AppConfig — real fcr.mandatory_headroom_mw is used unless
            demo_headroom_mw overrides it for illustration.
        demo_headroom_mw: if given, OVERRIDES cfg for demonstration purposes
            only. Callers must not present demo output as the production
            configuration's real behaviour.

    Returns FCRActivationSummary (rows = ISP-grain aggregate, ticks = full
    30s-grain detail). No settlement, no revenue (FCR is non-remunerated per
    Portuguese grid code / REN, verified in module docstring) — this is a
    physical/compliance simulation only.
    """
    reserved = (demo_headroom_mw if demo_headroom_mw is not None
                else max(0.0, cfg.plant.fcr.mandatory_headroom_mw))

    day = du.parse_date(delivery_date)
    isp_min = du.isp_duration_min(day)
    ticks_per_isp = (isp_min * 60) // FCR_TICK_SECONDS
    tick_h = FCR_TICK_SECONDS / 3600.0   # hours per tick, for energy integration

    rng = random.Random(f"fcr-{delivery_date}")
    daily_bias_mhz = rng.gauss(0.0, _FREQ_DAILY_BIAS_STD_MHZ)
    prev_dev_mhz = daily_bias_mhz
    hours = du.delivery_hours(day)

    ticks: List[FCRActivationTick] = []
    rows: List[FCRActivationRow] = []
    up_mwh = dn_mwh = 0.0
    n_deadband = n_responding = 0
    max_resp = 0.0
    tick_counter = 0

    for h in hours:
        for isp in du.hour_to_isps(h, day):
            isp_devs, isp_resps, isp_energies = [], [], []
            for _ in range(ticks_per_isp):
                tick_counter += 1
                dev = _simulate_frequency_deviation_mhz(rng, prev_dev_mhz, daily_bias_mhz)
                prev_dev_mhz = dev
                resp = fcr_droop_response_mw(dev, reserved)
                energy = resp * tick_h   # signed: + up, - down

                secs_since_midnight = (tick_counter - 1) * FCR_TICK_SECONDS
                hh = (secs_since_midnight // 3600) % 24
                mm = (secs_since_midnight % 3600) // 60
                ss = secs_since_midnight % 60
                time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

                if resp == 0.0:
                    n_deadband += 1
                    direction = "none"
                else:
                    n_responding += 1
                    direction = "up" if resp > 0 else "down"
                    if resp > 0:
                        up_mwh += abs(energy)
                    else:
                        dn_mwh += abs(energy)
                    max_resp = max(max_resp, abs(resp))

                ticks.append(FCRActivationTick(
                    tick=tick_counter, isp=isp, hour=h, time_hhmmss=time_str,
                    freq_deviation_mhz=round(dev, 2), direction=direction,
                    response_mw=round(resp, 4), energy_mwh=round(energy, 6),
                ))
                isp_devs.append(dev)
                isp_resps.append(resp)
                isp_energies.append(energy)

            # Honest ISP-grain aggregation of this ISP's real 30s ticks.
            net_energy = sum(isp_energies)
            n_up = sum(1 for r in isp_resps if r > 0)
            n_dn = sum(1 for r in isp_resps if r < 0)
            if n_up and n_dn:
                isp_direction = "mixed"
            elif n_up:
                isp_direction = "up"
            elif n_dn:
                isp_direction = "down"
            else:
                isp_direction = "none"
            nonzero_resps = [abs(r) for r in isp_resps if r != 0.0]
            mean_abs_dev = sum(abs(d) for d in isp_devs) / len(isp_devs)
            mean_abs_resp = sum(nonzero_resps) / len(nonzero_resps) if nonzero_resps else 0.0

            rows.append(FCRActivationRow(
                isp=isp, hour=h, freq_deviation_mhz=round(mean_abs_dev, 2),
                direction=isp_direction, response_mw=round(mean_abs_resp, 4),
                energy_mwh=round(net_energy, 6),
            ))

    return FCRActivationSummary(
        reserved_headroom_mw=reserved,
        n_isp_in_deadband=n_deadband,
        n_isp_responding=n_responding,
        up_mwh=round(up_mwh, 4),
        dn_mwh=round(dn_mwh, 4),
        max_response_mw=round(max_resp, 4),
        rows=rows,
        ticks=ticks,
    )
