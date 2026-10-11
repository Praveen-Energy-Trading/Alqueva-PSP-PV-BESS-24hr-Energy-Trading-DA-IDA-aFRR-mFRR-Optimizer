"""
dispatch_sheet_builder.py — builds the Dispatch_Hourly DataFrame (24 rows).

Reads from:
  ComponentStore  — per-unit PSP, BESS, PV, reservoir, efficiency, inflow,
                    initial_state (reservoir/BESS at gate-open)
  PositionStore   — DA / IDA1 / IDA2 / IDA3 / XBID committed positions + prices
  ReserveStore    — aFRR / mFRR hourly offers (up/dn MW + cap prices)
  ActivationStore — aFRR / mFRR per-ISP activations (grouped to hourly revenue)
  DeliveryStore   — per-ISP scheduled vs actual (grouped to hourly imbalance)

Returns a pandas DataFrame with columns grouped A–L as per the design spec.
Missing component data (if ComponentStore file absent) falls back to zeros.

PHYSICS INTEGRITY NOTES (verified against plant.yaml and MILP constraints):
  - Energy balance: p_net = PSP_net + pv_used + p_dis − p_chg  (p_chg = grid charge
    only; pv_to_bess is internal PV→BESS and does NOT cross the grid boundary)
  - Mass balance: ΔV_upper = (inflow + q_pump − q_turb − spill) × dt / M3_PER_HM3
  - BESS SOC%: referenced to plant capacity 2.0 MWh (plant.yaml: bess.capacity_mwh)
  - Reservoir fill%: referenced to operational bounds from plant.yaml
      upper: 830 hm³ (floor) → 3150 hm³ (usable ceiling)
      lower: 5 hm³ (floor)  → 54 hm³ (capacity)
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from common_layer.database import (
    PositionStore, ReserveStore, DeliveryStore, ActivationStore, ComponentStore,
)
from common_layer.configuration.config_loader import load_config
from common_layer.optimisation_model.reserve_offer_builder import _envelope
from common_layer.utilities import date_utils as du

# ── Alqueva physical limits — confirmed from plant.yaml / market.yaml ────────
# Total plant generation/pump envelope (FCR-aware) is now computed dynamically
# via _envelope(cfg) at the top of build_dispatch_hourly() — see that call site.
# The two constants below are PSP-only (not FCR-reduced), used solely for the
# CF_turbine/CF_pump capacity-factor denominators, which are physical nameplate
# ratios, not the tradable envelope, so they correctly stay fixed.
_PSP_MAX_GEN_MW  = 518.4    # PSP turbine only: 4×129.6 (for CF_turbine denominator)
_PSP_MAX_PUMP_MW = 446.4    # PSP pump only: 4×111.6 (for CF_pump denominator)
_BESS_CAP_MWH  = 2.0        # BESS capacity (plant.yaml: bess.capacity_mwh)
_M3_PER_HM3    = 1_000_000.0

# Reservoir operational bounds (plant.yaml)
_UPPER_MIN_HM3  = 830.0     # hard operational floor
_UPPER_MAX_HM3  = 3150.0    # usable ceiling (upper_usable_hm3)
_LOWER_MIN_HM3  = 5.0       # hard operational floor
_LOWER_MAX_HM3  = 54.0      # capacity (lower_capacity_hm3)

# Default initial state (plant.yaml initial_state) — used when ComponentStore
# does not yet have initial_state (old runs before this field was added)
_DEFAULT_UPPER_INIT_HM3 = 2490.0
_DEFAULT_LOWER_INIT_HM3 = 27.0


def build_dispatch_hourly(delivery_date: str) -> pd.DataFrame:
    """Builds the real-hourly (24-row) summary sheet by honestly aggregating
    each hour's real ISPs (4 per hour post-transition, 1 per hour pre-
    transition -- hour_to_isps()/isp_duration_min() handle both). Every
    store this reads (PositionStore, ComponentStore, ReserveStore) is now
    ISP-keyed since the DA/IDA/XBID gates solve natively at ISP resolution
    -- a plain `.get(h, ...)` for h in 1..24 would silently read ISP KEYS
    1..24 (the first ~6 hours of the day) instead of real hourly values,
    which is exactly the bug this aggregation replaces (surfaced as large
    spurious Energy_balance_check_MW / Mass_balance_error_hm3 on the Risk &
    Constraints page)."""
    day = du.parse_date(delivery_date)
    hours = du.delivery_hours(day)
    isp_dt_h = du.isp_duration_min(day) / 60.0

    def isps_for(h: int) -> List[int]:
        return du.hour_to_isps(h, day)

    def avg(d: dict, isps: List[int], field: str, default: float = 0.0) -> float:
        """Rate/power fields (MW, m3/h, %, ratios) -- time-weighted mean is
        exact since every ISP in an hour has equal duration."""
        vals = [d.get(isp, {}).get(field, default) for isp in isps]
        return sum(vals) / len(vals) if vals else default

    def avg_list(d: dict, isps: List[int], field: str, n: int = 4) -> List[float]:
        sums = [0.0] * n
        for isp in isps:
            arr = d.get(isp, {}).get(field, [0.0] * n)
            for i in range(n):
                sums[i] += arr[i] if i < len(arr) else 0.0
        cnt = max(len(isps), 1)
        return [s / cnt for s in sums]

    def last(d: dict, isps: List[int], field: str, default: float = 0.0) -> float:
        """Stock fields (reservoir level, head) -- the value at the END of
        the hour (last ISP), matching the original hourly semantics of
        'reservoir level at this hour's close', not an average level."""
        if not isps:
            return default
        return d.get(isps[-1], {}).get(field, default)

    def vol_and_price(pos: dict, isps: List[int]) -> tuple[float, float]:
        """Position dicts: volume_mwh sums to the real hourly MWh (== MW
        since the hour is always 1h regardless of ISP count); price is the
        volume-weighted average across the hour's ISPs, falling back to a
        simple average when total traded volume is ~0."""
        total_vol = 0.0
        weighted = 0.0
        abs_weight = 0.0
        for isp in isps:
            row = pos.get(isp, {})
            v = row.get("volume_mwh", 0.0)
            p = row.get("price_eur_mwh", 0.0)
            total_vol += v
            weighted += p * abs(v)
            abs_weight += abs(v)
        if abs_weight > 1e-9:
            price = weighted / abs_weight
        else:
            prices = [pos.get(isp, {}).get("price_eur_mwh", 0.0) for isp in isps]
            price = sum(prices) / len(prices) if prices else 0.0
        return total_vol, price

    def has_any(pos: dict, isps: List[int]) -> bool:
        return any(isp in pos for isp in isps)

    def sum_revenue(pos: dict, isps: List[int]) -> float:
        """True per-ISP revenue summed across the hour -- NOT
        avg_price * total_volume, which is only exact when every ISP in the
        hour has the same sign of volume. Real mid-hour mode switches
        (turbine -> pump within the same hour) do happen, so the two are
        not interchangeable -- confirmed against 2026-08-16, where using
        the average-price shortcut for DA alone understated revenue by
        ~16k EUR (2%) versus this exact sum."""
        return sum(
            pos.get(isp, {}).get("volume_mwh", 0.0) * pos.get(isp, {}).get("price_eur_mwh", 0.0)
            for isp in isps
        )

    def sum_delta_revenue(cur_pos: dict, prev_pos: dict, isps: List[int]) -> float:
        """True per-ISP incremental revenue for a re-bidding gate: each ISP's
        (cur_volume - effective_prior_volume) settled at cur's price, summed
        across the hour. ISPs the gate didn't touch (outside its delivery
        window) contribute zero, matching the original per-hour behaviour of
        only counting hours actually in the gate's window."""
        total = 0.0
        for isp in isps:
            cur = cur_pos.get(isp)
            if cur is None:
                continue
            prev_vol = prev_pos.get(isp, {}).get("volume_mwh", 0.0)
            cur_vol = cur.get("volume_mwh", prev_vol)
            cur_prc = cur.get("price_eur_mwh", 0.0)
            total += cur_prc * (cur_vol - prev_vol)
        return total

    # FCR-aware envelope: reuse the SAME function reserve_offer_builder.py uses
    # (single source of truth) instead of the hardcoded _P_MAX_GEN_MW/_P_MAX_PUMP_MW
    # constants below, which never subtracted live FCR headroom and would silently
    # mask a real PR-11 breach if plant.yaml's fcr.mandatory_headroom_mw is ever
    # set above its current 0.0 (see Codebase_Interview_Prep_Book.md for the fuller
    # writeup of this bug).
    p_gen_cap, p_pump_cap = _envelope(load_config())

    # ── Load all stores ────────────────────────────────────────────────────────
    pos  = PositionStore()
    rsvr = ReserveStore()
    comp = ComponentStore().load(delivery_date) or {}

    da_pos   = pos.load_position(delivery_date, "DA")
    ida1_pos = pos.load_position(delivery_date, "IDA1")
    ida2_pos = pos.load_position(delivery_date, "IDA2")
    ida3_pos = pos.load_position(delivery_date, "IDA3")
    xbid_pos = pos.load_position(delivery_date, "XBID")

    # Load cumulative committed position (DA + all IDA/XBID deltas) once — used per hour below
    committed_final: Dict[int, float] = pos.committed_position(delivery_date)

    # Effective per-ISP position after each gate (latest gate that touched an ISP
    # wins, falling back to the previous gate for ISPs outside that gate's
    # delivery window) -- needed as the "prior" side of each gate's true
    # per-ISP incremental revenue (sum_delta_revenue below), since a gate's own
    # dict only has rows for the ISPs it actually re-bid.
    eff_after_ida1 = {isp: (ida1_pos[isp] if isp in ida1_pos else da_pos.get(isp, {})) for isp in da_pos}
    eff_after_ida2 = {isp: (ida2_pos[isp] if isp in ida2_pos else eff_after_ida1.get(isp, {})) for isp in da_pos}
    eff_after_ida3 = {isp: (ida3_pos[isp] if isp in ida3_pos else eff_after_ida2.get(isp, {})) for isp in da_pos}

    afrr_off = rsvr.load_reserve(delivery_date, "aFRR")
    mfrr_off = rsvr.load_reserve(delivery_date, "mFRR")

    act_afrr = ActivationStore().load(delivery_date, "aFRR")
    act_mfrr = ActivationStore().load(delivery_date, "mFRR")
    rt_rows  = DeliveryStore().load(delivery_date)

    psp_sched  = comp.get("psp_schedule", {})
    bess_sched = comp.get("bess_schedule", {})
    pv_sched   = comp.get("pv_schedule", {})
    res_traj   = comp.get("reservoir_trajectory", {})
    eff_ph     = comp.get("efficiency_per_hour", {})
    inflow_m3h = comp.get("inflow_m3h", {})

    # Seed reservoir tracker from stored gate-open state, not from the h=1 trajectory value
    init_st = comp.get("initial_state", {})
    prev_upper_hm3 = float(init_st.get("upper_reservoir_hm3", _DEFAULT_UPPER_INIT_HM3))

    # ── Aggregate ISP activation revenue per hour ──────────────────────────────
    # Ramp-corrected ISP duration: eff_isp_h = (isp_min − fat_min/2) / 60
    #   aFRR FAT=5 min:    (15 − 2.5)  / 60 = 0.208333h  (face 0.25h overstates energy by 20%)
    #   mFRR FAT=12.5 min: (15 − 6.25) / 60 = 0.145833h  (face 0.25h overstates energy by 71%)
    # ActivationStore always carries the correct value; these defaults guard against rows
    # written before the effective_isp_h column was added to the schema.
    _EFF_ISP_H_AFRR = round((15 - 5.0 / 2) / 60, 6)    # 0.208333h
    _EFF_ISP_H_MFRR = round((15 - 12.5 / 2) / 60, 6)   # 0.145833h

    afrr_act_rev: Dict[int, float] = {h: 0.0 for h in hours}
    for row in act_afrr:
        h = row["hour"]
        eff_h = row.get("eff_isp_h", _EFF_ISP_H_AFRR)   # ramp-corrected: (15-FAT/2)/60
        rev = (row.get("up_mw", 0.0) * row.get("up_price_eur_mwh", 0.0)
             + row.get("dn_mw", 0.0) * row.get("dn_price_eur_mwh", 0.0)) * eff_h
        afrr_act_rev[h] = afrr_act_rev.get(h, 0.0) + rev

    mfrr_act_rev: Dict[int, float] = {h: 0.0 for h in hours}
    for row in act_mfrr:
        h = row["hour"]
        eff_h = row.get("eff_isp_h", _EFF_ISP_H_MFRR)   # ramp-corrected: (15-12.5/2)/60
        rev = (row.get("up_mw", 0.0) * row.get("up_price_eur_mwh", 0.0)
             + row.get("dn_mw", 0.0) * row.get("dn_price_eur_mwh", 0.0)) * eff_h
        mfrr_act_rev[h] = mfrr_act_rev.get(h, 0.0) + rev

    # ── Aggregate imbalance settlement per hour ────────────────────────────────
    # MIBEL dual-pricing (market.yaml: fallback_long_factor=0.85, fallback_short_factor=1.20)
    # Long (over-delivered): TSO accepts surplus but pays DA × 0.85 (discount)
    # Short (under-delivered): plant buys back shortfall at DA × 1.20 (premium)
    # ISP duration 15 min → energy per ISP = MW × 0.25 h
    _IMB_LONG_FACTOR  = 0.85   # matches market.yaml imbalance.fallback_long_factor
    _IMB_SHORT_FACTOR = 1.20   # matches market.yaml imbalance.fallback_short_factor
    da_price_h = {h: vol_and_price(da_pos, isps_for(h))[1] for h in hours}
    imb_rev: Dict[int, float] = {h: 0.0 for h in hours}
    for row in rt_rows:
        h = row["hour"]
        dev = row.get("actual_mw", 0.0) - row.get("scheduled_mw", 0.0)
        da_p = da_price_h.get(h, 0.0)
        if dev > 0:   # long: over-delivered → receives DA × 0.85 per surplus MWh
            imb_rev[h] = imb_rev.get(h, 0.0) + dev * da_p * _IMB_LONG_FACTOR * isp_dt_h
        elif dev < 0: # short: under-delivered → pays DA × 1.20 per missing MWh (dev is negative)
            imb_rev[h] = imb_rev.get(h, 0.0) + dev * da_p * _IMB_SHORT_FACTOR * isp_dt_h

    # ── Build one row per hour ─────────────────────────────────────────────────
    rows = []
    cum_rev = 0.0
    cum_net_mwh = 0.0

    for h in hours:
        isps = isps_for(h)

        # --- GROUP A: Inputs ---
        da_vol, da_price = vol_and_price(da_pos, isps)
        pv_avail = avg(pv_sched, isps, "available_mw")
        inflow   = sum(float(inflow_m3h.get(isp, 0.0)) for isp in isps) / max(len(isps), 1)

        # --- GROUP B: PSP plant totals ---
        psp_gen  = avg(psp_sched, isps, "turbine_mw")
        psp_pump = avg(psp_sched, isps, "pump_mw")
        psp_net_da   = da_vol   # real hourly MWh == MW equivalent (hour is always 1h)
        final_vol    = sum(committed_final.get(isp, psp_net_da) for isp in isps) / max(len(isps), 1)
        # units_on_turb/pump average to a fractional "on for X of the hour's ISPs" --
        # honest for a mid-hour mode switch, and reduces to the exact original 0/1 on
        # pre-transition dates (1 ISP per hour).
        units_turb   = round(sum(avg_list(psp_sched, isps, "units_on_turb")), 2)
        units_pump_n = round(sum(avg_list(psp_sched, isps, "units_on_pump")), 2)
        da_side = "SELL" if psp_net_da > 0.01 else ("BUY" if psp_net_da < -0.01 else "IDLE")

        # --- GROUP C: Per-unit PSP (4 units) ---
        u_gen  = avg_list(psp_sched, isps, "units_turbine")
        u_pump = avg_list(psp_sched, isps, "units_pump")
        u_on_t = avg_list(psp_sched, isps, "units_on_turb")
        u_on_p = avg_list(psp_sched, isps, "units_on_pump")
        u_qt   = avg_list(psp_sched, isps, "units_q_turb")
        u_qp   = avg_list(psp_sched, isps, "units_q_pump")
        q_turb_total = avg(psp_sched, isps, "q_turb_total_m3h")
        q_pump_total = avg(psp_sched, isps, "q_pump_total_m3h")

        # --- GROUP D: PV ---
        pv_used     = avg(pv_sched, isps, "used_mw")
        pv_to_bess  = avg(pv_sched, isps, "to_bess_mw")
        pv_curt     = avg(pv_sched, isps, "curtailed_mw")

        # --- GROUP E: BESS ---
        bess_chg     = avg(bess_sched, isps, "charge_mw")       # grid → BESS (p_chg in MILP)
        bess_tot_chg = avg(bess_sched, isps, "total_charge_mw") # p_chg + pv_to_bess
        bess_dis     = avg(bess_sched, isps, "discharge_mw")
        bess_soc     = last(bess_sched, isps, "soc_mwh")        # stock -- value at hour-end
        bess_soc_pct = round(100.0 * bess_soc / _BESS_CAP_MWH, 1)  # ref: 2.0 MWh capacity

        # --- GROUP F: Reservoir & hydraulics ---
        upper_hm3 = last(res_traj, isps, "upper_hm3")   # stock -- level at hour-end
        lower_hm3 = last(res_traj, isps, "lower_hm3")
        spill_m3h = avg(res_traj, isps, "spill_m3h")
        head_m    = last(res_traj, isps, "head_m")

        upper_rng = _UPPER_MAX_HM3 - _UPPER_MIN_HM3    # 3150 − 830 = 2320 hm³ usable range
        lower_rng = _LOWER_MAX_HM3 - _LOWER_MIN_HM3    # 54 − 5 = 49 hm³ usable range
        upper_pct = round(100.0 * (upper_hm3 - _UPPER_MIN_HM3) / upper_rng, 1) if upper_rng else 0.0
        lower_pct = round(100.0 * (lower_hm3 - _LOWER_MIN_HM3) / lower_rng, 1) if lower_rng else 0.0

        dV_actual = upper_hm3 - prev_upper_hm3   # Δ volume vs prior-hour end state
        # Mass balance theoretical: integrate real per-ISP rates over each ISP's own
        # duration (isp_dt_h) and sum across the hour's ISPs, rather than treating an
        # already hour-averaged rate as if a full hour elapsed at that single rate --
        # exact for any ISP count per hour (4 normally, 3/5 on DST change days).
        dV_theoretical = sum(
            (float(inflow_m3h.get(isp, 0.0))
             + psp_sched.get(isp, {}).get("q_pump_total_m3h", 0.0)
             - psp_sched.get(isp, {}).get("q_turb_total_m3h", 0.0)
             - res_traj.get(isp, {}).get("spill_m3h", 0.0)) * isp_dt_h
            for isp in isps
        ) / _M3_PER_HM3
        mass_balance_err = round(abs(dV_actual - dV_theoretical), 6)
        prev_upper_hm3   = upper_hm3

        # --- GROUP G: Efficiency & capacity factors ---
        eta_trb = round(avg(eff_ph, isps, "eta_trb_pw"), 4)
        eta_pmp = round(avg(eff_ph, isps, "eta_pmp_pw"), 4)
        # CF denominator is PSP-only (518.4/446.4 MW) — mixing in PV+BESS capacity would understate CF
        cf_trb  = round(psp_gen  / _PSP_MAX_GEN_MW,  4) if _PSP_MAX_GEN_MW  else 0.0
        cf_pmp  = round(psp_pump / _PSP_MAX_PUMP_MW, 4) if _PSP_MAX_PUMP_MW else 0.0

        # --- GROUP H: IDA re-optimisation ---
        ida1_vol, ida1_prc = vol_and_price(ida1_pos, isps) if has_any(ida1_pos, isps) else (psp_net_da, da_price)
        ida1_del = round(ida1_vol - psp_net_da, 4)
        ida1_spr = round(ida1_prc - da_price, 4)

        ida2_vol, ida2_prc = vol_and_price(ida2_pos, isps) if has_any(ida2_pos, isps) else (ida1_vol, ida1_prc)
        ida2_del = round(ida2_vol - ida1_vol, 4)
        # Zero spread for hours outside IDA2 delivery window (H1–H2 frozen; market.yaml delivery_hours [3,24])
        # Propagated fallback price would otherwise misrepresent a trade that never happened.
        ida2_spr = round(ida2_prc - da_price, 4) if has_any(ida2_pos, isps) else 0.0

        ida3_vol, ida3_prc = vol_and_price(ida3_pos, isps) if has_any(ida3_pos, isps) else (ida2_vol, ida2_prc)
        ida3_del = round(ida3_vol - ida2_vol, 4)
        # Zero spread for hours outside IDA3 delivery window (H1–H11 frozen; market.yaml delivery_hours [12,24])
        ida3_spr = round(ida3_prc - da_price, 4) if has_any(ida3_pos, isps) else 0.0

        xbid_vol, xbid_prc = vol_and_price(xbid_pos, isps) if has_any(xbid_pos, isps) else (ida3_vol, ida3_prc)
        xbid_del = round(xbid_vol - ida3_vol, 4)
        ida_cum  = round(final_vol - psp_net_da, 4)

        # --- GROUP I: aFRR ---
        afrr_up  = avg(afrr_off, isps, "up_mw")
        afrr_dn  = avg(afrr_off, isps, "dn_mw")
        afrr_cup = avg(afrr_off, isps, "cap_up_eur_mw")
        afrr_cdn = avg(afrr_off, isps, "cap_dn_eur_mw")

        # --- GROUP J: mFRR ---
        mfrr_up  = avg(mfrr_off, isps, "up_mw")
        mfrr_dn  = avg(mfrr_off, isps, "dn_mw")
        mfrr_cup = avg(mfrr_off, isps, "cap_up_eur_mw")
        mfrr_cdn = avg(mfrr_off, isps, "cap_dn_eur_mw")

        # --- GROUP K: Physical headroom checks ---
        # PR-11: use committed net position (final_vol) — matches reserve_offer_builder.py
        # EXACTLY (same _envelope(cfg) call, computed once above — FCR-aware).
        # gen_headroom  = p_gen_cap  - final_vol  - afrr_up - mfrr_up
        # pump_headroom = final_vol  + p_pump_cap - afrr_dn - mfrr_dn
        # In pump mode (final_vol=-300): gen_hr = p_gen_cap-(-300) (full ramp-up range)
        # In gen mode  (final_vol=+500): gen_hr = p_gen_cap-500     (near cap)
        gen_hr  = round(p_gen_cap  - final_vol - afrr_up - mfrr_up, 2)
        pump_hr = round(final_vol + p_pump_cap - afrr_dn - mfrr_dn, 2)

        # --- GROUP L: Energy balance verification ---
        # MILP INV-1: p_net = (PSP_gen − PSP_pump) + pv_used + p_dis − p_chg
        # pv_to_bess is internal (PV panels → BESS internal bus), not grid-crossing,
        # so bess_chg (grid charge only) enters the balance — not bess_tot_chg.
        net_components = psp_gen - psp_pump + pv_used + bess_dis - bess_chg
        energy_balance = round(psp_net_da - net_components, 4)

        # --- GROUP L: Revenue per hour ---
        # DA settlement: true per-ISP volume x price, summed -- NOT
        # da_price * psp_net_da, which uses an hour-average price and would
        # misstate revenue on any hour with a mid-hour mode switch (real,
        # confirmed sign flips within a single hour on this plant).
        rev_da          = round(sum_revenue(da_pos, isps), 2)
        # IDA incremental: each ISP's delta volume settles at that ISP's real
        # price, summed across the hour -- kept per-gate (not just the
        # Rev_IDA_EUR sum below) so callers that need to distinguish which
        # gate actually re-bid don't have to guess from delta_MW alone.
        rev_ida1        = round(sum_delta_revenue(ida1_pos, da_pos, isps), 2)
        rev_ida2        = round(sum_delta_revenue(ida2_pos, eff_after_ida1, isps), 2)
        rev_ida3        = round(sum_delta_revenue(ida3_pos, eff_after_ida2, isps), 2)
        rev_xbid        = round(sum_delta_revenue(xbid_pos, eff_after_ida3, isps), 2)
        rev_ida         = round(rev_ida1 + rev_ida2 + rev_ida3 + rev_xbid, 2)
        # aFRR capacity revenue: MW × EUR/MW/h × 1h
        rev_afrr_cap_up = round(afrr_up * afrr_cup, 2)
        rev_afrr_cap_dn = round(afrr_dn * afrr_cdn, 2)
        rev_afrr_cap    = round(rev_afrr_cap_up + rev_afrr_cap_dn, 2)
        rev_afrr_act    = round(afrr_act_rev.get(h, 0.0), 2)
        # mFRR capacity revenue: MW × EUR/MW/h × 1h
        rev_mfrr_cap_up = round(mfrr_up * mfrr_cup, 2)
        rev_mfrr_cap_dn = round(mfrr_dn * mfrr_cdn, 2)
        rev_mfrr_cap    = round(rev_mfrr_cap_up + rev_mfrr_cap_dn, 2)
        rev_mfrr_act    = round(mfrr_act_rev.get(h, 0.0), 2)
        rev_imbalance   = round(imb_rev.get(h, 0.0), 2)
        rev_total       = round(rev_da + rev_ida + rev_afrr_cap + rev_afrr_act
                                + rev_mfrr_cap + rev_mfrr_act + rev_imbalance, 2)
        cum_rev        += rev_total
        cum_net_mwh    += final_vol

        rows.append({
            # A — Inputs
            "Hour":                       h,
            "DA_price_EUR_MWh":           round(da_price, 4),
            "PV_available_MW":            round(pv_avail, 4),
            "Reservoir_inflow_m3h":       round(inflow, 2),
            # B — PSP plant totals
            "DA_side":                    da_side,
            "PSP_gen_MW":                 round(psp_gen, 4),
            "PSP_pump_MW":                round(psp_pump, 4),
            # Plant_net includes PSP + PV + BESS (e.g. 200+4.2+0.5 = 204.7 MW); distinct from PSP_gen_MW
            "Plant_net_DA_MW":            round(psp_net_da, 4),
            "Plant_net_final_MW":         round(final_vol, 4),
            "Units_turbining":            units_turb,
            "Units_pumping":              units_pump_n,
            # C — Per-unit PSP
            "PSP_gen_u1_MW":              round(u_gen[0]  if len(u_gen)  > 0 else 0.0, 4),
            "PSP_gen_u2_MW":              round(u_gen[1]  if len(u_gen)  > 1 else 0.0, 4),
            "PSP_gen_u3_MW":              round(u_gen[2]  if len(u_gen)  > 2 else 0.0, 4),
            "PSP_gen_u4_MW":              round(u_gen[3]  if len(u_gen)  > 3 else 0.0, 4),
            "PSP_pump_u1_MW":             round(u_pump[0] if len(u_pump) > 0 else 0.0, 4),
            "PSP_pump_u2_MW":             round(u_pump[1] if len(u_pump) > 1 else 0.0, 4),
            "PSP_pump_u3_MW":             round(u_pump[2] if len(u_pump) > 2 else 0.0, 4),
            "PSP_pump_u4_MW":             round(u_pump[3] if len(u_pump) > 3 else 0.0, 4),
            "On_turb_u1":                 int(u_on_t[0]  if len(u_on_t) > 0 else 0),
            "On_turb_u2":                 int(u_on_t[1]  if len(u_on_t) > 1 else 0),
            "On_turb_u3":                 int(u_on_t[2]  if len(u_on_t) > 2 else 0),
            "On_turb_u4":                 int(u_on_t[3]  if len(u_on_t) > 3 else 0),
            "On_pump_u1":                 int(u_on_p[0]  if len(u_on_p) > 0 else 0),
            "On_pump_u2":                 int(u_on_p[1]  if len(u_on_p) > 1 else 0),
            "On_pump_u3":                 int(u_on_p[2]  if len(u_on_p) > 2 else 0),
            "On_pump_u4":                 int(u_on_p[3]  if len(u_on_p) > 3 else 0),
            "q_turb_u1_m3h":              round(u_qt[0]  if len(u_qt) > 0 else 0.0, 2),
            "q_turb_u2_m3h":              round(u_qt[1]  if len(u_qt) > 1 else 0.0, 2),
            "q_turb_u3_m3h":             round(u_qt[2]  if len(u_qt) > 2 else 0.0, 2),
            "q_turb_u4_m3h":              round(u_qt[3]  if len(u_qt) > 3 else 0.0, 2),
            "q_pump_u1_m3h":              round(u_qp[0]  if len(u_qp) > 0 else 0.0, 2),
            "q_pump_u2_m3h":              round(u_qp[1]  if len(u_qp) > 1 else 0.0, 2),
            "q_pump_u3_m3h":              round(u_qp[2]  if len(u_qp) > 2 else 0.0, 2),
            "q_pump_u4_m3h":              round(u_qp[3]  if len(u_qp) > 3 else 0.0, 2),
            "q_turb_total_m3h":           round(q_turb_total, 2),
            "q_pump_total_m3h":           round(q_pump_total, 2),
            # D — PV
            "PV_used_MW":                 round(pv_used, 4),
            "PV_to_BESS_MW":              round(pv_to_bess, 4),
            "PV_curtailed_MW":            round(pv_curt, 4),
            # E — BESS
            "BESS_charge_MW":             round(bess_chg, 4),
            "BESS_total_charge_MW":       round(bess_tot_chg, 4),
            "BESS_discharge_MW":          round(bess_dis, 4),
            "BESS_SOC_MWh":               round(bess_soc, 4),
            "BESS_SOC_pct":               bess_soc_pct,  # ref to 2.0 MWh capacity
            # F — Reservoir & hydraulics
            "Reservoir_upper_hm3":        round(upper_hm3, 4),
            "Reservoir_lower_hm3":        round(lower_hm3, 4),
            "Reservoir_upper_pct":        upper_pct,   # ref to 830–3150 hm³ range
            "Reservoir_lower_pct":        lower_pct,   # ref to 5–54 hm³ range
            "Head_net_m":                 round(head_m, 2),
            "Spill_m3h":                  round(spill_m3h, 2),
            "dReservoir_upper_hm3":       round(dV_actual, 6),
            "dReservoir_theoretical_hm3": round(dV_theoretical, 6),
            "Mass_balance_error_hm3":     mass_balance_err,
            # G — Efficiency & capacity factors
            "Eta_turbine_pw":             eta_trb,
            "Eta_pump_pw":                eta_pmp,
            "CF_turbine":                 cf_trb,
            "CF_pump":                    cf_pmp,
            # H — IDA re-optimisation
            "IDA1_price_EUR_MWh":         round(ida1_prc, 4),
            "IDA1_spread_EUR_MWh":        ida1_spr,
            "IDA1_delta_MW":              ida1_del,
            "IDA2_price_EUR_MWh":         round(ida2_prc, 4),
            "IDA2_spread_EUR_MWh":        ida2_spr,
            "IDA2_delta_MW":              ida2_del,
            "IDA3_price_EUR_MWh":         round(ida3_prc, 4),
            "IDA3_spread_EUR_MWh":        ida3_spr,
            "IDA3_delta_MW":              ida3_del,
            "XBID_delta_MW":              xbid_del,
            "IDA_cumulative_delta_MW":    ida_cum,
            # I — aFRR
            "aFRR_up_MW":                 round(afrr_up, 4),
            "aFRR_dn_MW":                 round(afrr_dn, 4),
            "aFRR_capUp_EUR_MW":          round(afrr_cup, 4),
            "aFRR_capDn_EUR_MW":          round(afrr_cdn, 4),
            # J — mFRR
            "mFRR_up_MW":                 round(mfrr_up, 4),
            "mFRR_dn_MW":                 round(mfrr_dn, 4),
            "mFRR_capUp_EUR_MW":          round(mfrr_cup, 4),
            "mFRR_capDn_EUR_MW":          round(mfrr_cdn, 4),
            # K — Headroom checks (≥ 0 = physical capacity not exceeded)
            "Gen_headroom_MW":            gen_hr,
            "Pump_headroom_MW":           pump_hr,
            # L — Balance and revenue
            "Energy_balance_check_MW":    energy_balance,   # should be 0
            "Rev_DA_EUR":                 rev_da,
            "Rev_IDA_EUR":                rev_ida,
            "Rev_IDA1_EUR":               rev_ida1,
            "Rev_IDA2_EUR":               rev_ida2,
            "Rev_IDA3_EUR":               rev_ida3,
            "Rev_XBID_EUR":               rev_xbid,
            "Rev_aFRR_cap_up_EUR":        rev_afrr_cap_up,
            "Rev_aFRR_cap_dn_EUR":        rev_afrr_cap_dn,
            "Rev_aFRR_cap_EUR":           rev_afrr_cap,
            "Rev_aFRR_act_EUR":           rev_afrr_act,
            "Rev_mFRR_cap_up_EUR":        rev_mfrr_cap_up,
            "Rev_mFRR_cap_dn_EUR":        rev_mfrr_cap_dn,
            "Rev_mFRR_cap_EUR":           rev_mfrr_cap,
            "Rev_mFRR_act_EUR":           rev_mfrr_act,
            "Rev_imbalance_EUR":          rev_imbalance,
            "Rev_hour_total_EUR":         rev_total,
            "Cum_Rev_EUR":                round(cum_rev, 2),
            "Cum_Net_MWh":                round(cum_net_mwh, 4),
        })

    df = pd.DataFrame(rows)
    # Zero-out solver floating-point noise near zero
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].mask(df[num_cols].abs() < 1e-6, 0.0)
    # Replace any NaN/Inf that openpyxl cannot write (causes silent Excel failure)
    df[num_cols] = df[num_cols].fillna(0.0).replace([float("inf"), float("-inf")], 0.0)
    return df
