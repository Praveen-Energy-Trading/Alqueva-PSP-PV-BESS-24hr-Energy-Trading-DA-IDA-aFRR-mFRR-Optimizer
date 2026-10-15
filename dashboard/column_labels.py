"""column_labels.py - readable column headings for the raw report tables (Plant_net_DA_MW -> "Plant net, DA (MW)").

The data keeps its own column names; only what the table shows is renamed. A column that is not listed here is shown with its underscores as spaces."""
from __future__ import annotations

_LABELS = {
    "Hour": "Hour", "DA_price_EUR_MWh": "DA price (EUR/MWh)", "PV_available_MW": "PV available (MW)",
    "Reservoir_inflow_m3h": "Reservoir inflow (m³/h)", "DA_side": "DA side",
    "PSP_gen_MW": "PSP generation (MW)", "PSP_pump_MW": "PSP pumping (MW)",
    "Plant_net_DA_MW": "Plant net, DA (MW)", "Plant_net_final_MW": "Plant net, final (MW)",
    "Units_turbining": "Units turbining", "Units_pumping": "Units pumping",
    "q_turb_total_m3h": "Turbine flow, total (m³/h)", "q_pump_total_m3h": "Pump flow, total (m³/h)",
    "PV_used_MW": "PV used (MW)", "PV_to_BESS_MW": "PV to BESS (MW)", "PV_curtailed_MW": "PV curtailed (MW)",
    "BESS_charge_MW": "BESS charge (MW)", "BESS_total_charge_MW": "BESS total charge (MW)", "BESS_discharge_MW": "BESS discharge (MW)",
    "BESS_SOC_MWh": "BESS state of charge (MWh)", "BESS_SOC_pct": "BESS state of charge (%)",
    "Reservoir_upper_hm3": "Upper reservoir (hm³)", "Reservoir_lower_hm3": "Lower reservoir (hm³)",
    "Reservoir_upper_pct": "Upper reservoir (%)", "Reservoir_lower_pct": "Lower reservoir (%)",
    "Head_net_m": "Net head (m)", "Spill_m3h": "Spill (m³/h)",
    "dReservoir_upper_hm3": "Upper reservoir change (hm³)", "dReservoir_theoretical_hm3": "Upper reservoir change, expected (hm³)",
    "Mass_balance_error_hm3": "Water balance error (hm³)",
    "Eta_turbine_pw": "Turbine efficiency", "Eta_pump_pw": "Pump efficiency", "CF_turbine": "Turbine capacity factor", "CF_pump": "Pump capacity factor",
    "XBID_delta_MW": "XBID change (MW)", "IDA_cumulative_delta_MW": "IDA cumulative change (MW)",
    "aFRR_up_MW": "aFRR up capacity (MW)", "aFRR_dn_MW": "aFRR down capacity (MW)",
    "aFRR_capUp_EUR_MW": "aFRR up capacity price (EUR/MW)", "aFRR_capDn_EUR_MW": "aFRR down capacity price (EUR/MW)",
    "mFRR_up_MW": "mFRR up capacity (MW)", "mFRR_dn_MW": "mFRR down capacity (MW)",
    "mFRR_capUp_EUR_MW": "mFRR up capacity price (EUR/MW)", "mFRR_capDn_EUR_MW": "mFRR down capacity price (EUR/MW)",
    "Gen_headroom_MW": "Generation headroom (MW)", "Pump_headroom_MW": "Pumping headroom (MW)",
    "Energy_balance_check_MW": "Energy balance check (MW)",
    "Rev_DA_EUR": "DA revenue (EUR)", "Rev_IDA_EUR": "IDA revenue (EUR)", "Rev_XBID_EUR": "XBID revenue (EUR)",
    "Rev_aFRR_cap_up_EUR": "aFRR up capacity revenue (EUR)", "Rev_aFRR_cap_dn_EUR": "aFRR down capacity revenue (EUR)",
    "Rev_aFRR_cap_EUR": "aFRR capacity revenue (EUR)", "Rev_aFRR_act_EUR": "aFRR activation revenue (EUR)",
    "Rev_mFRR_cap_up_EUR": "mFRR up capacity revenue (EUR)", "Rev_mFRR_cap_dn_EUR": "mFRR down capacity revenue (EUR)",
    "Rev_mFRR_cap_EUR": "mFRR capacity revenue (EUR)", "Rev_mFRR_act_EUR": "mFRR activation revenue (EUR)",
    "Rev_imbalance_EUR": "Imbalance revenue (EUR)", "Rev_hour_total_EUR": "Total revenue in the hour (EUR)",
    "Cum_Rev_EUR": "Cumulative revenue (EUR)", "Cum_Net_MWh": "Cumulative net energy (MWh)",
    # the quarter-hour reserve activation table
    "ISP": "ISP", "Market": "Market", "Up_MW": "Up (MW)", "Dn_MW": "Down (MW)", "Up_price_EUR_MWh": "Up price (EUR/MWh)",
    "Dn_price_EUR_MWh": "Down price (EUR/MWh)", "Duration_h": "Duration (h)", "Revenue_EUR": "Revenue (EUR)",
}
for _g in ("IDA1", "IDA2", "IDA3"):
    _LABELS[f"{_g}_price_EUR_MWh"] = f"{_g} price (EUR/MWh)"
    _LABELS[f"{_g}_spread_EUR_MWh"] = f"{_g} spread (EUR/MWh)"
    _LABELS[f"{_g}_delta_MW"] = f"{_g} change (MW)"
    _LABELS[f"Rev_{_g}_EUR"] = f"{_g} revenue (EUR)"
for _u in range(1, 5):
    _LABELS[f"PSP_gen_u{_u}_MW"] = f"Unit {_u} generation (MW)"
    _LABELS[f"PSP_pump_u{_u}_MW"] = f"Unit {_u} pumping (MW)"
    _LABELS[f"On_turb_u{_u}"] = f"Unit {_u} turbine on"
    _LABELS[f"On_pump_u{_u}"] = f"Unit {_u} pump on"
    _LABELS[f"q_turb_u{_u}_m3h"] = f"Unit {_u} turbine flow (m³/h)"
    _LABELS[f"q_pump_u{_u}_m3h"] = f"Unit {_u} pump flow (m³/h)"


def label(column: str) -> str:
    return _LABELS.get(column, str(column).replace("_", " "))


def readable(df):
    """The same table with readable column headings."""
    return df.rename(columns={c: label(c) for c in df.columns})


# the hourly dispatch table is far too wide to read in one piece (98 columns), so it is shown one topic at a time
GROUPS = {
    "Market & position": ["Hour", "DA_price_EUR_MWh", "DA_side", "Plant_net_DA_MW", "IDA1_price_EUR_MWh", "IDA1_delta_MW", "IDA2_price_EUR_MWh", "IDA2_delta_MW",
                          "IDA3_price_EUR_MWh", "IDA3_delta_MW", "XBID_delta_MW", "IDA_cumulative_delta_MW", "Plant_net_final_MW"],
    "Pumped storage": ["Hour", "PSP_gen_MW", "PSP_pump_MW", "Units_turbining", "Units_pumping", "Gen_headroom_MW", "Pump_headroom_MW",
                       "Eta_turbine_pw", "Eta_pump_pw", "CF_turbine", "CF_pump", "Energy_balance_check_MW"],
    "Units (power)": ["Hour"] + [f"PSP_gen_u{u}_MW" for u in range(1, 5)] + [f"PSP_pump_u{u}_MW" for u in range(1, 5)],
    "Units (water flow)": ["Hour"] + [f"q_turb_u{u}_m3h" for u in range(1, 5)] + ["q_turb_total_m3h"] + [f"q_pump_u{u}_m3h" for u in range(1, 5)] + ["q_pump_total_m3h"],
    "Water & reservoirs": ["Hour", "Reservoir_inflow_m3h", "Spill_m3h", "Reservoir_upper_hm3", "Reservoir_upper_pct", "Reservoir_lower_hm3", "Reservoir_lower_pct",
                           "Head_net_m", "dReservoir_upper_hm3", "dReservoir_theoretical_hm3", "Mass_balance_error_hm3"],
    "PV & battery": ["Hour", "PV_available_MW", "PV_used_MW", "PV_to_BESS_MW", "PV_curtailed_MW", "BESS_charge_MW", "BESS_total_charge_MW", "BESS_discharge_MW",
                     "BESS_SOC_MWh", "BESS_SOC_pct"],
    "Reserves": ["Hour", "aFRR_up_MW", "aFRR_dn_MW", "aFRR_capUp_EUR_MW", "aFRR_capDn_EUR_MW", "mFRR_up_MW", "mFRR_dn_MW", "mFRR_capUp_EUR_MW", "mFRR_capDn_EUR_MW"],
    "Revenue": ["Hour", "Rev_DA_EUR", "Rev_IDA1_EUR", "Rev_IDA2_EUR", "Rev_IDA3_EUR", "Rev_XBID_EUR", "Rev_aFRR_cap_EUR", "Rev_aFRR_act_EUR",
                "Rev_mFRR_cap_EUR", "Rev_mFRR_act_EUR", "Rev_imbalance_EUR", "Rev_hour_total_EUR", "Cum_Rev_EUR", "Cum_Net_MWh"],
}


def topic(df, name):
    """The columns of one topic, with readable headings (a column the report does not have is left out)."""
    return readable(df[[c for c in GROUPS[name] if c in df.columns]])
