"""Risk & Constraints — how close to physical/regulatory limits the plant
ran, and a live check of the invariants the test suite defines as
"correct" (tests/test_bug_regressions.py BUG-1..8 and friends), computed
against real production output instead of only at test time."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import data
import theme

st.title("⚠️ Risk & Constraints")

selected_date = st.session_state.get("selected_date")
report_ready = st.session_state.get("report_ready", False)

if not selected_date:
    st.warning("No runs found yet — visit Run & Monitor to start one.")
    st.stop()
if not report_ready:
    st.info(data.no_report_message(selected_date))
    st.stop()

report = data.load_daily_report(selected_date)
dispatch, isp, kpis = report["dispatch"], report["isp"], report["kpis"]

plant_cfg = data.load_plant_config()
market_cfg = data.load_market_config()
fcr_mw = plant_cfg["fcr"]["mandatory_headroom_mw"]
gen_cap_mw = market_cfg["bid_limits"]["max_generation_mw"]
pump_cap_mw = market_cfg["bid_limits"]["max_pump_mw"]
gen_cap_fcr_mw = gen_cap_mw - fcr_mw
pump_cap_fcr_mw = pump_cap_mw - fcr_mw

st.caption(
    f"Physical envelope: {gen_cap_mw:.1f} MW generation / {pump_cap_mw:.1f} MW pumping "
    f"(`config/market.yaml` bid_limits). {fcr_mw:.1f} MW is reserved off the top for "
    f"mandatory, non-remunerated FCR (`config/plant.yaml`), leaving a tradable envelope "
    f"of {gen_cap_fcr_mw:.1f} / {pump_cap_fcr_mw:.1f} MW."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Utilization % — computed here, not stored anywhere in the pipeline output
# ---------------------------------------------------------------------------

st.subheader("Envelope utilization")
net = dispatch["Plant_net_final_MW"]
gen_util_pct = (net.clip(lower=0) / gen_cap_fcr_mw * 100)
pump_util_pct = ((-net).clip(lower=0) / pump_cap_fcr_mw * 100)

fig = go.Figure()
fig.add_trace(go.Bar(x=dispatch["Hour"], y=gen_util_pct, name="Generation utilization %", marker_color=theme.COLOR_GEN))
fig.add_trace(go.Bar(x=dispatch["Hour"], y=-pump_util_pct, name="Pump utilization % (mirrored)", marker_color=theme.COLOR_PUMP))
fig.add_hline(y=100, line_dash="dot", line_color=theme.STATUS_CRITICAL, annotation_text="tradable envelope limit")
fig.add_hline(y=-100, line_dash="dot", line_color=theme.STATUS_CRITICAL)
theme.style_fig(fig, height=380, yaxis_title="% of tradable envelope")
st.plotly_chart(fig, width="stretch")
peak_gen = gen_util_pct.max()
peak_pump = pump_util_pct.max()
c1, c2 = st.columns(2)
c1.metric("Peak generation utilization", f"{peak_gen:.1f} %")
c2.metric("Peak pump utilization", f"{peak_pump:.1f} %")

st.markdown("---")

# ---------------------------------------------------------------------------
# FCR headroom margin
# ---------------------------------------------------------------------------

st.subheader("FCR headroom margin")
if {"Gen_headroom_MW", "Pump_headroom_MW"}.issubset(dispatch.columns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["Gen_headroom_MW"], name="Gen headroom MW", line=dict(color=theme.COLOR_GEN)))
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["Pump_headroom_MW"], name="Pump headroom MW", line=dict(color=theme.COLOR_PUMP)))
    fig.add_hline(y=0, line_dash="dot", line_color=theme.STATUS_CRITICAL, annotation_text="breach")
    theme.style_fig(fig, height=320, yaxis_title="MW")
    st.plotly_chart(fig, width="stretch")
    min_gen_hr = dispatch["Gen_headroom_MW"].min()
    min_pump_hr = dispatch["Pump_headroom_MW"].min()
    if min_gen_hr < -0.01 or min_pump_hr < -0.01:
        st.error(f"🔴 Headroom breached — min gen headroom {min_gen_hr:.2f} MW, min pump headroom {min_pump_hr:.2f} MW")
    else:
        st.success(f"🟢 Headroom never breached — min gen {min_gen_hr:.2f} MW, min pump {min_pump_hr:.2f} MW")

st.markdown("---")

# ---------------------------------------------------------------------------
# Constraint verification panel — promoted from the bottom of Summary_KPIs
# ---------------------------------------------------------------------------

st.subheader("Constraint verification (from Summary_KPIs)")
verify_rows = kpis[kpis["Section"].astype(str).str.contains("Constraint", case=False, na=False)]
if verify_rows.empty:
    st.info("No 'Constraint Verification' section found in this report's Summary_KPIs sheet.")
else:
    cols = st.columns(len(verify_rows)) if len(verify_rows) <= 4 else st.columns(4)
    for i, (_, row) in enumerate(verify_rows.iterrows()):
        col = cols[i % len(cols)]
        val = row["Value"]
        is_pass = str(val).strip().upper() == "YES"
        col.metric(str(row["Metric"]).strip(), str(val), delta=None)
        if isinstance(val, str) and val.strip().upper() in ("YES", "NO"):
            (col.success if is_pass else col.error)("PASS" if is_pass else "FAIL")

st.markdown("---")

# ---------------------------------------------------------------------------
# Invariant health checklist — computed from stored data, not re-derived
# from the test suite itself. Documents what's checkable vs not.
# ---------------------------------------------------------------------------

st.subheader("Production health checklist")
st.caption(
    "Live checks of the invariants `tests/test_bug_regressions.py` and "
    "friends define as 'correct' — run here against real output, not just "
    "at test time."
)

checks = []

# BUG-3 style: no simultaneous up+down activation in the same ISP.
if {"ISP", "Market", "Up_MW", "Dn_MW"}.issubset(isp.columns):
    simultaneous = isp[(isp["Up_MW"] > 0) & (isp["Dn_MW"] > 0)]
    checks.append(("No simultaneous up+down activation in the same ISP", simultaneous.empty,
                    f"{len(simultaneous)} ISP(s) with both directions active" if not simultaneous.empty else ""))

# BUG-6 style: BESS SOC within bounds (10-95% per plant.yaml).
if "BESS_SOC_pct" in dispatch.columns:
    soc = dispatch["BESS_SOC_pct"]
    breach = dispatch[(soc < 9.99) | (soc > 95.01)]
    checks.append(("BESS SOC stays within 10-95% bounds", breach.empty,
                    f"{len(breach)} hour(s) outside bounds" if not breach.empty else ""))

# Envelope never breached (physical + FCR-adjusted), reusing the headroom columns above.
if {"Gen_headroom_MW", "Pump_headroom_MW"}.issubset(dispatch.columns):
    env_breach = dispatch[(dispatch["Gen_headroom_MW"] < -0.01) | (dispatch["Pump_headroom_MW"] < -0.01)]
    checks.append(("Physical + FCR envelope never breached", env_breach.empty,
                    f"{len(env_breach)} hour(s) breached" if not env_breach.empty else ""))

# Mass balance / energy balance from the report's own flagged columns.
if "Mass_balance_error_hm3" in dispatch.columns:
    mb_breach = dispatch[dispatch["Mass_balance_error_hm3"].abs() > 1e-4]
    checks.append(("Reservoir mass balance error within tolerance (1e-4 hm3)", mb_breach.empty,
                    f"max error {dispatch['Mass_balance_error_hm3'].abs().max():.2e} hm3" if not mb_breach.empty else ""))
if "Energy_balance_check_MW" in dispatch.columns:
    eb_breach = dispatch[dispatch["Energy_balance_check_MW"].abs() > 0.1]
    checks.append(("Hourly energy balance within tolerance (0.1 MW)", eb_breach.empty,
                    f"max error {dispatch['Energy_balance_check_MW'].abs().max():.2f} MW" if not eb_breach.empty else ""))

for label, ok, detail in checks:
    icon = "🟢" if ok else "🔴"
    st.markdown(f"{icon} **{label}**" + (f" — {detail}" if detail else ""))

st.info(
    "**Not checkable from current exports:** FAT-mode-deliverability (needs per-ISP unit-mode "
    "data not currently in `ISP_Activation`), and settlement total = capacity + activation "
    "reconciliation (needs joining `Gate_Decisions` against `phase_5b` settlement output, not "
    "yet exposed as a single exported field). Flagged here rather than silently skipped."
)
