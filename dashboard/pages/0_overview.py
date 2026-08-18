"""Overview — the one-glance boardroom page. Every element here mirrors
run_production.py's own 19-phase table, the same Summary_KPIs/Gate_Decisions
Excel sheets Trading Desk uses, and the same audit trail Decision Rationale
uses — nothing computed here that isn't already computed elsewhere; this
page is a compact front door to the other six, not a new data source."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import data
import gate_ticket
import delivery_ticket
import dispatch_ticket
import theme

st.title("⚡ Overview")

# Streamlit's default top padding on the main content block is ~35px,
# leaving a visible dead gap above the title before anything else on the
# page. Trimmed to match the standard 16px gap used between every other
# subsection below.
st.markdown(
    '<style>'
    'div[data-testid="stMainBlockContainer"] { padding-top: 1rem; }'
    # Tab strips with 6-11 wide labels (Market & Delivery, Optimization &
    # Physical Dispatch) overflow their container and require horizontal
    # scrolling to reach the later tabs. Wrapping onto 2-3 rows instead
    # keeps every button visible without side-scrolling to find one.
    'div[data-testid="stTabs"] [role="tablist"] { flex-wrap: wrap; overflow-x: visible; row-gap: 4px; }'
    'div[data-testid="stSegmentedControl"] div[role="radiogroup"] { flex-wrap: wrap; row-gap: 4px; }'
    '</style>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Asset strip — physical plant, straight from config/plant.yaml. Fixed
# regardless of which run/date is selected, since it's hardware, not a
# computed result — the same numbers run_production.py prints as its own
# "PLANT SPECS" header, just five cards instead of a text block.
# ---------------------------------------------------------------------------

plant_cfg = data.load_plant_config()
psp, pv, bess, fcr = plant_cfg["psp"], plant_cfg["pv"], plant_cfg["bess"], plant_cfg["fcr"]
n_units = psp["n_units"]

ASSET_CARDS = [
    ("🌀", "Turbines", f"{n_units} × {psp['p_turbine_max_mw']:.1f} MW", f"{n_units * psp['p_turbine_max_mw']:.1f} MW total"),
    ("🔽", "Pumps", f"{n_units} × {psp['p_pump_max_mw']:.1f} MW", f"{n_units * psp['p_pump_max_mw']:.1f} MW total"),
    ("☀️", "PV", f"{pv['peak_capacity_mw']:.1f} MWp", ""),
    ("🔋", "BESS", f"{bess['power_mw']:.1f} MW / {bess['capacity_mwh']:.1f} MWh", ""),
    ("🛡️", "FCR headroom", f"{fcr['mandatory_headroom_mw']:.1f} MW", "49.8–50.2 Hz, <30s"),
]

st.markdown(
    f'<div style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500; margin-bottom:8px;">'
    f'Alqueva PSP + PV + BESS</div>'
    '<div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:10px;">' +
    "".join(
        f'<div style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE}; border-radius:10px; padding:0.75rem 0.9rem;">'
        f'<div style="font-size:12px; color:{theme.INK_SECONDARY}; margin-bottom:4px;">{icon} {label}</div>'
        f'<div style="font-size:16px; font-weight:500; color:{theme.INK_PRIMARY};">{value}</div>'
        f'<div style="font-size:12px; color:{theme.INK_MUTED}; margin-top:2px;">{sub}&nbsp;</div>'
        f'</div>'
        for icon, label, value, sub in ASSET_CARDS
    ) +
    "</div>",
    unsafe_allow_html=True,
)

selected_date = st.session_state.get("selected_date")
report_ready = st.session_state.get("report_ready", False)

if not selected_date:
    st.warning("No runs found yet — visit Run & Monitor to start one.")
    st.stop()

run_status = data.load_run_status(selected_date)
state = data.run_phase_state(selected_date)

if state == "running":
    st.success(f"🟢 Pipeline is actively running right now for delivery **{selected_date}**.")
elif state == "idle_running":
    if run_status and run_status.get("mode") == "trader":
        st.warning(f"🟡 A run is in progress for **{selected_date}** but the log has "
                   f"gone quiet — most likely paused on an Approve/Reject or ENTER "
                   f"prompt waiting on you in trader mode. Check the terminal, or "
                   f"Console Log for the last line printed.")
    else:
        st.warning(f"🟡 A run is in progress for **{selected_date}** but the log has "
                   f"gone quiet — most likely still computing (e.g. a slow model "
                   f"fit) rather than stuck. Check Console Log for the last line printed.")
elif state == "stopped":
    st.error(f"⚫ A run for **{selected_date}** was started but the process is no longer "
             f"running — most likely Ctrl+C or a crash while it was paused waiting for "
             f"input. Start a fresh run when ready.")
elif state == "none":
    st.info(f"No run-status record for **{selected_date}** yet — see Run & Monitor.")
else:
    results = run_status["results"]
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    if n_fail:
        st.error(f"🔴 **{selected_date}** — {n_fail} phase(s) FAILED (finished {run_status['finished_at']}, mode={run_status['mode']})")
    else:
        st.success(f"🟢 **{selected_date}** — {n_pass}/{len(results)} phases passed cleanly (finished {run_status['finished_at']}, mode={run_status['mode']})")

# ---------------------------------------------------------------------------
# Live gate tickets — updates gate-by-gate as the pipeline runs, reading
# PositionStore/ReserveStore directly (SQLite), not the Excel report — so
# this shows real numbers even mid-run, before analytics has written
# anything. Every gate that has decided so far gets a compact strip card
# (so DA doesn't just vanish once aFRR becomes 'latest'); the most recent
# one also gets the full detailed chart below. See dashboard/gate_ticket.py.
# ---------------------------------------------------------------------------

all_tickets = data.all_gate_tickets(selected_date)
if all_tickets:
    gate_keys = [t["gate"] for t in all_tickets]
    state_key = f"overview_selected_gate_{selected_date}"
    if state_key not in st.session_state or st.session_state[state_key] not in gate_keys:
        st.session_state[state_key] = gate_keys[-1]  # default to the latest decision

    cols = st.columns(len(all_tickets))
    for i, t in enumerate(all_tickets):
        icon = {"good": "🟢", "neutral": "⚪", "critical": "🔴"}[t["status_class"]]
        rev = f"€{t['revenue_items'][0][1]:,.0f}" if t.get("revenue_items") else "—"
        label = f"{icon} {gate_ticket.gate_caption_name(t['gate'])}\n{t['decision']} · {rev}"
        is_selected = t["gate"] == st.session_state[state_key]
        if cols[i].button(label, key=f"gatebtn_{selected_date}_{t['gate']}",
                           use_container_width=True, type="primary" if is_selected else "secondary"):
            st.session_state[state_key] = t["gate"]

    selected_ticket = next(t for t in all_tickets if t["gate"] == st.session_state[state_key])
    fig_num = gate_keys.index(st.session_state[state_key]) + 1
    # Card height varies by content: standard DA/aFRR/mFRR bar-chart cards
    # are shorter (no tradable-hours section), rebid cards (IDA1-3) are
    # taller, and XBID's multi-window pill row adds a bit more on top of
    # that. One fixed height sized for the tallest case left a large blank
    # gap in the iframe below every shorter card.
    if selected_ticket.get("xbid_windows"):
        card_height = 325
    elif selected_ticket.get("is_rebid_gate"):
        card_height = 285
    else:
        card_height = 345
    components.html(gate_ticket.render(selected_ticket, selected_date, fig_num=fig_num), height=card_height)
    st.markdown("---")

# ---------------------------------------------------------------------------
# Delivery cards — RT dispatch, aFRR activation, mFRR activation, FCR droop
# response. Not decisions like the gates above, so they get their own tab
# strip rather than sitting in the gate-ticket buttons. Only rendered once
# each phase has actually produced data for this date. FCR is the odd one
# out: it's a standalone, non-remunerated compliance simulation, not read
# from a pipeline run's audit trail — see data.py::load_fcr_activation.
# ---------------------------------------------------------------------------

rt = data.load_rt_delivery(selected_date)
afrr_act = data.load_activation_summary(selected_date, "aFRR")
mfrr_act = data.load_activation_summary(selected_date, "mFRR")
fcr_act = data.load_fcr_activation(selected_date)
afrr_agc = data.load_agc_mechanism_demo(selected_date, "aFRR")
mfrr_agc = data.load_agc_mechanism_demo(selected_date, "mFRR")
reservoir_traj = data.load_reservoir_trajectory(selected_date)
pv_routing = data.load_pv_routing(selected_date)
multi_asset = data.load_multi_asset_dispatch(selected_date)
water_balance = data.load_water_balance(selected_date)
bess_soc_price = data.load_bess_soc_price(selected_date)
bess_charge_source = data.load_bess_charge_source(selected_date)
da_vs_activation = data.load_da_vs_activation(selected_date)
isp_dispatch = data.load_isp_dispatch(selected_date)
afrr_dispatch = data.load_afrr_dispatch(selected_date, "aFRR")
mfrr_dispatch = data.load_afrr_dispatch(selected_date, "mFRR")
market_cards = [
    ("ISP dispatch", lambda: components.html(delivery_ticket.render_rt_card(rt), height=374)) if rt else None,
    ("aFRR activation", lambda: components.html(delivery_ticket.render_activation_card(afrr_act, "aFRR"), height=448)) if afrr_act else None,
    ("mFRR activation", lambda: components.html(delivery_ticket.render_activation_card(mfrr_act, "mFRR"), height=448)) if mfrr_act else None,
    ("FCR response", lambda: components.html(delivery_ticket.render_fcr_card(fcr_act), height=485)) if fcr_act else None,
    ("aFRR AGC mechanism", lambda: components.html(delivery_ticket.render_agc_mechanism_card(afrr_agc, "aFRR"), height=497)) if afrr_agc else None,
    ("mFRR AGC mechanism", lambda: components.html(delivery_ticket.render_agc_mechanism_card(mfrr_agc, "mFRR"), height=497)) if mfrr_agc else None,
]
technical_cards = [
    ("Reservoir trajectory", lambda: components.html(dispatch_ticket.render_reservoir_trajectory_card(reservoir_traj), height=458)) if reservoir_traj else None,
    ("PV routing & curtailment", lambda: components.html(dispatch_ticket.render_pv_routing_card(pv_routing), height=347)) if pv_routing else None,
    ("Multi-asset dispatch", lambda: components.html(dispatch_ticket.render_multi_asset_dispatch_card(multi_asset), height=377)) if multi_asset else None,
    ("Water balance", lambda: components.html(dispatch_ticket.render_water_balance_card(water_balance), height=340)) if water_balance else None,
    ("BESS SOC vs price", lambda: components.html(dispatch_ticket.render_bess_soc_price_card(bess_soc_price), height=341)) if bess_soc_price else None,
    ("BESS charge source", lambda: components.html(dispatch_ticket.render_bess_charge_source_card(bess_charge_source), height=320)) if bess_charge_source else None,
    ("DA vs ISP activation", lambda: components.html(dispatch_ticket.render_da_vs_activation_card(da_vs_activation), height=296)) if da_vs_activation else None,
    ("ISP asset dispatch (96-pt)", lambda: components.html(dispatch_ticket.render_isp_dispatch_card(isp_dispatch), height=374)) if isp_dispatch else None,
    ("aFRR dispatch (BESS vs PSP)", lambda: components.html(dispatch_ticket.render_afrr_dispatch_card(afrr_dispatch), height=380)) if afrr_dispatch else None,
    ("mFRR dispatch (BESS vs PSP)", lambda: components.html(dispatch_ticket.render_afrr_dispatch_card(mfrr_dispatch), height=380)) if mfrr_dispatch else None,
    ("FCR dispatch (droop + headroom)", lambda: components.html(dispatch_ticket.render_fcr_dispatch_card(fcr_act), height=366)) if fcr_act else None,
]
market_cards = [c for c in market_cards if c is not None]
technical_cards = [c for c in technical_cards if c is not None]

# Plain st.tabs has no session_state key, so its selected tab is pure
# client-side DOM state — a full script rerun (e.g. every auto-refresh
# tick while a pipeline is live) always snaps it back to the first tab.
# st.segmented_control is keyed in session_state, so the selection
# survives reruns instead of visibly jumping back every couple of seconds.
def _render_section(title: str, cards: list, state_key: str) -> None:
    if not cards:
        return
    st.markdown(f"##### {title}")
    labels = [label for label, _ in cards]
    if st.session_state.get(state_key) not in labels:
        st.session_state[state_key] = labels[0]
    selected = st.session_state[state_key]
    if len(labels) > 1:
        selected = st.segmented_control(
            title, labels, key=state_key, label_visibility="collapsed",
        ) or labels[0]
    dict(cards)[selected]()
    st.markdown("---")


_render_section("Market & Delivery", market_cards, "overview_market_tab")
_render_section("Optimization & Physical Dispatch", technical_cards, "overview_dispatch_tab")

if not report_ready:
    st.info(data.no_report_message(selected_date))
    st.stop()

report = data.load_daily_report(selected_date)
kpis = report["kpis"]
dispatch = report["dispatch"]

# ---------------------------------------------------------------------------
# P&L breakdown — every real revenue line item as its own proportion bar,
# not folded into "IDA + XBID" or "Reserve" buckets, since those hide which
# gate/product actually moved the needle.
# ---------------------------------------------------------------------------

total_pnl = data.kpi_value(kpis, "Total daily P&L") or 0.0
reserve_pct = data.kpi_value(kpis, "Reserve share of P&L")
pnl_lines = [
    ("DA",                data.kpi_value(kpis, "DA energy revenue") or 0.0,               theme.COLOR_GEN),
    ("IDA1",              data.kpi_value(kpis, "IDA1 incremental revenue") or 0.0,         theme.COLOR_PRICE),
    ("IDA2",              data.kpi_value(kpis, "IDA2 incremental revenue") or 0.0,         theme.COLOR_PRICE),
    ("IDA3",              data.kpi_value(kpis, "IDA3 incremental revenue") or 0.0,         theme.COLOR_PRICE),
    ("XBID",              data.kpi_value(kpis, "XBID incremental revenue") or 0.0,         theme.STATUS_NEUTRAL),
    ("aFRR capacity",     data.kpi_value(kpis, "aFRR capacity revenue") or 0.0,            theme.COLOR_UP),
    ("aFRR activation",   data.kpi_value(kpis, "aFRR activation revenue") or 0.0,          theme.COLOR_UP),
    ("mFRR capacity",     data.kpi_value(kpis, "mFRR capacity revenue") or 0.0,            theme.COLOR_PUMP),
    ("mFRR activation",   data.kpi_value(kpis, "mFRR activation revenue") or 0.0,          theme.COLOR_PUMP),
    ("Imbalance settlement", data.kpi_value(kpis, "Imbalance settlement") or 0.0,          theme.STATUS_GOOD),
]
st.markdown("##### P&L Breakdown")
components.html(dispatch_ticket.render_pnl_breakdown_card(total_pnl, reserve_pct, pnl_lines), height=510)

# ---------------------------------------------------------------------------
# Dispatch profile + Reserve capacity offered -- same widgets as Trading
# Desk's (moved here too, ownership-wise this is the physical/delivery
# page; Trading Desk's copy left in place pending merge/dedup review).
# ---------------------------------------------------------------------------

st.markdown("##### Dispatch profile (hourly)")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                     row_heights=[0.6, 0.4])
fig.add_trace(go.Bar(x=dispatch["Hour"], y=dispatch["Plant_net_final_MW"],
                      name="Net dispatch MW", marker_color=theme.COLOR_GEN), row=1, col=1)
fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["DA_price_EUR_MWh"],
                          name="DA price EUR/MWh", line=dict(color=theme.COLOR_PRICE, width=2)), row=2, col=1)
theme.style_fig(fig, height=460)
fig.update_yaxes(title_text="MW", gridcolor=theme.GRIDLINE, row=1, col=1)
fig.update_yaxes(title_text="EUR/MWh", gridcolor=theme.GRIDLINE, row=2, col=1)
fig.update_xaxes(title_text="Hour", row=2, col=1)
st.plotly_chart(fig, width="stretch")

st.markdown("##### Reserve capacity offered (aFRR / mFRR, hourly)")
fig = go.Figure()
if "aFRR_up_MW" in dispatch.columns:
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["aFRR_up_MW"], name="aFRR up", line=dict(color=theme.COLOR_UP)))
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["aFRR_dn_MW"], name="aFRR dn", line=dict(color=theme.COLOR_DOWN)))
if "mFRR_up_MW" in dispatch.columns:
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["mFRR_up_MW"], name="mFRR up", line=dict(color=theme.COLOR_UP, dash="dot")))
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["mFRR_dn_MW"], name="mFRR dn", line=dict(color=theme.COLOR_DOWN, dash="dot")))
theme.style_fig(fig, height=380, yaxis_title="MW", xaxis_title="Hour")
st.plotly_chart(fig, width="stretch")
