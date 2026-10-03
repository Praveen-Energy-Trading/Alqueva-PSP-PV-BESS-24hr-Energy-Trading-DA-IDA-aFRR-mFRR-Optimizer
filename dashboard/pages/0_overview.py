"""Overview - the one-glance boardroom page. Every element here mirrors
run_production.py's own 19-phase table and the same Summary_KPIs/
Gate_Decisions Excel sheets Trading Desk uses - nothing computed here
that isn't already computed elsewhere; this page is a compact front
door to the other pages, not a new data source."""
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
# Asset strip - physical plant, straight from config/plant.yaml. Fixed
# regardless of which run/date is selected, since it's hardware, not a
# computed result - the same numbers run_production.py prints as its own
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

@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    selected_date = st.session_state.get("selected_date")
    report_ready = st.session_state.get("report_ready", False)

    if not selected_date:
        st.warning("No runs found yet - visit Run & Monitor to start one.")
        return

    run_status = data.load_run_status(selected_date)
    state = data.run_phase_state(selected_date)

    if state == "running":
        st.success(f"🟢 Pipeline is actively running right now for delivery **{selected_date}**.")
    elif state == "idle_running":
        if run_status and run_status.get("mode") == "trader":
            st.warning(f"🟡 A run is in progress for **{selected_date}** but the log has "
                       f"gone quiet - most likely paused on an Approve/Reject or ENTER "
                       f"prompt waiting on you in trader mode. Check the terminal, or "
                       f"Console Log for the last line printed.")
        else:
            st.warning(f"🟡 A run is in progress for **{selected_date}** but the log has "
                       f"gone quiet - most likely still computing (e.g. a slow model "
                       f"fit) rather than stuck. Check Console Log for the last line printed.")
    elif state == "stopped":
        st.error(f"⚫ A run for **{selected_date}** was started but the process is no longer "
                 f"running - most likely Ctrl+C or a crash while it was paused waiting for "
                 f"input. Start a fresh run when ready.")
    elif state == "none":
        st.info(f"No run-status record for **{selected_date}** yet - see Run & Monitor.")
    else:
        results = run_status["results"]
        n_fail = sum(1 for r in results if r["status"] == "FAIL")
        n_pass = sum(1 for r in results if r["status"] == "PASS")
        if n_fail:
            st.error(f"🔴 **{selected_date}** - {n_fail} phase(s) FAILED (finished {run_status['finished_at']}, mode={run_status['mode']})")
        else:
            st.success(f"🟢 **{selected_date}** - {n_pass}/{len(results)} phases passed cleanly (finished {run_status['finished_at']}, mode={run_status['mode']})")

    # ---------------------------------------------------------------------------
    # Live gate tickets - updates gate-by-gate as the pipeline runs, reading
    # PositionStore/ReserveStore directly (SQLite), not the Excel report - so
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

        # A plain `if cols[i].button(...): st.session_state[state_key] = ...`
        # only writes the new selection AFTER that button has already been
        # drawn -- so the just-clicked button still renders with its OLD
        # (now-stale) primary/secondary type, and the highlight only catches
        # up on the NEXT rerun. An on_click callback fires before the script
        # body re-runs, so session_state already holds the new gate by the
        # time `is_selected` is computed below -- the highlight and the
        # card switch land in the same click, not one apart.
        def _select_gate(gate: str) -> None:
            st.session_state[state_key] = gate

        cols = st.columns(len(all_tickets))
        for i, t in enumerate(all_tickets):
            icon = {"good": "🟢", "neutral": "⚪", "critical": "🔴"}[t["status_class"]]
            rev = f"€{t['revenue_items'][0][1]:,.0f}" if t.get("revenue_items") else " - "
            label = f"{icon} {gate_ticket.gate_caption_name(t['gate'])}\n{t['decision']} · {rev}"
            is_selected = t["gate"] == st.session_state[state_key]
            cols[i].button(label, key=f"gatebtn_{selected_date}_{t['gate']}",
                            use_container_width=True, type="primary" if is_selected else "secondary",
                            on_click=_select_gate, args=(t["gate"],))

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
        # +20px for the "Gate closes ..." line added below the decided
        # timestamp (see gate_ticket.py's gate_close_bit) -- every real gate
        # has one, but guard on presence anyway so a ticket without it
        # doesn't get a blank gap under it.
        if selected_ticket.get("gate_close"):
            card_height += 20
        components.html(gate_ticket.render(selected_ticket, selected_date, fig_num=fig_num), height=card_height)
        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Delivery cards - RT dispatch, aFRR activation, mFRR activation, FCR droop
    # response. Not decisions like the gates above, so they get their own tab
    # strip rather than sitting in the gate-ticket buttons. Only rendered once
    # each phase has actually produced data for this date. FCR is the odd one
    # out: it's a standalone, non-remunerated compliance simulation, not read
    # from a pipeline run's audit trail - see data.py::load_fcr_activation.
    # ---------------------------------------------------------------------------

    rt = data.load_rt_delivery(selected_date)
    imbalance = data.load_imbalance_settlement(selected_date)
    capacity_vs_activation = data.load_capacity_vs_activation(selected_date)
    afrr_act = data.load_activation_summary(selected_date, "aFRR")
    mfrr_act = data.load_activation_summary(selected_date, "mFRR")
    fcr_act = data.load_fcr_activation(selected_date)
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
    # aFRR/mFRR dispatch cards now carry the settlement stat row (revenue,
    # MWh, ISPs activated) formerly shown on their own separate "activation"
    # card in Market & Delivery -- that card's mini ACE/response chart just
    # duplicated this one in less detail. Taller container heights (650 vs
    # the old 520) account for the added stat row + caption.
    # Logical order (not the order these were added in): physical resources
    # -> storage -> aggregate dispatch -> real-time delivery accuracy ->
    # financial consequence -> reserve products -> reserve revenue summary.
    technical_cards = [
        ("Reservoir trajectory", lambda: components.html(dispatch_ticket.render_reservoir_trajectory_card(reservoir_traj), height=560)) if reservoir_traj else None,
        ("Water balance", lambda: components.html(dispatch_ticket.render_water_balance_card(water_balance), height=390)) if water_balance else None,
        ("PV routing & curtailment", lambda: components.html(dispatch_ticket.render_pv_routing_card(pv_routing), height=400)) if pv_routing else None,
        ("BESS charge source", lambda: components.html(dispatch_ticket.render_bess_charge_source_card(bess_charge_source), height=370)) if bess_charge_source else None,
        ("BESS SOC vs price", lambda: components.html(dispatch_ticket.render_bess_soc_price_card(bess_soc_price), height=400)) if bess_soc_price else None,
        ("Multi-asset dispatch", lambda: components.html(dispatch_ticket.render_multi_asset_dispatch_card(multi_asset), height=460)) if multi_asset else None,
        ("ISP asset dispatch (96-pt)", lambda: components.html(dispatch_ticket.render_isp_dispatch_card(isp_dispatch), height=490)) if isp_dispatch else None,
        ("DA vs ISP activation", lambda: components.html(dispatch_ticket.render_da_vs_activation_card(da_vs_activation), height=355)) if da_vs_activation else None,
        ("ISP dispatch", lambda: components.html(delivery_ticket.render_rt_card(rt), height=374)) if rt else None,
        ("Imbalance settlement", lambda: components.html(delivery_ticket.render_imbalance_settlement_card(imbalance), height=560)) if imbalance else None,
        ("aFRR dispatch", lambda: components.html(dispatch_ticket.render_afrr_dispatch_card(afrr_dispatch, afrr_act), height=650)) if afrr_dispatch else None,
        ("mFRR dispatch", lambda: components.html(dispatch_ticket.render_afrr_dispatch_card(mfrr_dispatch, mfrr_act), height=650)) if mfrr_dispatch else None,
        ("FCR dispatch", lambda: components.html(dispatch_ticket.render_fcr_dispatch_card(fcr_act), height=460)) if fcr_act else None,
        ("Capacity vs activation", lambda: components.html(delivery_ticket.render_capacity_vs_activation_card(capacity_vs_activation), height=340)) if capacity_vs_activation else None,
    ]
    technical_cards = [c for c in technical_cards if c is not None]

    # Plain st.tabs has no session_state key, so its selected tab is pure
    # client-side DOM state - a full script rerun (e.g. every auto-refresh
    # tick while a pipeline is live) always snaps it back to the first tab.
    # st.segmented_control is keyed in session_state, so the selection
    # survives reruns instead of visibly jumping back every couple of seconds.
    def _render_section(title: str, cards: list, state_key: str) -> None:
        if not cards:
            return
        st.markdown(f"##### {title}")
        labels = [label for label, _ in cards]
        if state_key not in st.session_state:
            st.session_state[state_key] = labels[0]  # only seed on first render

        if len(labels) > 1 and st.session_state[state_key] in labels:
            selected = st.segmented_control(
                title, labels, key=state_key, label_visibility="collapsed",
            )
        else:
            # The remembered selection's data isn't in this refresh tick's list
            # (e.g. a cache re-fetch briefly raced a mid-run write) -- render
            # whatever we can without touching session_state, so the real
            # selection is still there and resumes on its own once the data
            # reappears, instead of being permanently knocked onto another tab.
            selected = st.session_state[state_key]

        render_fn = dict(cards).get(selected) or cards[0][1]
        render_fn()
        st.markdown("---")


    _render_section("Optimization & Physical Dispatch", technical_cards, "overview_dispatch_tab")

    # ---------------------------------------------------------------------------
    # Gate Position Evolution -- net MW position as of each gate's close
    # (DA/IDA1/IDA2/IDA3/XBID), so a re-bid's actual physical effect is
    # visible, not just its revenue (each gate ticket card above already
    # shows its own improvement-vs-threshold rationale). Live PositionStore
    # read, same as the sections above, so it renders mid-run too.
    # ---------------------------------------------------------------------------

    gate_pos = data.load_gate_position_evolution(selected_date)
    if gate_pos:
        st.markdown("##### Gate Position Evolution")
        gate_order = gate_pos["gate_order"]
        gate_key = "overview_gate_pos_tab"
        if gate_key not in st.session_state:
            st.session_state[gate_key] = gate_order[0]
        if st.session_state[gate_key] in gate_order:
            selected_gate = st.segmented_control(
                "Gate Position Evolution", gate_order, key=gate_key, label_visibility="collapsed",
            )
        else:
            selected_gate = st.session_state[gate_key]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                             row_heights=[0.6, 0.4])
        if selected_gate == "DA":
            fig.add_trace(go.Bar(x=gate_pos["hours"], y=gate_pos["gates_mw"]["DA"],
                                  name="Net MW", marker_color=theme.COLOR_GEN), row=1, col=1)
            fig.add_trace(go.Scatter(x=gate_pos["hours"], y=gate_pos["gates_price"]["DA"],
                                      name="Price EUR/MWh", line=dict(color=theme.COLOR_PRICE, width=2)), row=2, col=1)
            theme.style_fig(fig, height=460)
            fig.update_yaxes(title_text="MW", gridcolor=theme.GRIDLINE, row=1, col=1)
            fig.update_yaxes(title_text="EUR/MWh", gridcolor=theme.GRIDLINE, row=2, col=1)
            fig.update_xaxes(title_text="Hour", row=2, col=1)
            st.plotly_chart(fig, width="stretch")
            st.caption("**DA** is the fixed reference position -- every other gate below is "
                       "shown as its deviation from this chart, not from each other.")
        else:
            n_diverged = gate_pos["diverged_isps"][selected_gate]
            net_delta = gate_pos["net_mw_delta"][selected_gate]
            if n_diverged == 0:
                # An all-zero delta drawn as a chart -- bar or line -- is
                # visually indistinguishable from an empty/broken widget (a
                # flat line at 0 sits exactly on top of the 0 gridline). There
                # is nothing to plot here, so say so plainly instead of
                # rendering a graph that looks empty.
                st.info(f"**{selected_gate}** held - position is unchanged from DA at every "
                        f"ISP. Nothing to plot; the DA chart above is still the live position.")
            else:
                mw_d = gate_pos["gates_mw_delta"][selected_gate]
                price_d = gate_pos["gates_price_delta"][selected_gate]
                fig.add_trace(go.Bar(x=gate_pos["hours"], y=mw_d,
                                      name="MW vs DA", marker_color=theme.COLOR_GEN), row=1, col=1)
                fig.add_trace(go.Scatter(x=gate_pos["hours"], y=price_d, mode="lines",
                                          name="Price vs DA (EUR/MWh)", line=dict(color=theme.COLOR_PRICE, width=2)), row=2, col=1)
                fig.add_hline(y=0, line_color=theme.GRIDLINE, row=1, col=1)
                fig.add_hline(y=0, line_color=theme.GRIDLINE, row=2, col=1)
                theme.style_fig(fig, height=460)
                fig.update_yaxes(title_text="MW vs DA", gridcolor=theme.GRIDLINE, row=1, col=1)
                fig.update_yaxes(title_text="EUR/MWh vs DA", gridcolor=theme.GRIDLINE, row=2, col=1)
                fig.update_xaxes(title_text="Hour", row=2, col=1)
                st.plotly_chart(fig, width="stretch")
                st.caption(f"**{selected_gate}** diverged from DA at **{n_diverged}** ISP(s), "
                           f"net **{net_delta:+.1f} MW** vs the DA baseline.")
        st.markdown("---")

    if not report_ready:
        st.info(data.no_report_message(selected_date))
        return

    report = data.load_daily_report(selected_date)
    kpis = report["kpis"]
    dispatch = report["dispatch"]

    # P&L Breakdown lives on Trading Desk only (the money page) -- removed from
    # here to avoid the same figures appearing on two pages.

    # ---------------------------------------------------------------------------
    # Reserve capacity offered -- physical/delivery detail, so this (not
    # Trading Desk) is the correct home for it. "Dispatch profile (hourly)"
    # used to live here too -- removed, superseded by Gate Position Evolution
    # above, which covers the same net-MW-vs-price shape per gate (including
    # the final/XBID position) instead of just one fixed snapshot.
    # ---------------------------------------------------------------------------

    st.markdown("##### Reserve capacity offered (aFRR / mFRR, hourly)")
    # aFRR (up to ~500 MW) and mFRR (up to ~85 MW) are ~6x apart in scale --
    # sharing one axis flattened mFRR into a barely-visible line hugging
    # zero. Split into two stacked panels, each with its own scale, so both
    # products are legible. "dn" (down-regulation) is still mirrored below
    # zero -- negative = capacity to reduce output, positive = capacity to
    # increase output, the plant's own physical direction, matching the
    # figures/ pipeline's reserve-capacity chart. Filled areas (instead of
    # bare lines) make the offered-capacity envelope read at a glance.
    if "aFRR_up_MW" in dispatch.columns and "mFRR_up_MW" in dispatch.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=("aFRR", "mFRR"))
        for row, (up_col, dn_col) in [(1, ("aFRR_up_MW", "aFRR_dn_MW")), (2, ("mFRR_up_MW", "mFRR_dn_MW"))]:
            fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch[up_col], name="Up",
                                      mode="lines", line=dict(color=theme.COLOR_UP, width=2),
                                      fill="tozeroy", fillcolor=theme.rgba(theme.COLOR_UP, 0.15),
                                      legendgroup="up", showlegend=(row == 1)), row=row, col=1)
            fig.add_trace(go.Scatter(x=dispatch["Hour"], y=-dispatch[dn_col], name="Down",
                                      mode="lines", line=dict(color=theme.COLOR_DOWN, width=2),
                                      fill="tozeroy", fillcolor=theme.rgba(theme.COLOR_DOWN, 0.15),
                                      legendgroup="down", showlegend=(row == 1)), row=row, col=1)
            fig.add_hline(y=0, line_color=theme.GRIDLINE, row=row, col=1)
        theme.style_fig(fig, height=460)
        fig.update_layout(margin=dict(t=32))  # subplot title ("aFRR") otherwise sits under the modebar icons
        fig.update_yaxes(title_text="aFRR (MW)", gridcolor=theme.GRIDLINE, row=1, col=1)
        fig.update_yaxes(title_text="mFRR (MW)", gridcolor=theme.GRIDLINE, row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=2, col=1)
        for ann in fig["layout"]["annotations"]:
            ann["font"] = dict(color=theme.INK_SECONDARY, size=13)
        st.plotly_chart(fig, width="stretch")
        st.caption("Up = capacity to increase output · Down = capacity to reduce output (shown negative).")


_render()
