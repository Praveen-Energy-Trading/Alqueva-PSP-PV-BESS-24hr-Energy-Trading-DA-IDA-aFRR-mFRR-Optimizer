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

_CARD = f"background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE}; border-radius:12px; box-shadow:0 1px 2px rgba(11,11,11,0.04);"
_TINTS = ["#e8f0fb", "#fdeee4", "#fff6dc", "#e3f5ee", "#efe9fb"]                  # one soft tint per icon tile

st.markdown(
    f'<div style="font-size:11.5px; letter-spacing:0.06em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600; margin-bottom:8px;">'
    f'Alqueva PSP + PV + BESS</div>'
    '<div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:12px;">' +
    "".join(
        f'<div style="{_CARD} padding:12px 14px; display:flex; align-items:center; gap:12px;">'
        f'<div style="flex:none; width:36px; height:36px; border-radius:10px; background:{_TINTS[n]}; display:flex; align-items:center; justify-content:center; font-size:18px;">{icon}</div>'
        f'<div style="min-width:0;">'
        f'<div style="font-size:11.5px; color:{theme.INK_MUTED}; font-weight:500; margin-bottom:1px;">{label}</div>'
        f'<div style="font-size:16px; font-weight:600; color:{theme.INK_PRIMARY}; line-height:1.25;">{value}</div>'
        + (f'<div style="font-size:11.5px; color:{theme.INK_MUTED}; margin-top:1px;">{sub}</div>' if sub else "") +
        f'</div></div>'
        for n, (icon, label, value, sub) in enumerate(ASSET_CARDS)
    ) +
    "</div>",
    unsafe_allow_html=True,
)

# Allocation of the plant's MW over the markets: a card with one segmented bar and a legend with share and MW (reserve figures are the largest offers).
_alloc = data.load_market_allocation()
_alloc_total = _alloc[-1][1]
_alloc_colours = [theme.COLOR_GEN, theme.COLOR_UP, theme.COLOR_PUMP, theme.COLOR_PRICE]
st.markdown(
    f'<div style="{_CARD} padding:12px 16px; margin-top:12px;">'
    f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;">'
    f'<span style="font-size:11.5px; letter-spacing:0.06em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600;">Allocation over the markets (largest offers)</span>'
    f'<span style="font-size:13px; color:{theme.INK_SECONDARY};"><b style="color:{theme.INK_PRIMARY}; font-weight:600;">{_alloc_total:.1f} MW</b> in total</span></div>'
    '<div style="display:flex; gap:3px; height:12px;">' +
    "".join(f'<div style="flex:{share:.2f} 1 0; min-width:6px; background:{colour}; border-radius:6px;"></div>' for (_, _, share), colour in zip(_alloc[:-1], _alloc_colours)) +
    '</div><div style="display:flex; flex-wrap:wrap; gap:6px 26px; margin-top:10px; font-size:13px; color:' + theme.INK_SECONDARY + ';">' +
    "".join(f'<span style="display:inline-flex; align-items:center;"><span style="width:9px; height:9px; border-radius:50%; background:{colour}; margin-right:7px;"></span>'
            f'<b style="color:{theme.INK_PRIMARY}; font-weight:600; margin-right:6px;">{name}</b>{share:.0f}% ({mw:.0f} MW)</span>'
            for (name, mw, share), colour in zip(_alloc[:-1], _alloc_colours)) +
    '</div></div>',
    unsafe_allow_html=True,
)

# Plain st.tabs has no session_state key, so its selected tab is pure
# client-side DOM state - a full script rerun (e.g. every auto-refresh
# tick while a pipeline is live) always snaps it back to the first tab.
# st.segmented_control is keyed in session_state, so the selection
# survives reruns instead of visibly jumping back every couple of seconds.
#
# Each section is its own fragment, nested in the page's run_every
# fragment: picking a tab reruns only that section. Before, a click reran
# the whole Overview (every gate ticket, chart and replay widget), which
# took seconds while the page rebuilt and scrolled up and down.
@st.fragment
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


def _crosshair_series(fig, hours, series, x_format) -> None:
    """A vertical line that follows the cursor through all panels of a figure, with one hover box for the hour.
    A spike drawn by plotly stops at the end of its own panel, so the line comes from an invisible full-height overlay axis
    (a transparent point per hour) that matches the x axis of the panels. The overlay is the hovered trace, so one transparent
    trace per value, `series` = [(label, values, number format), ...], carries that value for the hour into the shared hover box."""
    for label, values, fmt in series:
        fig.add_trace(go.Scatter(x=list(hours), y=[0] * len(hours), mode="markers", marker=dict(opacity=0), opacity=0, xaxis="x3", yaxis="y3",
                                 name=label, customdata=list(values), showlegend=False,
                                 hovertemplate="%{customdata:" + fmt + "}" if fmt else "%{customdata}"))
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",       # the overlay panel must not paint over the real panels; the paper has the same colour
        xaxis3=dict(overlaying="x", matches="x", hoverformat=x_format, showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks="",
                    showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=theme.INK_MUTED),
        yaxis3=dict(domain=[0, 1], anchor="x3", range=[-1, 1], showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks=""),
    )


@st.fragment
def _render_gate_positions(gate_pos: dict) -> None:
    """Gate Position Evolution -- its own fragment for the same reason as
    _render_section: switching gate tabs reruns only this chart."""
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
    # gate_pos["hours"] repeats each hour for its four 15-minute periods, so the day is drawn on a clock axis: one bar per quarter-hour
    # (ISP 1 = 00:00-00:15), centred in its period, and the price as a step line that starts at 00:00 and ends at 24:00
    n_isp = len(gate_pos["hours"])
    day_ms = 24 * 3600 * 1000
    step = day_ms / n_isp
    starts = [i * step for i in range(n_isp)]
    centres = [t + step / 2 for t in starts]
    bar_w = step * 0.88
    spans = [f"{int(t // 3600000):02d}:{int(t % 3600000 // 60000):02d}-{int((t + step) // 3600000):02d}:{int((t + step) % 3600000 // 60000):02d}" for t in starts]

    def _plot(mw, price, up_name, down_name, up_color, down_color, mw_title, price_title, crosshair_mw, crosshair_price) -> None:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.6, 0.4])
        pos = [v if v > 0 else None for v in mw]
        neg = [v if v < 0 else None for v in mw]
        fig.add_trace(go.Bar(x=centres, y=pos, name=up_name, marker_color=up_color, width=bar_w), row=1, col=1)
        fig.add_trace(go.Bar(x=centres, y=neg, name=down_name, marker_color=down_color, width=bar_w), row=1, col=1)
        fig.add_trace(go.Scatter(x=starts + [day_ms], y=list(price) + [price[-1]], name=price_title, mode="lines", line_shape="hv",
                                 line=dict(color=theme.COLOR_PRICE, width=2), fill="tozeroy" if min(price) >= 0 else None,
                                 fillcolor=theme.rgba(theme.COLOR_PRICE, 0.08)), row=2, col=1)
        theme.style_fig(fig, height=460, barmode="overlay")
        _crosshair_series(fig, centres, [("Quarter-hour", spans, ""), (crosshair_mw, mw, ",.0f"), (crosshair_price, price, ",.1f")], "Time of day")
        fig.add_hline(y=0, line_color=theme.AXIS_LINE, line_width=1, row=1, col=1)
        fig.update_yaxes(title_text=mw_title, gridcolor=theme.GRIDLINE, row=1, col=1)
        fig.update_yaxes(title_text=price_title, gridcolor=theme.GRIDLINE, row=2, col=1)
        fig.update_xaxes(type="date", range=[0, day_ms], tickvals=[h * 3600000 for h in range(0, 25, 4)],
                         ticktext=[f"{h:02d}:00" for h in range(0, 25, 4)], showgrid=False)
        fig.update_xaxes(title_text="Time of day (one bar = one quarter-hour, ISP 1-96)", row=2, col=1)
        st.plotly_chart(fig, width="stretch")

    if selected_gate == "DA":
        _plot(gate_pos["gates_mw"]["DA"], gate_pos["gates_price"]["DA"], "Selling (generating)", "Buying (pumping)",
              theme.COLOR_GEN, theme.COLOR_PUMP, "Net position (MW)", "Price (EUR/MWh)", "Net MW", "Price EUR/MWh")
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
            _plot(gate_pos["gates_mw_delta"][selected_gate], gate_pos["gates_price_delta"][selected_gate], "More than DA", "Less than DA",
                  theme.COLOR_UP, theme.COLOR_DOWN, "Change vs DA (MW)", "Price vs DA (EUR/MWh)", "MW vs DA", "Price vs DA (EUR/MWh)")
            st.caption(f"**{selected_gate}** diverged from DA at **{n_diverged}** ISP(s), "
                       f"net **{net_delta:+.1f} MW** vs the DA baseline.")
    st.markdown("---")


def _status_line(kind: str, html_text: str) -> None:
    """The run status as one slim line (st.success / st.error take about twice the height for the same sentence)."""
    bg, fg = ("#e6f4ea", "#1e6b33") if kind == "good" else ("#fdecea", "#9b2c2c")
    st.markdown(f'<div style="background:{bg}; color:{fg}; border-radius:12px; padding:6px 16px; font-size:14px; line-height:1.35; margin:14px 0;">{html_text}</div>',
                unsafe_allow_html=True)


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
            _status_line("bad", f"🔴 <b>{selected_date}</b> - {n_fail} phase(s) FAILED (finished {run_status['finished_at']}, mode={run_status['mode']})")
        else:
            _status_line("good", f"🟢 <b>{selected_date}</b> - {n_pass}/{len(results)} phases passed cleanly (finished {run_status['finished_at']}, mode={run_status['mode']})")

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

        # Two rows of buttons of the same width: the day-ahead and reserve gates (3, centred), then the intraday gates (4). Each keeps its name,
        # decision and amount on one line; a flat colour dot shows the status.
        st.markdown(
            '<style>'
            'div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button { min-height: 2.6rem; border-radius: 12px; font-weight: 500;'
            ' border: 1px solid ' + theme.GRIDLINE + '; box-shadow: 0 1px 2px rgba(11,11,11,0.05); transition: box-shadow .15s, border-color .15s; }'
            'div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover { border-color: ' + theme.BLUE + '; box-shadow: 0 2px 6px rgba(42,120,214,0.18); }'
            'div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="primary"] { font-weight: 600; box-shadow: 0 2px 8px rgba(42,120,214,0.35); }'
            '</style>', unsafe_allow_html=True)
        # One row of equal cards: the name with a status dot above, the amount below.
        row_cols = st.columns(len(all_tickets))
        for slot, t in zip(row_cols, all_tickets):
            dot = {"good": ":green[●]", "neutral": ":gray[●]", "critical": ":red[●]"}[t["status_class"]]
            value = t["revenue_items"][0][1] if t.get("revenue_items") else None
            rev = "–" if value is None else (f"−€{abs(value):,.0f}" if value < 0 else f"€{value:,.0f}")
            is_selected = t["gate"] == st.session_state[state_key]
            if is_selected:
                dot = "●"                                              # white on the blue selected button
            label = f"{dot} {gate_ticket.gate_caption_name(t['gate'])}  \n{rev}"      # a markdown hard break between the two lines
            slot.button(label, key=f"gatebtn_{selected_date}_{t['gate']}",
                        use_container_width=True, type="primary" if is_selected else "secondary",
                        on_click=_select_gate, args=(t["gate"],), help=f"{t['decision']}")

        selected_ticket = next(t for t in all_tickets if t["gate"] == st.session_state[state_key])
        # Card height varies by content: standard DA/aFRR/mFRR bar-chart cards
        # are shorter (no tradable-hours section), rebid cards (IDA1-3) are
        # taller, and XBID's multi-window pill row adds a bit more on top of
        # that. One fixed height sized for the tallest case left a large blank
        # gap in the iframe below every shorter card.
        if selected_ticket.get("xbid_windows"):
            card_height = 365
        elif selected_ticket.get("is_rebid_gate"):
            card_height = 340
        else:
            card_height = 405
        # +20px for the "Gate closes ..." line added below the decided
        # timestamp (see gate_ticket.py's gate_close_bit) -- every real gate
        # has one, but guard on presence anyway so a ticket without it
        # doesn't get a blank gap under it.
        if selected_ticket.get("gate_close"):
            card_height += 20
        components.html(gate_ticket.render(selected_ticket, selected_date), height=card_height)
        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Delivery cards - RT dispatch, aFRR activation, mFRR activation, FCR droop
    # response. Not decisions like the gates above, so they get their own tab
    # strip rather than sitting in the gate-ticket buttons. Only rendered once
    # each phase has actually produced data for this date. FCR is the odd one
    # out: it's a standalone, non-remunerated compliance simulation, not read
    # from a pipeline run's audit trail - see data.py::load_fcr_activation.
    # ---------------------------------------------------------------------------

    imbalance = data.load_imbalance_settlement(selected_date)
    afrr_act = data.load_activation_summary(selected_date, "aFRR")
    mfrr_act = data.load_activation_summary(selected_date, "mFRR")
    fcr_act = data.load_fcr_activation(selected_date)
    reservoir_traj = data.load_reservoir_trajectory(selected_date)
    pv_routing = data.load_pv_routing(selected_date)
    water_balance = data.load_water_balance(selected_date)
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
        ("ISP asset dispatch (96-pt)", lambda: components.html(dispatch_ticket.render_isp_dispatch_card(isp_dispatch), height=490)) if isp_dispatch else None,
        ("DA vs activation", lambda: components.html(dispatch_ticket.render_da_vs_activation_card(da_vs_activation), height=355)) if da_vs_activation else None,
        ("Imbalance settlement", lambda: components.html(delivery_ticket.render_imbalance_settlement_card(imbalance), height=560)) if imbalance else None,
        ("aFRR dispatch", lambda: components.html(dispatch_ticket.render_afrr_dispatch_card(afrr_dispatch, afrr_act), height=650)) if afrr_dispatch else None,
        ("mFRR dispatch", lambda: components.html(dispatch_ticket.render_afrr_dispatch_card(mfrr_dispatch, mfrr_act), height=650)) if mfrr_dispatch else None,
        ("FCR dispatch", lambda: components.html(dispatch_ticket.render_fcr_dispatch_card(fcr_act), height=460)) if fcr_act else None,
    ]
    technical_cards = [c for c in technical_cards if c is not None]

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
        _render_gate_positions(gate_pos)

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
        _crosshair_series(fig, dispatch["Hour"], [("aFRR up (MW)", dispatch["aFRR_up_MW"], ",.0f"), ("aFRR down (MW)", dispatch["aFRR_dn_MW"], ",.0f"),
                                                  ("mFRR up (MW)", dispatch["mFRR_up_MW"], ",.0f"), ("mFRR down (MW)", dispatch["mFRR_dn_MW"], ",.0f")], ".0f")
        fig.update_xaxes(range=[1, 24], tickvals=[1, 4, 8, 12, 16, 20, 24])
        fig.update_xaxes(title_text="Hour of the day (1-24)", row=2, col=1)
        for ann in fig["layout"]["annotations"]:
            ann["font"] = dict(color=theme.INK_SECONDARY, size=13)
        st.plotly_chart(fig, width="stretch")
        st.caption("Up = more injection or less pumping; down (shown negative) = less injection or more pumping. "
                   "Reserve capacity is contracted per hour.")


_render()
