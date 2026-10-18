"""Trading Desk - the money: P&L, per-gate performance, dispatch, reserves."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import html

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import column_labels
import data
import dispatch_ticket
import ml_ui
import theme


st.title("💰 Trading Desk")

CARD_LINE = "#e3e1d9"


def _head(title: str, hint: str = "") -> None:
    """A section heading: bold, sentence case, with a muted line under it on how to read the block (the divider lines of the old page are gone)."""
    st.markdown(f"<div style='margin:1.4rem 0 0.4rem;'><div style='font-size:1.3rem; font-weight:700; color:{theme.INK_PRIMARY}; line-height:1.25;'>{title}</div>"
                + (f"<div style='font-size:0.86rem; color:{theme.INK_MUTED}; margin-top:0.15rem;'>{hint}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _banner(kind: str, html_text: str) -> None:
    """One slim, coloured line: amber needs a look, blue is information."""
    bg, fg = {"warn": ("#fdf0d0", "#7a5200"), "info": ("#eef1f8", "#36415c")}[kind]
    st.markdown(f'<div style="background:{bg}; color:{fg}; border-radius:12px; padding:8px 16px; font-size:14px; line-height:1.45; margin:10px 0;">{html_text}</div>', unsafe_allow_html=True)


def _pill(text: str, color: str, bg: str) -> str:
    return f"<span style='background:{bg}; color:{color}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{text}</span>"


def _gates_table(gates, dot_of) -> None:
    """The gates as a table in the style of the ML pages: muted header, thin row lines, a coloured square per gate and a status pill."""
    grid = "display:grid; grid-template-columns:minmax(4.2rem, 0.6fr) minmax(9.5rem, 1.8fr) minmax(6.5rem, 1.1fr) minmax(5.5rem, 0.9fr) minmax(5rem, 0.8fr) minmax(5rem, 0.8fr) minmax(5.5rem, 0.9fr); align-items:center; column-gap:0.6rem;"
    heads = ["Gate", "Market", "Gate close", "Active quarter-hours (of 96)", "Status", "Net position (MWh)", "Net revenue (EUR)"]
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{theme.INK_MUTED}; font-weight:600; line-height:1.2;'>"
            + "".join(f"<span style='{'' if j in (0, 1, 2) else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(heads)) + "</div>")
    body = ""
    for _, r in gates.iterrows():
        active = int(r["Active quarter-hours (of 96)"])
        status = _pill("Active", "#006100", "#e3f1e3") if active else _pill("No position", "#5a5953", "#ecebe6")
        rev = float(r["Net revenue EUR"])
        cells = [f"<span style='color:{dot_of(r['Gate'])}'>■</span>&nbsp;<b>{html.escape(str(r['Gate']))}</b>", html.escape(str(r["Market description"])), html.escape(str(r["Gate close"])),
                 f"{active}", status, f"{float(r['Net position MWh']):,.2f}", f"−€{abs(rev):,.0f}" if rev < 0 else f"€{rev:,.0f}"]
        body += (f"<div style='{grid} min-height:2.5rem; font-size:0.86rem; border-bottom:1px solid #eeede8;'>"
                 + "".join(f"<span style='{'' if j in (0, 1, 2) else 'text-align:center;'} color:{theme.INK_PRIMARY};'>{c}</span>" for j, c in enumerate(cells)) + "</div>")
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{head}{body}</div>", unsafe_allow_html=True)


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    selected_date = st.session_state.get("selected_date")
    report_ready = st.session_state.get("report_ready", False)

    if not selected_date:
        _banner("warn", "No runs found yet. Open Run &amp; Monitor to start one.")
        return
    if not report_ready:
        _banner("info", data.no_report_message(selected_date))
        return

    report = data.load_daily_report(selected_date)
    dispatch, isp, gates, kpis = report["dispatch"], report["isp"], report["gates"], report["kpis"]

    total_pnl = data.kpi_value(kpis, "Total daily P&L") or 0.0
    da_rev    = data.kpi_value(kpis, "DA energy revenue") or 0.0
    afrr_cap  = data.kpi_value(kpis, "aFRR capacity revenue") or 0.0
    afrr_act  = data.kpi_value(kpis, "aFRR activation revenue") or 0.0
    mfrr_cap  = data.kpi_value(kpis, "mFRR capacity revenue") or 0.0
    mfrr_act  = data.kpi_value(kpis, "mFRR activation revenue") or 0.0
    imbalance = data.kpi_value(kpis, "Imbalance settlement") or 0.0

    # ---------------------------------------------------------------------------
    # P&L breakdown card - same widget as Overview's "P&L Breakdown" (moved here
    # too, ownership-wise this is the money page; Overview's copy left in place
    # pending merge/dedup review).
    # ---------------------------------------------------------------------------

    pnl_lines = [
        ("DA",                da_rev,   theme.COLOR_GEN),
        ("IDA1",              data.kpi_value(kpis, "IDA1 incremental revenue") or 0.0, theme.COLOR_PRICE),
        ("IDA2",              data.kpi_value(kpis, "IDA2 incremental revenue") or 0.0, theme.COLOR_PRICE),
        ("IDA3",              data.kpi_value(kpis, "IDA3 incremental revenue") or 0.0, theme.COLOR_PRICE),
        ("XBID",              data.kpi_value(kpis, "XBID incremental revenue") or 0.0, theme.STATUS_NEUTRAL),
        ("aFRR capacity",     afrr_cap, theme.COLOR_UP),
        ("aFRR activation",   afrr_act, theme.COLOR_UP),
        ("mFRR capacity",     mfrr_cap, theme.COLOR_PUMP),
        ("mFRR activation",   mfrr_act, theme.COLOR_PUMP),
        ("Imbalance settlement", imbalance, theme.STATUS_GOOD),
    ]
    _head("P&L breakdown", "Where the day's result comes from, by market")
    components.html(dispatch_ticket.render_pnl_breakdown_card(total_pnl, pnl_lines), height=510)

    # ---------------------------------------------------------------------------
    # Real-price re-settlement: the SAME committed bid above, valued against
    # the real archived settlement price instead of each gate's own
    # forecast/bid price -- a true apples-to-apples check, unlike the
    # separate Phase 7 backtest (which re-solves a different bid entirely).
    # Only rendered when a report has been generated for this date
    # (`python phase_7_backtesting_and_validation/run_live_resettlement.py
    # --date <date>`) -- never fabricated when absent.
    # ---------------------------------------------------------------------------
    live_resettle = data.load_live_resettlement_report(selected_date)
    if live_resettle is not None:
        with st.expander("🔁 Real-price re-settlement (same bid, no re-solve)", expanded=False):
            st.caption(
                "The same committed position as above, valued at the real archived settlement price instead of the forecast or bid price. "
                "Activation and imbalance revenue are excluded: no real activation-price source exists in this project yet."
            )
            gate_rows = live_resettle.get("LiveResettlement")
            if gate_rows is not None and not gate_rows.empty:
                ml_ui.table(gate_rows, fit=True)
            reserve_rows = live_resettle.get("ReserveResettlement")
            if reserve_rows is not None and not reserve_rows.empty:
                ml_ui.table(reserve_rows, fit=True)
            summary = live_resettle.get("Summary")
            if summary:
                for label, val, is_header in summary:
                    if not is_header:
                        st.caption(f"**{label}:** {val}")

    _head("Cumulative revenue through the day", "The running total of the day's revenue, hour by hour")
    if "Cum_Rev_EUR" in dispatch.columns:
        fig = go.Figure(go.Scatter(x=dispatch["Hour"], y=dispatch["Cum_Rev_EUR"], name="Cumulative revenue",
                                    mode="lines+markers", line=dict(width=3, color=theme.COLOR_GEN), hovertemplate="€%{y:,.0f}"))
        theme.style_fig(fig, height=380, xaxis_title="Hour of the day", yaxis_title="Cumulative revenue (EUR)", legend=False)
        # a vertical line that follows the cursor, with one hover box for the hour
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(hoverformat="d", showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=theme.INK_MUTED)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    _head("Per-gate trading decisions", "DA, IDA1 to IDA3 and XBID: when each gate closes and what it decided")
    # the position store holds one entry per quarter-hour (ISP 1-96); reports written before the column was renamed still say "Hours active"
    _g = gates.rename(columns={"Hours active": "Active quarter-hours (of 96)"})
    _dot = {"DA": theme.COLOR_GEN, "IDA1": theme.COLOR_PRICE, "IDA2": theme.COLOR_PRICE, "IDA3": theme.COLOR_PRICE, "XBID": theme.STATUS_NEUTRAL}
    _gates_table(_g, lambda gate: _dot.get(str(gate), theme.INK_MUTED))

    # Dispatch profile + Reserve capacity offered live on Overview only (the
    # physical/delivery page) -- removed from here to avoid the same charts
    # appearing on two pages.

    # a section of its own, not an expander: choosing a topic reruns the page and would fold an expander shut again
    _head("Hourly dispatch", "Choose a topic to see its columns, hour by hour")
    topic = st.segmented_control("Topic", list(column_labels.GROUPS), default="Market & position", key="desk_dispatch_topic", label_visibility="collapsed")
    ml_ui.table(column_labels.topic(dispatch, topic or "Market & position"), fit=True, height=500)

    with st.expander("ISP-level reserve activation (96 x 15-min)"):
        ml_ui.table(column_labels.readable(isp), fit=True, height=400)


_render()
