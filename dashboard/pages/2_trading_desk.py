"""Trading Desk - the money: P&L, per-gate performance, dispatch, reserves."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import column_labels
import data
import dispatch_ticket
import ml_ui
import theme


st.title("💰 Trading Desk")

@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    selected_date = st.session_state.get("selected_date")
    report_ready = st.session_state.get("report_ready", False)

    if not selected_date:
        st.warning("No runs found yet - visit Run & Monitor to start one.")
        return
    if not report_ready:
        st.info(data.no_report_message(selected_date))
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
    st.subheader("P&L Breakdown")
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
                "Same committed position shown above, valued at the real archived "
                "settlement price instead of the forecast/bid price. Activation and "
                "imbalance revenue are excluded — no real activation-price source "
                "exists in this project yet."
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

    st.markdown("---")

    st.subheader("Cumulative revenue through the day")
    if "Cum_Rev_EUR" in dispatch.columns:
        fig = go.Figure(go.Scatter(x=dispatch["Hour"], y=dispatch["Cum_Rev_EUR"], name="Cumulative revenue",
                                    mode="lines+markers", line=dict(width=3, color=theme.COLOR_GEN), hovertemplate="€%{y:,.0f}"))
        theme.style_fig(fig, height=420, xaxis_title="Hour", yaxis_title="Cumulative EUR", legend=False)
        # a vertical line that follows the cursor, with one hover box for the hour
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(hoverformat="d", showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=theme.INK_MUTED)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("Per-gate trading decisions (DA / IDA1-3 / XBID)")
    # the position store holds one entry per quarter-hour (ISP 1-96); reports written before the column was renamed still say "Hours active"
    ml_ui.table(gates.rename(columns={"Hours active": "Active quarter-hours (of 96)"}).drop(columns=["VWAP EUR/MWh"], errors="ignore"), fit=True, stretch=True)

    st.markdown("---")

    # Dispatch profile + Reserve capacity offered live on Overview only (the
    # physical/delivery page) -- removed from here to avoid the same charts
    # appearing on two pages.

    # a section of its own, not an expander: choosing a topic reruns the page and would fold an expander shut again
    st.subheader("Hourly dispatch")
    topic = st.segmented_control("Topic", list(column_labels.GROUPS), default="Market & position", key="desk_dispatch_topic", label_visibility="collapsed")
    ml_ui.table(column_labels.topic(dispatch, topic or "Market & position"), fit=True, height=500)

    with st.expander("ISP-level reserve activation (96 x 15-min)"):
        ml_ui.table(column_labels.readable(isp), fit=True, height=400)


_render()
