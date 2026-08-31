"""Trading Desk - the money: P&L, per-gate performance, dispatch, reserves."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

import data
import dispatch_ticket
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
    reserve_pct = data.kpi_value(kpis, "Reserve share of P&L")

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
    components.html(dispatch_ticket.render_pnl_breakdown_card(total_pnl, reserve_pct, pnl_lines), height=510)

    # ---------------------------------------------------------------------------
    # Real-price re-settlement: the SAME committed bid above, valued against
    # the real archived settlement price instead of each gate's own
    # forecast/bid price -- a true apples-to-apples check, unlike the
    # separate Phase 6 backtest (which re-solves a different bid entirely).
    # Only rendered when a report has been generated for this date
    # (`python phase_6_backtesting_and_validation/run_live_resettlement.py
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
                st.dataframe(gate_rows, use_container_width=True, hide_index=True)
            reserve_rows = live_resettle.get("ReserveResettlement")
            if reserve_rows is not None and not reserve_rows.empty:
                st.dataframe(reserve_rows, use_container_width=True, hide_index=True)
            summary = live_resettle.get("Summary")
            if summary:
                for label, val, is_header in summary:
                    if not is_header:
                        st.caption(f"**{label}:** {val}")

    st.markdown("---")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Revenue by product")
        # Same 10 line items as the P&L Breakdown card above -- previously
        # this lumped IDA1/IDA2/IDA3/XBID into one "IDA+XBID" bar, a coarser
        # split than the card sitting right next to it for the same numbers.
        wf_labels = [label for label, _, _ in pnl_lines] + ["Total P&L"]
        wf_values = [value for _, value, _ in pnl_lines] + [total_pnl]
        waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(pnl_lines) + ["total"],
            x=wf_labels,
            y=wf_values,
            connector={"line": {"color": theme.GRIDLINE}},
            increasing=dict(marker=dict(color=theme.COLOR_GEN)),
            decreasing=dict(marker=dict(color=theme.COLOR_DOWN)),
            totals=dict(marker=dict(color=theme.COLOR_PRICE)),
        ))
        theme.style_fig(waterfall, height=420, legend=False)
        waterfall.update_xaxes(tickangle=-35)
        st.plotly_chart(waterfall, width="stretch")

    with right:
        st.subheader("Cumulative revenue through the day")
        if "Cum_Rev_EUR" in dispatch.columns:
            fig = go.Figure(go.Scatter(x=dispatch["Hour"], y=dispatch["Cum_Rev_EUR"],
                                        mode="lines+markers", line=dict(width=3, color=theme.COLOR_GEN)))
            theme.style_fig(fig, height=420, xaxis_title="Hour", yaxis_title="Cumulative EUR", legend=False)
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("Per-gate trading decisions (DA / IDA1-3 / XBID)")
    gc1, gc2 = st.columns([1, 1])
    with gc1:
        st.dataframe(gates, width="stretch", height=360)
    with gc2:
        if {"Gate", "Net revenue EUR", "VWAP EUR/MWh"}.issubset(gates.columns):
            # VWAP sat in the table unused -- pairing it with net revenue
            # shows WHY a gate made money (price achieved), not just how much.
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                 row_heights=[0.55, 0.45])
            fig.add_trace(go.Bar(x=gates["Gate"], y=gates["Net revenue EUR"],
                                  marker_color=theme.COLOR_GEN), row=1, col=1)
            fig.add_trace(go.Bar(x=gates["Gate"], y=gates["VWAP EUR/MWh"],
                                  marker_color=theme.COLOR_PRICE), row=2, col=1)
            theme.style_fig(fig, height=360, legend=False)
            fig.update_yaxes(title_text="Net revenue EUR", gridcolor=theme.GRIDLINE, row=1, col=1)
            fig.update_yaxes(title_text="VWAP EUR/MWh", gridcolor=theme.GRIDLINE, row=2, col=1)
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Dispatch profile + Reserve capacity offered live on Overview only (the
    # physical/delivery page) -- removed from here to avoid the same charts
    # appearing on two pages.

    with st.expander("Full hourly dispatch table"):
        st.dataframe(dispatch, width="stretch", height=500)

    with st.expander("ISP-level reserve activation (96 x 15-min)"):
        st.dataframe(isp, width="stretch", height=400)


_render()
