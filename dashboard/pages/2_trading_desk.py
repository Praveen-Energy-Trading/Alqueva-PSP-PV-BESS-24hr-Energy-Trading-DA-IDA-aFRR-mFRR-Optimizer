"""Trading Desk — the money: P&L, per-gate performance, dispatch, reserves."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
import theme

st.title("💰 Trading Desk")

selected_date = st.session_state.get("selected_date")
report_ready = st.session_state.get("report_ready", False)

if not selected_date:
    st.warning("No runs found yet — visit Run & Monitor to start one.")
    st.stop()
if not report_ready:
    st.info(data.no_report_message(selected_date))
    st.stop()

report = data.load_daily_report(selected_date)
dispatch, isp, gates, kpis = report["dispatch"], report["isp"], report["gates"], report["kpis"]

total_pnl = data.kpi_value(kpis, "Total daily P&L") or 0.0
da_rev    = data.kpi_value(kpis, "DA energy revenue") or 0.0
ida_rev   = data.kpi_value(kpis, "IDA incremental revenue") or 0.0
afrr_cap  = data.kpi_value(kpis, "aFRR capacity revenue") or 0.0
afrr_act  = data.kpi_value(kpis, "aFRR activation revenue") or 0.0
mfrr_cap  = data.kpi_value(kpis, "mFRR capacity revenue") or 0.0
mfrr_act  = data.kpi_value(kpis, "mFRR activation revenue") or 0.0
imbalance = data.kpi_value(kpis, "Imbalance settlement") or 0.0
reserve_pct = data.kpi_value(kpis, "Reserve share of P&L")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total P&L", f"{total_pnl:,.0f} EUR")
c2.metric("DA", f"{da_rev:,.0f} EUR")
c3.metric("IDA + XBID", f"{ida_rev:,.0f} EUR")
c4.metric("aFRR (cap+act)", f"{afrr_cap + afrr_act:,.0f} EUR")
c5.metric("mFRR (cap+act)", f"{mfrr_cap + mfrr_act:,.0f} EUR")
c6, c7 = st.columns(2)
c6.metric("Imbalance settlement", f"{imbalance:,.0f} EUR")
c7.metric("Reserve share of P&L", f"{reserve_pct:.1f} %" if reserve_pct is not None else "n/a")

st.markdown("---")

left, right = st.columns([1, 1])
with left:
    st.subheader("Revenue by product")
    waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * 7 + ["total"],
        x=["DA", "IDA+XBID", "aFRR cap", "aFRR act", "mFRR cap", "mFRR act", "Imbalance", "Total P&L"],
        y=[da_rev, ida_rev, afrr_cap, afrr_act, mfrr_cap, mfrr_act, imbalance, total_pnl],
        connector={"line": {"color": theme.GRIDLINE}},
        increasing=dict(marker=dict(color=theme.COLOR_GEN)),
        decreasing=dict(marker=dict(color=theme.COLOR_DOWN)),
        totals=dict(marker=dict(color=theme.COLOR_PRICE)),
    ))
    theme.style_fig(waterfall, height=420, legend=False)
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
    if {"Gate", "Net revenue EUR"}.issubset(gates.columns):
        fig = go.Figure(go.Bar(x=gates["Gate"], y=gates["Net revenue EUR"], marker_color=theme.COLOR_GEN))
        theme.style_fig(fig, height=360, yaxis_title="Net revenue EUR", legend=False)
        st.plotly_chart(fig, width="stretch")

st.markdown("---")

st.subheader("Dispatch profile")
st.caption("Net MW and DA price share an hour axis but different scales, so they're "
           "stacked rather than sharing one plot with two y-axes — that reads unambiguously "
           "at a glance instead of inviting a mis-scaled visual correlation.")
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

st.subheader("Reserve capacity offered (aFRR / mFRR)")
st.caption("Color encodes direction (up/down); line style encodes market (aFRR solid, mFRR dashed).")
fig = go.Figure()
if "aFRR_up_MW" in dispatch.columns:
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["aFRR_up_MW"], name="aFRR up", line=dict(color=theme.COLOR_UP)))
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["aFRR_dn_MW"], name="aFRR dn", line=dict(color=theme.COLOR_DOWN)))
if "mFRR_up_MW" in dispatch.columns:
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["mFRR_up_MW"], name="mFRR up", line=dict(color=theme.COLOR_UP, dash="dot")))
    fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["mFRR_dn_MW"], name="mFRR dn", line=dict(color=theme.COLOR_DOWN, dash="dot")))
theme.style_fig(fig, height=380, yaxis_title="MW", xaxis_title="Hour")
st.plotly_chart(fig, width="stretch")

with st.expander("Full hourly dispatch table"):
    st.dataframe(dispatch, width="stretch", height=500)

with st.expander("ISP-level reserve activation (96 x 15-min)"):
    st.dataframe(isp, width="stretch", height=400)
