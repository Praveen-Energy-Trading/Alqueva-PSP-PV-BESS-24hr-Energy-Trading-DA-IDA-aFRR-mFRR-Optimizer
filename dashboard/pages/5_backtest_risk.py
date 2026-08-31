"""Backtest & Portfolio Risk - multi-day aggregate performance and risk
metrics (VaR/CVaR/Sharpe/drawdown, operational reliability, day-part
patterns). This data has existed in runtime/reports/backtest_*.xlsx since
before this dashboard rewrite but was never surfaced anywhere."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import data
import theme

st.title("📈 Backtest & Portfolio Risk")


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    files = data.list_backtest_reports()
    if not files:
        st.info("No backtest reports found in `runtime/reports/backtest_*.xlsx`. "
                "Run `phase_6_backtesting_and_validation/run_backtest.py` to generate one.")
        return

    chosen = st.selectbox("Backtest report", files, format_func=lambda p: p.name)
    report = data.load_backtest_report(chosen)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    st.subheader("Summary")
    summary_metrics = [(l, v) for l, v, is_hdr in report.get("Summary", []) if not is_hdr]
    by_label = dict(summary_metrics)
    days = by_label.get("Days")
    feasible = by_label.get("Feasible")
    passed = by_label.get("Checker passed")

    # A count on its own ("Feasible: 180") doesn't say whether that's good
    # or bad without the denominator -- same pass/fail banner pattern used
    # on Overview's own run-status line, so "is this backtest healthy" reads
    # the same way everywhere in the dashboard.
    if days and feasible is not None and passed is not None:
        n_days, n_feas, n_pass = int(days), int(feasible), int(passed)
        if n_feas == n_days and n_pass == n_days:
            st.success(f"🟢 {n_feas}/{n_days} days feasible, {n_pass}/{n_days} passed the physical checker")
        else:
            st.warning(f"🟡 {n_feas}/{n_days} days feasible, {n_pass}/{n_days} passed the physical checker "
                       f" -  {n_days - n_feas} infeasible, {n_days - n_pass} checker failure(s)")

    other_metrics = [(l, v) for l, v in summary_metrics if l not in ("Days", "Feasible", "Checker passed")]
    if other_metrics:
        theme.metric_cards(other_metrics, ncols=4)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Portfolio risk (VaR/CVaR/Sharpe/drawdown)
    # ---------------------------------------------------------------------------

    st.subheader("Portfolio risk")
    if "Risk" not in report:
        st.info("This backtest report has no Risk sheet - risk metrics weren't computed for this run.")
    else:
        section = None
        section_rows: dict[str, list[tuple[str, object]]] = {}
        for label, value, is_hdr in report["Risk"]:
            if is_hdr and label.startswith("---"):
                section = label.strip("- ").strip()
                section_rows[section] = []
            elif not is_hdr and section:
                section_rows[section].append((label, value))

        # VaR/CVaR at both confidence levels, historical vs Monte Carlo,
        # side by side -- turns 8 separate number cards into one chart a
        # reader can compare at a glance instead of cross-referencing labels.
        hist = dict(section_rows.get("Historical Simulation", []))
        mc = dict(section_rows.get("Monte Carlo Bootstrap (VaR 95%)", []))
        var_hist_95 = next((v for k, v in hist.items() if k.startswith("VaR(95%)")), None)
        cvar_hist_95 = next((v for k, v in hist.items() if k.startswith("CVaR(95%)")), None)
        var_hist_99 = next((v for k, v in hist.items() if k.startswith("VaR(99%)")), None)
        cvar_hist_99 = next((v for k, v in hist.items() if k.startswith("CVaR(99%)")), None)
        var_mc_95 = next((v for k, v in mc.items() if k.startswith("VaR(95%)") and "mean" in k), None)
        cvar_mc_95 = next((v for k, v in mc.items() if k.startswith("CVaR(95%)") and "mean" in k), None)

        if any(v is not None for v in (var_hist_95, cvar_hist_95, var_hist_99, cvar_hist_99)):
            st.markdown("**VaR / CVaR - how bad could a day get**")
            metrics_x = ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"]
            hist_y = [var_hist_95, cvar_hist_95, var_hist_99, cvar_hist_99]
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(x=metrics_x, y=hist_y, name="Historical", marker_color=theme.COLOR_GEN))
            if var_mc_95 is not None and cvar_mc_95 is not None:
                fig_var.add_trace(go.Bar(x=["VaR 95%", "CVaR 95%"], y=[var_mc_95, cvar_mc_95],
                                          name="Monte Carlo", marker_color=theme.COLOR_PUMP))
            theme.style_fig(fig_var, height=320, yaxis_title="EUR (downside from mean)", barmode="group")
            st.plotly_chart(fig_var, width="stretch")
            st.caption("Historical = worst days actually observed in the backtest. Monte Carlo = same estimate "
                       "from 10,000 bootstrap resamples, for a confidence check on the historical number.")

        for section, rows in section_rows.items():
            st.markdown(f"**{section}**")
            theme.metric_cards(rows, ncols=4)

        if "Backtest" in report:
            bt = report["Backtest"]
            if "objective_eur" in bt.columns:
                st.markdown("**Daily P&L distribution**")
                fig = go.Figure(go.Histogram(x=bt.loc[bt["feasible"] == 1, "objective_eur"], nbinsx=30,
                                              marker_color=theme.COLOR_GEN))
                theme.style_fig(fig, height=320, xaxis_title="Daily objective EUR", yaxis_title="Days", legend=False)
                st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Multi-day trends
    # ---------------------------------------------------------------------------

    st.subheader("Multi-day trends")
    if "Backtest" in report:
        bt = report["Backtest"]
        # Daily P&L alone hides the trajectory -- a run with two great weeks
        # and one bad one looks the same as a steadily-mediocre run until you
        # add the running total underneath it.
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             row_heights=[0.55, 0.45])
        fig.add_trace(go.Bar(x=bt["date"], y=bt["objective_eur"], name="Daily objective EUR",
                              marker_color=theme.COLOR_GEN), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["objective_eur"].cumsum(), name="Cumulative EUR",
                                  line=dict(color=theme.COLOR_PRICE, width=2)), row=2, col=1)
        theme.style_fig(fig, height=380, legend=False)
        fig.update_yaxes(title_text="Daily EUR", gridcolor=theme.GRIDLINE, row=1, col=1)
        fig.update_yaxes(title_text="Cumulative EUR", gridcolor=theme.GRIDLINE, row=2, col=1)
        st.plotly_chart(fig, width="stretch")

        # Two different scales (EUR/MWh vs MW) - stacked on a shared date axis
        # instead of a dual-y-axis overlay, so neither line's slope is read
        # against the wrong scale.
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        fig2.add_trace(go.Scatter(x=bt["date"], y=bt["price_mae"], name="Price MAE EUR/MWh", line=dict(color=theme.COLOR_GEN)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=bt["date"], y=bt["pv_mae"], name="PV MAE MW", line=dict(color=theme.COLOR_PUMP)), row=2, col=1)
        theme.style_fig(fig2, height=380, legend=False)
        fig2.update_yaxes(title_text="Price MAE EUR/MWh", gridcolor=theme.GRIDLINE, row=1, col=1)
        fig2.update_yaxes(title_text="PV MAE MW", gridcolor=theme.GRIDLINE, row=2, col=1)
        st.plotly_chart(fig2, width="stretch")

        with st.expander("Full per-day table"):
            st.dataframe(bt, width="stretch", height=400)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Operational reliability
    # ---------------------------------------------------------------------------

    st.subheader("Operational reliability")
    if "Operational" in report:
        op = report["Operational"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg turbine starts/day", f"{op['turbine_starts_total'].mean():.1f}")
        c2.metric("Avg pump starts/day", f"{op['pump_starts_total'].mean():.1f}")
        c3.metric("Avg turbine run length (h)", f"{op['turb_avg_run_h'].mean():.1f}")
        c4.metric("Avg pump run length (h)", f"{op['pump_avg_run_h'].mean():.1f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=op["date"], y=op["turb_hours_top25pct_price"], name="Turbine hrs @ top-25% price", line=dict(color=theme.COLOR_GEN)))
        fig.add_trace(go.Scatter(x=op["date"], y=op["pump_hours_bot25pct_price"], name="Pump hrs @ bottom-25% price", line=dict(color=theme.COLOR_PUMP)))
        theme.style_fig(fig, height=320, yaxis_title="Hours/day")
        st.plotly_chart(fig, width="stretch")
        st.caption("Price-timing quality: how much of the plant's running hours land in the "
                   "most favourable price windows for that action.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Extended KPIs - efficiency, capacity factors, reservoir head, revenue mix
    # ---------------------------------------------------------------------------

    st.subheader("Extended KPIs - efficiency and mix")
    if "KPI_Extended" in report:
        kx = report["KPI_Extended"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg turbine capacity factor", f"{kx['turbine_capacity_factor_pct'].mean():.1f} %")
        c2.metric("Avg pump capacity factor", f"{kx['pump_capacity_factor_pct'].mean():.1f} %")
        c3.metric("Avg PV utilisation", f"{kx['pv_utilisation_pct'].mean():.1f} %")
        c4.metric("Avg reservoir fill (end)", f"{kx['reservoir_fill_end_pct'].mean():.1f} %")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=kx["date"], y=kx["turbine_capacity_factor_pct"], name="Turbine CF %", line=dict(color=theme.COLOR_GEN)))
        fig.add_trace(go.Scatter(x=kx["date"], y=kx["pump_capacity_factor_pct"], name="Pump CF %", line=dict(color=theme.COLOR_PUMP)))
        theme.style_fig(fig, height=320, yaxis_title="Capacity factor %")
        st.plotly_chart(fig, width="stretch")
        st.caption("Capacity factor: actual energy moved vs. what running flat-out the whole day would have moved.")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=kx["date"], y=kx["da_revenue_share_pct"], name="DA revenue share %",
                                   stackgroup="mix", line=dict(color=theme.COLOR_GEN, width=0.5)))
        fig2.add_trace(go.Scatter(x=kx["date"], y=kx["frr_revenue_share_pct"], name="FRR revenue share %",
                                   stackgroup="mix", line=dict(color=theme.COLOR_PUMP, width=0.5)))
        theme.style_fig(fig2, height=280, yaxis_title="% of daily revenue")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Revenue mix: how much of each day's P&L came from energy trading (DA) vs. reserve capacity (aFRR/mFRR).")

        with st.expander("Reservoir head range (min/max per day)"):
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=kx["date"], y=kx["head_max_m"], name="Head max m", line=dict(color=theme.COLOR_GEN)))
            fig3.add_trace(go.Scatter(x=kx["date"], y=kx["head_min_m"], name="Head min m", line=dict(color=theme.COLOR_PUMP)))
            theme.style_fig(fig3, height=280, yaxis_title="Meters")
            st.plotly_chart(fig3, width="stretch")
    else:
        st.info("This backtest report has no KPI_Extended sheet.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Temporal (day-part) patterns
    # ---------------------------------------------------------------------------

    st.subheader("Day-part patterns")
    if "Temporal" in report:
        temp = report["Temporal"]
        agg = temp.groupby("band", as_index=False).agg(
            avg_profit_eur_h=("avg_profit_eur_h", "mean"),
            turbine_pct=("turbine_pct", "mean"),
            pump_pct=("pump_pct", "mean"),
        )
        band_order = ["night", "morning", "afternoon", "evening"]
        agg["band"] = pd.Categorical(agg["band"], categories=band_order, ordered=True)
        agg = agg.sort_values("band")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["band"], y=agg["avg_profit_eur_h"], name="Avg profit EUR/h", marker_color=theme.COLOR_GEN))
        theme.style_fig(fig, height=320, yaxis_title="EUR/h", legend=False)
        st.plotly_chart(fig, width="stretch")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=agg["band"], y=agg["turbine_pct"], name="Turbine %", marker_color=theme.COLOR_GEN))
        fig2.add_trace(go.Bar(x=agg["band"], y=agg["pump_pct"], name="Pump %", marker_color=theme.COLOR_PUMP))
        theme.style_fig(fig2, height=320, yaxis_title="% of hours", barmode="group")
        st.plotly_chart(fig2, width="stretch")


_render()
