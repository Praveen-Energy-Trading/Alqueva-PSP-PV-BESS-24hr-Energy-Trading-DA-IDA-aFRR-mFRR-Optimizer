"""Backtest & Portfolio Risk - summary and portfolio risk metrics
(VaR/CVaR/Sharpe/drawdown). This data has existed in
runtime/reports/backtest_*.xlsx since before this dashboard rewrite but was
never surfaced anywhere."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
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

    # "Real-X days" coverage counts (e.g. "Real-price days", "Real-mFRR-
    # capacity days") are excluded from the widget -- still in the
    # underlying Excel report, just not surfaced as dashboard cards.
    other_metrics = [(l, v) for l, v in summary_metrics
                      if l not in ("Days", "Feasible", "Checker passed", "NOTE — out of scope",
                                   "Avg solve (s)", "Avg price MAE (EUR/MWh)",
                                   "Avg PV MAE (MW) — PV actual always synthetic")
                      and not (l.startswith("Real-") and "days" in l)]
    if other_metrics:
        theme.metric_cards(other_metrics, ncols=4)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Portfolio risk (VaR/CVaR)
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
        mc_95 = dict(section_rows.get("Monte Carlo Bootstrap (VaR 95%)", []))
        mc_99 = dict(section_rows.get("Monte Carlo Bootstrap (VaR 99%)", []))
        var_hist_95 = next((v for k, v in hist.items() if k.startswith("VaR(95%)")), None)
        cvar_hist_95 = next((v for k, v in hist.items() if k.startswith("CVaR(95%)")), None)
        var_hist_99 = next((v for k, v in hist.items() if k.startswith("VaR(99%)")), None)
        cvar_hist_99 = next((v for k, v in hist.items() if k.startswith("CVaR(99%)")), None)
        var_mc_95 = next((v for k, v in mc_95.items() if k.startswith("VaR(95%)") and "mean" in k), None)
        cvar_mc_95 = next((v for k, v in mc_95.items() if k.startswith("CVaR(95%)") and "mean" in k), None)
        var_mc_99 = next((v for k, v in mc_99.items() if k.startswith("VaR(99%)") and "mean" in k), None)
        cvar_mc_99 = next((v for k, v in mc_99.items() if k.startswith("CVaR(99%)") and "mean" in k), None)

        if any(v is not None for v in (var_hist_95, cvar_hist_95, var_hist_99, cvar_hist_99)):
            st.markdown("**VaR / CVaR - how bad could a day get**")
            metrics_x = ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"]
            hist_y = [var_hist_95, cvar_hist_95, var_hist_99, cvar_hist_99]
            mc_x, mc_y = [], []
            for label, val in [("VaR 95%", var_mc_95), ("CVaR 95%", cvar_mc_95),
                                ("VaR 99%", var_mc_99), ("CVaR 99%", cvar_mc_99)]:
                if val is not None:
                    mc_x.append(label)
                    mc_y.append(val)
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(x=metrics_x, y=hist_y, name="Historical", marker_color=theme.COLOR_GEN))
            if mc_x:
                fig_var.add_trace(go.Bar(x=mc_x, y=mc_y, name="Monte Carlo", marker_color=theme.COLOR_PUMP))
            theme.style_fig(fig_var, height=320, yaxis_title="EUR (downside from mean)", barmode="group")
            st.plotly_chart(fig_var, width="stretch")
            st.caption("Historical = worst days actually observed in the backtest. Monte Carlo = same estimate "
                       "from 10,000 bootstrap resamples, for a confidence check on the historical number.")

        # "Risk-Adjusted" (Sharpe ratio, Max drawdown) is excluded from the
        # widget -- both are unstable/misleading with the small day-counts
        # this backtest currently has, still in the underlying Excel report.
        for section, rows in section_rows.items():
            if section == "Risk-Adjusted":
                continue
            st.markdown(f"**{section}**")
            theme.metric_cards(rows, ncols=4)


_render()
