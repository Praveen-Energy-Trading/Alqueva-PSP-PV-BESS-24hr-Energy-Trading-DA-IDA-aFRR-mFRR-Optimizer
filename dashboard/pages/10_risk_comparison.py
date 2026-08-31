"""EV vs CVaR Risk Comparison - does the CVaR-averse DA objective actually
help, on real settled prices? Reads runtime/reports/risk_comparison_*.xlsx,
written by phase_6_backtesting_and_validation/run_risk_comparison.py, which
solves the DA gate twice per real-price day (expected-value vs CVaR-averse
risk measure) on the IDENTICAL scenario fan and values both resulting bids
against the REAL settled price. Every number here is read straight from
that file - nothing recomputed or illustrative."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import data
import theme

st.title("⚖️ EV vs CVaR Risk Comparison")


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    files = data.list_risk_comparison_reports()
    if not files:
        st.info("No risk-comparison reports found in `runtime/reports/risk_comparison_*.xlsx`. "
                "Run `phase_6_backtesting_and_validation/run_risk_comparison.py` to generate one.")
        return

    chosen = st.selectbox("Risk comparison report", files, format_func=lambda p: p.name)
    report = data.load_backtest_report(chosen)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    st.subheader("Summary")
    summary_metrics = [(l, v) for l, v, is_hdr in report.get("Summary", []) if not is_hdr]
    if summary_metrics:
        theme.metric_cards(summary_metrics, ncols=3)
    else:
        st.info("This report has no Summary sheet.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # EV vs CVaR realized-outcome risk metrics, side by side
    # ---------------------------------------------------------------------------

    st.subheader("Realized-outcome risk metrics")
    if "RiskComparison" not in report:
        st.info("This report has no RiskComparison sheet.")
    else:
        rows = report["RiskComparison"]
        labels = [label for label, ev, cvar in rows]
        ev_vals = [ev for label, ev, cvar in rows]
        cvar_vals = [cvar for label, ev, cvar in rows]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=ev_vals, name="Expected-value strategy", marker_color=theme.COLOR_GEN))
        fig.add_trace(go.Bar(x=labels, y=cvar_vals, name="CVaR-averse strategy", marker_color=theme.COLOR_PUMP))
        theme.style_fig(fig, height=420, yaxis_title="EUR (or ratio for Sharpe)", barmode="group")
        st.plotly_chart(fig, width="stretch")
        st.caption("Both strategies solved on the IDENTICAL real scenario fan for each day, "
                   "only the risk measure (expected-value vs CVaR-averse) differs - isolates "
                   "the one variable that matters, valued against the REAL settled DA price.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Per-day realised revenue, both strategies
    # ---------------------------------------------------------------------------

    st.subheader("Per-day realised revenue")
    if "Comparison" in report:
        cmp = report["Comparison"]
        real_days = cmp[cmp["price_source"] == "OMIE_LIVE"] if "price_source" in cmp.columns else cmp
        if not real_days.empty and "realised_ev_eur" in real_days.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=real_days["date"], y=real_days["realised_ev_eur"],
                                   name="Expected-value strategy", marker_color=theme.COLOR_GEN))
            fig2.add_trace(go.Bar(x=real_days["date"], y=real_days["realised_cvar_eur"],
                                   name="CVaR-averse strategy", marker_color=theme.COLOR_PUMP))
            theme.style_fig(fig2, height=360, yaxis_title="Realised EUR", barmode="group")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No real-price days in this comparison window.")

        with st.expander("Full per-day table"):
            st.dataframe(cmp, width="stretch", height=300)


_render()
