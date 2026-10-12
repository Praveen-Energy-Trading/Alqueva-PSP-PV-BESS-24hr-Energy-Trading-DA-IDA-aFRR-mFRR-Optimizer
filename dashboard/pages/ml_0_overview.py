"""ML Lifecycle 0 - Overview: from data to a monitored model. The whole project on one screen."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("ML Lifecycle: from data to a monitored model",
             "Eight electricity-price forecasting targets of the Portuguese market, taken through every stage of a machine-learning "
             "project, with the evidence of each stage one click away.", step=0)
ml_ui.tour_bar(0)

results = ml_ui.cached(ml_data.results_table)()
stages = ml_ui.cached(ml_data.stage_status)()
facts = ml_ui.cached(lambda: ml_data.pack_json("facts"))() or {}
snap = ml_ui.cached(lambda: ml_data.pack_json("registry_snapshot"))() or {}

decided = results["champion"].notna().sum()
gains = results["gain_vs_reference"].dropna()
naive_gains = results["gain_vs_naive"].dropna()
locked_done = results["locked_mae"].notna().sum()
ml_ui.cards([
    ("Targets with a champion", f"{decided} of {len(results) - 1}"),
    ("Mean gain over the decision tree", f"{100 * gains.mean():.1f}%" if len(gains) else "-"),
    ("Mean gain over the strongest naive forecast", f"{100 * naive_gains.mean():.1f}%" if len(naive_gains) else "-"),
    ("Scored on the locked year", f"{locked_done} of {len(results) - 1}"),
    ("Experiments recorded", len(snap.get("runs", [])) or "-"),
    ("Automated tests", f"{facts.get('test_functions', '-')} in {facts.get('test_files', '-')} files"),
], ncols=6, pre_formatted=True)
st.caption("Gain = how much lower the champion's mean absolute error is than the reference's, on the same hours, in the development window. "
           "References: a single decision tree (the simplest of the five algorithms) and a naive forecast that needs no model "
           "(yesterday's price, last week's price, or for the intraday targets the day-ahead price itself); the strongest of them is used, so the "
           "naive forecast cannot be chosen to flatter the champion.")

st.subheader("The lifecycle")
st.caption("Each box is a stage of the project and its state today. Every one has a page in this section.")
for row in (stages[:5], stages[5:]):
    cols = st.columns(5)
    for col, s in zip(cols, row):
        with col:
            st.markdown(
                f'<div style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE}; border-radius:10px; padding:0.7rem 0.8rem; '
                f'margin-bottom:8px; min-height:112px;"><div style="font-weight:600; color:{theme.INK_PRIMARY}; margin-bottom:6px;">{s["stage"]}</div>'
                f'{ml_ui.badge(s["state"], s["state"])}<div style="color:{theme.INK_SECONDARY}; font-size:0.84rem; margin-top:8px;">{s["fact"]}</div></div>',
                unsafe_allow_html=True)

st.subheader("Where every target stands")
view = results.copy()
for c in ("gain_vs_reference", "gain_vs_naive", "locked_gain_vs_reference", "locked_gain_vs_naive"):
    view[c] = (100 * view[c]).round(1)
view["treatment"] = view["treatment"].map(ml_data.pretty)
st.markdown("**Development window**")
ml_ui.table(view[["label", "treatment", "dev_days", "champion_short", "dev_mae", "unit", "gain_vs_reference", "gain_vs_naive"]].rename(columns={
    "label": "Target", "treatment": "Treatment", "dev_days": "Days", "champion_short": "Champion", "dev_mae": "MAE", "unit": "Unit",
    "gain_vs_reference": "Gain over decision tree (%)", "gain_vs_naive": "Gain over strongest naive (%)"}), formats={"dev_mae": "%.2f"})
st.markdown("**Locked year**")
ml_ui.table(view[["label", "locked_mae", "unit", "locked_gain_vs_reference", "locked_gain_vs_naive"]].rename(columns={
    "label": "Target", "locked_mae": "MAE", "unit": "Unit", "locked_gain_vs_reference": "Gain over decision tree (%)",
    "locked_gain_vs_naive": "Gain over strongest naive (%)"}))

st.markdown("**Gain of the champion over the decision tree and over the strongest naive forecast: development window and locked year**")
g = results.dropna(subset=["gain_vs_reference"])
if len(g):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["target"].map(ml_data.short), y=100 * g["gain_vs_reference"], name="Over decision tree, development", marker_color=theme.BLUE))
    k = g.dropna(subset=["locked_gain_vs_reference"])
    if len(k):
        fig.add_trace(go.Bar(x=k["target"].map(ml_data.short), y=100 * k["locked_gain_vs_reference"], name="Over decision tree, locked year", marker_color=theme.ORANGE))
    gn = results.dropna(subset=["gain_vs_naive"])
    if len(gn):
        fig.add_trace(go.Bar(x=gn["target"].map(ml_data.short), y=100 * gn["gain_vs_naive"], name="Over strongest naive, development", marker_color=theme.CATEGORICAL[2]))
    kn = results.dropna(subset=["locked_gain_vs_naive"])
    if len(kn):
        fig.add_trace(go.Bar(x=kn["target"].map(ml_data.short), y=100 * kn["locked_gain_vs_naive"], name="Over strongest naive, locked year", marker_color=theme.CATEGORICAL[3]))
    fig.add_hline(y=0, line_color=theme.INK_MUTED)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(theme.style_fig(fig, height=380, yaxis_title="% lower MAE than the reference", barmode="group"), width="stretch")
    st.caption("The gain over the reference is measured on the same hours, so it does not depend on the price level. The raw locked-year MAE "
               "does: for aFRR capacity the price level fell by about 80% in 2025, so its locked-year MAE is lower for that reason, not "
               "because the model improved.")
    if not len(k):
        ml_ui.not_yet_run("the locked tests for the other targets")
else:
    ml_ui.not_yet_run("champion decisions")
