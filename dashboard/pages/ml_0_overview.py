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
             "Seven electricity-price forecasting targets of the Portuguese market, taken through every stage of a machine-learning "
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
theme.metric_cards([
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
view["dev_mae"] = view["dev_mae"].round(3)
view["locked_mae"] = view["locked_mae"].round(3)
view["gain_vs_reference"] = (100 * view["gain_vs_reference"]).round(1)
view["locked_gain_vs_reference"] = (100 * view["locked_gain_vs_reference"]).round(1)
view["gain_vs_naive"] = (100 * view["gain_vs_naive"]).round(1)
view["locked_gain_vs_naive"] = (100 * view["locked_gain_vs_naive"]).round(1)
st.dataframe(
    view[["label", "treatment", "dev_days", "champion", "dev_mae", "gain_vs_reference", "gain_vs_naive", "locked_mae", "locked_gain_vs_reference",
          "locked_gain_vs_naive", "unit"]].rename(columns={
        "label": "Target", "treatment": "Treatment", "dev_days": "Development days", "champion": "Champion", "dev_mae": "Development MAE",
        "gain_vs_reference": "Development gain over decision tree (%)", "gain_vs_naive": "Development gain over strongest naive (%)", "locked_mae": "Locked-year MAE",
        "locked_gain_vs_reference": "Locked-year gain over decision tree (%)", "locked_gain_vs_naive": "Locked-year gain over strongest naive (%)", "unit": "Unit"}),
    hide_index=True, width="stretch")

st.markdown("**Gain of the champion over the decision tree and over the strongest naive forecast: development window and locked year**")
g = results.dropna(subset=["gain_vs_reference"])
if len(g):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["label"], y=100 * g["gain_vs_reference"], name="over decision tree, development", marker_color=theme.BLUE))
    k = g.dropna(subset=["locked_gain_vs_reference"])
    if len(k):
        fig.add_trace(go.Bar(x=k["label"], y=100 * k["locked_gain_vs_reference"], name="over decision tree, locked year", marker_color=theme.ORANGE))
    gn = results.dropna(subset=["gain_vs_naive"])
    if len(gn):
        fig.add_trace(go.Bar(x=gn["label"], y=100 * gn["gain_vs_naive"], name="over strongest naive, development", marker_color=theme.CATEGORICAL[2]))
    kn = results.dropna(subset=["locked_gain_vs_naive"])
    if len(kn):
        fig.add_trace(go.Bar(x=kn["label"], y=100 * kn["locked_gain_vs_naive"], name="over strongest naive, locked year", marker_color=theme.CATEGORICAL[3]))
    fig.add_hline(y=0, line_color=theme.INK_MUTED)
    st.plotly_chart(theme.style_fig(fig, height=340, yaxis_title="% lower MAE than the reference", barmode="group"), width="stretch")
    st.caption("The gain over the reference is measured on the same hours, so it does not depend on the price level. The raw locked-year MAE "
               "does: for aFRR capacity the price level fell by about 80% in 2025, so its locked-year MAE is lower for that reason, not "
               "because the model improved.")
    if not len(k):
        ml_ui.not_yet_run("the locked tests for the other targets")
else:
    ml_ui.not_yet_run("champion decisions")

st.subheader("What this project demonstrates")
st.markdown("""
1. **Problem framing:** what is forecast, what is known when, and why a spread or a level.
2. **Honest evaluation:** a rolling origin, a locked test year used once, point-in-time features, and statistical tests of every difference.
3. **Fair tuning and comparison:** nested time-aware tuning with equal budgets, and Holm-corrected pairwise tests.
4. **A written decision rule,** fixed before the results, and a report of what was found even when it is unflattering.
5. **ML as operations:** a registry, reproducible experiments, a promotion rule, monitoring, retraining and a safe live switch.
""")

ml_ui.talk_track(
    "This is a price-forecasting system for the Portuguese electricity market, built end to end: seven forecasting targets from day-ahead "
    "to the reserve markets. I treated it as a real machine-learning project, not a notebook: each box above is a stage with its own "
    "code, tests and evidence, and I can show you any of them.",
    "Which result matters most?",
    "The day-ahead champion, tuned LightGBM, is 11% better than the decision-tree reference, and the difference is significant against "
    "every other algorithm. The intraday targets show smaller, mostly tied differences, and I say so: with about 300 days only larger "
    "differences can be established.")
ml_ui.limits("Seven targets, six with a champion: mFRR is deferred because its history is too short. The intraday targets are labelled "
             "limited evidence. Gains are measured against a decision tree and against the strongest of several naive forecasts; the intraday gains over a naive forecast are small.")
