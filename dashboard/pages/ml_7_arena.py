"""ML Lifecycle 7 - Model arena: which algorithm is better, where, and whether the difference is real."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Model arena", "Five tree algorithms are compared on the same days, scored the same way, and every pair is tested. The "
             "algorithms' settings are never compared, only their out-of-sample errors.", step=7)
ml_ui.tour_bar(7)

targets = [t for t in ml_data.TARGETS if t != "mfrr_price"]
target = ml_ui.target_picker(targets=targets)
tabs = ml_ui.cached(ml_data.comparison_tables)(target)
daily = ml_ui.cached(lambda: ml_data.pack("daily_loss"))()
board = tabs["scoreboard"]
if board.empty:
    ml_ui.not_yet_run(f"the comparison for {ml_data.label(target)}")
    st.stop()

board = board.sort_values("mae").reset_index(drop=True)
best = board.iloc[0]
pw = tabs["pairwise_dm"]
theme.metric_cards([("Best algorithm", ml_data.model_label(best["model"])), ("Its MAE", f"{best['mae']:.3f} {ml_data.UNITS[target]}"),
                    ("Hours compared", f"{int(best['n']):,}"), ("Pairs that differ (Holm, 5%)", f"{int(pw['significant'].sum())} of {len(pw)}" if not pw.empty else "-")],
                   ncols=4, pre_formatted=True)

st.subheader("Scoreboard")
l, r = st.columns([3, 2])
with l:
    fig = go.Figure(go.Bar(x=[ml_data.model_label(m) for m in board["model"]], y=board["mae"], marker_color=[theme.BLUE if i == 0 else theme.INK_MUTED for i in range(len(board))],
                           text=[f"{v:.2f}" for v in board["mae"]], textposition="outside"))
    st.plotly_chart(theme.style_fig(fig, height=300, yaxis_title=f"MAE ({ml_data.UNITS[target]})", legend=False), width="stretch")
with r:
    st.dataframe(board.assign(model=board["model"].map(ml_data.model_label))[["model", "mae", "rmse", "sse", "days_best"]].rename(columns={
        "model": "Algorithm", "mae": "MAE", "rmse": "RMSE", "sse": "SSE", "days_best": "Share of days best"}).round(3), hide_index=True, width="stretch")
st.caption("Tuned algorithms on the development window. 'Share of days best' is how often an algorithm had the smallest error of the day: "
           "the average error and the winning share can disagree.")

st.subheader("Is the difference real? Every pair, with the Holm correction")
if pw.empty:
    ml_ui.not_yet_run("the pairwise tests")
else:
    models = list(board["model"])
    stat = pd.DataFrame(np.nan, index=models, columns=models)
    star = pd.DataFrame("", index=models, columns=models)
    for _, x in pw.iterrows():
        stat.loc[x["a"], x["b"]], stat.loc[x["b"], x["a"]] = x["statistic"], -x["statistic"]
        mark = "*" if x["p_holm"] < 0.05 else ""
        star.loc[x["a"], x["b"]] = star.loc[x["b"], x["a"]] = mark
    lim = max(3.0, float(np.nanmax(np.abs(stat.values))))
    txt = [[("" if i == j or np.isnan(stat.iloc[i, j]) else f"{stat.iloc[i, j]:.1f}{star.iloc[i, j]}") for j in range(len(models))] for i in range(len(models))]
    fig = go.Figure(go.Heatmap(z=stat.values, x=[ml_data.model_label(m) for m in models], y=[ml_data.model_label(m) for m in models], colorscale="RdBu",
                               zmin=-lim, zmax=lim, text=txt, texttemplate="%{text}", colorbar=dict(title="DM")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(theme.style_fig(fig, height=380, legend=False), width="stretch")
    st.caption("Diebold-Mariano statistic of the row algorithm against the column algorithm. Negative (blue): the row algorithm has the smaller error. "
               "A star marks a difference that is still significant after correcting for the number of pairs tested (Holm).")
    vb = tabs["versus_best"]
    if not vb.empty:
        tied = [m for m, d in zip(vb["model"], vb["distinguishable_from_best"]) if not d]
        st.markdown(f"**Best by mean loss:** {ml_data.model_label(vb['best'].iloc[0])}. **Cannot be told apart from it:** "
                    + (", ".join(ml_data.model_label(m) for m in tied) if tied else "none, every other algorithm is significantly worse."))
        with st.expander("Each algorithm against the best (Holm over these comparisons)"):
            st.dataframe(vb[["model", "mean_loss", "best_loss", "mean_diff", "statistic", "p_value", "p_holm", "distinguishable_from_best"]].round(4), hide_index=True, width="stretch")

st.subheader("Where each algorithm wins")
by = st.radio("Group by", ["hour", "month", "regime"], horizontal=True, key="ml7_by", format_func=lambda x: {"hour": "Hour of day", "month": "Month of the year", "regime": "Market regime"}[x])
t = tabs[f"by_{by}"]
if t.empty:
    ml_ui.not_yet_run("the breakdown")
else:
    mcols = [c for c in t.columns if c in set(board["model"])]
    fig = go.Figure()
    for i, m in enumerate(mcols):
        if by == "hour":
            fig.add_trace(go.Scatter(x=t["group"], y=t[m], mode="lines", name=ml_data.model_label(m), line=dict(color=theme.CATEGORICAL[i % 8], width=1.8)))
        else:
            fig.add_trace(go.Bar(x=t["group"].astype(str), y=t[m], name=ml_data.model_label(m), marker_color=theme.CATEGORICAL[i % 8]))
    st.plotly_chart(theme.style_fig(fig, height=340, yaxis_title="MAE", barmode="group"), width="stretch")
    win = t[["group", "units", "lowest_mae", "winner", "winner_share"]].copy()
    win["lowest_mae"] = win["lowest_mae"].map(ml_data.model_label)
    win["winner"] = win["winner"].map(ml_data.model_label)
    with st.expander("Lowest average error and most frequent winner per group"):
        st.dataframe(win.rename(columns={"group": by.capitalize(), "units": "Units", "lowest_mae": "Lowest MAE", "winner": "Wins most units", "winner_share": "Share of units won"}).round(3),
                     hide_index=True, width="stretch")

st.subheader("Error through time")
if daily.empty:
    ml_ui.not_yet_run("the daily losses (data pack)")
else:
    d = daily[daily["target"] == target]
    opts = [m for m in d["model"].unique() if m not in ("champion", "reference")]
    pick = st.multiselect("Algorithms", opts, default=[o for o in ["lightgbm", "decision_tree"] if o in opts] or opts[:2], format_func=ml_data.model_label, key="ml7_pick")
    win_d = st.slider("Smoothing (days)", 1, 60, 28, key="ml7_smooth")
    fig = go.Figure()
    for i, m in enumerate(pick):
        x = d[d["model"] == m].sort_values("date")
        fig.add_trace(go.Scatter(x=x["date"], y=x["mae"].rolling(win_d, min_periods=1).mean(), mode="lines", name=ml_data.model_label(m),
                                 line=dict(color=theme.CATEGORICAL[i % 8], width=1.6)))
    st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title="daily MAE, smoothed"), width="stretch")
    st.caption("The errors of all algorithms rise and fall together over time, which is why the tests compare them on paired daily losses.")

