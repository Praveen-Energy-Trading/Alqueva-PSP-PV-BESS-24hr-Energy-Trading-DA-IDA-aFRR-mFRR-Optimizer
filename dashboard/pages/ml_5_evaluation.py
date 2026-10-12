"""ML Lifecycle 5 - Evaluation design: rolling-origin folds, the locked test year, the metrics and the statistical test."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Evaluation design", "A model is only as good as the way it is scored. The scoring contract is fixed before any model is fitted: "
             "which days, trained on what, scored how, and what counts as a real difference.", step=5)
ml_ui.tour_bar(5)

ev = ml_ui.cached(lambda: ml_data.config("evaluation"))()
roll = ml_ui.cached(lambda: ml_data.config("rollout"))()
daily = ml_ui.cached(lambda: ml_data.pack("daily_loss"))()
if not ev:
    ml_ui.not_yet_run("the evaluation configuration")
    st.stop()


def d(x):
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


dev, lk, sch = ev["development"], ev["locked_test"], ev["schedule"]
theme.metric_cards([("Development scoring", f"{dev['test_start']} to {dev['test_end']}"), ("Locked test year", f"{lk['start']} to {lk['end']}"),
                    ("Refit every", f"{sch['retrain_every_days']} days"), ("Training window", f"{sch['train_window_days']} days"),
                    ("Gap before the first test day", f"{sch['gap_days']} day")], ncols=5, pre_formatted=True)

st.subheader("Rolling-origin folds")
st.markdown("The model is refitted every few weeks on the most recent history and used to forecast the next days, exactly as it would have been "
            "run live. Each row below is one refit: a training window (blue), a gap, and the test days it forecasts (orange). The red band is the "
            "locked year: it is never used to choose anything.")
treat = st.radio("Treatment", ["Standard (day-ahead, aFRR)", "Limited evidence (IDA1-3, XBID)"], horizontal=True, key="ml5_treat")
limited = treat.startswith("Limited")
lim = roll.get("treatments", {}).get("limited_evidence", {}).get("schedule", {})
c1, c2, c3 = st.columns(3)
every = c1.slider("Refit every (days)", 7, 90, int(sch["retrain_every_days"]), key="ml5_every")
window_opts = ["rolling 730 days", "rolling 365 days", "expanding"]
default_w = 2 if limited else 0
w = c2.selectbox("Training window", window_opts, index=default_w, key=f"ml5_win_{limited}")
min_train = 180 if limited else int(sch["min_train_days"])
c3.metric("Minimum training days", min_train)
schedule = {"retrain_every_days": every, "gap_days": int(sch["gap_days"]), "min_train_days": min_train,
            "train_window_days": None if w == "expanding" else int(w.split()[1])}
first_data = dt.date(2024, 6, 13) if limited else dt.date(2020, 1, 1)
start = max(d(dev["test_start"]), first_data + dt.timedelta(days=min_train + schedule["gap_days"] + 1))
try:
    folds = ml_data.folds_for(first_data, start, d(dev["test_end"]), schedule)
except ValueError as e:
    st.error(str(e))
    folds = None
if folds is not None and len(folds):
    fig = go.Figure()
    for _, f in folds.iterrows():
        y = int(f["fold"])
        fig.add_shape(type="rect", x0=f["train_start"], x1=f["train_end"], y0=y - 0.4, y1=y + 0.4, fillcolor=theme.BLUE, opacity=0.55, line_width=0)
        fig.add_shape(type="rect", x0=f["test_start"], x1=f["test_end"] + dt.timedelta(days=1), y0=y - 0.4, y1=y + 0.4, fillcolor=theme.ORANGE, line_width=0)
    fig.add_shape(type="rect", x0=d(lk["start"]), x1=d(lk["end"]), y0=-1, y1=len(folds), fillcolor=theme.STATUS_CRITICAL, opacity=0.18, line_width=0)
    fig.add_trace(go.Scatter(x=folds["test_start"], y=folds["fold"], mode="markers", marker=dict(opacity=0), showlegend=False,
                             text=[f"fold {r.fold}: trained {r.train_start} to {r.train_end} ({r.train_days} days), forecasts {r.test_start} to {r.test_end}" for r in folds.itertuples()],
                             hoverinfo="text"))
    fig.update_yaxes(autorange="reversed", title_text="fold")
    st.plotly_chart(theme.style_fig(fig, height=420, legend=False), width="stretch")
    st.caption(f"{len(folds)} folds, first test day {folds['test_start'].min()}, last {folds['test_end'].max()}. Blue: training window. Orange: test days. "
               "Red: locked test year. Change the sliders to see how the schedule changes the folds.")

st.subheader("What is measured")
st.dataframe(__import__("pandas").DataFrame([
    {"Metric": "SSE", "Meaning": "sum of squared errors", "Reads as": "dominated by the largest misses (spikes)"},
    {"Metric": "MAE", "Meaning": "mean absolute error", "Reads as": "the typical miss in EUR/MWh"},
    {"Metric": "RMSE", "Meaning": "root mean squared error", "Reads as": "typical miss, weighting big misses more"}]), hide_index=True, width="stretch")
st.caption("Always in price units, never on the transformed scale: a model trained on a transformed price is judged in euros.")

st.subheader("Is a difference real? The Diebold-Mariano test, worked")
st.markdown("The 24 hours of a day are strongly related, so the test uses **one loss difference per delivery day**, a variance that allows for "
            "autocorrelation (Newey-West, 7 lags) and a small-sample correction. Below: the champion against the decision tree on the development window.")
if daily.empty:
    ml_ui.not_yet_run("the daily losses (data pack)")
else:
    tgt = ml_ui.target_picker(targets=[t for t in ml_data.TARGETS if t in set(daily["target"])])
    res = ml_data.dm_example(daily, tgt)
    if res is None:
        st.info("Not enough days for this target.")
    else:
        diff = res["differences"]
        diff["date"] = diff["date"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=diff["date"], y=diff["difference"], marker_color=[theme.BLUE if v < 0 else theme.ORANGE for v in diff["difference"]], name="daily difference"))
        fig.add_trace(go.Scatter(x=diff["date"], y=diff["difference"].expanding().mean(), mode="lines", line=dict(color=theme.INK_PRIMARY, width=2), name="running mean"))
        fig.add_hline(y=0, line_color=theme.INK_MUTED)
        st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title="champion MAE - reference MAE (EUR)"), width="stretch")
        theme.metric_cards([("Days compared", res["n_days"]), ("Mean daily difference", f"{res['mean_diff']:+.3f}"),
                            ("DM statistic", f"{res['statistic']:.2f}"), ("p-value", f"{res['p_value']:.2g}")], ncols=4, pre_formatted=True)
        st.caption("Negative differences (blue) mean the champion was better that day. A statistic far from zero with a small p-value means the "
                   "difference is unlikely to be luck; many pairwise comparisons are corrected with the Holm method on the next pages.")

st.subheader("The locked year")
st.markdown("""
- The development window **ends before** the locked year starts, and the tools refuse any development run that touches it.
- A final run must cover **exactly** the locked window and each algorithm may be scored on it **once**; every access is logged.
- The outcomes of the locked year are **frozen** by digest, so a changed outcome stops the test.
- Everything else (tuning, comparing, choosing the champion) uses the development window only.
""")

