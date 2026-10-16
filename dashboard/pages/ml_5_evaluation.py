"""ML Lifecycle 5 - Evaluation design: rolling-origin folds, the locked test year, the metrics and the statistical test."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Evaluation design", "", step=4)
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
tune = ml_ui.cached(lambda: ml_data.config("tuning"))().get("search", {})


_fmt = lambda x: f"{pd.Timestamp(x):%d-%m-%Y}"
_val = int(tune.get("n_blocks", 3)) * int(tune.get("validation_days", 30))          # the last days of every training window, where the settings are chosen

st.subheader("How the data is split")
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
    n = len(folds)
    v0 = pd.Timestamp(folds["test_start"].min())                                    # the first validation day of this schedule
    v1 = pd.Timestamp(dev["test_end"]) + pd.Timedelta(days=1)
    start0 = pd.Timestamp(folds["train_start"].min())                               # the first day any refit trains on
    l0, l1 = pd.Timestamp(lk["start"]), pd.Timestamp(lk["end"]) + pd.Timedelta(days=1)
    win, gap, step = schedule["train_window_days"], schedule["gap_days"], every
    train_label = f"{win - _val} days" if win else "all earlier days"
    # one colour, one meaning: blue = the days a model learns from, violet = the days its settings are tuned on, orange = the days it forecasts and is scored on, red = the locked year
    BLUE, VIOLET, ORANGE, RED, MUTED = theme.BLUE, theme.VIOLET, theme.ORANGE, theme.STATUS_CRITICAL, theme.INK_MUTED

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=folds["test_start"], y=folds["fold"], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    # ---- the rows: one refit each, in the same four parts
    for f in folds.itertuples():
        y = int(f.fold)
        t_end = pd.Timestamp(f.train_end) + pd.Timedelta(days=1)
        split = max(t_end - pd.Timedelta(days=_val), pd.Timestamp(f.train_start))
        fig.add_shape(type="rect", x0=f.train_start, x1=split, y0=y - 0.38, y1=y + 0.38, fillcolor=BLUE, opacity=0.55, line_width=0)
        fig.add_shape(type="rect", x0=split, x1=t_end, y0=y - 0.38, y1=y + 0.38, fillcolor=VIOLET, opacity=0.85, line_width=0)
        fig.add_shape(type="rect", x0=f.test_start, x1=pd.Timestamp(f.test_end) + pd.Timedelta(days=1), y0=y - 0.38, y1=y + 0.38, fillcolor=ORANGE, line_width=0)
    fig.add_trace(go.Scatter(x=folds["train_end"], y=folds["fold"], mode="markers", marker=dict(opacity=0), showlegend=False,
                             text=[f"refit {r.fold}: trains on {_fmt(r.train_start)} to {_fmt(r.train_end)} ({r.train_days} days), tunes on its last {_val} days, "
                                   f"forecasts {_fmt(r.test_start)} to {_fmt(r.test_end)}" for r in folds.itertuples()], hoverinfo="text"))
    # ---- the first row is named part by part, on the row itself
    f0 = folds.iloc[0]
    e0 = pd.Timestamp(f0["train_end"]) + pd.Timedelta(days=1)
    s0 = max(e0 - pd.Timedelta(days=_val), pd.Timestamp(f0["train_start"]))
    mid = lambda a_, b_: a_ + (b_ - a_) / 2
    top = -0.42
    span = (l1 + pd.Timedelta(days=20)) - start0
    x_train = min(max(pd.Timestamp(f0["train_start"]) + (s0 - pd.Timestamp(f0["train_start"])) * 0.35, start0 + span * 0.08), s0 - pd.Timedelta(days=2))      # a label is 150-200 px wide: keep it off the left edge
    # two label levels: the lower one holds training, the gap and the forecast, the upper one the tuning; every label sits above the part it names
    callouts = [(x_train, 0, -46, BLUE, f"<b>Training · {train_label}</b><br>the model learns here"),
                (mid(s0, e0), 0, -90, VIOLET, f"<b>Tuning · last {_val} days of the window</b><br>its settings are chosen here"),
                (e0, 24, -46, MUTED, f"{gap} day gap"),
                (mid(pd.Timestamp(f0["test_start"]), pd.Timestamp(f0["test_end"]) + pd.Timedelta(days=1)), 150, -46, ORANGE,
                 f"<b>Validation forecast · {step} days</b><br>forecast, then scored")]
    for x, ax, ay, c, txt in callouts:
        fig.add_annotation(x=x, y=top, ax=ax, ay=ay, axref="pixel", ayref="pixel", text=txt, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=c, font=dict(size=12, color=c),
                           yanchor="bottom", standoff=1, align="center")
    # ---- the three periods, under the rows: what happens in each, and where the results come from
    ys, ye = n + 0.8, n + 4.4
    strips = [(start0, v0, BLUE, 0.55, theme.INK_PRIMARY, f"<b>Training period</b><br>{_fmt(start0)} to {_fmt(v0 - pd.Timedelta(days=1))}<br>used only for learning"),
              (v0, v1, ORANGE, 0.95, "white", f"<b>Validation period</b> &nbsp;{_fmt(v0)} to {_fmt(v1 - pd.Timedelta(days=1))}<br>all the orange validation forecasts together<br>"
                                               "the algorithms are compared and the champion is chosen"),
              (l0, l1, RED, 0.95, "white", f"<b>Locked test</b><br>{_fmt(l0)} to {_fmt(l1 - pd.Timedelta(days=1))}<br>scored once: the final result")]
    for x0, x1, c, op, tc, txt in strips:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=ys, y1=ye, fillcolor=c, opacity=op, line_width=0)
        fig.add_annotation(x=mid(x0, x1), y=(ys + ye) / 2, text=txt, showarrow=False, font=dict(size=11, color=tc), align="center")
    for x in (v0, l0):                                                              # where one period ends and the next begins
        fig.add_shape(type="line", x0=x, x1=x, y0=-0.6, y1=ys, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=l0 + (l1 - l0) / 2, y=n / 2, text="no refit trains on<br>or tunes on this year", showarrow=False, font=dict(size=12, color=RED))
    fig.add_annotation(x=l1, y=-10.2, xanchor="right", yanchor="top", text=f"{n} refits, one row each, a new one every {step} days ↓", showarrow=False, font=dict(size=12, color=MUTED))
    theme.style_fig(fig, height=int((n + 16.5) * 12.5 + 40), xaxis_title=None, legend=False)          # 12.5 px a row whatever the number of rows, so the labels keep their places
    fig.update_yaxes(autorange=False, range=[n + 5.0, -11.5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True)
    fig.update_xaxes(type="date", range=[start0, l1 + pd.Timedelta(days=20)], showgrid=False)
    fig.update_layout(margin=dict(t=8, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.subheader("What is measured")
ml_ui.table(__import__("pandas").DataFrame([{"SSE": "sum of squared errors", "MAE": "mean absolute error", "RMSE": "root mean squared error"}]), stretch=True)
st.caption("MAE is lowest for the median, so the forecasts are median-type: the typical price, not the expected price, which a spike lifts.")

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
        ml_ui.cards([("Days compared", res["n_days"]), ("Mean daily difference", f"{res['mean_diff']:+.3f}"),
                            ("DM statistic", f"{res['statistic']:.2f}"), ("p-value", f"{res['p_value']:.2g}")], ncols=4, pre_formatted=True)
        st.caption("Negative differences (blue) mean the champion was better that day. A statistic far from zero with a small p-value means the "
                   "difference is unlikely to be luck; many pairwise comparisons are corrected with the Holm method on the next pages.")

st.subheader("A short history: mFRR")
expl = ml_data.exploratory("mfrr_price")
if not expl:
    ml_ui.not_yet_run("the mFRR exploration")
else:
    st.info(f"Exploratory, not a decision. mFRR prices start on 27-11-2024, so only {expl['scored_days']} days can be scored against the {expl['required_days']} the "
            "evidence rule requires. Nothing is tuned, no champion is chosen and the locked year is not read.")
    sc = expl["scores"]
    rows = [(n, sc[n]) for n in sorted([n for n in sc if n not in expl["naive_lags_days"]], key=ml_data.model_index)]
    rows += [(n, sc[n]) for n in expl["naive_lags_days"]]
    names = {"naive_week": "Naive: same hour a week earlier", "naive_3days": "Naive: same hour three days earlier"}
    t = __import__("pandas").DataFrame([{"Forecast": names.get(n, ml_data.MODEL_LABELS.get(n, ml_data.pretty(n))), "MAE (EUR/MWh)": s_["mae"], "RMSE (EUR/MWh)": s_["rmse"]} for n, s_ in rows])
    c_left, c_right = st.columns([3, 2])
    with c_left:
        ml_ui.table(t, formats={"MAE (EUR/MWh)": "%.1f", "RMSE (EUR/MWh)": "%.1f"})
    with c_right:
        ml_ui.cards([("Days scored", f"{expl['scored_days']} of {expl['required_days']} required"),
                     ("Best algorithm over the best naive", f"{theme.pct(100 * expl['gain_of_best_over_best_naive'])} lower MAE"),
                     ("Diebold-Mariano p-value", f"{expl['dm_best_vs_best_naive']['p_value']:.2f}")], ncols=1, pre_formatted=True)
    st.caption("The algorithms are ahead of the naive forecasts, but with this few days the difference cannot be told from luck (p above 0.05). "
               "mFRR joins the full process when the windows are rolled forward and the history is long enough.")
