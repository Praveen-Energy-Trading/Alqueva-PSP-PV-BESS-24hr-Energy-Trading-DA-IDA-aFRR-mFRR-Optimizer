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


def _strip(parts, height="4.6rem") -> str:
    """A row of coloured boxes, as wide as their number of days: (days, colour, title, small text)."""
    boxes = "".join(
        f'<div style="flex:{max(days, 1)} 1 0; min-width:0; background:{c}22; border:1.5px solid {c}; border-radius:10px; padding:0.45rem 0.7rem; min-height:{height}; overflow:hidden;">'
        f'<div style="font-weight:600; font-size:0.9rem; color:{theme.INK_PRIMARY}; line-height:1.25;">{title}</div>'
        f'<div style="font-size:0.78rem; color:{theme.INK_SECONDARY}; line-height:1.3; margin-top:0.15rem;">{small}</div></div>'
        for days, c, title, small in parts)
    return f'<div style="display:flex; gap:0.35rem; margin:0.2rem 0 0.5rem 0;">{boxes}</div>'


_fmt = lambda x: f"{d(x):%d-%m-%Y}"
_first, _vs, _ve, _ls, _le = dt.date(2020, 1, 1), d(dev["test_start"]), d(dev["test_end"]), d(lk["start"]), d(lk["end"])
st.markdown('<div style="font-size:0.8rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:' + theme.INK_MUTED + ';">The whole history</div>'
            + _strip([((_vs - _first).days, theme.BLUE, "Training history", f"{_fmt(_first)} to {_fmt(_vs - dt.timedelta(days=1))}<br>only used to fit models"),
                      ((_ve - _vs).days + 1, theme.ORANGE, "Validation",
                       f"{_fmt(_vs)} to {_fmt(_ve)}<br>rolling forecasts, refitted every {sch['retrain_every_days']} days; the algorithms are compared and the champion is chosen here"),
                      ((_le - _ls).days + 1, theme.STATUS_CRITICAL, "Locked test", f"{_fmt(_ls)} to {_fmt(_le)}<br>read once, never used for any choice")]),
            unsafe_allow_html=True)
_win, _val, _gap, _step = int(sch["train_window_days"]), int(tune.get("n_blocks", 3)) * int(tune.get("validation_days", 30)), int(sch["gap_days"]), int(sch["retrain_every_days"])
st.markdown('<div style="font-size:0.8rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:' + theme.INK_MUTED + ';">Every refit</div>'
            + _strip([(6, theme.BLUE, "Train", f"{_win - _val} days"),
                      (3, theme.VIOLET, "Tuning validation", f"the last {_val} days of the window: the hyperparameters are chosen here"),
                      (1, theme.INK_MUTED, "Gap", f"{_gap} day"),
                      (2, theme.ORANGE, "Forecast", f"the next {_step} days")], height="3.4rem"), unsafe_allow_html=True)

st.subheader("Rolling-origin folds")
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
    st.caption(f"{len(folds)} folds, first validation day {folds['test_start'].min()}, last {folds['test_end'].max()}. Blue: training window. Orange: validation days. "
               "Red: locked test year. Change the sliders to see how the schedule changes the folds.")

st.subheader("What is measured")
ml_ui.table(__import__("pandas").DataFrame([{"SSE": "sum of squared errors", "MAE": "mean absolute error", "RMSE": "root mean squared error"}]), stretch=True)

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
