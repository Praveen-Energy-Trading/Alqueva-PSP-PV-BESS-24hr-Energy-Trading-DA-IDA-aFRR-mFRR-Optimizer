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
from maths import tex as M

ml_ui.header("Evaluation design", "", step=4)
ml_ui.tour_bar(5)

ev = ml_ui.cached(lambda: ml_data.config("evaluation"))()
roll = ml_ui.cached(lambda: ml_data.config("rollout"))()
if not ev:
    ml_ui.not_yet_run("the evaluation configuration")
    st.stop()


def d(x):
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


dev, lk, sch = ev["development"], ev["locked_test"], ev["schedule"]
tune = ml_ui.cached(lambda: ml_data.config("tuning"))().get("search", {})


_fmt = lambda x: f"{pd.Timestamp(x):%d-%m-%Y}"
_val = int(tune.get("n_blocks", 3)) * int(tune.get("validation_days", 30))          # the last days of every training window, where the settings are chosen

INK, NEUTRAL, CARD_LINE = theme.INK_PRIMARY, "#6b6a64", "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731


def _section(title: str, hint: str = "", num: str = "") -> None:
    """A section heading: a number (the steps of the story), a bold title and, under it, one muted line on how to read the section."""
    _n = f"<span style='font-weight:400; color:{NEUTRAL};'>{num}</span>&nbsp; " if num else ""
    st.markdown(f"<div style='margin:1.7rem 0 0.45rem;'><div style='font-size:1.55rem; font-weight:700; color:{INK}; line-height:1.25;'>{_n}{title}</div>"
                + (f"<div style='font-size:0.88rem; color:{theme.INK_MUTED}; margin-top:0.2rem;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _mini(title: str, hint: str = "") -> None:
    """The title of a chart or table inside a section."""
    st.markdown(f"<div style='margin:0.6rem 0 0.3rem;'><div style='font-size:1.02rem; font-weight:700; color:{INK};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{theme.INK_MUTED}; margin-top:0.1rem; line-height:1.35;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _line(html_: str) -> None:
    """One short note under a table or chart, the key numbers in bold."""
    st.markdown(f"<div style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin:0.25rem 0 0.9rem;'>{html_}</div>", unsafe_allow_html=True)


def _panel(inner: str) -> str:
    """A white rounded card, as on the Tuning lab."""
    return (f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.55rem 1.1rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); "
            f"font-size:0.9rem; color:{theme.INK_SECONDARY};'>{inner}</div>")


def _table(headers: list, rows: list, widths: str | None = None, left: tuple = (0,)) -> None:
    """A table as on the Tuning lab: muted header, thin row lines, the first column left, the others centred."""
    grid = f"display:grid; grid-template-columns:{widths or f'minmax(9rem, 1.5fr) repeat({len(headers) - 1}, minmax(0, 1fr))'}; align-items:center; column-gap:0.6rem;"
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{theme.INK_MUTED}; font-weight:600; line-height:1.2;'>"
            + "".join(f"<span style='{'' if j in left else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(headers)) + "</div>")
    body = "".join(f"<div style='{grid} min-height:2.5rem; font-size:0.86rem; border-bottom:1px solid #eeede8;'>"
                   + "".join(f"<span style='{'' if j in left else 'text-align:center;'} color:{INK};'>{c}</span>" for j, c in enumerate(row)) + "</div>" for row in rows)
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{head}{body}</div>", unsafe_allow_html=True)

_section("How the data is split", "Each refit trains on the days before it, tunes on the last days of its window, and forecasts the days after it", num="①")
c_evid, c_win = st.columns([2, 1], gap="medium")
treat = c_evid.radio("Evidence", ["Full (day-ahead, aFRR)", "Limited (IDA1-3, XBID)", "Very limited (mFRR)"], horizontal=True, key="ml5_treat")
tkey = {"Full": "standard", "Limited": "limited_evidence", "Very": "very_limited_evidence"}[treat.split()[0]]
tr_cfg = roll.get("treatments", {}).get(tkey, {}).get("schedule", {})
first_data = {"standard": dt.date(2020, 1, 1), "limited_evidence": dt.date(2024, 6, 13), "very_limited_evidence": dt.date(2024, 11, 27)}[tkey]       # where each group's prices begin
c1, _, _ = st.columns(3, gap="medium")
every = c1.slider("Refit every (days)", 7, 90, int(sch["retrain_every_days"]), key="ml5_every")
window_opts = ["rolling 730 days", "rolling 365 days", "expanding"]
default_w = 0 if tkey == "standard" else 2
w = c_win.selectbox("Training window", window_opts, index=default_w, key=f"ml5_win_{tkey}")
min_train = int(tr_cfg.get("min_train_days", sch["min_train_days"]))
schedule = {"retrain_every_days": every, "gap_days": int(sch["gap_days"]), "min_train_days": min_train,
            "train_window_days": None if w == "expanding" else int(w.split()[1])}
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
    span = (l1 + pd.Timedelta(days=20)) - start0
    px_day = 1150.0 / span.days                                                     # about 1150 px of plot: used only to round the corners by a few pixels

    def pill(x0, x1, y0, y1, color, opacity=1.0, r_px=3.5, row_px=14.0):
        """A rounded bar as a path (the axis is a date axis: x in milliseconds since 1970)."""
        ms = lambda ts: pd.Timestamp(ts).value / 1e6
        xa, xb = ms(x0), ms(x1)
        rx = min(r_px / px_day * 86400000.0, (xb - xa) / 2)
        ry = min(r_px / row_px, (y1 - y0) / 2)
        path = (f"M {xa + rx},{y0} L {xb - rx},{y0} Q {xb},{y0} {xb},{y0 + ry} L {xb},{y1 - ry} Q {xb},{y1} {xb - rx},{y1} "
                f"L {xa + rx},{y1} Q {xa},{y1} {xa},{y1 - ry} L {xa},{y0 + ry} Q {xa},{y0} {xa + rx},{y0} Z")
        fig.add_shape(type="path", path=path, fillcolor=color, opacity=opacity, line_width=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=folds["test_start"], y=folds["fold"], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    one_day = pd.Timedelta(days=1)
    for yr in range(start0.year + 1, l1.year + 1):                                  # faint year guides, so a date can be followed across the rows
        fig.add_shape(type="line", x0=pd.Timestamp(yr, 1, 1), x1=pd.Timestamp(yr, 1, 1), y0=-0.7, y1=n + 0.6, line=dict(color="#eeede7", width=1), layer="below")
    f_first = folds.iloc[0]
    fig.add_shape(type="rect", x0=start0 - pd.Timedelta(days=4), x1=pd.Timestamp(f_first["test_end"]) + pd.Timedelta(days=24), y0=-0.7, y1=0.7, fillcolor="#e4ebf8", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=l0 + pd.Timedelta(days=2), x1=l1 - pd.Timedelta(days=2), y0=-0.7, y1=n + 0.4, fillcolor=RED, opacity=0.05,
                  line=dict(color=RED, width=1, dash="dash"), layer="below")                # the locked year: closed off, nothing trains or tunes in it
    # ---- the rows: one refit each, in the same four parts; the first row, the one that is named, at full strength
    for f in folds.itertuples():
        y = int(f.fold)
        first_row = y == int(folds["fold"].min())
        t_end = pd.Timestamp(f.train_end) + one_day
        split = max(t_end - pd.Timedelta(days=_val), pd.Timestamp(f.train_start))
        k = 1.0 if first_row else 0.72
        pill(f.train_start, split - pd.Timedelta(hours=10), y - 0.36, y + 0.36, BLUE, 0.62 * k if not first_row else 0.75)
        pill(split + pd.Timedelta(hours=10), t_end - pd.Timedelta(hours=10), y - 0.36, y + 0.36, VIOLET, 0.9 * k)
        pill(f.test_start, pd.Timestamp(f.test_end) + one_day, y - 0.36, y + 0.36, ORANGE, k)
    fig.add_trace(go.Scatter(x=folds["train_end"], y=folds["fold"], mode="markers", marker=dict(opacity=0), showlegend=False,
                             text=[f"refit {r.fold}: trains on {_fmt(r.train_start)} to {_fmt(r.train_end)} ({r.train_days} days), tunes on its last {_val} days, "
                                   f"forecasts {_fmt(r.test_start)} to {_fmt(r.test_end)}" for r in folds.itertuples()], hoverinfo="text"))
    # ---- the first row is named part by part, on the row itself
    f0 = folds.iloc[0]
    e0 = pd.Timestamp(f0["train_end"]) + one_day
    s0 = max(e0 - pd.Timedelta(days=_val), pd.Timestamp(f0["train_start"]))
    mid = lambda a_, b_: a_ + (b_ - a_) / 2
    top = -0.42
    x_train = min(max(pd.Timestamp(f0["train_start"]) + (s0 - pd.Timestamp(f0["train_start"])) * 0.35, start0 + span * 0.08), s0 - pd.Timedelta(days=2))      # a label is 150-200 px wide: keep it off the left edge
    # two label levels: the lower one holds training, the gap and the forecast, the upper one the tuning; every label sits above the part it names
    callouts = [(x_train, 0, -46, BLUE, f"<b>Training: {train_label}</b><br>The model learns here"),
                (mid(s0, e0), 0, -90, VIOLET, f"<b>Tuning: last {_val} days of the window</b><br>Its settings are chosen here"),
                (e0, 24, -46, MUTED, f"{gap} day gap"),
                (mid(pd.Timestamp(f0["test_start"]), pd.Timestamp(f0["test_end"]) + one_day), 150, -46, ORANGE,
                 f"<b>Validation forecast: {step} days</b><br>Forecast, then scored")]
    for x, ax, ay, c, txt in callouts:                                             # tags: a thin border in the colour of the part they name, a dot where the line meets it
        fig.add_annotation(x=x, y=top, ax=ax, ay=ay, axref="pixel", ayref="pixel", text=txt, showarrow=True, arrowhead=3, arrowsize=0.9, arrowwidth=1, arrowcolor=c,
                           font=dict(size=12, color=c), yanchor="bottom", standoff=1, align="center", bgcolor="white", bordercolor=c, borderwidth=1, borderpad=4)
    # ---- the three periods, under the rows: what happens in each, and where the results come from
    ys, ye = n + 0.8, n + 4.4
    strips = [(start0, v0, BLUE, 0.7, theme.INK_PRIMARY, f"<b>Training period</b><br>{_fmt(start0)} to {_fmt(v0 - one_day)}<br>used only for learning and tuning"),
              (v0, v1, ORANGE, 1.0, "white", f"<b>Validation period</b> &nbsp;{_fmt(v0)} to {_fmt(v1 - one_day)}<br>all the orange validation forecasts together<br>"
                                              "algorithms compared, champion chosen"),
              (l0, l1, RED, 1.0, "white", f"<b>Locked test</b><br>{_fmt(l0)} to<br>{_fmt(l1 - one_day)}<br>scored once")]
    for x0, x1, c, op, tc, txt in strips:
        pill(x0 + pd.Timedelta(days=1.5), x1 - pd.Timedelta(days=1.5), ys, ye, c, op, r_px=8)
        fig.add_annotation(x=mid(x0, x1), y=(ys + ye) / 2, text=txt, showarrow=False, font=dict(size=11, color=tc), align="center")
    # the tuning days are not a period of their own: the last days before each forecast, so they move with every refit and lie in the training and the validation period alike
    t0 = min(max(pd.Timestamp(r.train_end) + one_day - pd.Timedelta(days=_val), pd.Timestamp(r.train_start)) for r in folds.itertuples())
    t1 = max(pd.Timestamp(r.train_end) + one_day for r in folds.itertuples())
    pill(t0, t1, ye + 0.7, ye + 4.3, VIOLET, 0.9, r_px=8)
    fig.add_annotation(x=mid(t0, t1), y=ye + 2.5, align="center", showarrow=False, font=dict(size=11, color="white"),
                       text=f"<b>Tuning days of all refits together</b><br>{_fmt(t0)} to {_fmt(t1 - one_day)}<br>the {_val} days just before each forecast; they move with every refit")
    for x in (v0, l0):                                                              # where one period ends and the next begins
        fig.add_shape(type="line", x0=x, x1=x, y0=-0.6, y1=ys - 0.2, line=dict(color="#c4c2b9", width=1, dash="dot"))
    fig.add_annotation(x=l0 + (l1 - l0) / 2, y=n / 2, text="<i>no refit trains on<br>or tunes on this year</i>", showarrow=False, font=dict(size=12, color=RED))
    fig.add_annotation(x=l1, y=-10.1, xanchor="right", yanchor="top", text=f"{n} refits, one every {step} days ↓", showarrow=False, font=dict(size=11, color=MUTED), bgcolor="#eef1f8", bordercolor="#d5dcec", borderwidth=1, borderpad=5)
    theme.style_fig(fig, height=int((n + 19.5) * 14 + 60), xaxis_title=None, legend=False)          # 14 px a row whatever the number of rows, so the labels keep their places
    fig.update_yaxes(autorange=False, range=[n + 9.1, -10.4], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
    fig.update_xaxes(type="date", range=[start0, l1 + pd.Timedelta(days=20)], showgrid=False, dtick="M12", tickformat="%Y", ticks="outside", ticklen=4, tickcolor="#c4c2b9",
                     tickfont=dict(size=11, color=MUTED), showline=False)
    fig.update_layout(margin=dict(t=2, b=34, l=10, r=10))
    fig.layout.shapes = tuple(sh for sh in fig.layout.shapes if not (sh.line.color == theme.AXIS_LINE and sh.x0 == sh.x1 == 0))          # no dark axis line down the left: the rows need none
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    sq = lambda c_, t_: f"<span style='white-space:nowrap;'><span style='color:{c_}'>■</span>&nbsp;{t_}</span>"           # noqa: E731
    _line(f"{sq(BLUE, 'Training days: the model learns from them')} &nbsp;&nbsp; {sq(VIOLET, 'Tuning days: its settings are chosen on them')} &nbsp;&nbsp; "
          f"{sq(ORANGE, 'Validation forecast days: forecast, then scored')} &nbsp;&nbsp; {sq(RED, 'The locked year: scored once')}")
    _mini("The schedule in numbers")
    vdays = (v1 - v0).days
    ldays = (l1 - l0).days
    st.markdown(_panel(
        f"<div style='display:grid; grid-template-columns:11rem 1fr; row-gap:0.45rem; align-items:baseline;'>"
        f"<span style='font-weight:600; color:{INK};'>Validation period</span><span>{_fmt(v0)} to {_fmt(v1 - one_day)}, <b style='color:{INK};'>{vdays:,}</b> days</span>"
        f"<span style='font-weight:600; color:{INK};'>Refits</span><span>{M(rf"\frac{{{vdays}}}{{{step}}}={vdays / step:.2f}")}, so <b style='font-size:1.1rem; color:{INK};'>{n}</b> refits, one new row every {step} days</span>"
        f"<span style='font-weight:600; color:{INK};'>Each refit</span><span>Trains on {train_label}, tunes on the last {_val} days of its window, leaves a {gap}-day gap, then forecasts {step} days</span>"
        f"<span style='font-weight:600; color:{INK};'>Locked year</span><span>{_fmt(l0)} to {_fmt(l1 - one_day)}, <b style='color:{INK};'>{ldays:,}</b> days, never trained or tuned on</span></div>"), unsafe_allow_html=True)

_section("What is measured", "Every forecast is judged by the size of its errors, an error being the actual price minus the forecast", num="②")
_mini("The scores", "An illustration with four hourly errors, not data: " + M(r"e = 2,\ -4,\ 1,\ 9"))
_errs = [2, -4, 1, 9]
_sse = sum(e_ ** 2 for e_ in _errs)
_mse = _sse / len(_errs)
_rmse = _mse ** 0.5
_mae = sum(abs(e_) for e_ in _errs) / len(_errs)
_table(["Score", "Name", "Formula", "Worked out on the four errors", "Result"],
       [["SSE", "Sum of squared errors", M(r"\sum_{i=1}^{n} e_i^{2}"), M("+".join(f"{e_ ** 2}" for e_ in _errs)), f"{_sse:,.0f}"],
        ["MSE", "Mean squared error", M(r"\frac{1}{n}\sum_{i=1}^{n} e_i^{2}"), M(rf"\frac{{{_sse}}}{{{len(_errs)}}}"), f"{_mse:,.1f}"],
        ["RMSE", "Root mean squared error", M(r"\sqrt{\mathrm{MSE}}"), M(rf"\sqrt{{{_mse:.1f}}}"), f"{_rmse:,.2f}"],
        ["<b>MAE</b>", "<b>Mean absolute error</b>", M(r"\frac{1}{n}\sum_{i=1}^{n} \left| e_i \right|"),
         M(rf"\frac{{{'+'.join(str(abs(e_)) for e_ in _errs)}}}{{{len(_errs)}}}=\frac{{{sum(abs(e_) for e_ in _errs)}}}{{{len(_errs)}}}"), f"<b>{_mae:,.2f}</b>"]],
       widths="minmax(4rem, 0.6fr) minmax(11rem, 1.6fr) minmax(7rem, 1fr) minmax(12rem, 2fr) minmax(4rem, 0.7fr)", left=(0, 1))
_line(f"MAE is the score the tuning and the comparison use, so the large miss of 9 counts as 9, not as 81: it makes up {100 * 81 / _sse:.0f}% of the SSE but {100 * 9 / sum(abs(e_) for e_ in _errs):.0f}% of the absolute errors. "
      "MAE is lowest for the median, so the forecasts are median-type: the typical price, not the expected price, which a spike lifts.")
