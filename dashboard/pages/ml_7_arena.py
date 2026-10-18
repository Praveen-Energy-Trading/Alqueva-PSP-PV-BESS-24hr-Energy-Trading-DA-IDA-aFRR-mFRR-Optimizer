"""ML Lifecycle 7 - Model arena: which algorithm is better, where, and whether the difference is real. Told as a story in numbered steps."""
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
             "algorithms' settings are never compared, only their out-of-sample errors.", step=6)
ml_ui.tour_bar(7)

# The colours mean one thing each, on every picture of this page:
#   an algorithm keeps its own colour; the champion is near-black; a real difference is green, a gap that could be luck is grey.
MUTED, INK, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#2b2a28"
NEUTRAL, T_DONE, GREY, REAL, LUCK = "#6b6a64", "#8791ab", "#b9b7ae", theme.GREEN, "#9a9890"
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # a hint starts with a capital
_alg = lambda m_: CHAMP if m_ == "champion" else theme.CATEGORICAL[ml_data.model_index(m_) % 8]                       # noqa: E731
_name = lambda m_: "Champion" if m_ == "champion" else ml_data.model_label(m_)                                         # noqa: E731
_dot = lambda m_: f"<span style='color:{_alg(m_)}'>■</span>&nbsp;"                                                      # noqa: E731


def _section(title: str, hint: str = "", num: str = "") -> None:
    """A section heading: a number (the steps of the story), a bold title and, under it, one muted line on how to read the section."""
    _n = f"<span style='font-weight:400; color:{NEUTRAL};'>{num}</span>&nbsp; " if num else ""
    st.markdown(f"<div style='margin:1.7rem 0 0.45rem;'><div style='font-size:1.55rem; font-weight:700; color:{INK}; line-height:1.25;'>{_n}{title}</div>"
                + (f"<div style='font-size:0.88rem; color:{MUTED}; margin-top:0.2rem;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _mini(title: str, hint: str = "") -> None:
    """The title of a chart or table inside a section."""
    st.markdown(f"<div style='margin:0.6rem 0 0.3rem;'><div style='font-size:1.02rem; font-weight:700; color:{INK};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{MUTED}; margin-top:0.1rem; line-height:1.35;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _line(html: str) -> None:
    """One short note under a table or chart, the key numbers in bold."""
    st.markdown(f"<div style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin:0.25rem 0 0.9rem;'>{html}</div>", unsafe_allow_html=True)


def _panel(inner: str, min_height: str = "") -> str:
    """A white rounded card for a calculation, as on the Tuning lab."""
    return (f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.55rem 1.1rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); "
            f"font-size:0.9rem; color:{theme.INK_SECONDARY}; {min_height}'>{inner}</div>")


def _table(headers: list, rows: list, highlight: int | None = None, first: str = "minmax(9rem, 1.5fr)") -> None:
    """A table as on the Tuning lab: muted header, thin row lines, the first column left, the others centred; one row can be shaded (the best, or the mean)."""
    grid = f"display:grid; grid-template-columns:{first} repeat({len(headers) - 1}, minmax(0, 1fr)); align-items:center; column-gap:0.6rem;"
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{MUTED}; font-weight:600; line-height:1.2;'>"
            + "".join(f"<span style='{'' if j == 0 else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(headers)) + "</div>")
    body = ""
    for i, row in enumerate(rows):
        hl = i == highlight
        body += (f"<div style='{grid} min-height:2.5rem; font-size:0.86rem; "
                 + ("border-top:1px solid #e3e1d9; background:#f8f7f3; font-weight:700;" if hl else "border-bottom:1px solid #eeede8;") + "'>"
                 + "".join(f"<span style='{'' if j == 0 else 'text-align:center;'} color:{INK if str(c) != '–' else MUTED};'>{c}</span>" for j, c in enumerate(row)) + "</div>")
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{head}{body}</div>", unsafe_allow_html=True)


def _pill(text: str, color: str, bg: str) -> str:
    return f"<span style='background:{bg}; color:{color}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{text}</span>"


def _legend_top(fig, y=1.08) -> None:
    """The legend of a chart on one row above it, at the left, as on the Tuning lab."""
    fig.update_layout(legend=dict(orientation="h", x=0, y=y, xanchor="left", yanchor="bottom", font=dict(size=12, color=theme.INK_SECONDARY)))


targets = list(ml_data.TARGETS)
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
units = ml_data.UNITS[target]
models = list(board["model"])
n_hours = int(best["n"])
n_days_all = round(n_hours / 24)
res0 = ml_data.dm_example(daily, target) if not daily.empty else None                                                # the champion against the decision tree: the example of the figure

# ============================================================================================ 1. the arena, drawn
SH, AN = [], []                                                                                  # shapes and annotations, handed to the figure once at the end
ROW, X_END = 34.0, 1010
px_x = 1150.0 / X_END


def _shp(x0, x1, y0, y1, color, opacity=1.0, r_px=6.0):
    """A rounded bar as a path; x in the units of the figure, y in rows."""
    rx, ry = min(r_px / px_x, (x1 - x0) / 2), min(r_px / ROW, (y1 - y0) / 2)
    SH.append(dict(type="path", fillcolor=color, opacity=opacity, line_width=0,
                   path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                         f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")))


def _fcard(y0, y1, r_px=10.0):
    rx, ry = r_px / px_x, r_px / ROW
    x0, x1 = -12, X_END + 12
    SH.append(dict(type="path", fillcolor="#ffffff", line=dict(color=CARD_LINE, width=1), layer="below",
                   path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                         f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")))


def _fhead(y, num, title, sub_=""):
    AN.append(dict(x=6, y=y, xanchor="left", yanchor="middle", showarrow=False, align="left", font=dict(size=14, color=INK),
                   text=f"<b style='font-size:17px;color:{NEUTRAL}'>{num}</b>&nbsp; <b>{title}</b>" + (f"<br><span style='font-size:12px;color:{MUTED}'>{_cap(sub_)}</span>" if sub_ else "")))


def _fnote(x, y, text, size=12, color=INK, anchor="center", **kw):
    AN.append(dict(x=x, y=y, xanchor=anchor, showarrow=False, text=text, font=dict(size=size, color=color), **kw))


def _farrow(x0, y0, x1, y1, color=MUTED):
    AN.append(dict(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=color, text=""))


piv = None
if not daily.empty:
    dd = daily[(daily["target"] == target) & (daily["model"].isin(models))]
    piv = dd.pivot_table(index="model", columns="date", values="mae").reindex(models).dropna(axis=1)
if piv is not None and piv.shape[1] >= 20:
    dates = list(piv.columns)
    n_d = len(dates)
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=[0, X_END], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    # ---- card 1: five lanes over the same days
    _fcard(-0.6, 7.45)
    _fhead(0.4, "①", "The same days for every algorithm", f"{ml_data.label(target)}, development window: {n_d:,} delivery days from {pd.Timestamp(dates[0]):%d-%m-%Y} to {pd.Timestamp(dates[-1]):%d-%m-%Y}")
    pitch, y_first, xs0, xs1 = 0.95, 1.7, 215.0, 1000.0
    ys = [y_first + i * pitch for i in range(len(models))]
    xs = np.linspace(xs0, xs1, n_d)
    z = piv.to_numpy(dtype=float)
    lo, hi = float(np.nanquantile(z, 0.05)), float(np.nanquantile(z, 0.95))
    fig_a.add_trace(go.Heatmap(x=xs, y=ys, z=np.clip(z, lo, hi), zmin=lo, zmax=hi, showscale=False, ygap=5, colorscale=[[0, "#3d4d78"], [0.5, T_DONE], [1, "#e9ecf3"]],
                               text=[[f"{ml_data.model_label(m)}, {pd.Timestamp(dt):%d-%m-%Y}: {v:.1f}" for dt, v in zip(dates, row)] for m, row in zip(models, z)],
                               hoverinfo="text", hoverlabel=dict(bgcolor="white", font=dict(size=12, color=INK), bordercolor=CARD_LINE)))
    for m, y in zip(models, ys):
        _fnote(8, y, f"{_dot(m)}<b>{ml_data.model_label(m)}</b>", anchor="left")
    _fnote(xs0, ys[-1] + 0.72, f"{pd.Timestamp(dates[0]):%d-%m-%Y}", size=11, color=MUTED, anchor="left")
    _fnote(xs1, ys[-1] + 0.72, f"{pd.Timestamp(dates[-1]):%d-%m-%Y}", size=11, color=MUTED, anchor="right")
    _fnote(8, ys[-1] + 1.3, "One column is one delivery day, the same day in every row. Darker = smaller error that day.", size=11, color=MUTED, anchor="left")
    # ---- card 2: one score per day
    y2 = 7.8
    _fcard(y2, y2 + 4.5)
    _fhead(y2 + 0.8, "②", "One score per day", "A forecast error is the gap to the real price, in absolute value")
    cy = y2 + 2.75
    x = 8.0
    for h in range(24):
        _shp(x, x + 15, cy - 0.4, cy + 0.4, GREY, 0.9, r_px=2)
        x += 18
    _fnote(8, cy - 0.95, "24 hourly errors of one delivery day", size=11, color=MUTED, anchor="left")
    _fnote(8, cy + 0.85, "Hour 1", size=10, color=MUTED, anchor="left")
    _fnote(8 + 24 * 18 - 3, cy + 0.85, "Hour 24", size=10, color=MUTED, anchor="right")
    ex = piv.loc[models[0]]
    ex_day = ex.index[int((ex - ex.median()).abs().to_numpy().argmin())]                    # a typical day of the best algorithm
    _farrow(448, cy, 497, cy)
    _shp(500, 700, cy - 0.62, cy + 0.62, NEUTRAL, 0.12, r_px=8)
    _fnote(600, cy, "<b>Day score</b><br>mean of the 24", size=12)
    _fnote(600, cy + 1.05, f"{float(ex[ex_day]):.1f} on {pd.Timestamp(ex_day):%d-%m-%Y}", size=10, color=MUTED)
    _farrow(703, cy, 752, cy)
    _shp(755, 1000, cy - 0.62, cy + 0.62, NEUTRAL, 0.12, r_px=8)
    _fnote(877, cy, "<b>MAE</b><br>mean of all hourly errors", size=12)
    _fnote(877, cy + 1.05, f"{float(best['mae']):.3f} {units} over {n_hours:,} hours", size=10, color=MUTED)
    end = y2 + 4.5
    # ---- card 3: one gap per day for a pair (the champion against the decision tree)
    if res0 is not None:
        y3 = end + 0.35
        _fcard(y3, y3 + 4.4)
        _fhead(y3 + 0.8, "③", "One gap per day for a pair", "The two models are scored on the same day; their difference is that day's gap")
        lastd = res0["losses"].tail(21).reset_index(drop=True)
        kk = int((lastd["a"] - lastd["b"] - (lastd["a"] - lastd["b"]).mean()).abs().idxmin())               # a typical day, as on the picture of step 5
        va, vb = round(float(lastd["a"][kk]), 1), round(float(lastd["b"][kk]), 1)                           # the gap is the difference of the numbers shown
        cya, cyb = y3 + 2.55, y3 + 3.55
        scale = 470.0 / max(va, vb)
        xa, xb = 190 + va * scale, 190 + vb * scale
        _fnote(8, cya, f"{_dot('champion')}<b>Champion</b>", anchor="left")
        _fnote(8, cyb, f"{_dot('decision_tree')}<b>Decision tree</b>", anchor="left")
        _shp(190, xa, cya - 0.32, cya + 0.32, _alg("champion"), 0.9, r_px=3)
        _shp(190, xb, cyb - 0.32, cyb + 0.32, _alg("decision_tree"), 0.9, r_px=3)
        _fnote(xa + 8, cya, f"{va:.1f}", size=11, color=_alg("champion"), anchor="left")
        _fnote(xb + 8, cyb, f"{vb:.1f}", size=11, color=_alg("decision_tree"), anchor="left")
        _fnote(190, y3 + 1.75, f"Day score on {pd.Timestamp(lastd['date'][kk]):%d-%m-%Y}", size=11, color=MUTED, anchor="left")
        for xx in (xa, xb):
            SH.append(dict(type="line", x0=xx, x1=xx, y0=cya - 0.45, y1=cyb + 0.5, line=dict(color="#a8a69d", width=1.2, dash="dot")))
        _shp(730, 1000, y3 + 2.0, y3 + 4.0, NEUTRAL, 0.12, r_px=8)
        _fnote(865, y3 + 2.5, f"<b>Gap = {va:.1f} − {vb:.1f} = {va - vb:+.1f}</b>", size=12)
        _fnote(865, y3 + 3.1, f"{'Champion' if va < vb else 'Decision tree'} closer", size=10, color=MUTED)
        _fnote(865, y3 + 3.5, f"One gap per day, {res0['n_days']:,} days", size=10, color=MUTED)
        end = y3 + 4.4
    fig_a.layout.shapes = ()
    theme.style_fig(fig_a, height=int((end + 0.9) * ROW + 12), xaxis_title=None, legend=False)
    fig_a.update_yaxes(autorange=False, range=[end + 0.15, -0.75], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
    fig_a.update_xaxes(range=[-15, X_END + 15], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
    fig_a.update_layout(margin=dict(t=2, b=2, l=10, r=10), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    fig_a.layout.shapes = SH
    fig_a.layout.annotations = AN
    st.plotly_chart(fig_a, width="stretch", config={"displayModeBar": False})

# ============================================================================================ 2. the scoreboard
_section("The scoreboard", "Lower is better. Counting days instead of size of errors can give a different winner.", num="④")
if piv is not None and piv.shape[1] >= 20:
    _cnt = {m_: int((piv.idxmin() == m_).sum()) for m_ in models}
    _nd = piv.shape[1]
else:
    _nd = n_days_all
    _cnt = {m_: int(round(float(board.loc[board["model"] == m_, "days_best"].iloc[0]) * _nd)) for m_ in models}
_names = [ml_data.model_label(m) for m in models]
_mini("Average error (MAE)")
f_s = go.Figure(go.Bar(x=_names, y=board["mae"], marker_color=[_alg(m) for m in models], text=[f"<b>{v:.3f}</b>" if i == 0 else f"{v:.3f}" for i, v in enumerate(board["mae"])],
                       textposition="outside", cliponaxis=False, textfont=dict(size=12, color=theme.INK_SECONDARY), hoverinfo="skip"))
theme.style_fig(f_s, height=290, yaxis_title=f"MAE ({units})", legend=False)
f_s.update_xaxes(tickangle=0)
f_s.update_yaxes(range=[0, float(board["mae"].max()) * 1.18])
f_s.update_layout(margin=dict(t=24), bargap=0.55)
st.plotly_chart(f_s, width="stretch", config={"displayModeBar": False})
_mini("Days on which each algorithm had the smallest error")
f_b = go.Figure(go.Bar(x=_names, y=[100 * _cnt[m] / _nd for m in models], marker_color=[_alg(m) for m in models], text=[f"{100 * _cnt[m] / _nd:.1f}%" for m in models],
                       textposition="outside", cliponaxis=False, textfont=dict(size=12, color=theme.INK_SECONDARY), hoverinfo="skip"))
theme.style_fig(f_b, height=290, yaxis_title="Share of days (%)", legend=False)
f_b.update_xaxes(tickangle=0)
f_b.update_yaxes(range=[0, max(100 * _cnt[m] / _nd for m in models) * 1.18])
f_b.update_layout(margin=dict(t=24), bargap=0.55)
st.plotly_chart(f_b, width="stretch", config={"displayModeBar": False})
_mini("The scoreboard in numbers", f"{n_hours:,} hourly forecasts on {_nd:,} days. RMSE is the same score with squared errors, so large misses weigh more.")
_table(["Algorithm", f"MAE ({units})", f"RMSE ({units})", "Days with the smallest error", "Share of days"],
       [[f"{_dot(m)}{ml_data.model_label(m)}", f"{float(r['mae']):.3f}", f"{float(r['rmse']):.3f}", f"{_cnt[m]:,} of {_nd:,}", f"{100 * _cnt[m] / _nd:.1f}%"] for m, (_, r) in zip(models, board.iterrows())], highlight=0)
_line("The best algorithm by MAE is shaded.")

_mini("Calculation for one algorithm")
_rc, _rd = st.columns([1, 3], gap="medium")
with _rc:
    _rsel = st.selectbox("Algorithm", models, index=0, format_func=ml_data.model_label, key="ml7_row", label_visibility="collapsed")
_rb = board[board["model"] == _rsel].iloc[0]
with _rd:
    st.markdown(_panel(
        f"<div style='display:grid; grid-template-columns:9rem 1fr; row-gap:0.45rem; align-items:baseline;'>"
        f"<span style='font-weight:600; color:{INK};'>MAE</span><span>The mean of the {n_hours:,} hourly errors = <b style='font-size:1.1rem; color:{_alg(_rsel)};'>{_rb['mae']:.3f}</b> {units}</span>"
        f"<span style='font-weight:600; color:{INK};'>Smallest error</span><span>{_cnt[_rsel]:,} of {_nd:,} days = <b style='font-size:1.1rem; color:{INK};'>{100 * _cnt[_rsel] / _nd:.1f}%</b> of the days</span></div>"), unsafe_allow_html=True)
_line("A day counts for the algorithm with the smallest error of that day. A high share of days can go with a worse average when the misses on the other days are large.")

# ============================================================================================ 3-4. is the gap real
if daily.empty:
    _section("Is the edge real or luck? Day by day", num="⑤")
    ml_ui.not_yet_run("the daily losses (data pack)")
    res, A, B = None, "champion", "decision_tree"
else:
    _section("Is the edge real or luck? Day by day", "Pick any two to compare on the same days", num="⑤")
    _pa, _pb, _px = st.columns([2, 2, 4], gap="medium")
    _opts = ["champion"] + models
    with _pa:
        A = st.selectbox("Compare", _opts, index=0, format_func=_name, key="ml7_a")
    with _pb:
        B = st.selectbox("Against", _opts, index=_opts.index("decision_tree") if "decision_tree" in _opts else 1, format_func=_name, key="ml7_b")
    la, lb = _name(A), _name(B)
    res = ml_data.dm_example(daily, target, A, B) if A != B else None
    if res is not None and not (np.isfinite(res["statistic"]) and np.isfinite(res["p_value"]) and float(res["statistic"]) != 0.0):
        res = None
        st.info(f"{la} and {lb} give the same errors on these days, so there is no gap to test.")
    elif A == B:
        st.info("Pick two different algorithms.")
    elif res is None:
        st.info("Not enough days for this target.")
if res is not None and A != B:
    ca, cb = _alg(A), _alg(B)
    diff, los = res["differences"].copy(), res["losses"].copy()
    diff["date"] = diff["date"].astype(str)
    mean, stat, p, n_d = float(res["mean_diff"]), float(res["statistic"]), float(res["p_value"]), int(res["n_days"])
    wins = int((diff["difference"] < 0).sum())
    _mini("Every day, both are scored", "Each bar is the average error of one delivery day. The shorter bar was closer. The last 21 days are shown.")
    last = los.tail(21).reset_index(drop=True)
    lab = [f"{pd.Timestamp(x):%d-%m}" for x in last["date"]]
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=lab, y=last["a"], name=la, marker_color=ca))
    f1.add_trace(go.Bar(x=lab, y=last["b"], name=lb, marker_color=cb))
    k = int((last["a"] - last["b"] - (last["a"] - last["b"]).mean()).abs().idxmin())
    ka, kb = round(float(last["a"][k]), 1), round(float(last["b"][k]), 1)
    f1.add_annotation(x=lab[k], y=max(last["a"][k], last["b"][k]), ax=0, ay=-48, showarrow=True, arrowhead=3, arrowwidth=1, arrowcolor=INK, font=dict(size=12, color=INK),
                      text=f"<b>This day</b><br>{la} {ka:.1f}, {lb} {kb:.1f}<br>Gap = {ka - kb:+.1f}", align="center",
                      bgcolor="white", bordercolor=INK, borderwidth=1, borderpad=4, yanchor="bottom")
    theme.style_fig(f1, height=300, yaxis_title="Average error of the day", barmode="group")
    f1.update_xaxes(type="category", tickangle=0, tickfont=dict(size=10, color=MUTED))
    f1.update_yaxes(range=[0, float(max(last["a"].max(), last["b"].max())) * 1.6])
    _legend_top(f1)
    f1.update_layout(margin=dict(t=34))
    st.plotly_chart(f1, width="stretch", config={"displayModeBar": False})
    _mini("One gap per day, over the whole window", f"The gap of a day is {la}'s error minus {lb}'s. Below zero, {la} was closer. The line is the running average.")
    neg, pos = diff["difference"].clip(upper=0), diff["difference"].clip(lower=0)
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=diff["date"], y=neg, name=f"{la} closer", marker_color=ca))
    f2.add_trace(go.Bar(x=diff["date"], y=pos, name=f"{lb} closer", marker_color=cb))
    f2.add_trace(go.Scatter(x=diff["date"], y=diff["difference"].expanding().mean(), mode="lines", line=dict(color=INK, width=2), name="Running average"))
    f2.add_hline(y=0, line_color=MUTED)
    theme.style_fig(f2, height=300, yaxis_title=f"{la} − {lb} ({units})", barmode="relative")
    _legend_top(f2)
    f2.update_layout(margin=dict(t=34))
    st.plotly_chart(f2, width="stretch", config={"displayModeBar": False})
    _line(f"{la} was closer on <b>{wins:,} of {n_d:,} days ({100 * wins / n_d:.1f}%)</b>. Closer on most days is not yet proof: the days are not independent, and luck can produce a lead.")

    _section("The average gap against luck, and the verdict", num="⑥")
    se = abs(mean / stat) if stat else float(diff["difference"].std() / n_d ** 0.5)
    sd_gap = float(diff["difference"].std())
    band = 1.96 * se
    m = max(abs(mean), band, 3.2 * se) * 1.18
    c_obs = ca if mean < 0 else cb
    closer = la if mean < 0 else lb
    _l, _r = st.columns([5, 3], gap="medium")
    with _l:
        xs_b = np.linspace(-m, m, 600)
        ys_b = np.exp(-0.5 * (xs_b / se) ** 2)
        inz = np.abs(xs_b) <= band
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=xs_b, y=ys_b, mode="lines", line=dict(color="#a8a69d", width=1.5), fill="tozeroy", fillcolor="rgba(196,194,185,0.22)", hoverinfo="skip", showlegend=False))
        f3.add_trace(go.Scatter(x=xs_b[inz], y=ys_b[inz], mode="lines", line=dict(width=0), fill="tozeroy", fillcolor="rgba(196,194,185,0.75)", hoverinfo="skip", showlegend=False))
        f3.add_trace(go.Scatter(x=[mean], y=[0], mode="markers", marker=dict(symbol="diamond", size=17, color=c_obs, line=dict(color="white", width=1.5)), cliponaxis=False,
                                showlegend=False, hovertemplate=f"Observed average gap {mean:+.4f}<extra></extra>"))
        f3.add_annotation(x=mean, y=0, ax=(105 if mean < -0.5 * m else -105 if mean > 0.5 * m else 0), ay=-70, axref="pixel", ayref="pixel", showarrow=True, arrowhead=2, arrowwidth=1.6,
                          arrowcolor=c_obs, font=dict(size=12, color=c_obs), align="center",
                          text=f"<b>Observed average gap {mean:+.3f}</b><br>{abs(stat):.1f} luck-widths from zero, {closer} closer", bgcolor="white", bordercolor=c_obs, borderwidth=1, borderpad=4)
        f3.add_annotation(x=0, y=1.0, yanchor="bottom", showarrow=False, font=dict(size=12, color=MUTED), align="center",
                          text=f"<b>The luck zone</b> (±{band:.3f})<br>If the two were equal, the average would land here 95% of the time")
        kmax = int(m / se)
        stp = max(1, int(np.ceil(2 * kmax / 12)))
        tv = [kk * se for kk in range(-kmax, kmax + 1) if kk % stp == 0]
        tt = [f"{kk}".replace("-", "−") for kk in range(-kmax, kmax + 1) if kk % stp == 0]
        theme.style_fig(f3, height=300, xaxis_title="Average gap in luck-widths", legend=False)
        f3.update_xaxes(range=[-m, m], tickmode="array", tickvals=tv, ticktext=tt, zeroline=False, showgrid=False)
        f3.update_yaxes(range=[-0.02, 1.4], visible=False, fixedrange=True)
        st.plotly_chart(f3, width="stretch", config={"displayModeBar": False})
    with _r:
        real = p < 0.05
        badge_col = REAL if real else LUCK
        ph = None
        if A in models and B in models and not pw.empty:
            _hit = pw[((pw["a"] == A) & (pw["b"] == B)) | ((pw["a"] == B) & (pw["b"] == A))]
            if len(_hit):
                ph = float(_hit["p_holm"].iloc[0])
        _hd = lambda txt_: f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; margin-top:0.55rem;'>{txt_}</div>"                    # noqa: E731
        st.markdown(_panel(
            f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; padding:0.2rem 0;'>① One luck-width</div>"
            f"<div>= <b style='font-size:1.1rem; color:{INK};'>{se:.4f}</b>, how much the average gap moves by chance</div>"
            f"<div style='font-size:0.8rem; line-height:1.45; margin-top:0.2rem;'>Plain: the spread of the {n_d:,} daily gaps ÷ √{n_d:,} = {sd_gap:.3f} ÷ {n_d ** 0.5:.2f} = {sd_gap / n_d ** 0.5:.4f}.<br>"
            f"Neighbouring days move together, so they hold less news than {n_d:,} independent days: the corrected figure is {se:.4f}.</div>"
            + _hd("② The gap in luck-widths: the DM statistic")
            + f"<div>{mean:+.4f} ÷ {se:.4f} = <b style='font-size:1.1rem; color:{INK};'>{stat:.2f}</b></div>"
            + _hd("③ The chance of a gap this big if the two were equal")
            + f"<div>p = <b style='font-size:1.1rem; color:{INK};'>{p:.2g}</b>" + (f", after the Holm correction {ph:.2g}" if ph is not None else "") + "</div>"
            + f"<div style='margin-top:0.7rem; font-size:1.25rem; font-weight:800; color:{badge_col};'>{'Real, not luck' if real else 'Not distinguishable from luck'}</div>"), unsafe_allow_html=True)
    _line("Beyond about 2 luck-widths, the edge of the luck zone, a gap is called real. The shaded middle is where chance alone would put the average.")

# ============================================================================================ 5. every pair, with Holm
_section("Every pair, with the Holm correction", f"{len(pw)} pairs are tested, so some would look real by chance alone" if not pw.empty else "", num="⑦")
if pw.empty:
    ml_ui.not_yet_run("the pairwise tests")
else:
    stat_m = pd.DataFrame(np.nan, index=models, columns=models)
    star = pd.DataFrame("", index=models, columns=models)
    for _, x in pw.iterrows():
        stat_m.loc[x["a"], x["b"]], stat_m.loc[x["b"], x["a"]] = x["statistic"], -x["statistic"]
        mark = "*" if x["p_holm"] < 0.05 else ""
        star.loc[x["a"], x["b"]] = star.loc[x["b"], x["a"]] = mark
    lim = max(3.0, float(np.nanmax(np.abs(stat_m.values))))
    txt = [[("" if i == j or np.isnan(stat_m.iloc[i, j]) else f"{stat_m.iloc[i, j]:.2f}{star.iloc[i, j]}") for j in range(len(models))] for i in range(len(models))]
    _mini("The test statistic of every pair", "Row against column. Blue: the row algorithm has the smaller error. Red: the column algorithm has. A star: still real after the correction.")
    fig = go.Figure(go.Heatmap(z=stat_m.values, x=_names, y=_names, colorscale="RdBu_r", zmin=-lim, zmax=lim, text=txt, texttemplate="%{text}", colorbar=dict(title="DM", thickness=12),
                               hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    theme.style_fig(fig, height=340, legend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    _mini("Why the correction", "The p-values, smallest first, against what each is allowed to be")
    pws = pw.sort_values("p_value").reset_index(drop=True)
    mm = len(pws)
    ranks = list(range(1, mm + 1))
    thr = [0.05 / (mm - i + 1) for i in ranks]
    okk, stop = [], False
    for pv_, th_ in zip(pws["p_value"], thr):                                       # Holm: walk up from the smallest p-value, stop at the first one that is too large
        stop = stop or pv_ > th_
        okk.append(not stop)
    fh = go.Figure()
    fh.add_trace(go.Scatter(x=[0.5, mm + 0.5], y=[0.05, 0.05], mode="lines", line=dict(color=GREY, width=1.5, dash="dot"), hoverinfo="skip", showlegend=False))
    fh.add_trace(go.Scatter(x=[r - 0.5 for r in ranks] + [mm + 0.5], y=thr + [thr[-1]], mode="lines", line=dict(color=NEUTRAL, width=2, shape="hv"), hoverinfo="skip", showlegend=False))
    fh.add_trace(go.Scatter(x=ranks, y=[max(float(v), 1e-12) for v in pws["p_value"]], mode="markers", showlegend=False,
                            marker=dict(size=12, color=[REAL if o else LUCK for o in okk], line=dict(color="white", width=1.2)),
                            text=[f"{ml_data.model_label(a)} against {ml_data.model_label(b)}: p = {v:.2g}" for a, b, v in zip(pws["a"], pws["b"], pws["p_value"])], hoverinfo="text"))
    fh.add_annotation(x=mm + 0.4, y=0.05, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=11, color=MUTED), text="Without correction: 0.05")
    fh.add_annotation(x=0.6, y=thr[0], xanchor="left", yanchor="top", yshift=-4, showarrow=False, font=dict(size=11, color=NEUTRAL), text="Allowed after correction")
    fh.add_annotation(x=0.5, y=1.0, xref="x", yref="paper", xanchor="left", yanchor="bottom", showarrow=False, align="left", font=dict(size=12, color=theme.INK_SECONDARY),
                      text=f"<span style='color:{REAL}'>●</span> Real after the correction &nbsp;&nbsp; <span style='color:{LUCK}'>●</span> Could be luck")
    theme.style_fig(fh, height=320, xaxis_title="Pairs, from the smallest p-value", yaxis_title="p-value", legend=False)
    fh.update_yaxes(type="log", range=[-12.3, 0.1], tickvals=[1, 1e-3, 1e-6, 1e-9, 1e-12], ticktext=["1", "0.001", "1e-6", "1e-9", "1e-12"])
    fh.update_xaxes(tickvals=ranks, range=[0.4, mm + 0.6])
    fh.update_layout(margin=dict(t=34))
    st.plotly_chart(fh, width="stretch", config={"displayModeBar": False})
    _line("A green dot sits below the step Holm allows for its rank. A grey dot could be luck. Dots are floored at 1e-12 for display.")
    _pa_, _pb_ = globals().get("A"), globals().get("B")
    _from_pick = _pa_ in models and _pb_ in models and _pa_ != _pb_
    if _from_pick:
        _pick = pws[((pws["a"] == _pa_) & (pws["b"] == _pb_)) | ((pws["a"] == _pb_) & (pws["b"] == _pa_))]
    else:
        _pick = pws.iloc[:1]
    if len(_pick):
        _ri = int(_pick.index[0])
        _r_, _pv, _th = _ri + 1, float(pws["p_value"].iloc[_ri]), thr[_ri]
        _na, _nb2 = ml_data.model_label(pws["a"].iloc[_ri]), ml_data.model_label(pws["b"].iloc[_ri])
        _own = _pv <= _th
        _good = okk[_ri]
        _verdict = "Real after the correction" if _good else ("Could be luck: a smaller p-value failed before it, so Holm stops" if _own else "Could be luck")
        _mini("Calculation for one pair", f"{_na} against {_nb2}. " + ("The pair follows the choice in step ⑤." if _from_pick else "The pair with the smallest p-value; pick two algorithms in step ⑤ to change it."))
        st.markdown(_panel(
            f"<div style='display:grid; grid-template-columns:7.5rem 1fr; row-gap:0.4rem; align-items:baseline;'>"
            f"<span style='font-weight:600; color:{INK};'>① Rank</span><span>The p-value is number <b style='color:{INK};'>{_r_}</b> of {mm}, smallest first.</span>"
            f"<span style='font-weight:600; color:{INK};'>② Limit</span><span>0.05 ÷ ({mm} − {_r_} + 1) = 0.05 ÷ {mm - _r_ + 1} = <b style='color:{INK};'>{_th:.4g}</b></span>"
            f"<span style='font-weight:600; color:{INK};'>③ Compare</span><span>p = <b style='color:{INK};'>{_pv:.2g}</b> {'≤' if _own else '>'} {_th:.4g}, so this pair {'passes' if _own else 'fails'} its limit. After the correction: p = {float(_pick['p_holm'].iloc[0]):.2g}.</span></div>"
            f"<div style='margin-top:0.6rem; font-size:1.15rem; font-weight:800; color:{REAL if _good else LUCK};'>{_verdict}</div>"), unsafe_allow_html=True)

# ============================================================================================ 6. where each algorithm wins
_section("Where each algorithm wins", num="⑧")
by = st.radio("Group by", ["hour", "month", "regime"], horizontal=True, key="ml7_by", format_func=lambda x: {"hour": "Hour of day", "month": "Month of the year", "regime": "Market regime"}[x], label_visibility="collapsed")
t = tabs[f"by_{by}"]
if t.empty:
    ml_ui.not_yet_run("the breakdown")
else:
    mcols = ml_data.ordered_models(c for c in t.columns if c in set(board["model"]))
    _mini(f"Average error by {by}", f"MAE in {units}")
    fig = go.Figure()
    for m in mcols:
        if by == "hour":
            fig.add_trace(go.Scatter(x=t["group"], y=t[m], mode="lines", name=ml_data.model_label(m), line=dict(color=_alg(m), width=2)))
        else:
            fig.add_trace(go.Bar(x=t["group"].astype(str), y=t[m], name=ml_data.model_label(m), marker_color=_alg(m)))
    theme.style_fig(fig, height=320, yaxis_title=f"MAE ({units})", barmode="group")
    _legend_top(fig)
    fig.update_layout(margin=dict(t=34, l=62, r=12))
    if by == "hour":
        fig.update_xaxes(range=[float(t["group"].min()) - 0.5, float(t["group"].max()) + 0.5])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    _mini("Lowest average error in each slot", "The colour of the cell is the algorithm with the lowest MAE in that slot")
    strip = go.Figure()
    for m in mcols:
        hit = t[t["lowest_mae"] == m]
        if len(hit):
            strip.add_trace(go.Bar(x=hit["group"] if by == "hour" else hit["group"].astype(str), y=[1] * len(hit), name=ml_data.model_label(m), marker_color=_alg(m),
                                   hovertemplate=f"{ml_data.model_label(m)}: lowest average error<extra></extra>", showlegend=True))
    theme.style_fig(strip, height=110, barmode="stack")
    strip.update_traces(marker_line_width=2, marker_line_color="white")
    strip.update_yaxes(visible=False, fixedrange=True, range=[0, 1])
    strip.update_xaxes(showticklabels=False)
    if by == "hour":
        strip.update_xaxes(range=[float(t["group"].min()) - 0.5, float(t["group"].max()) + 0.5])
    strip.update_layout(bargap=0, margin=dict(t=34, b=4, l=62, r=12))
    _legend_top(strip)
    st.plotly_chart(strip, width="stretch", config={"displayModeBar": False})
    with st.expander("Lowest average error and most frequent winner per slot"):
        _table([by.capitalize(), "Units", "Lowest average error", "Wins most units", "Share of units won"],
               [[str(g), f"{int(u):,}", f"{_dot(lm)}{ml_data.model_label(lm)}", f"{_dot(w)}{ml_data.model_label(w)}", f"{100 * float(sh):.1f}%" if float(sh) <= 1 else f"{float(sh):.1f}%"]
                for g, u, lm, w, sh in zip(t["group"], t["units"], t["lowest_mae"], t["winner"], t["winner_share"])], first="minmax(6rem, 1fr)")

# ============================================================================================ 7. error through time
_section("Error through time", "The errors of all algorithms rise and fall together, which is why the tests compare them on paired daily losses", num="⑨")
if daily.empty:
    ml_ui.not_yet_run("the daily losses (data pack)")
else:
    d = daily[daily["target"] == target]
    opts = ml_data.ordered_models(m for m in d["model"].unique() if m not in ("champion", "reference"))
    _pc, _sc = st.columns([3, 2], gap="large")
    with _pc:
        pick = st.multiselect("Algorithms", opts, default=[o for o in ["lightgbm", "decision_tree"] if o in opts] or opts[:2], format_func=ml_data.model_label, key="ml7_pick")
    with _sc:
        win_d = st.slider("Smoothing (days)", 1, 60, 28, key="ml7_smooth")
    fig = go.Figure()
    for m in ml_data.ordered_models(pick):
        x = d[d["model"] == m].sort_values("date")
        fig.add_trace(go.Scatter(x=x["date"], y=x["mae"].rolling(win_d, min_periods=1).mean(), mode="lines", name=ml_data.model_label(m), line=dict(color=_alg(m), width=1.8)))
    theme.style_fig(fig, height=320, yaxis_title=f"Daily MAE, smoothed ({units})")
    _legend_top(fig)
    fig.update_layout(margin=dict(t=34))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ============================================================================================ 8. the outcome
_section("The outcome", num="⑩")
_vb = tabs["versus_best"]
_ex = lambda m_, v_: float(piv.loc[m_].mean()) if (piv is not None and m_ in piv.index) else float(v_)               # the mean daily error from the daily table, not from the 4-decimal file
_tied = []
if not _vb.empty:
    _tied = [ml_data.model_label(m_) for m_, d_ in zip(_vb["model"], _vb["distinguishable_from_best"]) if not d_]
_sig = f"{int(pw['significant'].sum())} of {len(pw)}" if not pw.empty else "–"
_lc, _rc2 = st.columns([3, 2], gap="medium")
with _lc:
    if not _vb.empty:
        _best_m = _vb["best"].iloc[0]
        _best_l = _ex(_best_m, _vb["best_loss"].iloc[0])
        lad = [(_best_m, _best_l, "best", None)] + [(m_, _ex(m_, l_), "tied" if not d_ else "worse", float(ph_)) for m_, l_, d_, ph_ in zip(_vb["model"], _vb["mean_loss"], _vb["distinguishable_from_best"], _vb["p_holm"])]
        lad.sort(key=lambda r_: r_[1])
        fl = go.Figure()
        k_tie = sum(1 for r_ in lad if r_[2] != "worse")
        if k_tie > 1:
            fl.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=-0.5, y1=k_tie - 0.5, fillcolor="rgba(154,152,144,0.14)", line_width=0, layer="below")
        fl.add_shape(type="line", x0=_best_l, x1=_best_l, y0=-0.5, y1=len(lad) - 0.5, line=dict(color=MUTED, width=1, dash="dot"))
        fl.add_trace(go.Scatter(x=[r_[1] for r_ in lad], y=[ml_data.model_label(r_[0]) for r_ in lad], mode="markers+text", showlegend=False,
                                marker=dict(size=15, color=[_alg(r_[0]) for r_ in lad], line=dict(color="white", width=1.5)),
                                text=[f"{r_[1]:.3f}" + ("" if r_[3] is None else f"   {'Cannot be told apart' if r_[2] == 'tied' else 'Significantly worse'} (p {r_[3]:.2g})") for r_ in lad],
                                textposition="middle right", textfont=dict(size=11, color=theme.INK_SECONDARY), cliponaxis=False, hoverinfo="skip"))
        theme.style_fig(fl, height=int(46 * len(lad)) + 70, xaxis_title=f"Mean daily error ({units})", legend=False)
        fl.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, fixedrange=True)
        fl.update_xaxes(range=[_best_l - 0.6, max(r_[1] for r_ in lad) + 3.4], showgrid=False, zeroline=False)
        fl.update_layout(margin=dict(t=6, b=44, l=120, r=10))
        st.plotly_chart(fl, width="stretch", config={"displayModeBar": False})
with _rc2:
    st.markdown(_panel(
        f"<div style='font-size:1.7rem; font-weight:800; color:{_alg(best['model'])}; line-height:1.2;'>{ml_data.model_label(best['model'])}</div>"
        f"<div style='line-height:1.6; font-size:0.9rem; margin-top:0.4rem;'>"
        f"<div><b>Lowest MAE:</b> {best['mae']:.3f} {units} over {n_days_all:,} days.</div>"
        f"<div><b>Pairs that differ after the Holm correction (5%):</b> {_sig}.</div>"
        f"<div><b>Cannot be told apart from it:</b> {', '.join(_tied) if _tied else 'none, every other algorithm is significantly worse'}.</div></div>"), unsafe_allow_html=True)
if not _vb.empty:
    _mini("Each algorithm against the best", "The Holm correction is applied over these comparisons")
    _rows = []
    for _, r_ in _vb.iterrows():
        worse = bool(r_["distinguishable_from_best"])
        _rows.append([f"{_dot(r_['model'])}{ml_data.model_label(r_['model'])}", f"{_ex(r_['model'], r_['mean_loss']):.3f}", f"{_ex(r_['best'], r_['best_loss']):.3f}", f"{round(_ex(r_['model'], r_['mean_loss']), 3) - round(_ex(r_['best'], r_['best_loss']), 3):+.3f}", f"{float(r_['statistic']):.2f}",
                      f"{float(r_['p_value']):.2g}", f"{float(r_['p_holm']):.2g}",
                      _pill("Real: worse than the best", "#006100", "#e3f1e3") if worse else _pill("Cannot be told apart", "#5a5953", "#ecebe6")])
    _table(["Algorithm", "Mean daily error", "Error of the best", "Difference", "DM statistic", "p-value", "p-value (Holm)", "Verdict"], _rows)
    _line("The mean daily error averages the day scores. The MAE of the scoreboard averages every hour, so the two differ in the third decimal.")
_line("The champion is chosen on the next page, by a rule written before the results existed.")
st.caption("The plant inputs (solar irradiance, temperature, natural inflow) are compared on their own page, against simple forecasts and, for irradiance and temperature, the weather forecast.")
