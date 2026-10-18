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
from maths import tex as M

ml_ui.header("Model arena", "", step=6)
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
        kk = int((lastd["a"] - lastd["b"] - (lastd["a"] - lastd["b"]).mean()).abs().idxmin())               # a typical day of the last three weeks
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
        _fnote(865, y3 + 2.5, f"<b>Gap = {va:.1f} - {vb:.1f} = {va - vb:+.1f}</b>", size=12)
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
st.markdown(f"<style>[data-testid='stVerticalBlockBorderWrapper']:has(.ml7-board) {{ background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,0.04); padding:0.4rem 0.7rem 0.2rem; }}</style>", unsafe_allow_html=True)
_hbar = lambda x_, txt_: (                                                                                          # noqa: E731
    go.Figure(go.Bar(y=_names, x=x_, orientation="h", marker_color=[_alg(m) for m in models], text=txt_, textposition="outside", cliponaxis=False,
                     textfont=dict(size=11, color=theme.INK_SECONDARY), hoverinfo="skip")))
with st.container(border=True):
    st.markdown("<span class='ml7-board'></span>", unsafe_allow_html=True)
    _c_mae, _c_days = st.columns([1.3, 1], gap="small")                                                                  # side by side: names on the vertical axis, so they never collide
    with _c_mae:
        _mini("Average error (MAE)", "Gap to the best, beside each bar")
        _m3 = [round(float(v), 3) for v in board["mae"]]
        f_s = _hbar(_m3, [f"<b>{v:.3f}</b>" if i == 0 else f"{v:.3f}  <span style='color:{MUTED};'>+{v - _m3[0]:.3f}</span>" for i, v in enumerate(_m3)])
        theme.style_fig(f_s, height=270, xaxis_title=f"MAE ({units})", legend=False)
        f_s.update_xaxes(range=[0, max(_m3) * 1.55], showgrid=False, dtick=5)
        f_s.update_yaxes(autorange="reversed", showgrid=False)
        f_s.update_layout(margin=dict(t=6, r=2), bargap=0.4)
        st.plotly_chart(f_s, width="stretch", config={"displayModeBar": False})
    with _c_days:
        _mini("Days with the smallest error", f"Dotted line: equal share, {100 / len(models):.0f}%")
        _sh = [100 * _cnt[m] / _nd for m in models]
        f_b = _hbar(_sh, [f"{v:.1f}%  <span style='color:{MUTED};'>{_cnt[m]} days</span>" for v, m in zip(_sh, models)])
        f_b.add_vline(x=100 / len(models), line_dash="dot", line_color="#9a9890", line_width=1)
        theme.style_fig(f_b, height=270, xaxis_title="Share of days (%)", legend=False)
        f_b.update_xaxes(range=[0, max(_sh) * 1.5], showgrid=False, dtick=10)
        f_b.update_yaxes(autorange="reversed", showgrid=False, showticklabels=False)
        f_b.update_layout(margin=dict(t=6, r=2), bargap=0.4)
        st.plotly_chart(f_b, width="stretch", config={"displayModeBar": False})
_mini("The scoreboard in numbers", f"{n_hours:,} hourly forecasts on {_nd:,} days. RMSE is the same score with squared errors, so large misses weigh more.")
_table(["Algorithm", f"MAE ({units})", f"RMSE ({units})", "Days with the smallest error", "Share of days"],
       [[f"{_dot(m)}{ml_data.model_label(m)}", f"{float(r['mae']):.3f}", f"{float(r['rmse']):.3f}", f"{_cnt[m]:,} of {_nd:,}", f"{100 * _cnt[m] / _nd:.1f}%"] for m, (_, r) in zip(models, board.iterrows())], highlight=0)
_line("The best algorithm by MAE is shaded.")

# ============================================================================================ 3-4. is the gap real
if daily.empty:
    _section("How the two compare, day by day", num="⑤")
    ml_ui.not_yet_run("the daily losses (data pack)")
    res, A, B = None, "champion", "decision_tree"
else:
    _section("How the two compare, day by day", "Pick any two to compare on the same days", num="⑤")
    _pa, _pb, _px = st.columns([2, 2, 4], gap="medium")
    _opts = ["champion"] + models
    with _pa:
        A = st.selectbox("Compare", _opts, index=0, format_func=_name, key="ml7_a")
    with _pb:
        B = st.selectbox("Against", _opts, index=_opts.index("decision_tree") if "decision_tree" in _opts else 1, format_func=_name, key="ml7_b")
    with _px:
        smooth_d = st.slider("Smoothing (days)", 1, 60, 28, key="ml7_smooth")
    _line(M(r"\text{Smoothed error}(t)=\frac{\text{error}(t)+\text{error}(t-1)+\cdots+\text{error}(t-N+1)}{N}", size=0.95) + ", N = number of days")
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
    _mini("Both models over the whole window", f"The daily error of each, smoothed over {smooth_d} days. Where {la}'s line is below {lb}'s, {la} was closer.")
    sm_a = los["a"].rolling(smooth_d, min_periods=1).mean()
    sm_b = los["b"].rolling(smooth_d, min_periods=1).mean()
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=los["date"], y=sm_a, mode="lines", name=la, line=dict(color=ca, width=1.8)))
    f2.add_trace(go.Scatter(x=los["date"], y=sm_b, mode="lines", name=lb, line=dict(color=cb, width=1.8), fill="tonexty", fillcolor="rgba(154,152,144,0.16)"))
    theme.style_fig(f2, height=320, yaxis_title=f"Daily error, smoothed ({units})")
    f2.update_layout(hovermode="x unified")
    _legend_top(f2)
    f2.update_layout(margin=dict(t=34))
    st.plotly_chart(f2, width="stretch", config={"displayModeBar": False})
    _line(f"{la} was closer on <b>{wins:,} of {n_d:,} days ({100 * wins / n_d:.1f}%)</b>.")
