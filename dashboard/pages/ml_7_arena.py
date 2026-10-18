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

BLUE, ORANGE, VIOLET, MUTED, INK = theme.BLUE, theme.ORANGE, theme.VIOLET, theme.INK_MUTED, theme.INK_PRIMARY          # blue = the champion, orange = the decision tree, on every picture
NEUTRAL, T_DONE, GREY = "#6b6a64", "#8791ab", "#b9b7ae"
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # a hint starts with a capital


def _section(title: str, hint: str = "", num: str = "") -> None:
    """A section heading: a number (the steps of the story), a bold title and, under it, one muted line on how to read the section."""
    _n = f"<span style='font-weight:400; color:{NEUTRAL};'>{num}</span>&nbsp; " if num else ""
    st.markdown(f"<div style='margin:1.7rem 0 0.45rem;'><div style='font-size:1.55rem; font-weight:700; color:{INK}; line-height:1.25;'>{_n}{title}</div>"
                + (f"<div style='font-size:0.88rem; color:{MUTED}; margin-top:0.2rem;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _mini(title: str, hint: str = "") -> None:
    """The title of a chart that shares a row with another one."""
    st.markdown(f"<div style='margin:0.2rem 0 0.3rem;'><div style='font-size:1.02rem; font-weight:700; color:{INK};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{MUTED}; margin-top:0.1rem; line-height:1.35;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _line(html: str) -> None:
    """One short note under a table or chart, the key numbers in bold."""
    st.markdown(f"<div style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin:0.25rem 0 0.9rem;'>{html}</div>", unsafe_allow_html=True)


def _panel(inner: str, min_height: str = "") -> str:
    """A white rounded card for a calculation, as on the Tuning lab."""
    return (f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.55rem 1.1rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); "
            f"font-size:0.9rem; color:{theme.INK_SECONDARY}; {min_height}'>{inner}</div>")


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
n_days_all = round(best["n"] / 24)
res = ml_data.dm_example(daily, target) if not daily.empty else None

# ============================================================================================ 1. the arena, drawn
SH, AN = [], []                                                                                  # shapes and annotations, handed to the figure once at the end
ROW, X_END = 34.0, 1010
px_x = 1150.0 / X_END


def _pill(x0, x1, y0, y1, color, opacity=1.0, r_px=6.0):
    """A rounded bar as a path; x in the units of the figure, y in rows."""
    rx, ry = min(r_px / px_x, (x1 - x0) / 2), min(r_px / ROW, (y1 - y0) / 2)
    SH.append(dict(type="path", fillcolor=color, opacity=opacity, line_width=0,
                   path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                         f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")))


def _card(y0, y1, r_px=10.0):
    rx, ry = r_px / px_x, r_px / ROW
    x0, x1 = -12, X_END + 12
    SH.append(dict(type="path", fillcolor="#ffffff", line=dict(color=CARD_LINE, width=1), layer="below",
                   path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                         f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")))


def _head(y, num, title, sub_=""):
    AN.append(dict(x=6, y=y, xanchor="left", yanchor="middle", showarrow=False, align="left", font=dict(size=14, color=INK),
                   text=f"<b style='font-size:17px;color:{NEUTRAL}'>{num}</b>&nbsp; <b>{title}</b>" + (f"<br><span style='font-size:12px;color:{MUTED}'>{_cap(sub_)}</span>" if sub_ else "")))


def _note(x, y, text, size=12, color=INK, anchor="center", **kw):
    AN.append(dict(x=x, y=y, xanchor=anchor, showarrow=False, text=text, font=dict(size=size, color=color), **kw))


def _arrow(x0, y0, x1, y1, color=MUTED):
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
    _card(-0.6, 6.75)
    _head(0.4, "①", "The same days for every algorithm", f"{ml_data.label(target)}, development window: {n_d:,} delivery days from {pd.Timestamp(dates[0]):%d-%m-%Y} to {pd.Timestamp(dates[-1]):%d-%m-%Y}")
    pitch, y_first, xs0, xs1 = 0.95, 1.7, 190.0, 1000.0
    ys = [y_first + i * pitch for i in range(len(models))]
    xs = np.linspace(xs0, xs1, n_d)
    z = piv.to_numpy(dtype=float)
    lo, hi = float(np.nanquantile(z, 0.05)), float(np.nanquantile(z, 0.95))
    fig_a.add_trace(go.Heatmap(x=xs, y=ys, z=np.clip(z, lo, hi), zmin=lo, zmax=hi, showscale=False, ygap=5, colorscale=[[0, "#3d4d78"], [0.5, T_DONE], [1, "#e9ecf3"]],
                               text=[[f"{ml_data.model_label(m)}, {pd.Timestamp(dt):%d-%m-%Y}: {v:.1f}" for dt, v in zip(dates, row)] for m, row in zip(models, z)],
                               hoverinfo="text", hoverlabel=dict(bgcolor="white", font=dict(size=12, color=INK), bordercolor=CARD_LINE)))
    for m, y in zip(models, ys):
        _note(8, y, f"<b>{ml_data.model_label(m)}</b>", anchor="left")
    _note(xs0, ys[-1] + 0.72, f"{pd.Timestamp(dates[0]):%d-%m-%Y}", size=11, color=MUTED, anchor="left")
    _note(xs1, ys[-1] + 0.72, f"{pd.Timestamp(dates[-1]):%d-%m-%Y}", size=11, color=MUTED, anchor="right")
    _note((xs0 + xs1) / 2, ys[-1] + 0.72, "One column is one delivery day, the same day in every row. Darker = smaller error that day.", size=11, color=MUTED)
    # ---- card 2: one score per day
    y2 = 7.1
    _card(y2, y2 + 4.2)
    _head(y2 + 0.8, "②", "One score per day", "A forecast error is the gap to the real price, in absolute value (MAE)")
    cy = y2 + 2.75
    x = 8.0
    for h in range(24):
        _pill(x, x + 15, cy - 0.4, cy + 0.4, GREY, 0.9, r_px=2)
        x += 18
    _note(8, cy - 0.95, "24 hourly errors of one delivery day", size=11, color=MUTED, anchor="left")
    _note(8, cy + 0.85, "Hour 1", size=10, color=MUTED, anchor="left")
    _note(8 + 24 * 18 - 3, cy + 0.85, "Hour 24", size=10, color=MUTED, anchor="right")
    ex = piv.loc[models[0]]
    ex_day = ex.index[int((ex - ex.median()).abs().to_numpy().argmin())]                    # a typical day of the best algorithm
    _arrow(448, cy, 497, cy)
    _pill(500, 700, cy - 0.5, cy + 0.5, NEUTRAL, 0.12, r_px=8)
    _note(600, cy, "<b>Day score</b>: mean of the 24", size=12)
    _note(600, cy + 0.85, f"{float(ex[ex_day]):.1f} on {pd.Timestamp(ex_day):%d-%m-%Y}, {ml_data.model_label(models[0])}", size=10, color=MUTED)
    _arrow(703, cy, 752, cy)
    _pill(755, 1000, cy - 0.5, cy + 0.5, NEUTRAL, 0.12, r_px=8)
    _note(877, cy, f"<b>MAE</b>: mean of the {n_d:,} day scores", size=12)
    _note(877, cy + 0.85, f"{float(best['mae']):.2f} {units}, {ml_data.model_label(models[0])}", size=10, color=MUTED)
    end = y2 + 4.2
    # ---- card 3: one gap per day for a pair
    if res is not None:
        y3 = end + 0.35
        _card(y3, y3 + 4.4)
        _head(y3 + 0.8, "③", "One gap per day for a pair", "The two models are scored on the same day; their difference is that day's gap")
        los = res["losses"]
        lastd = los.tail(21).reset_index(drop=True)
        kk = int((lastd["a"] - lastd["b"] - (lastd["a"] - lastd["b"]).mean()).abs().idxmin())               # a typical day, as on the picture of step 4
        va, vb = float(lastd["a"][kk]), float(lastd["b"][kk])
        cya, cyb = y3 + 2.55, y3 + 3.55
        scale = 520.0 / max(va, vb)
        xa, xb = 190 + va * scale, 190 + vb * scale
        _note(8, cya, "<b>Champion</b>", color=BLUE, anchor="left")
        _note(8, cyb, "<b>Decision tree</b>", color=ORANGE, anchor="left")
        _pill(190, xa, cya - 0.32, cya + 0.32, BLUE, 0.9, r_px=3)
        _pill(190, xb, cyb - 0.32, cyb + 0.32, ORANGE, 0.9, r_px=3)
        _note(xa + 8, cya, f"{va:.1f}", size=11, color=BLUE, anchor="left")
        _note(xb + 8, cyb, f"{vb:.1f}", size=11, color=ORANGE, anchor="left")
        _note(190, y3 + 1.75, f"Day score on {pd.Timestamp(lastd['date'][kk]):%d-%m-%Y}", size=11, color=MUTED, anchor="left")
        for xx in (xa, xb):
            SH.append(dict(type="line", x0=xx, x1=xx, y0=cya - 0.45, y1=cyb + 0.5, line=dict(color="#a8a69d", width=1.2, dash="dot")))
        _pill(790, 1000, y3 + 2.15, y3 + 3.95, NEUTRAL, 0.12, r_px=8)
        _note(895, y3 + 2.65, f"<b>Gap = {va:.1f} − {vb:.1f} = {va - vb:+.1f}</b>", size=13)
        _note(895, y3 + 3.4, f"{'Champion' if va < vb else 'Decision tree'} closer. One gap per day, {res['n_days']:,} days", size=10, color=MUTED)
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
_l, _r = st.columns(2, gap="large")
_names = [ml_data.model_label(m) for m in models]
with _l:
    _mini("Average error (MAE)")
    f_s = go.Figure(go.Bar(x=_names, y=board["mae"], marker_color=[BLUE if i == 0 else GREY for i in range(len(board))], text=[f"{v:.2f}" for v in board["mae"]],
                           textposition="outside", cliponaxis=False, textfont=dict(size=12, color=theme.INK_SECONDARY), hoverinfo="skip"))
    theme.style_fig(f_s, height=300, yaxis_title=f"MAE ({units})", legend=False)
    f_s.update_xaxes(tickangle=0)
    f_s.update_yaxes(range=[0, float(board["mae"].max()) * 1.18])
    f_s.update_layout(margin=dict(t=24))
    st.plotly_chart(f_s, width="stretch", config={"displayModeBar": False})
with _r:
    _mini("Days on which it had the smallest error")
    f_b = go.Figure(go.Bar(x=_names, y=100 * board["days_best"], marker_color=[VIOLET if i == int(board["days_best"].idxmax()) else GREY for i in range(len(board))],
                           text=[f"{100 * v:.0f}%" for v in board["days_best"]], textposition="outside", cliponaxis=False, textfont=dict(size=12, color=theme.INK_SECONDARY), hoverinfo="skip"))
    theme.style_fig(f_b, height=300, yaxis_title="Share of days (%)", legend=False)
    f_b.update_xaxes(tickangle=0)
    f_b.update_yaxes(range=[0, float(100 * board["days_best"].max()) * 1.18])
    f_b.update_layout(margin=dict(t=24))
    st.plotly_chart(f_b, width="stretch", config={"displayModeBar": False})
_line(f"Tuned algorithms on the development window, {n_days_all:,} days. The RMSE of each algorithm, the same score with squared errors, is "
      + ", ".join(f"{ml_data.model_label(m)} {v:.2f}" for m, v in zip(board["model"], board["rmse"])) + ".")

# ============================================================================================ 3-4. is the gap real
if daily.empty:
    _section("Is the gap real?", num="⑤")
    ml_ui.not_yet_run("the daily losses (data pack)")
elif res is None:
    _section("Is the gap real?", num="⑤")
    st.info("Not enough days for this target.")
else:
    diff, los = res["differences"].copy(), res["losses"].copy()
    diff["date"] = diff["date"].astype(str)
    mean, stat, p, n_d = float(res["mean_diff"]), float(res["statistic"]), float(res["p_value"]), int(res["n_days"])
    wins = int((diff["difference"] < 0).sum())
    _section("Is the champion's edge real or luck? Day by day", "The champion is compared with the decision tree on the same days", num="⑤")
    _l, _r = st.columns(2, gap="large")
    with _l:
        _mini("Every day, both are scored", "Each bar is the average error of one delivery day. The shorter bar was closer. The last 21 days are shown.")
        last = los.tail(21).reset_index(drop=True)
        lab = [f"{pd.Timestamp(x):%d-%m}" for x in last["date"]]
        f1 = go.Figure()
        f1.add_trace(go.Bar(x=lab, y=last["a"], name="Champion", marker_color=BLUE))
        f1.add_trace(go.Bar(x=lab, y=last["b"], name="Decision tree", marker_color=ORANGE))
        k = int((last["a"] - last["b"] - (last["a"] - last["b"]).mean()).abs().idxmin())
        f1.add_annotation(x=lab[k], y=max(last["a"][k], last["b"][k]), ax=0, ay=-48, showarrow=True, arrowhead=3, arrowwidth=1, arrowcolor=INK, font=dict(size=12, color=INK),
                          text=f"<b>This day</b><br>Champion {last['a'][k]:.1f}, tree {last['b'][k]:.1f}<br>Gap = {last['a'][k] - last['b'][k]:+.1f}", align="center",
                          bgcolor="white", bordercolor=INK, borderwidth=1, borderpad=4, yanchor="bottom")
        theme.style_fig(f1, height=320, yaxis_title="Average error of the day", barmode="group")
        f1.update_xaxes(type="category", tickangle=-90, tickfont=dict(size=10, color=MUTED))
        f1.update_yaxes(range=[0, float(max(last["a"].max(), last["b"].max())) * 1.6])
        st.plotly_chart(f1, width="stretch", config={"displayModeBar": False})
    with _r:
        _mini("One gap per day, over the whole window", "The gap of a day is the champion's error minus the tree's. The line is the running average.")
        neg, pos = diff["difference"].clip(upper=0), diff["difference"].clip(lower=0)
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=diff["date"], y=neg, name="Champion closer", marker_color=BLUE))
        f2.add_trace(go.Bar(x=diff["date"], y=pos, name="Tree closer", marker_color=ORANGE))
        f2.add_trace(go.Scatter(x=diff["date"], y=diff["difference"].expanding().mean(), mode="lines", line=dict(color=INK, width=2), name="Running average"))
        f2.add_hline(y=0, line_color=MUTED)
        theme.style_fig(f2, height=320, yaxis_title="Champion − tree (error of the day)", barmode="relative")
        st.plotly_chart(f2, width="stretch", config={"displayModeBar": False})
    _line(f"The champion was closer on <b>{wins:,} of {n_d:,} days ({wins / n_d:.0%})</b>. Closer on most days is not yet proof: the days are not independent, and luck can produce a lead.")

    _section("The average gap against luck, and the verdict", num="⑥")
    se = abs(mean / stat) if stat else float(diff["difference"].std() / n_d ** 0.5)
    band = 1.96 * se
    m = max(abs(mean), band) * 1.3
    c_obs = BLUE if mean < 0 else ORANGE
    _l, _r = st.columns([5, 3], gap="medium")
    with _l:
        f3 = go.Figure()
        f3.add_shape(type="rect", x0=-band, x1=band, y0=0.5, y1=1.0, fillcolor="#c4c2b9", opacity=0.45, line_width=0)
        f3.add_shape(type="line", x0=0, x1=0, y0=0.42, y1=1.08, line=dict(color=MUTED, width=1.5, dash="dot"))
        f3.add_annotation(x=0, y=1.1, yanchor="bottom", showarrow=False, font=dict(size=12, color=MUTED), align="center",
                          text=f"<b>The luck zone</b> (±{band:.2f})<br>If the two models were equal, the average would land here")
        f3.add_annotation(x=mean, y=0.75, ax=0, ay=0.75, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=c_obs, text="")
        f3.add_annotation(x=mean / 2, y=0.75, yshift=15, showarrow=False, font=dict(size=13, color=c_obs), text=f"<b>{abs(stat):.1f} luck-widths</b>")
        f3.add_trace(go.Scatter(x=[mean], y=[0.75], mode="markers", marker=dict(symbol="diamond", size=17, color=c_obs, line=dict(color="white", width=1.5)), showlegend=False,
                                hovertemplate=f"Observed average gap {mean:+.3f}<extra></extra>"))
        f3.add_annotation(x=mean, y=0.75, yshift=-30, showarrow=False, font=dict(size=12, color=c_obs), align="center",
                          text=f"<b>Observed average gap {mean:+.2f}</b><br>{'Champion' if mean < 0 else 'Tree'} lower by {abs(mean):.2f}")
        kmax = int(abs(stat))
        step = 1 if kmax <= 12 else 5
        ks = list(range(step, kmax + 1, step))
        sgn = -1 if mean < 0 else 1
        if ks:
            f3.add_trace(go.Scatter(x=[sgn * k_ * se for k_ in ks], y=[0.22] * len(ks), mode="markers+text", text=[str(k_) for k_ in ks], textposition="top center",
                                    textfont=dict(size=10, color=MUTED), marker=dict(symbol="line-ns", size=11, color=MUTED, line=dict(width=1.5, color=MUTED)),
                                    showlegend=False, hoverinfo="skip"))
        f3.add_annotation(x=m, y=0.06, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=11, color=MUTED), align="right",
                          text=f"One luck-width = {se:.2f}: how much the average gap moves by chance<br>(allowing for hours and neighbouring days that move together)")
        theme.style_fig(f3, height=270, xaxis_title="Champion − tree, average over all days", legend=False)
        f3.update_xaxes(range=[-m, m], zeroline=False, showgrid=False)
        f3.update_yaxes(range=[0, 1.55], visible=False, fixedrange=True)
        f3.layout.shapes = tuple(sh for sh in f3.layout.shapes if not (sh.line.color == theme.AXIS_LINE and sh.x0 == sh.x1 and sh.y0 != 0.42))
        st.plotly_chart(f3, width="stretch", config={"displayModeBar": False})
    with _r:
        real = p < 0.05
        badge_col = theme.GREEN if real else MUTED
        _row = lambda name_, val_: (f"<div style='display:flex; justify-content:space-between; align-items:baseline; padding:0.35rem 0; border-bottom:1px solid #eeede8;'>"      # noqa: E731
                                    f"<span>{name_}</span><b style='font-size:1.02rem; color:{INK};'>{val_}</b></div>")
        st.markdown(_panel(
            f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; padding:0.2rem 0;'>The verdict in four numbers</div>"
            + _row("Days compared", f"{n_d:,}") + _row("Average gap", f"{mean:+.3f}") + _row("DM statistic (gap in luck-widths)", f"{stat:.2f}") + _row("p-value", f"{p:.2g}")
            + f"<div style='margin-top:0.7rem; font-size:1.25rem; font-weight:800; color:{badge_col};'>{'Real, not luck' if real else 'Not distinguishable from luck'}</div>"
            + f"<div style='margin-top:0.2rem; font-size:0.84rem; line-height:1.4;'>"
            + (f"If the models were truly equal, a gap this big would appear with probability {p:.2g}." if real else
               f"Equal models would show a gap this big with probability {p:.2g}; the average is only {abs(stat):.1f} luck-widths from zero.") + "</div>"), unsafe_allow_html=True)
    _line("Beyond about 2 luck-widths, the edge of the luck zone, a gap is called real. The DM statistic is that distance in one number.")

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
    txt = [[("" if i == j or np.isnan(stat_m.iloc[i, j]) else f"{stat_m.iloc[i, j]:.1f}{star.iloc[i, j]}") for j in range(len(models))] for i in range(len(models))]
    _l, _r = st.columns(2, gap="large")
    with _l:
        _mini("The test statistic of every pair", "Row against column. Blue: the row algorithm has the smaller error. A star: still real after the correction.")
        fig = go.Figure(go.Heatmap(z=stat_m.values, x=_names, y=_names, colorscale="RdBu", zmin=-lim, zmax=lim, text=txt, texttemplate="%{text}", colorbar=dict(title="DM", thickness=12),
                                   hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        theme.style_fig(fig, height=360, legend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with _r:
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
        fh.add_trace(go.Scatter(x=[r - 0.5 for r in ranks] + [mm + 0.5], y=thr + [thr[-1]], mode="lines", line=dict(color=VIOLET, width=2, shape="hv"), hoverinfo="skip", showlegend=False))
        fh.add_trace(go.Scatter(x=ranks, y=[max(float(v), 1e-12) for v in pws["p_value"]], mode="markers", showlegend=False,
                                marker=dict(size=11, color=[theme.GREEN if o else ORANGE for o in okk], line=dict(color="white", width=1.2)),
                                text=[f"{ml_data.model_label(a)} against {ml_data.model_label(b)}: p = {v:.2g}" for a, b, v in zip(pws["a"], pws["b"], pws["p_value"])], hoverinfo="text"))
        fh.add_annotation(x=mm + 0.4, y=0.05, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=10, color=MUTED), text="Without correction: 0.05")
        fh.add_annotation(x=0.6, y=thr[0], xanchor="left", yanchor="top", yshift=-4, showarrow=False, font=dict(size=10, color=VIOLET), text="Allowed after correction")
        theme.style_fig(fh, height=360, xaxis_title="Pairs, from the smallest p-value", yaxis_title="p-value", legend=False)
        fh.update_yaxes(type="log", range=[-12.3, 0.1], tickvals=[1, 1e-3, 1e-6, 1e-9, 1e-12], ticktext=["1", "0.001", "1e-6", "1e-9", "1e-12"])
        fh.update_xaxes(tickvals=ranks, range=[0.4, mm + 0.6])
        st.plotly_chart(fh, width="stretch", config={"displayModeBar": False})
    _line("A green dot is real after the correction: it sits below the violet step, the limit Holm allows for its rank. Orange dots could be luck. "
          "Dots are floored at 1e-12 for display.")
    vb = tabs["versus_best"]
    if not vb.empty:
        tied = [m_ for m_, d_ in zip(vb["model"], vb["distinguishable_from_best"]) if not d_]
        st.markdown(f"**Best by mean loss:** {ml_data.model_label(vb['best'].iloc[0])}. **Cannot be told apart from it:** "
                    + (", ".join(ml_data.model_label(m_) for m_ in tied) if tied else "none, every other algorithm is significantly worse."))
        with st.expander("Each algorithm against the best (Holm over these comparisons)"):
            t2 = vb[["model", "mean_loss", "best_loss", "mean_diff", "statistic", "p_value", "p_holm", "distinguishable_from_best"]].copy()
            t2["model"] = t2["model"].map(ml_data.model_label)
            ml_ui.table(t2.rename(columns={"model": "Algorithm", "mean_loss": "Mean loss", "best_loss": "Loss of the best", "mean_diff": "Difference", "statistic": "Statistic",
                                            "p_value": "p-value", "p_holm": "p-value (Holm)", "distinguishable_from_best": "Significantly worse"}).round(4))

# ============================================================================================ 6. where each algorithm wins
_section("Where each algorithm wins", num="⑧")
by = st.radio("Group by", ["hour", "month", "regime"], horizontal=True, key="ml7_by", format_func=lambda x: {"hour": "Hour of day", "month": "Month of the year", "regime": "Market regime"}[x], label_visibility="collapsed")
t = tabs[f"by_{by}"]
if t.empty:
    ml_ui.not_yet_run("the breakdown")
else:
    mcols = ml_data.ordered_models(c for c in t.columns if c in set(board["model"]))
    fig = go.Figure()
    for m in mcols:
        color = theme.CATEGORICAL[ml_data.model_index(m) % 8]
        if by == "hour":
            fig.add_trace(go.Scatter(x=t["group"], y=t[m], mode="lines", name=ml_data.model_label(m), line=dict(color=color, width=1.8)))
        else:
            fig.add_trace(go.Bar(x=t["group"].astype(str), y=t[m], name=ml_data.model_label(m), marker_color=color))
    st.plotly_chart(theme.style_fig(fig, height=340, yaxis_title="MAE", barmode="group"), width="stretch", config={"displayModeBar": False})
    win = t[["group", "units", "lowest_mae", "winner", "winner_share"]].copy()
    win["lowest_mae"] = win["lowest_mae"].map(ml_data.model_label)
    win["winner"] = win["winner"].map(ml_data.model_label)
    with st.expander("Lowest average error and most frequent winner per group"):
        ml_ui.table(win.rename(columns={"group": by.capitalize(), "units": "Units", "lowest_mae": "Lowest MAE", "winner": "Wins most units", "winner_share": "Share of units won"}).round(3))

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
        fig.add_trace(go.Scatter(x=x["date"], y=x["mae"].rolling(win_d, min_periods=1).mean(), mode="lines", name=ml_data.model_label(m),
                                 line=dict(color=theme.CATEGORICAL[ml_data.model_index(m) % 8], width=1.6)))
    st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title="Daily MAE, smoothed"), width="stretch", config={"displayModeBar": False})

# ============================================================================================ 8. the outcome
_section("The outcome", num="⑩")
_tied = []
_vb = tabs["versus_best"]
if not _vb.empty:
    _tied = [ml_data.model_label(m_) for m_, d_ in zip(_vb["model"], _vb["distinguishable_from_best"]) if not d_]
_sig = f"{int(pw['significant'].sum())} of {len(pw)}" if not pw.empty else "–"
st.markdown(
    f"<div style='display:flex; align-items:center; gap:1.3rem; background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.8rem 1.2rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
    f"<div style='font-size:1.7rem; font-weight:800; color:{INK}; line-height:1.2;'>{ml_data.model_label(best['model'])}</div>"
    f"<div style='line-height:1.55; font-size:0.9rem; color:{theme.INK_SECONDARY};'>"
    f"<div><b>Lowest MAE:</b> {best['mae']:.3f} {units} over {n_days_all:,} days.</div>"
    f"<div><b>Pairs that differ after the Holm correction (5%):</b> {_sig}.</div>"
    f"<div><b>Cannot be told apart from it:</b> {', '.join(_tied) if _tied else 'none, every other algorithm is significantly worse'}.</div></div></div>", unsafe_allow_html=True)
_line("The champion is chosen on the next page, by a rule written before the results existed.")
st.caption("The plant inputs (solar irradiance, temperature, natural inflow) are compared on their own page, against simple forecasts and, for irradiance and temperature, the weather forecast.")
