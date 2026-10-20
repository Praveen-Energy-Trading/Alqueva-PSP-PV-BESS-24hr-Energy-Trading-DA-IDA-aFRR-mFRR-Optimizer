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

ml_ui.header("Model arena", "", step=5)
ml_ui.tour_bar(6)

# The colours mean one thing each, on every picture of this page:
#   an algorithm keeps its own colour; the champion is near-black; a real difference is green, a gap that could be luck is grey.
MUTED, INK, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#2b2a28"
NEUTRAL, T_DONE, GREY, REAL, LUCK = "#6b6a64", "#8791ab", "#b9b7ae", theme.GREEN, "#9a9890"
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # a hint starts with a capital
_alg = lambda m_: CHAMP if m_ == "champion" else theme.CATEGORICAL[ml_data.model_index(m_) % 8]                       # noqa: E731
_name = lambda m_: ml_data.champion_label(target) if m_ == "champion" else ml_data.model_label(m_)                                         # noqa: E731
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

_cfg_t, _cfg_e = ml_data.config("tuning"), ml_data.config("evaluation")
if _cfg_t and _cfg_e:
    try:
        st.page_link("pages/ml_5_evaluation_tuning.py", label="Evaluation design and tuning lab", icon="🎛️")
    except Exception:                                                                                                       # a single page run on its own
        pass

board = board.sort_values("mae").reset_index(drop=True)
best = board.iloc[0]
pw = tabs["pairwise_dm"]
units = ml_data.UNITS[target]
models = list(board["model"])
n_hours = int(best["n"])
n_days_all = round(n_hours / 24)
res0 = ml_data.dm_example(daily, target) if not daily.empty else None                                                # the champion against the decision tree: the example of the figure

# ============================================================================================ 0. from the tuning lab to the backtest
_tl = ml_data.pack("tuning_log")
_rr = sorted(pd.to_datetime(_tl.loc[(_tl["target"] == target) & (_tl["search"] == "tpe"), "tuned_at"]).unique()) if len(_tl) else []
_dd0 = pd.to_datetime(daily.loc[daily["target"] == target, "date"]) if not daily.empty else pd.Series(dtype="datetime64[ns]")
if len(_rr) and len(_dd0) and _cfg_t:
    import ast
    _b0, _b1 = _dd0.min(), _dd0.max()                                                                                       # the first and the last day of the backtest
    _rt, _gp = int(_cfg_t["search"]["retune_every_days"]), int(_cfg_t["search"].get("gap_days", 1))
    _st = []                                                                                                                # the days each round's settings forecast: after the gap day, for retune days
    for k_, e_ in enumerate(_rr):
        a_, b_ = max(pd.Timestamp(e_) + pd.Timedelta(days=_gp + 1), _b0), min(pd.Timestamp(e_) + pd.Timedelta(days=_gp + _rt), _b1)
        if b_ >= a_:
            _st.append((k_, pd.Timestamp(e_), a_, b_))
    _iso = lambda d_: pd.Timestamp(d_).strftime("%Y-%m-%d")                                                                  # noqa: E731
    _mini("From the tuning lab to the backtest",
          f"A block is the best setting one algorithm found in one tuning round. It forecasts only its own {_rt} days. The blocks of an algorithm are joined into its backtest series, and everything below is scored on those series")
    _s7, _a7 = [], []
    for i_, m_ in enumerate(models):
        for j_, (k_, e_, a_, b_) in enumerate(_st):
            _s7.append(dict(type="rect", x0=_iso(a_), x1=_iso(b_), y0=i_ - 0.34, y1=i_ + 0.34, fillcolor=_alg(m_), opacity=0.92 if j_ % 2 == 0 else 0.62, line=dict(color="white", width=2.5), layer="below"))
            _a7.append(dict(x=_iso(a_ + (b_ - a_) / 2), y=i_, text=f"<b>R{k_ + 1}</b>", showarrow=False, font=dict(size=11, color="white")))
    for k_, e_, a_, b_ in _st:                                                                                              # the head of every round
        _a7.append(dict(x=_iso(a_ + (b_ - a_) / 2), y=-0.95, text=f"<b>Round {k_ + 1}</b><br>tuned {e_:%d-%m-%Y}", showarrow=False, font=dict(size=11, color=INK)))
        _s7.append(dict(type="line", x0=_iso(a_), x1=_iso(a_), y0=-0.55, y1=len(models) - 0.45, line=dict(color="#c9c7be", width=1, dash="dot")))
    _yj = len(models) + 0.95                                                                                                # the joined series
    _s7.append(dict(type="rect", x0=_iso(_b0), x1=_iso(_b1), y0=_yj - 0.5, y1=_yj + 0.5, fillcolor=theme.ORANGE, opacity=1.0, line_width=0, layer="above"))
    _a7.append(dict(x=_iso(_b0 + (_b1 - _b0) / 2), y=_yj - 0.14, text="<b>Backtest validation</b>: the blocks joined, one error per day", showarrow=False, font=dict(size=12, color="white")))
    _a7.append(dict(x=_iso(_b0 + (_b1 - _b0) / 2), y=_yj + 0.2, text=f"{_b0:%d-%m-%Y} to {_b1:%d-%m-%Y} · {len(_dd0.unique()):,} days", showarrow=False, font=dict(size=11, color="white")))
    for k_, e_, a_, b_ in _st:                                                                                              # every block flows down into the joined series
        _a7.append(dict(x=_iso(a_ + (b_ - a_) / 2), y=_yj - 0.5, ax=_iso(a_ + (b_ - a_) / 2), ay=len(models) - 0.5, axref="x", ayref="y", xref="x", yref="y", showarrow=True, arrowhead=2,
                        arrowsize=1.0, arrowwidth=1.4, arrowcolor="#8a8880", text=""))
    _hx, _hy, _ht = [], [], []
    for i_, m_ in enumerate(models):                                                                                        # hover: the settings of every block
        for k_, e_, a_, b_ in _st:
            row_ = _tl[(_tl["target"] == target) & (_tl["search"] == "tpe") & (_tl["model"] == m_) & (pd.to_datetime(_tl["tuned_at"]) == e_)]
            try:
                d_ = ast.literal_eval(row_.iloc[0]["params"]) if len(row_) and isinstance(row_.iloc[0]["params"], str) else {}
            except (ValueError, SyntaxError):
                d_ = {}
            _hx.append(_iso(a_ + (b_ - a_) / 2)); _hy.append(i_)
            _ht.append(f"<b>{ml_data.model_label(m_)}, round {k_ + 1}</b><br>forecasts {a_:%d-%m-%Y} to {b_:%d-%m-%Y}<br>" + ("<br>".join(f"{ml_data.setting_label(k2_)}: {v2_}" for k2_, v2_ in d_.items()) or "the defaults"))
    fa7 = go.Figure(go.Scatter(x=_hx, y=_hy, mode="markers", marker=dict(size=28, opacity=0), text=_ht, hoverinfo="text", showlegend=False,
                               hoverlabel=dict(align="left", bgcolor="white", bordercolor="#e3e1d9", font=dict(size=12, color=INK))))
    theme.style_fig(fa7, height=int(50 * (len(models) + 2) + 90), legend=False)
    _x0_, _x1_ = _b0 - pd.Timedelta(days=15), _b1 + pd.Timedelta(days=15)
    _yrs7 = range(_x0_.year, _x1_.year + 1)
    fa7.update_xaxes(type="date", range=[_iso(_x0_), _iso(_x1_)], tickmode="array", tickvals=[f"{y_}-01-01" for y_ in _yrs7], ticktext=[""] * len(_yrs7), showgrid=True, gridcolor="#eceae3", zeroline=False)
    fa7.update_yaxes(range=[len(models) + 1.7, -1.45], tickvals=list(range(len(models))) + [_yj], showgrid=False, zeroline=False, fixedrange=True,
                     ticktext=[f"<span style='color:{_alg(m_)}'>■</span> <b>{ml_data.model_label(m_)}</b>" for m_ in models] + ["<b>Scored on</b>"])
    for y_ in _yrs7:                                                                                                        # the year in the middle of its own column
        _a7.append(dict(x=f"{y_}-07-01", xref="x", y=0, yref="paper", yshift=-14, text=str(y_), showarrow=False, font=dict(size=11, color=NEUTRAL)))
    fa7.update_layout(margin=dict(t=4, b=24, l=10, r=10))
    fa7.layout.shapes = _s7                                                                                                 # replace (not merge with) the axis line the theme added
    fa7.layout.annotations = _a7
    st.plotly_chart(fa7, width="stretch", config={"displayModeBar": False})
    _line(f"R1 to R{len(_rr)} are the tuning rounds of the Tuning lab. A day is never forecast with settings from a round that comes after it. Hover a block to read the settings.")

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
        _fnote(8, cya, f"{_dot('champion')}<b>{_name('champion')}</b>", anchor="left")
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
    # ---- ⑥ is the gap real or luck: the Diebold-Mariano test, built from the daily gaps of this pair
    stp = ml_data.dm_steps(res)
    se, lags = stp["se"], stp["lags"]
    real_gap = p < 0.05
    better = la if mean < 0 else lb
    _section("Is the gap real or luck?", "The Diebold-Mariano (DM) test: the standard way to compare two forecasts. It asks whether the average gap between them is larger than its own day-to-day noise", num="⑥")
    # 1. the gap per day: how it is made, and what it looks like
    _gl, _gr = st.columns([5, 4], gap="large")
    with _gl:
        _dts = pd.to_datetime(diff["date"])
        _mini("One gap per day", f"{la} minus {lb}, in {units}, on every day from {_dts.min():%d-%m-%Y} to {_dts.max():%d-%m-%Y} ({n_d:,} days). Below 0 {la} was closer, above 0 {lb} was")
        _g = diff["difference"].to_numpy(dtype=float)
        lo_, hi_ = np.quantile(_g, [0.005, 0.995])
        cnt_, edg_ = np.histogram(np.clip(_g, lo_, hi_), bins=48, range=(lo_, hi_))
        ctr_ = (edg_[:-1] + edg_[1:]) / 2
        fh = go.Figure(go.Bar(x=ctr_, y=cnt_, marker_color=[ca if c_ < 0 else cb for c_ in ctr_], width=float(edg_[1] - edg_[0]) * 0.9, hoverinfo="skip"))
        theme.style_fig(fh, height=300, xaxis_title=f"Gap per day ({units})", yaxis_title="Number of days", legend=False)
        fh.add_vline(x=0, line_dash="dot", line_color="#9a9890", line_width=1.2)
        fh.add_vline(x=mean, line_color=INK, line_width=2)
        fh.add_annotation(x=mean, y=1.0, yref="paper", text=f"<b>Average {mean:+.2f}</b>", showarrow=False, yanchor="bottom", font=dict(size=11, color=INK))
        fh.update_layout(margin=dict(t=26, r=10), bargap=0.05)
        st.plotly_chart(fh, width="stretch", config={"displayModeBar": False})
        _line(f"The bars count days by the size of the gap; the dark line is their average. The 0.5% most extreme days on each side are drawn at the edge.")
    with _gr:
        _mini("How one gap is made", "The first five days of the window")
        _t5 = los.head(5)
        _table(["Date", f"Error {la}", f"Error {lb}", "Gap"],
               [[f"{pd.Timestamp(r_.date):%d-%m-%Y}", f"{r_.a:.3f}", f"{r_.b:.3f}", f"<b>{r_.a - r_.b:+.3f}</b>"] for r_ in _t5.itertuples()], first="minmax(6rem, 1.2fr)")
        _line(M(r"d_t=e_t^{A}-e_t^{B}", size=0.95) + f", with A = {la} and B = {lb}. The same for all {n_d:,} days.")
    # 2. the calculation, step by step
    _mini("The calculation", "Four steps from the daily gaps to one ratio")
    _hd = f"font-weight:600; color:{INK};"
    st.markdown(_panel(
        f"<div style='display:grid; grid-template-columns:9.5rem 1fr; row-gap:0.7rem; align-items:baseline;'>"
        f"<span style='{_hd}'>① Average gap</span><span>{M(r'\bar d=\frac{1}{N}\sum_{t=1}^{N} d_t', size=0.95)} = <b style='font-size:1.1rem; color:{INK};'>{mean:+.3f}</b> {units} over N = {n_d:,} days</span>"
        f"<span style='{_hd}'>② Noise</span><span>{M(r'\text{SE}=\sqrt{\frac{\hat{\sigma}^{2}}{N-1}}', size=0.95)} = <b style='font-size:1.1rem; color:{INK};'>{se:.3f}</b> {units}"
        f"<br><span style='color:{MUTED}; font-size:0.8rem;'>How far the average could move by chance, counting days up to {lags} days apart together, because one bad day tends to follow another. Open the calculation below</span></span>"
        f"<span style='{_hd}'>③ DM ratio</span><span>{M(rf'\text{{DM}}=\frac{{\bar d}}{{\text{{SE}}}}=\frac{{{mean:.3f}}}{{{se:.3f}}}=', size=0.95)} <b style='font-size:1.1rem; color:{INK};'>{stat:.2f}</b>"
        f"<br><span style='color:{MUTED}; font-size:0.8rem;'>The average gap measured in units of its own noise</span></span>"
        f"<span style='{_hd}'>④ Chance</span><span><b style='font-size:1.1rem; color:{INK};'>p = {p:.2g}</b>"
        f"<br><span style='color:{MUTED}; font-size:0.8rem;'>If the two forecasts were equally good, a gap this large would almost never happen. Below 5% (a ratio beyond about 2) is read as real, not luck. Open the calculation below</span></span></div>"), unsafe_allow_html=True)
    # ---- the two derivations, as a cheat sheet: step, calculation with the numbers, and the reason in a few words
    _gm, _wt = stp["gammas"], stp["weights"]
    _lr = stp["long_run"]
    _r1 = _gm[1] / _gm[0]
    _se_ind = float(np.sqrt(_gm[0] / stp["df"]))
    _df, _x, _z, _ib, _tl = stp["df"], abs(stat), stp["z"], stp["incomplete_beta"], stp["tail"]
    _sci = lambda v_: (lambda m_, e_: rf"{float(m_):.1f}\times10^{{{int(e_)}}}")(*f"{v_:.1e}".split("e"))                     # noqa: E731  a number as LaTeX in powers of ten
    _b = lambda v_: f"<b style='color:{INK};'>{v_}</b>"                                                                     # noqa: E731
    _G = "display:grid; grid-template-columns:2.2rem minmax(0, 1fr) minmax(0, 16rem); column-gap:0.9rem; align-items:baseline;"


    def _cheat(rows: list) -> str:
        """A white card of numbered steps: what is done (with the numbers) and why, in a few words."""
        head = (f"<div style='{_G} padding:0.1rem 0 0.35rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; font-weight:600; color:{MUTED};'>"
                "<span>Step</span><span>Calculation</span><span>Why</span></div>")
        body = "".join(f"<div style='{_G} padding:0.5rem 0; border-bottom:1px solid #eeede8; font-size:0.9rem; color:{INK};'>"
                       f"<span style='color:{MUTED};'>{i}</span><span style='line-height:1.9;'>{calc}</span><span style='font-size:0.8rem; color:{MUTED}; line-height:1.45;'>{why}</span></div>"
                       for i, (calc, why) in enumerate(rows, 1))
        return _panel(head + body)


    with st.expander(f"② How the noise is calculated: SE = {se:.3f} {units}"):
        st.markdown(_cheat([
            (f"{M(r'd_t=e_t^{A}-e_t^{B}', size=0.9)} for each of N = {n_d:,} days; {M(r'\bar d=\frac{1}{N}\sum_{t=1}^{N} d_t', size=0.9)} = {_b(f'{mean:+.3f}')}",
             "one gap per day; its average is what we judge"),
            (M(r'x_t=d_t-\bar d', size=0.9), "spread is measured from the middle"),
            (f"{M(r'\gamma_0=\frac{1}{N}\sum_{t=1}^{N} x_t^{2}', size=0.9)} = {_b(f'{_gm[0]:.3f}')}", "squares: a day above and a day below must not cancel"),
            (f"{M(r'\gamma_k=\frac{1}{N}\sum_{t=k+1}^{N} x_t\,x_{t-k}', size=0.9)}, k = 1 to {lags}: {M(r'\gamma_1', size=0.85)} = {_b(f'{_gm[1]:+.3f}')}, correlation {M(r'\gamma_1/\gamma_0', size=0.85)} = {_b(f'{_r1:+.2f}')}",
             "today resembles yesterday, so the days are not independent and the average is noisier"),
            (f"{M(r'\hat{\sigma}^{2}=\gamma_0+2\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)\gamma_k', size=0.9)}, L = {lags}: {_b(f'{_lr:.3f}')} (terms below)",
             f"far days count less (weights); factor 2: each pair is met from both sides; L = {lags}: one week, the project's setting"),
            (f"{M(rf'\text{{SE}}=\sqrt{{\frac{{\hat{{\sigma}}^{{2}}}}{{N-1}}}}=\sqrt{{\frac{{{_lr:.3f}}}{{{_df}}}}}=', size=0.9)} {_b(f'{se:.3f}')} {units}",
             f"N - 1: small-sample adjustment. If days were independent: {M(rf'\sqrt{{{_gm[0]:.3f}/{_df}}}', size=0.85)} = {_se_ind:.3f}"),
        ]), unsafe_allow_html=True)
        _table(["Days apart k", "Weight 1 - k/(L+1)", "Product of two gaps k days apart (average)", "Added to the variance"],
               [[f"{k_}", f"{_wt[k_]:.3f}", f"{_gm[k_]:.3f}", f"{_gm[0]:.3f}" if k_ == 0 else f"{2 * _wt[k_] * _gm[k_]:+.3f}"] for k_ in range(len(_gm))]
               + [["<b>Total</b>", "", "", f"<b>{stp['long_run']:.3f}</b>"]], first="minmax(7rem, 1fr)")

    with st.expander(f"④ How p is calculated: p = {p:.2g}"):
        st.markdown(_cheat([
            (f"If the two were equally good the true gap is 0, and the ratio follows the t curve with {M(rf'\nu=N-1={_df}', size=0.9)} (almost the normal bell)",
             "p asks one thing: could this gap be luck?"),
            (f"{M(r'x=\left|\text{DM}\right|', size=0.9)} = {_b(f'{_x:.2f}')}", "the distance from 0, in noise units"),
            (f"{M(rf'z=\frac{{\nu}}{{\nu+x^{{2}}}}=\frac{{{_df}}}{{{_df}+{_x:.2f}^{{2}}}}=', size=0.9)} {_b(f'{_z:.4f}')}", "turns x into a number between 0 and 1"),
            (f"{M(rf'P(T>x)=\frac{{1}}{{2}}\,I_z\left(\frac{{\nu}}{{2}},\frac{{1}}{{2}}\right)=\frac{{1}}{{2}}\times {_sci(_ib)}=', size=0.9)} {_b(f'{_tl:.1e}')}",
             "the area of the curve beyond x on one side; I is the incomplete beta function, evaluated by software"),
            (f"{M(rf'p=2\times {_sci(_tl)}=', size=0.9)} {_b(f'{p:.2g}')}", "doubled: a ratio just as far on the other side would surprise us equally"),
            (f"p {'&lt;' if p < 0.05 else '&ge;'} 5%, that is |DM| {'&gt;' if abs(stat) > stp['t_crit'] else '&le;'} {stp['t_crit']:.2f}: " + _b("real, not luck" if p < 0.05 else "could be luck"),
             "5% is fixed in the written rule before the results"),
        ]), unsafe_allow_html=True)
    # 3. the picture of the answer: is the average far enough from zero?
    _vl, _vr = st.columns([5, 4], gap="large")
    with _vl:
        _mini("Is the average far from zero?", "The average gap with its noise band (average ± 2 noise units)")
        lo_i, hi_i = mean - 2 * se, mean + 2 * se
        pad_ = max(hi_i, 0) - min(lo_i, 0)
        fi = go.Figure()
        fi.add_shape(type="line", x0=lo_i, x1=hi_i, y0=0, y1=0, line=dict(color=REAL if real_gap else LUCK, width=7))
        fi.add_trace(go.Scatter(x=[mean], y=[0], mode="markers", marker=dict(size=15, color=INK, line=dict(color="white", width=2)), hoverinfo="skip", showlegend=False))
        fi.add_vline(x=0, line_dash="dot", line_color="#9a9890", line_width=1.2)
        fi.add_annotation(x=mean, y=0, text=f"<b>{mean:+.3f}</b>", yshift=22, showarrow=False, font=dict(size=12, color=INK))
        fi.add_annotation(x=0, y=0, text="0 = no gap", yshift=-26, showarrow=False, font=dict(size=11, color=MUTED))
        theme.style_fig(fi, height=170, xaxis_title=f"Gap ({units})", legend=False)
        fi.update_xaxes(range=[min(lo_i, 0) - 0.12 * pad_, max(hi_i, 0) + 0.12 * pad_], showgrid=False, zeroline=False)
        fi.update_yaxes(range=[-1, 1], visible=False, fixedrange=True)
        fi.update_layout(margin=dict(t=6, b=40, l=10, r=10))
        st.plotly_chart(fi, width="stretch", config={"displayModeBar": False})
        _line(("The band does not reach 0, so the gap cannot be put down to luck." if real_gap else "The band reaches 0, so the gap could be luck."))
    with _vr:
        st.markdown(_panel(
            f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; padding:0.2rem 0;'>Reading</div>"
            f"<div style='font-size:1.15rem; font-weight:800; color:{REAL if real_gap else LUCK}; line-height:1.3;'>"
            + (f"{better} is better: the gap is real" if real_gap else "No clear winner: the gap could be luck") + "</div>"
            f"<div style='margin-top:0.5rem; line-height:1.55;'>DM ratio <b style='color:{INK};'>{stat:.2f}</b>: the average gap is <b style='color:{INK};'>{abs(stat):.1f}</b> noise units from 0. "
            f"Beyond 2 in either direction is read as real.</div>"), unsafe_allow_html=True)
