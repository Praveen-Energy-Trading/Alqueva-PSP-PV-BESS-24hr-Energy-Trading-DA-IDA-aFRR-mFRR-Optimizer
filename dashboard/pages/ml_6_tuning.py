"""ML Lifecycle 6 - Tuning lab: nested time-aware tuning, every trial, Optuna against random search, what the search found."""
from __future__ import annotations

import math
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

ml_ui.header("Tuning lab", "", step=5)
ml_ui.tour_bar(6)

OBJECTIVE_NAME = {"regression": "squared error (L2)", "reg:squarederror": "squared error (L2)", "squared_error": "squared error (L2)",
                  "regression_l1": "absolute error (L1)", "reg:absoluteerror": "absolute error (L1)", "absolute_error": "absolute error (L1)",
                  "huber": "Huber", "reg:pseudohubererror": "pseudo-Huber"}
RULE_SETTINGS = ("objective", "loss")                                                # the settings whose values are learning rules


def _std(v) -> str:
    return OBJECTIVE_NAME.get(str(v), str(v))


def _std_text(text: str) -> str:
    """Library spellings inside a text, as standard names."""
    import re
    alt = "|".join(re.escape(k) for k in sorted(OBJECTIVE_NAME, key=len, reverse=True))
    return re.sub(rf"(?<![\w:])({alt})(?![\w])", lambda m: OBJECTIVE_NAME[m.group(1)], text)


_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # a hint starts with a capital


def _section(title: str, hint: str = "", num: str = "") -> None:
    """A section heading: a number (the steps of the story), a bold title and, under it, one muted line on how to read the section."""
    _n = f"<span style='font-weight:400; color:{NEUTRAL};'>{num}</span>&nbsp; " if num else ""
    st.markdown(f"<div style='margin:1.7rem 0 0.45rem;'><div style='font-size:1.55rem; font-weight:700; color:{theme.INK_PRIMARY}; line-height:1.25;'>{_n}{title}</div>"
                + (f"<div style='font-size:0.88rem; color:{theme.INK_MUTED}; margin-top:0.2rem;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _mini(title: str, hint: str = "") -> None:
    """The title of a chart that shares a row with another one."""
    st.markdown(f"<div style='margin:0.2rem 0 0.3rem;'><div style='font-size:1.02rem; font-weight:700; color:{theme.INK_PRIMARY};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{theme.INK_MUTED}; margin-top:0.1rem; line-height:1.35;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _line(html: str) -> None:
    """One short note under a table or chart, the key numbers in bold."""
    st.markdown(f"<div style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin:0.25rem 0 0.9rem;'>{html}</div>", unsafe_allow_html=True)


cfg = ml_ui.cached(lambda: ml_data.config("tuning"))()
trials = ml_ui.cached(lambda: ml_data.pack("trials"))()
tlog = ml_ui.cached(lambda: ml_data.pack("tuning_log"))()
if trials.empty or not cfg:
    ml_ui.not_yet_run("the tuning trials (data pack)")
    st.stop()

target = ml_ui.target_picker(targets=sorted(set(trials["target"]), key=lambda t: ml_data.TARGETS.index(t)))
s = cfg["search"]
ev = ml_ui.cached(lambda: ml_data.config("evaluation"))()
L, nb, gap = int(s["validation_days"]), int(s["n_blocks"]), int(s.get("gap_days", 1))
_log = tlog[(tlog["target"] == target) & (tlog["search"] == "tpe")]
expanding = ml_data.config("rollout").get("targets", {}).get(target, {}).get("treatment", "standard") != "standard"        # short histories train on every day they have
win, win_end = int(ev["schedule"]["train_window_days"]), None                        # a rolling window has the length it is configured with
if len(_log):
    win_end = pd.Timestamp(_log["tuned_at"].max())
    if expanding:                                                                    # an expanding window is as long as the days the latest round had: the latest block's training + that block + the gap
        win = int(_log[_log["tuned_at"] == _log["tuned_at"].max()]["inner_train_days"].median()) + L + gap
n_t, retune, refit = int(s["n_trials"]), int(s["retune_every_days"]), int(ev["schedule"]["retrain_every_days"])
BLUE, VIOLET, ORANGE, AQUA, RED, MUTED, INK = theme.BLUE, theme.VIOLET, theme.ORANGE, theme.AQUA, theme.STATUS_CRITICAL, theme.INK_MUTED, theme.INK_PRIMARY
GREY = "#d9d7cf"
T_DEF, T_DONE, T_STOP, T_WIN, NEUTRAL = "#3d3c39", "#8791ab", theme.MAGENTA, theme.GREEN, "#6b6a64"
ROW = 34.0                                                                           # pixels per row: the labels keep their places whatever the content
X_END = 1010                                                                         # the width of the figure in its own units
DAY = X_END / (1.402 * win)                                                          # units per day: the window always fills the same share, the rest is room for the labels
px_day = 1150.0 / X_END


SH, AN = [], []                                                                      # shapes and annotations, handed to the figure once at the end


def _shape(**kw):
    SH.append(kw)


def _annot(**kw):
    AN.append(kw)


def _pill(fig, x0, x1, y0, y1, color, opacity=1.0, r_px=4.0):
    """A rounded bar as a path; x in days of the window, y in rows."""
    rx, ry = min(r_px / px_day, (x1 - x0) / 2), min(r_px / ROW, (y1 - y0) / 2)
    _shape(type="path", fillcolor=color, opacity=opacity, line_width=0,
                  path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"))


CARD_LINE = "#e3e1d9"
CARD_X0, CARD_X1 = -12, X_END + 12                                                    # the cards reach a little beyond the content


def _card(y0, y1, r_px=10.0):
    """A white rounded card with a thin border, drawn under everything else."""
    rx, ry = r_px / px_day, r_px / ROW
    x0, x1 = CARD_X0, CARD_X1
    SH.append(dict(type="path", fillcolor="#ffffff", line=dict(color=CARD_LINE, width=1), layer="below",
                   path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                         f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")))


def _head(y, num, title, sub_="", two=False):
    """The header of a section: a number, a bold title, and a muted line of explanation."""
    AN.append(dict(x=6, y=y, xanchor="left", yanchor="middle", showarrow=False, align="left", font=dict(size=14, color=INK),
                   text=f"<b style='font-size:17px;color:{NEUTRAL}'>{num}</b>&nbsp; <b>{title}</b>" + (f"{'<br>' if two else ' &nbsp;'}<span style='font-size:12px;color:{MUTED}'>{_cap(sub_)}</span>" if sub_ else "")))


def _note(fig, x, y, text, size=12, color=INK, anchor="center", **kw):
    _annot(x=x, y=y, xanchor=anchor, showarrow=False, text=text, font=dict(size=size, color=color), **kw)


def _arrow(fig, x0, y0, x1, y1, color=MUTED):
    _annot(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=color, text="")


body = win - nb * L                                                                  # the days before the tuning blocks
xw = win * DAY                                                                       # where the window ends, in units
ft = go.Figure()
ft.add_trace(go.Scatter(x=[0, X_END], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
legend = (f"<span style='color:{BLUE}'>■</span> Training days (the model learns from them) &nbsp;&nbsp; <span style='color:{VIOLET}'>■</span> Tuning days ({nb} blocks of {L} days, scored) &nbsp;&nbsp; "
          f"<span style='color:{ORANGE}'>■</span> The refit's forecast days &nbsp;&nbsp; <span style='color:#c9c7be'>■</span> Days not used by that round")
_head(-1.95, "①", "Tuning only looks inside the refit's own window",
      f"{ml_data.label(target)}, latest tuning round (" + (f"window ending {pd.Timestamp(win_end):%d-%m-%Y}, " if win_end is not None else "") + (f"expanding window of all {win} days available" if expanding else f"rolling {win}-day window") + ")", two=True)
# ---- 1. the window of one refit: training, then the tuning days cut into blocks
rh = 0.8
_pill(ft, 0, body * DAY, 0, rh, BLUE, 0.62)
_pill(ft, body * DAY + 0.6, xw, 0, rh, VIOLET, 0.9)
for k in range(1, nb):                                                               # the block boundaries inside the tuning days
    _shape(type="line", x0=(body + k * L) * DAY, x1=(body + k * L) * DAY, y0=0.08, y1=rh - 0.08, line=dict(color="white", width=2))
_pill(ft, xw + 1, xw + 1 + 30 * DAY, 0, rh, ORANGE, 1.0)
_note(ft, 8, rh / 2, f"<b>One refit:</b> {body} days of training", anchor="left", align="left")
for k in range(nb):
    _note(ft, (body + k * L + L / 2) * DAY, rh / 2, f"<b>{k + 1}</b>", color="white")
_note(ft, xw + 1 + 30 * DAY + 10, rh / 2, "Forecast days", color=ORANGE, anchor="left")
# ---- 2. the tuning rounds: each block is predicted by a model that has seen only the days before it
for k in range(1, nb + 1):
    y0 = 1.55 + (k - 1) * 1.0
    b0 = body + (k - 1) * L                                                          # first day of block k
    tr_days = b0 - gap
    _pill(ft, 0, tr_days * DAY, y0, y0 + rh, BLUE, 0.62)
    _pill(ft, b0 * DAY, (b0 + L) * DAY, y0, y0 + rh, VIOLET, 0.9)
    if b0 + L < win:
        _pill(ft, (b0 + L) * DAY + 0.6, xw, y0, y0 + rh, GREY, 0.8)
    _note(ft, 8, y0 + rh / 2, f"<b>Round {k}:</b> trains on {tr_days} days", anchor="left", align="left")
    _note(ft, (b0 + L / 2) * DAY, y0 + rh / 2, f"<b>{k}</b>", color="white")
    _note(ft, xw + 36, y0 + rh / 2, f"Predicts block {k}: <b>error {k}</b>" + ("<br>Scored first: the pruner looks here" if k == nb else ""), color=VIOLET, anchor="left", align="left")
ys = 1.55 + nb * 1.0 + 0.75
_note(ft, 8, ys + 1.5, legend, size=11, color=theme.INK_SECONDARY, anchor="left", yanchor="middle", align="left", bgcolor="#f7f6f2", bordercolor="#e8e6df", borderwidth=1, borderpad=6)    # what the colours mean, on one row of its own under the rounds
_note(ft, xw + 36, ys + 0.3, "<b>Score</b> = average of<br>errors " + ", ".join(str(k) for k in range(1, nb + 1)), size=13, color=VIOLET, anchor="left",
      align="left", bgcolor="white", bordercolor=VIOLET, borderwidth=1, borderpad=6)
# ---- 3. the days tuning never looks at, and where they start
ysn = ys + 2.4
_pill(ft, xw + 1, xw + 130, ysn, ysn + rh, ORANGE, 0.35)
_pill(ft, xw + 134, X_END, ysn, ysn + rh, RED, 0.3)
_note(ft, xw + 65, ysn + rh / 2, "Backtest<br>validation", size=11)
_note(ft, xw + 134 + (X_END - xw - 134) / 2, ysn + rh / 2, "Locked<br>test year", size=11, color=RED)
_note(ft, xw - 8, ysn + rh / 2, "<b>Never used by the tuning</b> →", color=RED, anchor="right", align="right")
_shape(type="line", x0=xw + 0.5, x1=xw + 0.5, y0=-0.55, y1=ysn + rh + 0.1, line=dict(color="#a8a69d", width=1.2, dash="dot"))
_note(ft, xw - 6, -0.55, "The only days the tuning can use ←", size=11, color=MUTED, anchor="right")
_card(-2.7, ysn + rh + 0.6)

# ---- 4. the route of one setting
yr = ysn + rh + 1.9
_head(yr, "②", "The route of one trial", "one trial = one candidate setting of the algorithm")
cy, nh = yr + 1.5, 1.0
def _node(x0, x1, label, badges=()):
    _pill(ft, x0, x1, cy - nh / 2, cy + nh / 2, NEUTRAL, 0.10, r_px=6)
    _pill(ft, x0, x0 + 4, cy - nh / 2, cy + nh / 2, NEUTRAL, 1.0, r_px=2)
    _note(ft, x0 + 14, cy, label, size=12, anchor="left")
    for i, b in enumerate(badges):                                                   # the blocks it is scored on: violet, as the blocks above
        bx = x1 - 12 - (len(badges) - 1 - i) * 26
        _pill(ft, bx - 11, bx + 11, cy - 0.28, cy + 0.28, VIOLET, 0.9, r_px=3)
        _note(ft, bx, cy, f"<b>{b}</b>", size=11, color="white")
_node(0, 150, "Next trial")
_node(205, 365, "Latest block", badges=(nb,))
_arrow(ft, 152, cy, 203, cy)
_arrow(ft, 367, cy, 383, cy)
dx, dh, dw = 545, 1.15, 160
_dia = f"M {dx - dw},{cy} L {dx},{cy - dh} L {dx + dw},{cy} L {dx},{cy + dh} Z"
_shape(type="path", fillcolor=T_STOP, opacity=0.13, line_width=0, path=_dia)
_shape(type="path", fillcolor="rgba(0,0,0,0)", line=dict(color=T_STOP, width=1.5), path=_dia)
_note(ft, dx, cy, "Block 3: worse<br>than the median<br>of earlier trials?", size=11)
_arrow(ft, dx + dw + 2, cy, 738, cy)
_note(ft, dx + dw + 18, cy - 0.5, "No", size=11, color=MUTED)
_node(740, 915, "Older blocks", badges=tuple(range(nb - 1, 0, -1)))
_arrow(ft, 917, cy, 930, cy)
oy = cy + 2.25
_arrow(ft, dx, cy + dh + 0.02, dx, oy - 0.45, color=T_STOP)
_note(ft, dx + 22, cy + dh + 0.28, "Yes", size=11, color=MUTED, anchor="left")
_pill(ft, dx - 90, dx + 90, oy - 0.45, oy + 0.45, T_STOP, 0.9, r_px=6)
_note(ft, dx, oy, "<b>✕ Stopped</b>", size=12, color="white")
_pill(ft, 930, X_END, cy - nh / 2, cy + nh / 2, T_DONE, 0.95, r_px=6)
_note(ft, (930 + X_END) / 2, cy, "<b>Score</b>", size=12, color="white")
_arrow(ft, (930 + X_END) / 2, cy + nh / 2 + 0.02, (930 + X_END) / 2, oy - 0.45, color=T_WIN)
_pill(ft, 880, X_END, oy - 0.45, oy + 0.45, T_WIN, 0.95, r_px=6)
_note(ft, (880 + X_END) / 2, oy, "<b>Best score</b>", size=12, color="white")
_pill(ft, 0, 150, oy - 0.45, oy + 0.45, T_DEF, 0.95, r_px=6)
_note(ft, 75, oy, "<b>Trial 0</b>", size=12, color="white")
_note(ft, 165, oy, "Defaults, never stopped", size=11, color=MUTED, anchor="left")

# ---- 5. the same budget for every algorithm: the trials of one real round, TPE and the random control
_card(yr - 0.7, oy + 0.8)
ydt = oy + 2.0
_head(ydt + 0.1, "③", f"The same budget for every algorithm: up to {n_t} trials each", f"TPE = Tree-structured Parzen Estimator (Optuna's search). {ml_data.label(target)}, latest round. Each square is one trial ({len(trials):,} logged).", two=True)
_leg = " &nbsp;&nbsp; ".join(f"<span style='color:{c}'>■</span> {x}" for c, x in ((T_DEF, "Defaults (trial 0)"), (T_DONE, "All 3 blocks scored (darker = lower error)"), (T_STOP, "Stopped after block 3"), (T_WIN, "The winner"), ("#c9c7be", "Trial not needed")))
_note(ft, 6, ydt + 1.1, _leg, size=11, color=theme.INK_SECONDARY, anchor="left")
xs0, xs1 = 190, 905                                                                  # the right-hand end of the rows is kept for each row's best score
_dg = ml_ui.cached(ml_data.prepared_trials)(trials, target)                           # every search of this target, with the score relative to the defaults
_fin = _dg.loc[~_dg["pruned"].astype(bool) & ~_dg["is_default"].astype(bool) & _dg["rel"].notna(), "rel"]
_lo, _hi = (float(_fin.min()), max(float(_fin.quantile(0.95)), float(_fin.min()) + 0.02)) if len(_fin) else (0.9, 1.1)


def _shade(rel):
    """The colour of a finished trial: the lower its error (relative to the defaults), the darker."""
    u = min(max((_hi - float(rel)) / (_hi - _lo), 0.0), 1.0) if pd.notna(rel) else 0.5
    a_, b_ = (211, 216, 228), (84, 100, 134)
    return "#%02x%02x%02x" % tuple(round(x + (y_ - x) * u) for x, y_ in zip(a_, b_))


_hx, _hy, _ht = [], [], []                                                           # what a square says when the pointer is on it
_note(ft, xs1 + 10, ydt + 1.45, "Best score, stopped", size=9, color=MUTED, anchor="left")
cw = (xs1 - xs0) / n_t
bh, pitch = 0.42, 0.52
y = ydt + 1.75
n_pr_all = 0
combos_dt = math.prod(len(v) for v in cfg["space"].get("decision_tree", {}).values()) if cfg.get("space") else 0
for mdl in ml_data.ordered_models(sorted(set(trials["model"]))):
    for method, tag in (("tpe", ""), ("random", "random")):
        d = _dg[(_dg["model"] == mdl) & (_dg["search"] == method)]
        if d.empty:
            continue
        d = d[d["tuned_at"] == d["tuned_at"].max()].sort_values("trial")
        if method == "tpe":
            _note(ft, 0, y + bh / 2, f"<b>{ml_data.model_label(mdl)}</b>", size=12, anchor="left")
            _note(ft, xs0 - 6, y + bh / 2, "TPE", size=10, color=MUTED, anchor="right")                   # the first row of an algorithm is the smart search, the second the control
            n_pr_all += int(d["pruned"].sum())
        else:
            _note(ft, 0, y + bh / 2, "↳ Random control", size=10, color=MUTED, anchor="left")
        used = set(int(t) for t in d["trial"])
        for i in range(n_t):
            x0 = xs0 + i * cw
            if i not in used:                                                         # a slot of the budget that was not needed
                _shape(type="rect", x0=x0, x1=x0 + cw - 3, y0=y, y1=y + bh, line=dict(color="#c9c7be", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")
        for r in d.itertuples():
            x0 = xs0 + int(r.trial) * cw
            if bool(r.is_default):
                col, mark = T_DEF, ""
            elif bool(r.selected):
                col, mark = T_WIN, "★"
            elif bool(r.pruned):
                col, mark = T_STOP, "×"
            else:
                col, mark = _shade(r.rel), ""
            _pill(ft, x0, x0 + cw - 3, y, y + bh, col, 0.95, r_px=2)
            if mark:
                _note(ft, x0 + (cw - 3) / 2, y + bh / 2, f"<b>{mark}</b>", size=10, color="white")
            state = ("the defaults" if bool(r.is_default) else "the winner of this search" if bool(r.selected) else "stopped after block 3" if bool(r.pruned) else "scored on all 3 blocks")
            _hx.append(x0 + (cw - 3) / 2)
            _hy.append(y + bh / 2)
            _ht.append(f"<b>{ml_data.model_label(mdl)}, {'TPE' if method == 'tpe' else 'random control'}</b><br>Trial {int(r.trial)}: {state}<br>"
                       + (f"Score {float(r.rel):.3f} of the defaults<br>" if pd.notna(r.rel) else (f"Its error on the latest block: {float(r.block1):.2f}, worse than the median of the earlier trials<br>" if bool(r.pruned) and pd.notna(r.block1) else ""))
                       + ("<br>".join(", ".join(_pairs[i_:i_ + 2]) for i_ in range(0, len(_pairs), 2)) if not bool(r.is_default) and (_pairs := [x_.strip() for x_ in _std_text(ml_data.plain_params(r.params)).split(", ")]) else "Default settings"))
        _done = d[~d["pruned"].astype(bool) & d["rel"].notna()]
        _note(ft, xs1 + 10, y + bh / 2, f"<b>{float(_done['rel'].min()):.3f}</b>, {int(d['pruned'].sum())}" + ("✕" if method == "tpe" else ""), size=10, color=theme.INK_SECONDARY, anchor="left")
        if mdl == "decision_tree" and method == "tpe" and len(used) < n_t:
            _note(ft, xs0 + (len(used) + n_t) / 2 * cw, y + bh / 2, f"Whole grid ({combos_dt} combinations) already tried", size=10, color=MUTED)
        y += pitch
    y += 0.2
yd_end = y
ft.add_trace(go.Scatter(x=_hx, y=_hy, mode="markers", marker=dict(size=13, opacity=0), text=_ht, hoverinfo="text", showlegend=False,
                        hoverlabel=dict(align="left", bgcolor="white", bordercolor="#e3e1d9", font=dict(size=12, color=theme.INK_PRIMARY))))

_card(ydt - 0.7, yd_end + 0.05)
end = yd_end + 0.4
theme.style_fig(ft, height=int((end + 2.85) * ROW + 12), xaxis_title=None, legend=False)
ft.update_yaxes(autorange=False, range=[end, -2.85], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
ft.update_xaxes(range=[CARD_X0 - 3, CARD_X1 + 3], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
ft.update_layout(margin=dict(t=2, b=2, l=10, r=10))
ft.layout.shapes = SH                                                                   # replace (not merge with) the axis line the theme added
ft.layout.annotations = AN
st.plotly_chart(ft, width="stretch", config={"displayModeBar": False})

tt = ml_ui.cached(ml_data.prepared_trials)(trials, target, "tpe")
models = ml_data.ordered_models(tt["model"].unique())

_section(f"Random control and TPE <span style='font-weight:400; color:{theme.INK_MUTED};'>— {ml_data.label(target)}</span>", num="④")
model = st.selectbox("Algorithm", models, index=models.index("lightgbm") if "lightgbm" in models else 0, format_func=ml_data.model_label, key="ml6_model")
da = ml_ui.cached(ml_data.prepared_trials)(trials, target)
has_rnd = "random" in set(da["search"])                                              # the control was run for the day-ahead price only
if da.empty:
    ml_ui.not_yet_run("the tuning trials of this target")
else:
    from plotly.subplots import make_subplots
    ms = ml_data.ordered_models(da["model"].unique())
    rc_model = model if model in ms else ms[0]
    sp_ = cfg["space"].get(rc_model, {})
    allx = da[(da["model"] == rc_model) & ~da["is_default"].astype(bool) & da["rel"].notna()]
    infl = {}                                                                          # how much the score depends on each setting: the two that matter most make the map
    for k_ in sp_:
        g_ = allx.groupby(allx["params_d"].map(lambda p, k_=k_: str(p.get(k_))))["rel"].mean()
        if len(g_) > 1:
            infl[k_] = float(g_.max() - g_.min())
    ka, kb = sorted(infl, key=infl.get, reverse=True)[:2]
    va, vb = [str(v) for v in sp_[ka]], [str(v) for v in sp_[kb]]
    latest = {}
    for key in ("random", "tpe"):
        d_ = da[(da["search"] == key) & (da["model"] == rc_model) & ~da["is_default"].astype(bool)]
        latest[key] = d_[d_["tuned_at"] == d_["tuned_at"].max()]
    # the landscape of this tuning round: how good each cell turned out to be (the best score any finished trial of either search reached in it)
    both = pd.concat(latest.values())
    both = both[~both["pruned"].astype(bool) & both["rel"].notna()]
    land = np.full((len(vb), len(va)), np.nan)
    for iy in range(len(vb)):
        for ix in range(len(va)):
            hit = both[both["params_d"].map(lambda p: str(p.get(ka)) == va[ix] and str(p.get(kb)) == vb[iy])]
            if len(hit):
                land[iy, ix] = float(hit["rel"].min())
    lo_, hi_ = np.nanmin(land), np.nanmax(land)
    quality = (hi_ - land) / (hi_ - lo_) if hi_ > lo_ else np.zeros_like(land)           # 1 = the best cells
    GOLD = theme.YELLOW
    rng = np.random.default_rng(0)
    n_early = min(10, n_t - 1)                                              # TPE's first trials are random draws too
    rf = make_subplots(rows=2, cols=3, horizontal_spacing=0.03, vertical_spacing=0.16)
    for i, key in enumerate(("random", "tpe"), start=1):
        r_ = latest[key]
        for j, (lo_t, hi_t) in enumerate(((0, n_early), (n_early, 10 ** 6)), start=1):
            part = r_[(r_["trial"] > lo_t) & (r_["trial"] <= hi_t)]
            rf.add_trace(go.Heatmap(x=list(range(len(va))), y=list(range(len(vb))), z=quality, zmin=0, zmax=1, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
                                    colorscale=[[0, "#f7f5ee"], [1, GOLD]]), row=i, col=j)
            fin, stop = part[~part["pruned"].astype(bool)], part[part["pruned"].astype(bool)]
            for sub_, sym, col_, sz in ((fin, "circle", T_DONE, 12), (stop, "x", T_STOP, 13)):
                if len(sub_):
                    rf.add_trace(go.Scatter(x=[va.index(str(p.get(ka))) + rng.uniform(-0.3, 0.3) for p in sub_["params_d"]],
                                            y=[vb.index(str(p.get(kb))) + rng.uniform(-0.3, 0.3) for p in sub_["params_d"]], mode="markers", showlegend=False,
                                            marker=dict(symbol=sym, size=sz, color=col_, line=dict(color="white", width=1)), customdata=sub_["rel"],
                                            hovertemplate="Score %{customdata:.3f} of the defaults<extra></extra>"), row=i, col=j)
            win_ = part[part["selected"].astype(bool)]
            if len(win_):
                p = win_["params_d"].iloc[0]
                rf.add_trace(go.Scatter(x=[va.index(str(p.get(ka)))], y=[vb.index(str(p.get(kb)))], mode="markers", showlegend=False, hoverinfo="skip",
                                        marker=dict(symbol="star", size=27, color=T_WIN, line=dict(color="white", width=1.5))), row=i, col=j)
    # the result: the best score found so far, averaged over the tuning rounds
    curves = {k_: ml_data.best_so_far(da[da["search"] == k_]) for k_ in (("tpe", "random") if has_rnd else ("tpe",))}
    ends_ = {}
    for key, name, colr, dash, wd in (("random", "random control", theme.INK_MUTED, "dash", 3), ("tpe", "TPE", theme.INK_PRIMARY, "solid", 3.2)):
        if key not in curves:
            continue
        cc = curves[key][curves[key]["model"] == rc_model].sort_values("trial")
        rf.add_trace(go.Scatter(x=cc["trial"], y=cc["best_rel"], mode="lines", showlegend=False, line=dict(color=colr, width=wd, dash=dash),
                                hovertemplate=name + ": %{y:.3f} after trial %{x}<extra></extra>"), row=1, col=3)
        ends_[key] = (float(cc["trial"].iloc[-1]), float(cc["best_rel"].iloc[-1]))
    rf.add_hline(y=1.0, line=dict(color=theme.INK_MUTED, width=1, dash="dot"), row=1, col=3)
    # the work: a trial stopped after block 3 skips the other blocks
    tp = da[(da["search"] == "tpe") & (da["model"] == rc_model)]
    saved = float(tp["pruned"].sum()) * (nb - 1) / nb / max(len(tp), 1)
    rf.add_trace(go.Bar(y=["TPE", "random" if has_rnd else "no stopping"], x=[100 * (1 - saved), 100], orientation="h", marker_color=[T_DEF, theme.INK_MUTED], showlegend=False,
                        text=[f"{100 * (1 - saved):.0f}%", "100%"], textposition="inside", insidetextanchor="middle", textangle=0, textfont=dict(color="white", size=14), hoverinfo="skip"), row=2, col=3)
    if saved > 0:
        rf.add_trace(go.Bar(y=["TPE"], x=[100 * saved], orientation="h", marker=dict(color=T_STOP), showlegend=False, text=[f"{100 * saved:.0f}%<br>saved"],
                            textposition="inside", insidetextanchor="middle", textangle=0, constraintext="none", textfont=dict(color="white", size=12), hoverinfo="skip"), row=2, col=3)
    theme.style_fig(rf, height=516, legend=False, barmode="stack")
    M1, M2, R0, R1 = (0.105, 0.355), (0.375, 0.625), 0.745, 0.975                         # where the two maps and the result charts sit across the figure
    for i in (1, 2):
        for j, (d0, d1) in ((1, M1), (2, M2)):
            rf.update_xaxes(domain=[d0, d1], tickvals=list(range(len(va))), ticktext=[_std(v).replace(" (", "<br>(") if ka in RULE_SETTINGS else v for v in va], tickangle=0,
                            range=[-0.6, len(va) - 0.4], showgrid=False, zeroline=False, showticklabels=(i == 2), tickfont=dict(size=11), row=i, col=j)
            rf.update_yaxes(tickvals=list(range(len(vb))), ticktext=[_std(v).replace(" (", "<br>(") if kb in RULE_SETTINGS else v for v in vb], range=[-0.6, len(vb) - 0.4],
                            showgrid=False, zeroline=False, showticklabels=(j == 1), tickfont=dict(size=11), row=i, col=j)
        rf.update_xaxes(domain=[R0, R1], row=i, col=3)
    for j in (1, 2):
        rf.update_xaxes(title_text=ml_data.setting_label(ka), title_font=dict(size=11, color=theme.INK_SECONDARY), title_standoff=4, row=2, col=j)
    for i in (1, 2):
        rf.update_yaxes(title_text=ml_data.setting_label(kb), title_font=dict(size=11, color=theme.INK_SECONDARY), title_standoff=2, row=i, col=1)
    rf.update_xaxes(range=[0, n_t - 1], tickvals=[0, n_t - 1], showgrid=False, row=1, col=3)
    rf.update_yaxes(range=[min(v for _, v in ends_.values()) - 0.02, 1.01], showticklabels=True, tickformat='.2f', nticks=4, tickfont=dict(size=10), row=1, col=3)
    rf.update_xaxes(range=[0, 104], showticklabels=False, showgrid=False, row=2, col=3)
    rf.update_yaxes(showgrid=False, tickfont=dict(size=13), automargin=True, row=2, col=3)
    rf.update_layout(margin=dict(t=88, b=76, l=10, r=10))
    rf.layout.shapes = tuple(rf.layout.shapes) + tuple(                                  # two white cards: the two searches, and the verdict
        dict(type="rect", xref="paper", yref="paper", x0=x0_, x1=x1_, y0=-0.19, y1=1.165, fillcolor="#ffffff", line=dict(color="#e3e1d9", width=1), layer="below")
        for x0_, x1_ in ((-0.004, 0.652), (0.662, 1.004)))
    dom = {(1, 1): "x domain", (1, 2): "x2 domain", (1, 3): "x3 domain", (2, 1): "x4 domain", (2, 2): "x5 domain", (2, 3): "x6 domain"}
    # column titles, one header per search, the titles of the verdict
    for (i, j), title in (((1, 1), "First 10 trials"), ((1, 2), f"The other {n_t - 1 - n_early} trials"), ((1, 3), "Result: best score so far")):
        rf.add_annotation(x=0.5, y=1.0, xref=dom[(i, j)], yref="paper", yshift=30, yanchor="bottom", showarrow=False, text=f"<b>{title}</b>", font=dict(size=13, color=theme.INK_PRIMARY))
    rf.add_annotation(x=0.5, y=0.42, xref=dom[(2, 3)], yref="paper", yshift=6, yanchor="bottom", showarrow=False, text="<b>Work done</b>", font=dict(size=13, color=theme.INK_PRIMARY))
    for ypos, ysh, lab in ((1.0, 6, "<b style='font-size:13px;color:#0b0b0b'>Random control:</b> picks cells blindly and never stops"),
                           (0.42, 6, "<b style='font-size:13px;color:#0b0b0b'>TPE:</b> guesses for 10 trials, then learns from the scores and stops weak trials")):
        rf.add_annotation(x=0.012, y=ypos, xref="paper", yref="paper", yshift=ysh, xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=12, color=theme.INK_SECONDARY), text=lab)
    # end labels of the two lines
    rf.add_annotation(x=ends_["tpe"][0], y=ends_["tpe"][1], xref="x3", yref="y3", xanchor="right", yanchor="top", yshift=-8, showarrow=False, font=dict(size=12, color=theme.INK_PRIMARY),
                      text=f"TPE <b>{ends_['tpe'][1]:.3f}</b>")
    if has_rnd:
        rf.add_annotation(x=ends_["random"][0], y=ends_["random"][1], xref="x3", yref="y3", xanchor="right", yanchor="bottom", yshift=8, showarrow=False, font=dict(size=12, color=theme.INK_MUTED),
                          text=f"Random <b>{ends_['random'][1]:.3f}</b>")
    else:                                                                                # no control for this target: say so on the empty maps
        rf.add_annotation(x=0.365, y=0.79, xref="paper", yref="paper", showarrow=False, align="center", font=dict(size=13, color=theme.INK_SECONDARY), bgcolor="rgba(255,255,255,0.92)",
                          bordercolor="#e3e1d9", borderwidth=1, borderpad=8, text="<b>No random control for this target yet</b>")
    rf.add_annotation(x=0, y=1.0, xref="x3", yref="y3", xanchor="left", yanchor="bottom", yshift=3, showarrow=False, font=dict(size=11, color=theme.INK_MUTED), text="Defaults = 1.000")
    rf.add_annotation(x=0, y=1.0, yshift=70, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=12, color=theme.INK_SECONDARY), align="left",
                      text=f"<span style='color:{GOLD}'>■</span> Good combination &nbsp; <span style='color:{T_DONE}'>●</span> Trial &nbsp; <span style='color:{T_STOP}'>✕</span> Stopped early &nbsp; "
                           f"<span style='color:{T_WIN}'>★</span> Best found &nbsp;&nbsp; <span style='color:{theme.INK_MUTED}'>Each cell is one combination of the axes.</span>")
    st.plotly_chart(rf, width="stretch", config={"displayModeBar": False})
    if rc_model == "decision_tree":
        st.caption("This algorithm's grid is small, so both searches try every cell.")

    # ---- the formulas of this algorithm's learning rules (row 1) and the scores that judge a forecast (row 2)
    rule_raw = [str(v) for k_, vs in sp_.items() if k_ in RULE_SETTINGS for v in vs]
    fixed = not rule_raw                                                                # trees and forests: squared error only, nothing to choose
    rule_raw = rule_raw or ["squared_error"]
    CARD = {"squared error (L2)": ("Squared error (L2)", r"e^{2}", "900", "Follows spikes: large misses dominate"),
            "absolute error (L1)": ("Absolute error (L1)", r"\left| e \right|", "30", "Follows the typical day"),
            "Huber": ("Huber", r"\frac{1}{2}e^{2}\ \text{if}\ \left| e \right|\le\delta,\ \text{else}\ \delta\left(\left| e \right|-\frac{1}{2}\delta\right)", "29.5", "Squared if small, absolute if large (delta equals 1 here)"),
            "pseudo-Huber": ("Pseudo-Huber", r"\delta^{2}\left(\sqrt{1+\left(\frac{e}{\delta}\right)^{2}}-1\right)", "29.0", "A smooth Huber (delta equals 1 here)")}
    SCORES = [("SSE", r"\sum_{i=1}^{n} e_i^{2}", "905", "Sum of squared errors"), ("MSE", r"\frac{1}{n}\sum_{i=1}^{n} e_i^{2}", "301.7", "Mean squared error"),
              ("RMSE", r"\sqrt{\mathrm{MSE}}", "17.4", "Back in EUR/MWh"),
              ("MAE", r"\frac{1}{n}\sum_{i=1}^{n} \left| e_i \right|", "11", "The score the tuning compares settings by")]
    # two thin rows of chips, for reference only: the learning rules of this algorithm, and the scores
    FONT = "'Cambria Math', 'STIX Two Math', 'Times New Roman', serif"
    CHIP = "display:inline-flex; align-items:baseline; gap:0.45rem; background:#fff; border:1px solid #e3e1d9; border-radius:999px; padding:0.18rem 0.8rem; font-size:0.85rem; white-space:nowrap;"
    LAB = f"font-size:0.66rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600; min-width:5.2rem;"
    chips = "".join(
        f"<span style='{CHIP}' title='{raw}: {note}'><b style='color:{theme.INK_PRIMARY};'>{CARD[OBJECTIVE_NAME.get(raw, raw)][0]}</b>"
        f"<span style='color:{theme.INK_PRIMARY};'>{M(CARD[OBJECTIVE_NAME.get(raw, raw)][1])}</span>"
        f"</span>"
        for raw, note in ((r_, CARD[OBJECTIVE_NAME.get(r_, r_)][3]) for r_ in rule_raw))
    if fixed:
        chips += f"<span style='color:{theme.INK_MUTED}; font-size:0.72rem;'>Trees and forests always split on squared error</span>"
    score_chips = "".join(
        f"<span style='{CHIP}{'background:#fff7e6; border-color:#eda100;' if n == 'MAE' else ''}' title='{d}'><b style='color:{theme.INK_PRIMARY};'>{n}</b>"
        f"<span style='color:{theme.INK_PRIMARY};'>{M(f)}</span></span>" for n, f, v, d in SCORES)
    st.markdown(f"<div style='display:flex; flex-direction:column; gap:0.35rem; margin:0.1rem 0 0.6rem;'>"
                f"<div style='display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center;'><span style='{LAB}'>{ml_data.model_label(rc_model)} rules</span>{chips}</div>"
                f"<div style='display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center;'><span style='{LAB}'>scores</span>{score_chips}"
                f"<span style='color:{theme.INK_MUTED}; font-size:0.72rem;'>{M(r"e=\text{actual}-\text{forecast}")}; MAE is the tuning's score</span></div></div>", unsafe_allow_html=True)

io = ml_data.target_evidence("inner_vs_outer", target, "inner_vs_outer_tpe")
imp = ml_data.target_evidence("importance", target, "importance_random_search")

# ---- 5. what was searched, and what did it find
_section(f"What was searched and what was found <span style='font-weight:400; color:{theme.INK_MUTED};'>— {ml_data.model_label(model)}</span>", num="⑤")
from collections import Counter
import ast
space = cfg["space"].get(model, {})
_lg = tlog[(tlog["target"] == target) & (tlog["model"] == model)]
_nr = {k_: int((_lg["search"] == k_).sum()) for k_ in ("tpe", "random")}


def _chosen(key):
    cnt = {k_: Counter() for k_ in space}
    for pr_ in _lg[_lg["search"] == key]["params"]:
        try:
            d_ = ast.literal_eval(pr_) if isinstance(pr_, str) else dict(pr_)
        except (ValueError, SyntaxError):
            d_ = {}
        for k_, v_ in d_.items():
            if k_ in cnt:
                cnt[k_][str(v_)] += 1
    return cnt


_RS = {"squared error (L2)": "L2", "absolute error (L1)": "L1", "Huber": "Huber", "pseudo-Huber": "pseudo-Huber"}
_opt = lambda k_, v_: _RS.get(_std(v_), _std(v_)) if k_ in RULE_SETTINGS else str(v_)                                  # noqa: E731
_ch = {"tpe": _chosen("tpe"), "random": _chosen("random")}
_eta = dict(zip(imp[imp["model"] == model]["setting"], imp[imp["model"] == model]["eta2"])) if not imp.empty else {}
_keys5 = list(space.keys())
_n5 = len(_keys5)
_pitch = 1.25
cf = go.Figure()
cf.add_trace(go.Scatter(x=[0, 1010], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
_X0, _DX = 250, 100                                                                  # where the first option sits, and the distance between options
for i_, k_ in enumerate(_keys5):
    y_ = i_ * _pitch
    if i_:
        cf.add_shape(type="line", x0=10, x1=1000, y0=y_ - _pitch / 2, y1=y_ - _pitch / 2, line=dict(color="#eeede8", width=1))
    cf.add_annotation(x=14, y=y_, xanchor="left", showarrow=False, text=f"<b>{ml_data.setting_label(k_)}</b>", font=dict(size=12, color=theme.INK_PRIMARY))
    opts = [str(v_) for v_ in space[k_]]
    cf.add_trace(go.Scatter(x=[_X0 + j_ * _DX for j_ in range(len(opts))], y=[y_ + 0.5] * len(opts), mode="text", text=[_opt(k_, v_) for v_ in opts], textfont=dict(size=10, color=MUTED),
                            showlegend=False, hoverinfo="skip"))
    for key_, sym_, col_, dx_ in (("tpe", "circle", "#2b2a28", -13 if _nr["random"] else 0), ("random", "circle-open", theme.INK_MUTED, 13)):
        if not _nr[key_]:
            continue
        xs_, ys_, sz_, tx_ = [], [], [], []
        for j_, v_ in enumerate(opts):
            n_ = _ch[key_][k_].get(v_, 0)
            if n_:
                xs_.append(_X0 + j_ * _DX + dx_)
                ys_.append(y_)
                sz_.append(7 + 4.5 * n_)
                tx_.append(f"{'TPE' if key_ == 'tpe' else 'random control'} chose {_opt(k_, v_)} in {n_} of {_nr[key_]} rounds")
        cf.add_trace(go.Scatter(x=xs_, y=ys_, mode="markers", showlegend=False, text=tx_, hoverinfo="text", opacity=1,
                                marker=dict(symbol=sym_, size=sz_, color=col_, line=dict(color="white" if sym_ == "circle" else col_, width=1.5 if sym_ == "circle" else 2))))
    e_ = _eta.get(k_)
    if e_ is not None:
        w_ = min(e_ / 0.5, 1.0) * 230
        cf.add_shape(type="rect", x0=730, x1=730 + w_, y0=y_ - 0.22, y1=y_ + 0.22, fillcolor=T_DONE, line_width=0)
        cf.add_annotation(x=730 + w_ + 8, y=y_, xanchor="left", showarrow=False, text=f"<b>{100 * e_:.0f}%</b>", font=dict(size=11, color=theme.INK_SECONDARY))
cf.add_annotation(x=14, y=-0.95, xanchor="left", showarrow=False, text="Setting", font=dict(size=11, color=MUTED))
cf.add_annotation(x=_X0, y=-0.95, xanchor="left", showarrow=False,
                  text=f"Options: <span style='color:{T_DEF}'>●</span> TPE ({_nr['tpe']} rounds)" + (f" &nbsp; <span style='color:{theme.INK_MUTED}'>○</span> Random control ({_nr['random']} rounds)" if _nr["random"] else ""), font=dict(size=11, color=MUTED))
cf.add_annotation(x=730, y=-0.95, xanchor="left", showarrow=False, text="How much of the score it explains", font=dict(size=11, color=MUTED))
theme.style_fig(cf, height=int((_n5 * _pitch + 1.9) * 38 + 20), xaxis_title=None, legend=False)
cf.update_yaxes(autorange=False, range=[(_n5 - 1) * _pitch + 0.85, -1.4], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
cf.update_xaxes(range=[0, 1010], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
cf.update_layout(margin=dict(t=4, b=4, l=6, r=6), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
cf.layout.shapes = tuple(sh for sh in cf.layout.shapes if not (sh.line.color == theme.AXIS_LINE and sh.x0 == sh.x1))
st.plotly_chart(cf, width="stretch", config={"displayModeBar": False})
n_comb = math.prod(len(v) for v in space.values())
_n_trials = int(cfg["search"]["n_trials"])
_what = (f"Every combination is tried: {n_comb} + the defaults = {n_comb + 1} trials, fewer than the {_n_trials} allowed" if n_comb < _n_trials else
         f"{_n_trials} trials look at {100 * _n_trials / n_comb:.1f}% of them")
_line(f"{M(r' \times '.join(str(len(v)) for v in space.values()) + rf'={n_comb}')} combinations (the options of every setting, multiplied). {_what}.")

# ---- 6. it changes from round to round, and the gain afterwards is smaller
_section(f"The choice changes from round to round, and the gain afterwards is smaller <span style='font-weight:400; color:{theme.INK_MUTED};'>— {ml_data.model_label(model)}</span>", num="⑥")
_mini("What the search chose, round by round")
lg = tlog[(tlog["target"] == target) & (tlog["search"] == "tpe") & (tlog["model"] == model)].copy()
if len(lg):
    import ast
    _keys = list(space.keys())                                                         # the settings of this algorithm are the columns, every round one row
    _SHORT = {"squared error (L2)": "L2", "absolute error (L1)": "L1", "Huber": "Huber", "pseudo-Huber": "pseudo-Huber"}
    _grid = f"display:grid; grid-template-columns:5.3rem 5.2rem repeat({len(_keys)}, minmax(0, 1fr)); align-items:center; column-gap:0.5rem;"
    _head_ = (f"<div style='{_grid} padding:0.55rem 0 0.35rem; border-bottom:1px solid #e3e1d9; font-size:0.7rem; color:{theme.INK_MUTED}; font-weight:600; line-height:1.2;'>"
              "<span>Round</span><span>Gain</span>" + "".join(f"<span style='text-align:center;'>{ml_data.setting_label(k_)}</span>" for k_ in _keys) + "</div>")
    _rows = ""
    for i_, r_ in enumerate(lg.itertuples()):
        try:
            d_ = ast.literal_eval(r_.params) if isinstance(r_.params, str) else dict(r_.params)
        except (ValueError, SyntaxError):
            d_ = {}
        gain_ = 100 * (1 - r_.score / r_.default_score)
        cells = "".join(f"<span style='text-align:center; font-size:0.84rem; color:{theme.INK_PRIMARY}; white-space:nowrap;'>"
                        f"<span style='background:#f4f3ef; border-radius:999px; padding:0.1rem 0.5rem;'>{(_SHORT.get(_std(d_[k_]), d_[k_]) if k_ in RULE_SETTINGS else d_[k_]) if k_ in d_ else '–'}</span></span>" for k_ in _keys)
        _rows += (f"<div style='{_grid} height:2.9rem; {'' if i_ == len(lg) - 1 else 'border-bottom:1px solid #eeede8;'}'>"
                  f"<span style='font-weight:700; font-size:0.84rem; color:{theme.INK_PRIMARY};'>{pd.Timestamp(r_.tuned_at):%d-%m-%Y}</span>"
                  f"<span style='line-height:1.2;'><b style='color:{theme.INK_PRIMARY if gain_ >= 0 else T_STOP}; font-size:0.88rem;'>{gain_:+.1f}%</b>"
                  f"<br><span style='font-size:0.68rem; color:{theme.INK_MUTED};'>{r_.default_score:.2f} → {r_.score:.2f}</span></span>{cells}</div>")
    st.markdown(f"<div style='background:#fff; border:1px solid #e3e1d9; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{_head_}{_rows}</div>", unsafe_allow_html=True)

    # how one gain comes about: the three blocks, their mean, the formula (the numbers of the round picked here)
    _mini("Gain calculation")
    _rd = [f"{pd.Timestamp(x_):%d-%m-%Y}" for x_ in lg["tuned_at"]]
    _l, _r = st.columns([5, 4], gap="medium")
    with _r:
        _rpick = st.selectbox("Round", _rd, index=len(_rd) - 1, key="ml6_round")
    _tp = trials[(trials["target"] == target) & (trials["search"] == "tpe") & (trials["model"] == model) & (pd.to_datetime(trials["tuned_at"]) == pd.to_datetime(_rpick, format="%d-%m-%Y"))]
    _bc = [c_ for c_ in ("block1", "block2", "block3") if c_ in _tp and _tp[c_].notna().any()]
    _td, _tc = _tp[_tp["is_default"]], _tp[_tp["selected"]]
    if len(_td) and len(_tc) and _bc:
        _bd = [float(_td.iloc[0][c_]) for c_ in _bc]
        _bs = [float(_tc.iloc[0][c_]) for c_ in _bc]
        _md, _ms = sum(_bd) / len(_bd), sum(_bs) / len(_bs)
        _gn = 100 * (1 - _ms / _md)
        _GREY = "#b9b7ae"
        _cats = [f"Block {j_ + 1}" for j_ in range(len(_bc))] + [" ", "Mean"]
        bf = go.Figure()
        for nm_, ys_, col_ in (("Defaults", _bd + [None, _md], _GREY), ("Chosen setting", _bs + [None, _ms], theme.VIOLET)):
            bf.add_trace(go.Bar(x=_cats, y=ys_, name=nm_, marker_color=col_, text=[f"{v_:.2f}" if v_ is not None else "" for v_ in ys_], textposition="outside", cliponaxis=False,
                                textfont=dict(size=11, color=theme.INK_SECONDARY), hoverinfo="skip"))
        theme.style_fig(bf, height=260, yaxis_title="MAE", barmode="group")
        bf.update_yaxes(range=[0, max(_bd + _bs) * 1.18])
        bf.update_xaxes(tickangle=0)
        bf.update_layout(legend=dict(orientation="h", x=0, y=1.14, xanchor="left"), margin=dict(t=30, b=30, l=50, r=10))
        with _l:
            st.plotly_chart(bf, width="stretch", config={"displayModeBar": False})
        _sumtxt = lambda vs_: "+".join(f"{v_:.2f}" for v_ in vs_)                                                     # noqa: E731
        _k = len(_bc)
        _row = lambda dot_, name_, vs_, m_, c_: (f"<div style='display:grid; grid-template-columns:5.2rem 1fr; align-items:baseline; padding:0.35rem 0; border-bottom:1px solid #eeede8;'>"   # noqa: E731
                                                 f"<span style='color:{theme.INK_PRIMARY}; font-weight:600;'><span style='color:{dot_};'>●</span> {name_}</span>"
                                                 f"<span>{M(rf'\frac{{{_sumtxt(vs_)}}}{{{_k}}}=')} <b style='font-size:1.02rem; color:{c_};'>{m_:.2f}</b></span></div>")
        with _r:
            st.markdown(
                f"<div style='background:#fff; border:1px solid #e3e1d9; border-radius:12px; padding:0.5rem 1.1rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); font-size:0.86rem; color:{theme.INK_SECONDARY};'>"
                f"<div style='font-size:0.72rem; font-weight:600; color:{theme.INK_MUTED}; padding:0.2rem 0;'>① The mean of the blocks</div>"
                + _row(_GREY, "Defaults", _bd, _md, theme.INK_PRIMARY) + _row(theme.VIOLET, "Chosen", _bs, _ms, theme.VIOLET)
                + f"<div style='font-size:0.72rem; font-weight:600; color:{theme.INK_MUTED}; margin-top:0.55rem;'>② The gain</div>"
                f"<div style='display:flex; align-items:baseline; justify-content:space-between; gap:0.6rem; margin-top:0.1rem;'><span>{M(rf'\frac{{{_md:.2f}-{_ms:.2f}}}{{{_md:.2f}}}=')}</span>"
                f"<b style='font-size:1.7rem; line-height:1.1; color:{theme.INK_PRIMARY if _gn >= 0 else T_STOP};'>{_gn:+.1f}%</b></div></div>", unsafe_allow_html=True)

if io.empty:
    _line(f"The gain chart is not built for {ml_data.label(target)} yet.")
_ir = io[io["model"] == model].sort_values("tuned_at") if not io.empty else io
if len(_ir):
    # the rounds of this algorithm: the gain on the blocks it was chosen on, and the gain of the same setting afterwards; the last row is the mean (the bars below)
    _lab = [f"{pd.Timestamp(x_):%d-%m-%Y}" for x_ in _ir["tuned_at"]] + ["Mean of the rounds"]
    _gi = list(100 * _ir["inner_gain"]) + [100 * float(_ir["inner_gain"].mean())]
    _go = list(100 * _ir["outer_gain"]) + [100 * float(_ir["outer_gain"].mean())]
    _yy = list(range(len(_ir))) + [len(_ir) + 0.45]
    sf = go.Figure()
    sf.add_vline(x=0, line=dict(color=theme.INK_MUTED, width=1, dash="dot"))
    for y_, a_, b_ in zip(_yy, _gi, _go):
        sf.add_shape(type="line", x0=a_, x1=b_, y0=y_, y1=y_, line=dict(color="#c9c7bf", width=2.5), layer="below")
    sf.add_shape(type="line", xref="paper", x0=0, x1=1, y0=len(_ir) - 0.25, y1=len(_ir) - 0.25, line=dict(color="#e3e1d9", width=1))
    for nm_, xs_, col_ in (("On the tuning blocks", _gi, theme.VIOLET), ("On the days that followed", _go, theme.ORANGE)):
        sf.add_trace(go.Scatter(x=xs_, y=_yy, mode="markers+text", name=nm_, text=[f"{v_:+.1f}%" for v_ in xs_], textposition=[("middle right" if (g_ >= o_) == (col_ == theme.VIOLET) else "middle left") for g_, o_ in zip(_gi, _go)],
                                textfont=dict(size=10, color=col_), cliponaxis=False, hoverinfo="skip",
                                marker=dict(size=[12] * len(_ir) + [16], color=col_, line=dict(color="white", width=1.5))))
    _mini(f"Round by round — {ml_data.model_label(model)}")
    _cl, _cr = st.columns([5, 4], gap="medium")
    theme.style_fig(sf, height=int(42 * len(_yy)) + 78, xaxis_title="MAE gain over the defaults (%)", legend=False)
    sf.update_xaxes(zeroline=False, showgrid=False, range=[min(0.0, min(_gi + _go)) - 2.6, max(_gi + _go) + 2.6])
    sf.update_yaxes(tickmode="array", tickvals=_yy, ticktext=[f"<b>{x_}</b>" if x_.startswith("Mean") else x_ for x_ in _lab], autorange="reversed", showgrid=False, zeroline=False, fixedrange=True)
    sf.update_layout(margin=dict(t=34, b=40, l=130, r=40))
    sf.add_annotation(xref="paper", yref="paper", x=0, y=1, xanchor="left", yanchor="bottom", yshift=6, showarrow=False, font=dict(size=12, color=theme.INK_SECONDARY),
                      text=f"<span style='color:{theme.VIOLET}'>●</span> On the tuning blocks &nbsp;&nbsp;&nbsp; <span style='color:{theme.ORANGE}'>●</span> On the days that followed")
    with _cl:
        st.plotly_chart(sf, width="stretch", config={"displayModeBar": False})
    _fmt = lambda v_: f"{v_:.2f}" if v_ >= 0 else f"(-{abs(v_):.2f})"                                                    # noqa: E731
    _calc = lambda vs_: rf"\frac{{{'+'.join(_fmt(v_) for v_ in vs_)}}}{{{len(vs_)}}}="                                      # noqa: E731
    _mi, _mo = _gi[-1], _go[-1]
    _row2 = lambda col_, name_, vs_, m_: (f"<div style='padding:0.45rem 0; border-bottom:1px solid #eeede8;'><div style='font-weight:600; color:{theme.INK_PRIMARY};'><span style='color:{col_};'>●</span> {name_}</div>"   # noqa: E731
                                          f"<div style='margin-top:0.15rem;'>{M(_calc(vs_))} <b style='font-size:1.05rem; color:{col_};'>{m_:.2f}</b></div></div>")
    with _cr:
        st.markdown(
            f"<div style='background:#fff; border:1px solid #e3e1d9; border-radius:12px; padding:0.5rem 1.1rem 0.9rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); font-size:0.9rem; color:{theme.INK_SECONDARY}; min-height:{int(42 * len(_yy)) + 40}px; display:flex; flex-direction:column; justify-content:space-between;'>"
            f"<div style='font-size:0.74rem; font-weight:600; color:{theme.INK_MUTED}; padding:0.2rem 0;'>① The mean of the {len(_ir)} rounds</div>"
            + _row2(theme.VIOLET, "On the tuning blocks", _gi[:-1], _mi) + _row2(theme.ORANGE, "On the days that followed", _go[:-1], _mo)
            + f"<div style='font-size:0.74rem; font-weight:600; color:{theme.INK_MUTED}; margin-top:0.2rem;'>② The gap between the two bars below</div>"
            f"<div style='margin-top:0.1rem;'>{M(rf'{_mi:.2f}-{_mo:.2f}=')} "
            f"<b style='font-size:1.5rem; line-height:1.1; color:{theme.INK_PRIMARY};'>{_mi - _mo:.2f} points lost</b></div></div>", unsafe_allow_html=True)

if not io.empty:
    _mini("The gain when choosing, against the gain afterwards", "choosing the best of 40 on the same blocks flatters it: the gain on the days that followed is smaller (selection optimism, measured)")
    g = io.groupby("model")[["inner_gain", "outer_gain"]].mean().reindex(ml_data.ordered_models(io["model"].unique())).reset_index()
    fig = go.Figure()
    _xl = [ml_data.model_label(x).replace(" ", "<br>", 1) for x in g["model"]]
    for nm_, col_, key_ in (("On the tuning blocks", theme.VIOLET, "inner_gain"), ("On the days that followed", theme.ORANGE, "outer_gain")):
        fig.add_trace(go.Bar(x=_xl, y=100 * g[key_], name=nm_, marker_color=col_, text=[f"{v_:.1f}%" for v_ in 100 * g[key_]], textposition="outside", cliponaxis=False,
                             textfont=dict(size=12, color=col_), hoverinfo="skip"))
    theme.style_fig(fig, height=330, yaxis_title="MAE gain over the defaults (%)", barmode="group")
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(range=[min(0.0, float(100 * g[["inner_gain", "outer_gain"]].min().min()) * 1.3), float(100 * g[["inner_gain", "outer_gain"]].max().max()) * 1.15])
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.1, xanchor="left", yanchor="bottom"), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---- 7. the outcome
_sum = ml_ui.cached(ml_data.tuning_summary)(trials, nb)
_section("The outcome", "what tuning gave, and the same comparison for every target", num="⑦")
_line(M(r"\text{Fits saved}=\frac{\text{trials stopped}\times\frac{2}{3}}{\text{all trials}}") + ". A stopped trial skips 2 of its 3 block fits.")
_hdr = ["Target", "TPE best", "Random best", "Difference", "Fits saved", "Gain when choosing", "Gain afterwards"]
_g7 = "display:grid; grid-template-columns:minmax(9rem, 1.5fr) repeat(6, minmax(0, 1fr)); align-items:center; column-gap:0.6rem;"
_f = lambda v, fmt: "–" if pd.isna(v) else fmt.format(v)                                                              # noqa: E731
_d4 = lambda a_, b_: float("nan") if (pd.isna(a_) or pd.isna(b_)) else round(a_, 4) - round(b_, 4)                      # the difference of the two numbers as shown (4 decimals), so the table adds up
_fd = lambda v: "–" if pd.isna(v) else ("0.0000" if abs(v) < 5e-5 else f"{v:+.4f}")                                    # noqa: E731
_srows = ""
for i_, r_ in enumerate(_sum.itertuples()):
    allrow = r_.target == "all"
    cells = [("All targets" if allrow else ml_data.label(r_.target)), _f(r_.tpe, "{:.4f}"), _f(r_.random, "{:.4f}"), _fd(_d4(r_.tpe, r_.random)),
             _f(100 * r_.saved, "{:.0f}%"), _f(r_.inner, "{:.1f}%"), _f(r_.outer, "{:.1f}%")]
    _srows += (f"<div style='{_g7} height:2.5rem; font-size:0.86rem; {'border-top:1px solid #e3e1d9; background:#f8f7f3; font-weight:700;' if allrow else 'border-bottom:1px solid #eeede8;'}'>"
               + "".join(f"<span style='{'' if j == 0 else 'text-align:center;'} color:{theme.INK_PRIMARY if (j == 0 or c != '–') else theme.INK_MUTED};'>{c}</span>" for j, c in enumerate(cells)) + "</div>")
_shead = (f"<div style='{_g7} padding:0.6rem 0 0.4rem; border-bottom:1px solid #e3e1d9; font-size:0.72rem; color:{theme.INK_MUTED}; font-weight:600; line-height:1.2;'>"
          + "".join(f"<span style='{'' if j == 0 else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(_hdr)) + "</div>")
_mini("TPE against the random control, target by target")
st.markdown(f"<div style='background:#fff; border:1px solid #e3e1d9; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{_shead}{_srows}</div>", unsafe_allow_html=True)


# the calculation behind one row of the table: the five algorithms, then their mean (= the table row)
_mini("Calculation of one row")
_rt = [t_ for t_ in ml_data.TARGETS if t_ in set(trials["target"])]
_rc, _ = st.columns([1, 3])
with _rc:
    _rsel = st.selectbox("Row", _rt, index=_rt.index(target) if target in _rt else 0, format_func=ml_data.label, label_visibility="collapsed")
_bd = ml_ui.cached(ml_data.tuning_breakdown)(trials, _rsel, nb)
if len(_bd):
    _bd = _bd.assign(difference=_bd["tpe"] - _bd["random"])
    _mn = _bd.mean()
    _cl = lambda r_, fmt_: "–" if pd.isna(r_) else fmt_.format(r_)                                                      # noqa: E731
    _cells = lambda r_: [_cl(r_["tpe"], "{:.4f}"), _cl(r_["random"], "{:.4f}"), _fd(_d4(r_["tpe"], r_["random"])),
                         _cl(100 * r_["saved"], "{:.0f}%"), _cl(r_["inner"], "{:.1f}%"), _cl(r_["outer"], "{:.1f}%")]       # noqa: E731
    _crow = lambda name_, cs_, mean_: (f"<div style='{_g7} height:2.4rem; font-size:0.86rem; " + ("border-top:1px solid #e3e1d9; background:#f8f7f3; font-weight:700;" if mean_ else "border-bottom:1px solid #eeede8;") + "'>"   # noqa: E731
                                       + f"<span style='color:{theme.INK_PRIMARY};'>{name_}</span>"
                                       + "".join(f"<span style='text-align:center; color:{theme.INK_PRIMARY if c_ != '–' else theme.INK_MUTED};'>{c_}</span>" for c_ in cs_) + "</div>")
    _body = "".join(_crow(ml_data.model_label(m_), _cells(r_), False) for m_, r_ in _bd.iterrows()) + _crow(f"Mean of the {len(_bd)}", _cells(_mn), True)
    _hd = [h_ if j_ else "Algorithm" for j_, h_ in enumerate(_hdr)]
    _hh = (f"<div style='{_g7} padding:0.6rem 0 0.4rem; border-bottom:1px solid #e3e1d9; font-size:0.72rem; color:{theme.INK_MUTED}; font-weight:600; line-height:1.2;'>"
           + "".join(f"<span style='{'' if j_ == 0 else 'text-align:center;'}'>{h_}</span>" for j_, h_ in enumerate(_hd)) + "</div>")
    st.markdown(f"<div style='background:#fff; border:1px solid #e3e1d9; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{_hh}{_body}</div>", unsafe_allow_html=True)
    _line("<b>Mean</b> = the row of the table above. " + M(r"\text{Difference}=\text{TPE best}-\text{random best}") + ". <b>Fits saved</b> leaves out the decision tree, because all its combinations are tried.")
