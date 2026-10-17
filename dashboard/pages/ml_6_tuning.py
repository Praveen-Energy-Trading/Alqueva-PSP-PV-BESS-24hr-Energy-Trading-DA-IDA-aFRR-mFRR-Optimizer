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


def _section(title: str, hint: str = "") -> None:
    """A section heading: a bold title and, under it, one muted line on how to read the section."""
    st.markdown(f"<div style='margin:1.7rem 0 0.45rem;'><div style='font-size:1.55rem; font-weight:700; color:{theme.INK_PRIMARY}; line-height:1.25;'>{title}</div>"
                + (f"<div style='font-size:0.88rem; color:{theme.INK_MUTED}; margin-top:0.2rem;'>{hint}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _mini(title: str, hint: str = "") -> None:
    """The title of a chart that shares a row with another one."""
    st.markdown(f"<div style='margin:0.2rem 0 0.3rem;'><div style='font-size:1.02rem; font-weight:700; color:{theme.INK_PRIMARY};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{theme.INK_MUTED}; margin-top:0.1rem; line-height:1.35;'>{hint}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


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
                   text=f"<b style='font-size:17px;color:{NEUTRAL}'>{num}</b>&nbsp; <b>{title}</b>" + (f"{'<br>' if two else ' &nbsp;'}<span style='font-size:12px;color:{MUTED}'>{sub_}</span>" if sub_ else "")))


def _note(fig, x, y, text, size=12, color=INK, anchor="center", **kw):
    _annot(x=x, y=y, xanchor=anchor, showarrow=False, text=text, font=dict(size=size, color=color), **kw)


def _arrow(fig, x0, y0, x1, y1, color=MUTED):
    _annot(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=color, text="")


body = win - nb * L                                                                  # the days before the tuning blocks
xw = win * DAY                                                                       # where the window ends, in units
ft = go.Figure()
ft.add_trace(go.Scatter(x=[0, X_END], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
legend = (f"<span style='color:{BLUE}'>■</span> learns from &nbsp;&nbsp; <span style='color:{VIOLET}'>■</span> scored on ({nb} blocks of {L})<br>"
          f"<span style='color:{ORANGE}'>■</span> the refit's forecast &nbsp;&nbsp; <span style='color:#c9c7be'>■</span> not used by that round")
_head(-1.95, "①", "Tuning only looks inside the refit's own window",
      f"{ml_data.label(target)} · latest tuning round" + (f", window ending {pd.Timestamp(win_end):%d-%m-%Y}" if win_end is not None else "") + (f" · expanding window: all {win} days it has" if expanding else f" · rolling {win}-day window"), two=True)
# ---- 1. the window of one refit: training, then the tuning days cut into blocks
rh = 0.8
_pill(ft, 0, body * DAY, 0, rh, BLUE, 0.62)
_pill(ft, body * DAY + 0.6, xw, 0, rh, VIOLET, 0.9)
for k in range(1, nb):                                                               # the block boundaries inside the tuning days
    _shape(type="line", x0=(body + k * L) * DAY, x1=(body + k * L) * DAY, y0=0.08, y1=rh - 0.08, line=dict(color="white", width=2))
_pill(ft, xw + 1, xw + 1 + 30 * DAY, 0, rh, ORANGE, 1.0)
_note(ft, 8, rh / 2, f"<b>One refit</b> &nbsp;·&nbsp; {body} days of training", anchor="left", align="left")
for k in range(nb):
    _note(ft, (body + k * L + L / 2) * DAY, rh / 2, f"<b>{k + 1}</b>", color="white")
_note(ft, xw + 1 + 30 * DAY + 10, rh / 2, "forecast days", color=ORANGE, anchor="left")
# ---- 2. the tuning rounds: each block is predicted by a model that has seen only the days before it
for k in range(1, nb + 1):
    y0 = 1.55 + (k - 1) * 1.0
    b0 = body + (k - 1) * L                                                          # first day of block k
    tr_days = b0 - gap
    _pill(ft, 0, tr_days * DAY, y0, y0 + rh, BLUE, 0.62)
    _pill(ft, b0 * DAY, (b0 + L) * DAY, y0, y0 + rh, VIOLET, 0.9)
    if b0 + L < win:
        _pill(ft, (b0 + L) * DAY + 0.6, xw, y0, y0 + rh, GREY, 0.8)
    _note(ft, 8, y0 + rh / 2, f"<b>Round {k}</b> &nbsp;·&nbsp; trains on {tr_days} days", anchor="left", align="left")
    _note(ft, (b0 + L / 2) * DAY, y0 + rh / 2, f"<b>{k}</b>", color="white")
    _note(ft, xw + 36, y0 + rh / 2, f"predicts block {k}: <b>error {k}</b>" + ("<br>scored first: pruner looks here" if k == nb else ""), color=VIOLET, anchor="left", align="left")
ys = 1.55 + nb * 1.0 + 0.75
_note(ft, 8, ys + 0.25, legend, color=theme.INK_SECONDARY, anchor="left", yanchor="middle", align="left")    # what the colours mean, under the rounds
_note(ft, xw + 36, ys + 0.25, "<b>score</b> = average of<br>error " + ", ".join(str(k) for k in range(1, nb + 1)), size=13, color=VIOLET, anchor="left",
      align="left", bgcolor="white", bordercolor=VIOLET, borderwidth=1, borderpad=6)
# ---- 3. the days tuning never looks at, and where they start
ysn = ys + 1.25
_pill(ft, xw + 1, xw + 130, ysn, ysn + rh, ORANGE, 0.35)
_pill(ft, xw + 134, X_END, ysn, ysn + rh, RED, 0.3)
_note(ft, xw + 65, ysn + rh / 2, "backtest<br>validation", size=11)
_note(ft, xw + 134 + (X_END - xw - 134) / 2, ysn + rh / 2, "locked<br>test year", size=11, color=RED)
_note(ft, xw - 8, ysn + rh / 2, "<b>never looked at by the tuning</b> →", color=RED, anchor="right", align="right")
_shape(type="line", x0=xw + 0.5, x1=xw + 0.5, y0=-0.55, y1=ysn + rh + 0.1, line=dict(color="#a8a69d", width=1.2, dash="dot"))
_note(ft, xw - 6, -0.55, "the only days the tuning can use ←", size=11, color=MUTED, anchor="right")
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
_node(0, 150, "next trial")
_node(205, 365, "latest block", badges=(nb,))
_arrow(ft, 152, cy, 203, cy)
_arrow(ft, 367, cy, 383, cy)
dx, dh, dw = 545, 1.15, 160
_dia = f"M {dx - dw},{cy} L {dx},{cy - dh} L {dx + dw},{cy} L {dx},{cy + dh} Z"
_shape(type="path", fillcolor=T_STOP, opacity=0.13, line_width=0, path=_dia)
_shape(type="path", fillcolor="rgba(0,0,0,0)", line=dict(color=T_STOP, width=1.5), path=_dia)
_note(ft, dx, cy, "block 3: worse<br>than the median<br>of earlier trials?", size=11)
_arrow(ft, dx + dw + 2, cy, 738, cy)
_note(ft, dx + dw + 18, cy - 0.5, "no", size=11, color=MUTED)
_node(740, 915, "older blocks", badges=tuple(range(nb - 1, 0, -1)))
_arrow(ft, 917, cy, 930, cy)
oy = cy + 2.25
_arrow(ft, dx, cy + dh + 0.02, dx, oy - 0.45, color=T_STOP)
_note(ft, dx + 22, cy + dh + 0.28, "yes", size=11, color=MUTED, anchor="left")
_pill(ft, dx - 90, dx + 90, oy - 0.45, oy + 0.45, T_STOP, 0.9, r_px=6)
_note(ft, dx, oy, "<b>✕ stopped</b>", size=12, color="white")
_pill(ft, 930, X_END, cy - nh / 2, cy + nh / 2, T_DONE, 0.95, r_px=6)
_note(ft, (930 + X_END) / 2, cy, "<b>score</b>", size=12, color="white")
_arrow(ft, (930 + X_END) / 2, cy + nh / 2 + 0.02, (930 + X_END) / 2, oy - 0.45, color=T_WIN)
_pill(ft, 880, X_END, oy - 0.45, oy + 0.45, T_WIN, 0.95, r_px=6)
_note(ft, (880 + X_END) / 2, oy, "<b>best score</b>", size=12, color="white")
_pill(ft, 0, 150, oy - 0.45, oy + 0.45, T_DEF, 0.95, r_px=6)
_note(ft, 75, oy, "<b>trial 0</b>", size=12, color="white")
_note(ft, 165, oy, "defaults, never stopped", size=11, color=MUTED, anchor="left")

# ---- 5. the same budget for every algorithm: the trials of one real round, TPE and the random control
_card(yr - 0.7, oy + 0.8)
ydt = oy + 2.0
_head(ydt + 0.1, "③", f"The same budget for every algorithm: up to {n_t} trials each", f"{ml_data.label(target)}, its latest tuning round · every square is a trial that was kept ({len(trials):,} in all)", two=True)
_leg = " &nbsp;&nbsp; ".join(f"<span style='color:{c}'>■</span> {x}" for c, x in ((T_DEF, "defaults (trial 0)"), (T_DONE, "all 3 blocks scored"), (T_STOP, "stopped after block 3"), (T_WIN, "the winner"), ("#c9c7be", "trial not needed")))
_note(ft, 6, ydt + 1.1, _leg, size=11, color=theme.INK_SECONDARY, anchor="left")
xs0, xs1 = 130, 1000
cw = (xs1 - xs0) / n_t
bh, pitch = 0.42, 0.52
y = ydt + 1.75
n_pr_all = 0
combos_dt = math.prod(len(v) for v in cfg["space"].get("decision_tree", {}).values()) if cfg.get("space") else 0
for mdl in ml_data.ordered_models(sorted(set(trials["model"]))):
    for method, tag in (("tpe", ""), ("random", "random")):
        d = trials[(trials["target"] == target) & (trials["model"] == mdl) & (trials["search"] == method)]
        if d.empty:
            continue
        d = d[d["tuned_at"] == d["tuned_at"].max()].sort_values("trial")
        if method == "tpe":
            _note(ft, 0, y + bh / 2, f"<b>{ml_data.model_label(mdl)}</b>", size=12, anchor="left")
            n_pr_all += int(d["pruned"].sum())
        else:
            _note(ft, 0, y + bh / 2, "↳ random control", size=10, color=MUTED, anchor="left")
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
                col, mark = T_DONE, ""
            _pill(ft, x0, x0 + cw - 3, y, y + bh, col, 0.95, r_px=2)
            if mark:
                _note(ft, x0 + (cw - 3) / 2, y + bh / 2, f"<b>{mark}</b>", size=10, color="white")
        if mdl == "decision_tree" and method == "tpe" and len(used) < n_t:
            _note(ft, xs0 + (len(used) + n_t) / 2 * cw, y + bh / 2, f"whole grid ({combos_dt} combinations) already tried", size=10, color=MUTED)
        y += pitch
    y += 0.2
yd_end = y

# ---- 6. how often: a tuning round every 180 days, a refit every 30
_card(ydt - 0.7, yd_end + 0.05)
yt = yd_end + 1.15
_head(yt, "④", "When it happens", f"a tuning round every {retune} days, a refit every {refit}")
n_rounds, span_d = 4, 3 * retune
x0s, x1s = 130, 985
per_day = (x1s - x0s) / span_d
yb = yt + 1.05
for k in range(3):                                                                   # the settings of one round serve every refit until the next
    _pill(ft, x0s + k * retune * per_day, x0s + (k + 1) * retune * per_day - 3, yb, yb + 2.0, VIOLET, 0.09 + 0.05 * (k % 2), r_px=6)
for d_ in range(0, span_d, refit):
    _pill(ft, x0s + d_ * per_day + 1, x0s + d_ * per_day + 9, yb + 1.25, yb + 1.8, BLUE, 0.7, r_px=2)
for k in range(n_rounds):
    if k * retune <= span_d:
        _pill(ft, x0s + k * retune * per_day - 6, x0s + k * retune * per_day + 6, yb + 0.12, yb + 0.9, VIOLET, 0.95, r_px=3)
_note(ft, 0, yb + 0.5, "<b>tuning round</b>", size=12, color=VIOLET, anchor="left")
_note(ft, 0, yb + 1.52, "<b>refit</b>", size=12, color=BLUE, anchor="left")
_note(ft, x0s + 14, yb + 0.5, f"every {retune} days: a new search", size=11, color=VIOLET, anchor="left")
_note(ft, x0s + 14, yb + 1.0, f"{retune // refit} refits use its settings", size=11, color=MUTED, anchor="left")
_note(ft, x0s + 14, yb + 1.95, f"a refit every {refit} days", size=11, color=BLUE, anchor="left", yanchor="top")
_card(yt - 0.7, yb + 2.55)
end = yb + 2.85
theme.style_fig(ft, height=int((end + 2.85) * ROW + 12), xaxis_title=None, legend=False)
ft.update_yaxes(autorange=False, range=[end, -2.85], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
ft.update_xaxes(range=[CARD_X0 - 3, CARD_X1 + 3], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
ft.update_layout(margin=dict(t=2, b=2, l=10, r=10))
ft.layout.shapes = SH                                                                   # replace (not merge with) the axis line the theme added
ft.layout.annotations = AN
st.plotly_chart(ft, width="stretch", config={"displayModeBar": False})

tt = ml_ui.cached(ml_data.prepared_trials)(trials, target, "tpe")
models = ml_data.ordered_models(tt["model"].unique())

_section(f"Random control and TPE, from scratch · {ml_data.label(target)}")
model = st.selectbox("Algorithm (for this section and for the ones below)", models, index=models.index("lightgbm") if "lightgbm" in models else 0, format_func=ml_data.model_label, key="ml6_model")
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
                                            hovertemplate="score %{customdata:.3f} of the defaults<extra></extra>"), row=i, col=j)
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
    for (i, j), title in (((1, 1), "first 10 trials"), ((1, 2), f"the other {n_t - 1 - n_early} trials"), ((1, 3), "result: best score so far")):
        rf.add_annotation(x=0.5, y=1.0, xref=dom[(i, j)], yref="paper", yshift=30, yanchor="bottom", showarrow=False, text=f"<b>{title}</b>", font=dict(size=13, color=theme.INK_PRIMARY))
    rf.add_annotation(x=0.5, y=0.42, xref=dom[(2, 3)], yref="paper", yshift=6, yanchor="bottom", showarrow=False, text="<b>work done</b>", font=dict(size=13, color=theme.INK_PRIMARY))
    for ypos, ysh, lab in ((1.0, 6, "<b style='font-size:13px;color:#0b0b0b'>RANDOM CONTROL</b> &nbsp;picks cells blindly and never stops"),
                           (0.42, 6, "<b style='font-size:13px;color:#0b0b0b'>TPE</b> &nbsp;guesses for 10 trials, then learns from the scores and stops weak trials")):
        rf.add_annotation(x=0.012, y=ypos, xref="paper", yref="paper", yshift=ysh, xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=12, color=theme.INK_SECONDARY), text=lab)
    # end labels of the two lines
    rf.add_annotation(x=ends_["tpe"][0], y=ends_["tpe"][1], xref="x3", yref="y3", xanchor="right", yanchor="top", yshift=-8, showarrow=False, font=dict(size=12, color=theme.INK_PRIMARY),
                      text=f"TPE <b>{ends_['tpe'][1]:.3f}</b>")
    if has_rnd:
        rf.add_annotation(x=ends_["random"][0], y=ends_["random"][1], xref="x3", yref="y3", xanchor="right", yanchor="bottom", yshift=8, showarrow=False, font=dict(size=12, color=theme.INK_MUTED),
                          text=f"random <b>{ends_['random'][1]:.3f}</b>")
    else:                                                                                # no control for this target: say so on the empty maps
        rf.add_annotation(x=0.365, y=0.79, xref="paper", yref="paper", showarrow=False, align="center", font=dict(size=13, color=theme.INK_SECONDARY), bgcolor="rgba(255,255,255,0.92)",
                          bordercolor="#e3e1d9", borderwidth=1, borderpad=8, text="<b>no random control for this target</b><br>it was run for the day-ahead price only")
    rf.add_annotation(x=0, y=1.0, xref="x3", yref="y3", xanchor="left", yanchor="bottom", yshift=3, showarrow=False, font=dict(size=11, color=theme.INK_MUTED), text="defaults = 1.000")
    rf.add_annotation(x=0, y=1.0, yshift=70, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=12, color=theme.INK_SECONDARY), align="left",
                      text=f"<span style='color:{GOLD}'>■</span> gold = a good combination &nbsp; <span style='color:{T_DONE}'>●</span> a trial &nbsp; <span style='color:{T_STOP}'>✕</span> stopped early &nbsp; "
                           f"<span style='color:{T_WIN}'>★</span> the best found &nbsp;·&nbsp; <span style='color:{theme.INK_MUTED}'>a cell = one combination of the axes</span>")
    st.plotly_chart(rf, width="stretch", config={"displayModeBar": False})
    if rc_model == "decision_tree":
        st.caption("This algorithm's grid is small, so both searches try every cell.")

    # ---- the formulas of this algorithm's learning rules (row 1) and the scores that judge a forecast (row 2)
    rule_raw = [str(v) for k_, vs in sp_.items() if k_ in RULE_SETTINGS for v in vs]
    fixed = not rule_raw                                                                # trees and forests: squared error only, nothing to choose
    rule_raw = rule_raw or ["squared_error"]
    CARD = {"squared error (L2)": ("Squared error (L2)", "e²", "900", "follows spikes: big misses dominate"),
            "absolute error (L1)": ("Absolute error (L1)", "|e|", "30", "follows the typical day"),
            "Huber": ("Huber", "½e² if |e| ≤ δ,  else δ(|e| − ½δ)", "29.5", "squared if small, absolute if big (δ = 1 here)"),
            "pseudo-Huber": ("Pseudo-Huber", "δ²(√(1 + (e/δ)²) − 1)", "29.0", "a smooth Huber (δ = 1 here)")}
    SCORES = [("SSE", "Σ e²", "905", "sum of squared errors"), ("MSE", "(1/n) Σ e²", "301.7", "mean squared error"), ("RMSE", "√MSE", "17.4", "back in €/MWh"),
              ("MAE", "(1/n) Σ |e|", "11", "the score the tuning compares settings by")]
    # two thin rows of chips, for reference only: the learning rules of this algorithm, and the scores
    FONT = "'Cambria Math', 'STIX Two Math', 'Times New Roman', serif"
    CHIP = "display:inline-flex; align-items:baseline; gap:0.45rem; background:#fff; border:1px solid #e3e1d9; border-radius:999px; padding:0.18rem 0.8rem; font-size:0.85rem; white-space:nowrap;"
    LAB = f"font-size:0.66rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600; min-width:5.2rem;"
    chips = "".join(
        f"<span style='{CHIP}' title='{raw} · {note}'><b style='color:{theme.INK_PRIMARY};'>{CARD[OBJECTIVE_NAME.get(raw, raw)][0]}</b>"
        f"<span style='font-family:{FONT}; font-size:1rem; color:{theme.INK_PRIMARY};'>{CARD[OBJECTIVE_NAME.get(raw, raw)][1]}</span>"
        f"</span>"
        for raw, note in ((r_, CARD[OBJECTIVE_NAME.get(r_, r_)][3]) for r_ in rule_raw))
    if fixed:
        chips += f"<span style='color:{theme.INK_MUTED}; font-size:0.72rem;'>trees and forests always split on squared error</span>"
    score_chips = "".join(
        f"<span style='{CHIP}{'background:#fff7e6; border-color:#eda100;' if n == 'MAE' else ''}' title='{d}'><b style='color:{theme.INK_PRIMARY};'>{n}</b>"
        f"<span style='font-family:{FONT}; font-size:1rem; color:{theme.INK_PRIMARY};'>{f}</span></span>" for n, f, v, d in SCORES)
    st.markdown(f"<div style='display:flex; flex-direction:column; gap:0.35rem; margin:0.1rem 0 0.6rem;'>"
                f"<div style='display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center;'><span style='{LAB}'>{ml_data.model_label(rc_model)} rules</span>{chips}</div>"
                f"<div style='display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center;'><span style='{LAB}'>scores</span>{score_chips}"
                f"<span style='color:{theme.INK_MUTED}; font-size:0.72rem;'>e = actual − forecast · MAE is the tuning's score</span></div></div>", unsafe_allow_html=True)

_section(f"Search space <span style='font-weight:400; color:{theme.INK_MUTED};'>· {ml_data.model_label(model)}</span>", "every combination of settings the search may try")
space = cfg["space"].get(model, {})
_RULE_SHORT = {"squared error (L2)": "squared L2", "absolute error (L1)": "absolute L1", "Huber": "Huber", "pseudo-Huber": "pseudo-Huber"}      # the full names are in the strip of formulas above


def _space_row(k, v, last):
    chips = "".join(f"<span style='background:#f4f3ef; border-radius:999px; padding:0.12rem 0.6rem; font-size:0.82rem; color:{theme.INK_PRIMARY}; white-space:nowrap;'>{_RULE_SHORT.get(_std(x), _std(x)) if k in RULE_SETTINGS else x}</span>" for x in v)
    return (f"<div style='display:flex; align-items:center; gap:0.7rem; min-height:2.7rem; {'' if last else 'border-bottom:1px solid #eeede8;'}'>"
            f"<span style='flex:none; width:8.2rem; font-weight:600; font-size:0.86rem; color:{theme.INK_PRIMARY};'>{ml_data.setting_label(k)}</span>"
            f"<span style='display:flex; flex-wrap:wrap; gap:0.3rem; flex:1;'>{chips}</span>"
            f"<span style='flex:none; font-size:0.76rem; color:{theme.INK_MUTED}; white-space:nowrap;'>×{len(v)}</span></div>")


_items = list(space.items())                                                         # all the settings in two columns side by side: half as many rows
_cut = (len(_items) + 1) // 2 if len(_items) >= 4 else len(_items)
_cols = [_items[:_cut], _items[_cut:]] if _items[_cut:] else [_items]
_html = (f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(min(21rem, 100%), 1fr)); column-gap:2.2rem;'>"
         + "".join("<div>" + "".join(_space_row(k, v, last=(j == len(col) - 1)) for j, (k, v) in enumerate(col)) + "</div>" for col in _cols) + "</div>")
st.markdown(f"<div style='background:#fff; border:1px solid #e3e1d9; border-radius:12px; padding:0.1rem 1.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{_html}</div>", unsafe_allow_html=True)
n_comb = math.prod(len(v) for v in space.values())
_line(f"<b>{n_comb:,}</b> combinations. " + (f"Fewer than the {n_t} trials, so every one is tried." if n_comb < int(cfg["search"]["n_trials"]) else
      f"With {int(cfg['search']['n_trials'])} trials the space is searched thinly: TPE steers toward the good regions."))

_section(f"Every trial of one tuning round <span style='font-weight:400; color:{theme.INK_MUTED};'>· {ml_data.model_label(model)}</span>", "one dot per trial, scored on all 3 blocks; below the dotted line beats the defaults")
m = tt[tt["model"] == model]
rounds = sorted(m["tuned_at"].unique())
rnd = st.select_slider("Round (training window ends)", rounds, value=rounds[-1], key="ml6_round")
r = m[m["tuned_at"] == rnd].sort_values("trial")
ok = r[~r["pruned"].astype(bool) & ~r["is_default"].astype(bool) & r["score"].notna()]
fig = go.Figure()
fig.add_trace(go.Scatter(x=ok["trial"], y=ok["rel"], mode="markers", name="trial", marker=dict(color=theme.BLUE, size=8),
                         text=ok["params"].map(lambda p: _std_text(ml_data.plain_params(p))), hovertemplate="trial %{x}<br>relative score %{y:.3f}<br>%{text}<extra></extra>"))
best = r[r["selected"].astype(bool)]
fig.add_trace(go.Scatter(x=best["trial"], y=best["rel"], mode="markers", name="chosen", marker=dict(color=theme.YELLOW, size=15, symbol="star", line=dict(width=1, color="black"))))
dflt = r[r["is_default"].astype(bool)]
fig.add_trace(go.Scatter(x=dflt["trial"], y=dflt["rel"], mode="markers", name="defaults", marker=dict(color="black", size=11, symbol="diamond")))
fig.add_hline(y=1.0, line_dash="dot", line_color=theme.INK_MUTED)
st.plotly_chart(theme.style_fig(fig, height=330, yaxis_title="score / score of the defaults (below 1 is better)", xaxis_title="trial"), width="stretch")
npruned = int(r["pruned"].astype(bool).sum())
_line(f"<b>{len(r)}</b> trials &nbsp;·&nbsp; <b>{npruned}</b> stopped after block 3 (not shown) &nbsp;·&nbsp; <b>{int(r['duplicate'].astype(bool).sum())}</b> repeated an earlier setting "
      "&nbsp;·&nbsp; ★ the one chosen: the best mean score over the three blocks")

_section(f"What moves the score <span style='font-weight:400; color:{theme.INK_MUTED};'>· {ml_data.model_label(model)}</span>", "boxes that sit apart: that setting matters &nbsp;·&nbsp; overlapping boxes: it hardly matters &nbsp;·&nbsp; every non-default trial of every round")
exp = tt[(tt["model"] == model) & ~tt["is_default"].astype(bool) & tt["rel"].notna()]
if len(exp):
    keys = sorted({k for p in exp["params_d"] for k in p})
    k = st.selectbox("Setting", keys, key="ml6_setting", format_func=ml_data.setting_label)
    vals = exp["params_d"].map(lambda p: str(p.get(k, "")))
    fig = go.Figure()
    for i, v in enumerate(sorted(vals.unique())):
        fig.add_trace(go.Box(y=exp.loc[vals == v, "rel"], name=_std(v) if k in RULE_SETTINGS else v, marker_color=theme.CATEGORICAL[i % 8], boxpoints="all", jitter=0.4, pointpos=0))
    fig.add_hline(y=1.0, line_dash="dot", line_color=theme.INK_MUTED)
    st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title="score / defaults", legend=False), width="stretch")

_section(f"What the search chose, round by round <span style='font-weight:400; color:{theme.INK_MUTED};'>· {ml_data.model_label(model)}</span>", "the chosen setting changes from round to round: no single tuned setting is 'the' best one &nbsp;·&nbsp; under each gain: inner MAE of the defaults, then of the chosen setting")
lg = tlog[(tlog["target"] == target) & (tlog["search"] == "tpe") & (tlog["model"] == model)].copy()
if len(lg):
    import ast
    _keys = list(space.keys())                                                         # the settings of this algorithm are the columns, every round one row
    _SHORT = {"squared error (L2)": "L2", "absolute error (L1)": "L1", "Huber": "Huber", "pseudo-Huber": "pseudo-Huber"}
    _grid = f"display:grid; grid-template-columns:5.3rem 5.2rem repeat({len(_keys)}, minmax(0, 1fr)); align-items:center; column-gap:0.5rem;"
    _head_ = (f"<div style='{_grid} padding:0.55rem 0 0.35rem; border-bottom:1px solid #e3e1d9; font-size:0.7rem; color:{theme.INK_MUTED}; font-weight:600; line-height:1.2;'>"
              "<span>round</span><span>gain</span>" + "".join(f"<span style='text-align:center;'>{ml_data.setting_label(k_)}</span>" for k_ in _keys) + "</div>")
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


io = ml_data.target_evidence("inner_vs_outer", target, "inner_vs_outer_tpe")
imp = ml_data.target_evidence("importance", target, "importance_random_search")
if io.empty and imp.empty:
    _line(f"The two evidence charts (the gain when choosing against the gain afterwards, and what each setting explains) are not built for {ml_data.label(target)} yet.")
else:
    if not io.empty:
        _section("The gain when choosing, against the gain afterwards", "choosing the best of 40 on the same blocks flatters it: the gain on the days that followed is smaller (selection optimism, measured)")
        g = io.groupby("model")[["inner_gain", "outer_gain"]].mean().reindex(ml_data.ordered_models(io["model"].unique())).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[ml_data.model_label(x) for x in g["model"]], y=100 * g["inner_gain"], name="on the tuning blocks", marker_color=theme.VIOLET))
        fig.add_trace(go.Bar(x=[ml_data.model_label(x) for x in g["model"]], y=100 * g["outer_gain"], name="on the days that followed", marker_color=theme.ORANGE))
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(theme.style_fig(fig, height=340, yaxis_title="MAE gain over defaults (%)", barmode="group"), width="stretch")
    if not imp.empty:
        top = imp.sort_values("eta2", ascending=False).groupby("model").head(3)
        _best = top.sort_values("eta2", ascending=False).iloc[0]
        _section("How much of the score each setting explains", f"{'no single setting dominates' if _best['eta2'] < 0.5 else 'one setting stands out'}: at most <b>{100 * _best['eta2']:.0f}%</b> ({ml_data.setting_label(_best['setting'])}), "
              f"typically <b>{100 * top['eta2'].quantile(0.25):.0f}%</b> to <b>{100 * top['eta2'].quantile(0.75):.0f}%</b> (random-search trials)")
        fig = go.Figure()
        for mm in ml_data.ordered_models(top["model"].unique()):
            x = top[top["model"] == mm]
            fig.add_trace(go.Bar(y=[f"{ml_data.model_label(mm)}: {ml_data.setting_label(s_)}" for s_ in x["setting"]], x=x["eta2"], orientation="h",
                                 marker_color=theme.CATEGORICAL[ml_data.model_index(mm) % 8], showlegend=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(theme.style_fig(fig, height=420, xaxis_title="share of variance explained", legend=False), width="stretch")

_line("The plant inputs (solar irradiance, temperature, natural inflow) are not tuned here; their comparison is on the Plant inputs page.")
