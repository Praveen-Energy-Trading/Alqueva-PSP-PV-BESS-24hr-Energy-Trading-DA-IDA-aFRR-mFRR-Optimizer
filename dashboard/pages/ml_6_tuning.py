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


cfg = ml_ui.cached(lambda: ml_data.config("tuning"))()
trials = ml_ui.cached(lambda: ml_data.pack("trials"))()
tlog = ml_ui.cached(lambda: ml_data.pack("tuning_log"))()
if trials.empty or not cfg:
    ml_ui.not_yet_run("the tuning trials (data pack)")
    st.stop()

target = ml_ui.target_picker(targets=sorted(set(trials["target"]), key=lambda t: ml_data.TARGETS.index(t)))
s = cfg["search"]
ev = ml_ui.cached(lambda: ml_data.config("evaluation"))()
win, L, nb, gap = int(ev["schedule"]["train_window_days"]), int(s["validation_days"]), int(s["n_blocks"]), int(s.get("gap_days", 1))
n_t, retune, refit = int(s["n_trials"]), int(s["retune_every_days"]), int(ev["schedule"]["retrain_every_days"])
BLUE, VIOLET, ORANGE, AQUA, RED, MUTED, INK = theme.BLUE, theme.VIOLET, theme.ORANGE, theme.AQUA, theme.STATUS_CRITICAL, theme.INK_MUTED, theme.INK_PRIMARY
GREY = "#d9d7cf"
T_DEF, T_DONE, T_STOP, T_WIN, NEUTRAL = "#3d3c39", "#8791ab", theme.MAGENTA, theme.GREEN, "#6b6a64"
ROW = 34.0                                                                           # pixels per row: the labels keep their places whatever the content
X_END = win + 280                                                                    # room on the right for the labels and the days tuning never looks at
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


def _note(fig, x, y, text, size=12, color=INK, anchor="center", **kw):
    _annot(x=x, y=y, xanchor=anchor, showarrow=False, text=text, font=dict(size=size, color=color), **kw)


def _arrow(fig, x0, y0, x1, y1, color=MUTED):
    _annot(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=color, text="")


body = win - nb * L                                                                  # the days before the tuning blocks
ft = go.Figure()
ft.add_trace(go.Scatter(x=[0, X_END], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
legend = (f"<span style='color:{BLUE}'>■</span> learns from &nbsp;&nbsp; <span style='color:{VIOLET}'>■</span> scored on ({nb} blocks of {L}) &nbsp;&nbsp; "
          f"<span style='color:{ORANGE}'>■</span> the refit's forecast &nbsp;&nbsp; <span style='color:#c9c7be'>■</span> not used by that round")
_note(ft, 0, -1.55, legend, color=theme.INK_SECONDARY, anchor="left", yanchor="middle")
# ---- 1. the window of one refit: training, then the tuning days cut into blocks
rh = 0.8
_pill(ft, 0, body, 0, rh, BLUE, 0.62)
_pill(ft, body + 0.6, win, 0, rh, VIOLET, 0.9)
for k in range(1, nb):                                                               # the block boundaries inside the tuning days
    _shape(type="line", x0=body + k * L, x1=body + k * L, y0=0.08, y1=rh - 0.08, line=dict(color="white", width=2))
_pill(ft, win + gap, win + gap + 30, 0, rh, ORANGE, 1.0)
_note(ft, 8, rh / 2, f"<b>One refit: its own {win}-day window</b> &nbsp;·&nbsp; {body} days of training", anchor="left", align="left")
for k in range(nb):
    _note(ft, body + k * L + L / 2, rh / 2, f"<b>{k + 1}</b>", color="white")
_note(ft, win + gap + 36, rh / 2, "the refit's forecast days", color=ORANGE, anchor="left")
# ---- 2. the tuning rounds: each block is predicted by a model that has seen only the days before it
for k in range(1, nb + 1):
    y0 = 1.55 + (k - 1) * 1.0
    b0 = body + (k - 1) * L                                                          # first day of block k
    tr_days = b0 - gap
    _pill(ft, 0, tr_days, y0, y0 + rh, BLUE, 0.62)
    _pill(ft, b0, b0 + L, y0, y0 + rh, VIOLET, 0.9)
    if b0 + L < win:
        _pill(ft, b0 + L + 0.6, win, y0, y0 + rh, GREY, 0.8)
    _note(ft, 8, y0 + rh / 2, f"<b>Round {k}</b> &nbsp;·&nbsp; a model trains on the {tr_days} days before block {k}", anchor="left", align="left")
    _note(ft, b0 + L / 2, y0 + rh / 2, f"<b>{k}</b>", color="white")
    _note(ft, win + gap + 36, y0 + rh / 2, f"predicts block {k}: <b>error {k}</b>" + ("<br>scored first: pruner looks here" if k == nb else ""), color=VIOLET, anchor="left", align="left")
ys = 1.55 + nb * 1.0 + 0.75
_note(ft, win + gap + 36, ys + 0.25, "<b>score</b> = average of<br>error " + ", ".join(str(k) for k in range(1, nb + 1)), size=13, color=VIOLET, anchor="left",
      align="left", bgcolor="white", bordercolor=VIOLET, borderwidth=1, borderpad=6)
# ---- 3. the days tuning never looks at, and where they start
ysn = ys + 1.25
_pill(ft, win + gap, win + 140, ysn, ysn + rh, ORANGE, 0.35)
_pill(ft, win + 144, X_END, ysn, ysn + rh, RED, 0.3)
_note(ft, win + 70, ysn + rh / 2, "backtest<br>validation", size=11)
_note(ft, win + 144 + (X_END - win - 144) / 2, ysn + rh / 2, "locked<br>test year", size=11, color=RED)
_note(ft, win - 8, ysn + rh / 2, "<b>never looked at by the tuning</b> →", color=RED, anchor="right", align="right")
_shape(type="line", x0=win + 0.5, x1=win + 0.5, y0=-0.55, y1=ysn + rh + 0.1, line=dict(color="#a8a69d", width=1.2, dash="dot"))
_note(ft, win - 6, -0.55, "the only days the tuning can use ←", size=11, color=MUTED, anchor="right")

# ---- 4. the route of one setting
yr = ysn + rh + 0.95
_note(ft, 0, yr, "<b>The route of one trial</b> &nbsp;<span style='color:%s'>(one trial = one candidate setting of the algorithm)</span>" % MUTED, size=13, anchor="left")
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
ydt = oy + 1.3
_note(ft, 0, ydt + 0.3, f"<b>The same budget for every algorithm: up to {n_t} trials each</b><br><span style='color:{MUTED}'>{ml_data.label(target)}, its latest tuning round &nbsp;·&nbsp; every square is a trial that was kept ({len(trials):,} in all)</span>", size=13, anchor="left", align="left")
_leg = " &nbsp;&nbsp; ".join(f"<span style='color:{c}'>■</span> {x}" for c, x in ((T_DEF, "defaults (trial 0)"), (T_DONE, "all 3 blocks scored"), (T_STOP, "stopped after block 3"), (T_WIN, "the winner"), ("#c9c7be", "trial not needed")))
_note(ft, 0, ydt + 1.25, _leg, size=11, color=theme.INK_SECONDARY, anchor="left")
xs0, xs1 = 130, 1000
cw = (xs1 - xs0) / n_t
bh, pitch = 0.42, 0.52
y = ydt + 1.85
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
yt = yd_end + 0.55
_note(ft, 0, yt, "<b>When it happens</b>", size=13, anchor="left")
n_rounds, span_d = 4, 3 * retune
x0s, x1s = 130, 985
per_day = (x1s - x0s) / span_d
yb = yt + 0.7
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
end = yb + 2.4
theme.style_fig(ft, height=int((end + 2.3) * ROW + 12), xaxis_title=None, legend=False)
ft.update_yaxes(autorange=False, range=[end, -2.3], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
ft.update_xaxes(range=[-4, X_END], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
ft.update_layout(margin=dict(t=2, b=2, l=10, r=10))
ft.layout.shapes = SH                                                                   # replace (not merge with) the axis line the theme added
ft.layout.annotations = AN
st.plotly_chart(ft, width="stretch", config={"displayModeBar": False})

tt = ml_ui.cached(ml_data.prepared_trials)(trials, target, "tpe")
models = ml_data.ordered_models(tt["model"].unique())

st.subheader("Search space")
model = st.selectbox("Algorithm", models, format_func=ml_data.model_label, key="ml6_model")
space = cfg["space"].get(model, {})
ml_ui.table(pd.DataFrame([{"Setting": k.replace("_", " "), "Values": ", ".join(_std(x) if k in RULE_SETTINGS else str(x) for x in v), "Choices": len(v)} for k, v in space.items()]))
import math
st.caption(f"{math.prod(len(v) for v in space.values()):,} combinations. With 40 trials the large spaces are searched thinly (Optuna steers toward good "
           "regions); the decision tree's 25 combinations are searched exhaustively.")

st.markdown("##### The formulas: the learning rules the search chooses between, and the scores that judge the result")
ml_ui.table(pd.DataFrame([
    {"": "Learning rule", "Name": "Squared error (L2)", "Formula": "e²", "Example: errors 1, 2 and 30": "the miss of 30 costs 900",
     "Library names, and the score it goes with": "regression · reg:squarederror · squared_error. Goes with SSE, MSE and RMSE"},
    {"": "Learning rule", "Name": "Absolute error (L1)", "Formula": "|e|", "Example: errors 1, 2 and 30": "the miss of 30 costs 30",
     "Library names, and the score it goes with": "regression_l1 · reg:absoluteerror · absolute_error. Goes with MAE"},
    {"": "Learning rule", "Name": "Huber", "Formula": "½e² if |e| ≤ δ, else δ(|e| − ½δ)", "Example: errors 1, 2 and 30": "the miss of 30 costs 29.5 (δ = 1, for the example only)",
     "Library names, and the score it goes with": "huber (XGBoost: the smooth pseudo-Huber, reg:pseudohubererror). No score of its own"},
    {"": "Score", "Name": "SSE", "Formula": "Σ e²", "Example: errors 1, 2 and 30": "905", "Library names, and the score it goes with": "sum of squared errors"},
    {"": "Score", "Name": "MSE", "Formula": "(1/n) Σ e²", "Example: errors 1, 2 and 30": "301.7", "Library names, and the score it goes with": "mean squared error"},
    {"": "Score", "Name": "RMSE", "Formula": "√MSE", "Example: errors 1, 2 and 30": "17.4", "Library names, and the score it goes with": "back in the unit of the price (€/MWh)"},
    {"": "Score", "Name": "MAE", "Formula": "(1/n) Σ |e|", "Example: errors 1, 2 and 30": "11", "Library names, and the score it goes with": "the score the tuning uses to compare settings"},
]), stretch=True)
st.caption("e = actual price − forecast. A learning rule is what the model minimises while it learns, so it is one of the choices being tuned; a score judges the finished forecast. "
           "The two need not match: here every setting is scored with MAE, whichever rule trained the model.")

st.subheader("Every trial of one tuning round")
m = tt[tt["model"] == model]
rounds = sorted(m["tuned_at"].unique())
rnd = st.select_slider("Round (training window ends)", rounds, value=rounds[-1], key="ml6_round")
r = m[m["tuned_at"] == rnd].sort_values("trial")
ok = r[~r["pruned"].astype(bool) & ~r["is_default"].astype(bool) & r["score"].notna()]
fig = go.Figure()
fig.add_trace(go.Scatter(x=ok["trial"], y=ok["rel"], mode="markers", name="trial", marker=dict(color=theme.BLUE, size=8),
                         text=ok["params"], hovertemplate="trial %{x}<br>relative score %{y:.3f}<br>%{text}<extra></extra>"))
best = r[r["selected"].astype(bool)]
fig.add_trace(go.Scatter(x=best["trial"], y=best["rel"], mode="markers", name="chosen", marker=dict(color=theme.YELLOW, size=15, symbol="star", line=dict(width=1, color="black"))))
dflt = r[r["is_default"].astype(bool)]
fig.add_trace(go.Scatter(x=dflt["trial"], y=dflt["rel"], mode="markers", name="defaults", marker=dict(color="black", size=11, symbol="diamond")))
fig.add_hline(y=1.0, line_dash="dot", line_color=theme.INK_MUTED)
st.plotly_chart(theme.style_fig(fig, height=330, yaxis_title="score / score of the defaults (below 1 is better)", xaxis_title="trial"), width="stretch")
npruned = int(r["pruned"].astype(bool).sum())
st.caption(f"{len(r)} trials in this round; {npruned} were stopped after block 3, the latest block (not shown), and {int(r['duplicate'].astype(bool).sum())} repeated an earlier setting. "
           "The star is the setting chosen: the best mean score over the three blocks.")

st.subheader("What moves the score")
exp = tt[(tt["model"] == model) & ~tt["is_default"].astype(bool) & tt["rel"].notna()]
if len(exp):
    keys = sorted({k for p in exp["params_d"] for k in p})
    k = st.selectbox("Setting", keys, key="ml6_setting", format_func=lambda x: x.replace("_", " "))
    vals = exp["params_d"].map(lambda p: str(p.get(k, "")))
    fig = go.Figure()
    for i, v in enumerate(sorted(vals.unique())):
        fig.add_trace(go.Box(y=exp.loc[vals == v, "rel"], name=_std(v) if k in RULE_SETTINGS else v, marker_color=theme.CATEGORICAL[i % 8], boxpoints="all", jitter=0.4, pointpos=0))
    fig.add_hline(y=1.0, line_dash="dot", line_color=theme.INK_MUTED)
    st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title="score / defaults", legend=False), width="stretch")
    st.caption("All non-default trials of all rounds for this algorithm, by the value of the chosen setting. Differences between boxes show what the "
               "setting does; overlapping boxes show it hardly matters.")

st.subheader("Random control and TPE, from scratch (day-ahead price)")
da = ml_ui.cached(ml_data.prepared_trials)(trials, "da_price")
if da["search"].nunique() < 2:
    ml_ui.not_yet_run("the random-search control")
else:
    from plotly.subplots import make_subplots
    ms = ml_data.ordered_models(da["model"].unique())
    rc_model = st.selectbox("Algorithm", ms, index=ms.index("lightgbm") if "lightgbm" in ms else 0, format_func=ml_data.model_label, key="ml6_rc_model")
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
    rf = make_subplots(rows=2, cols=4, column_widths=[0.13, 0.27, 0.27, 0.33], horizontal_spacing=0.03, vertical_spacing=0.2)
    for i, key in enumerate(("random", "tpe"), start=1):
        r_ = latest[key]
        for j, (lo_t, hi_t) in enumerate(((0, n_early), (n_early, 10 ** 6)), start=1):
            part = r_[(r_["trial"] > lo_t) & (r_["trial"] <= hi_t)]
            rf.add_trace(go.Heatmap(x=list(range(len(va))), y=list(range(len(vb))), z=quality, zmin=0, zmax=1, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
                                    colorscale=[[0, "#f7f5ee"], [1, GOLD]]), row=i, col=j + 1)
            fin, stop = part[~part["pruned"].astype(bool)], part[part["pruned"].astype(bool)]
            for sub_, sym, col_, sz in ((fin, "circle", T_DONE, 10), (stop, "x", T_STOP, 11)):
                if len(sub_):
                    rf.add_trace(go.Scatter(x=[va.index(str(p.get(ka))) + rng.uniform(-0.3, 0.3) for p in sub_["params_d"]],
                                            y=[vb.index(str(p.get(kb))) + rng.uniform(-0.3, 0.3) for p in sub_["params_d"]], mode="markers", showlegend=False,
                                            marker=dict(symbol=sym, size=sz, color=col_, line=dict(color="white", width=1)), customdata=sub_["rel"],
                                            hovertemplate="score %{customdata:.3f} of the defaults<extra></extra>"), row=i, col=j + 1)
            win_ = part[part["selected"].astype(bool)]
            if len(win_):
                p = win_["params_d"].iloc[0]
                rf.add_trace(go.Scatter(x=[va.index(str(p.get(ka)))], y=[vb.index(str(p.get(kb)))], mode="markers", showlegend=False, hoverinfo="skip",
                                        marker=dict(symbol="star", size=22, color=T_WIN, line=dict(color="white", width=1.5))), row=i, col=j + 1)
    # the result: the best score found so far, averaged over the tuning rounds
    curves = {k_: ml_data.best_so_far(da[da["search"] == k_]) for k_ in ("tpe", "random")}
    ends_ = {}
    for key, name, colr, dash, wd in (("random", "random control", theme.INK_MUTED, "dash", 3), ("tpe", "TPE", theme.INK_PRIMARY, "solid", 3.2)):
        cc = curves[key][curves[key]["model"] == rc_model].sort_values("trial")
        rf.add_trace(go.Scatter(x=cc["trial"], y=cc["best_rel"], mode="lines", showlegend=False, line=dict(color=colr, width=wd, dash=dash),
                                hovertemplate=name + ": %{y:.3f} after trial %{x}<extra></extra>"), row=1, col=4)
        ends_[key] = (float(cc["trial"].iloc[-1]), float(cc["best_rel"].iloc[-1]))
    rf.add_hline(y=1.0, line=dict(color=theme.INK_MUTED, width=1, dash="dot"), row=1, col=4)
    # the work: a trial stopped after block 3 skips the other blocks
    tp = da[(da["search"] == "tpe") & (da["model"] == rc_model)]
    saved = float(tp["pruned"].sum()) * (nb - 1) / nb / max(len(tp), 1)
    rf.add_trace(go.Bar(y=["TPE", "random"], x=[100 * (1 - saved), 100], orientation="h", marker_color=[T_DEF, theme.INK_MUTED], showlegend=False,
                        text=[f"{100 * (1 - saved):.0f}%", "100%"], textposition="inside", textfont=dict(color="white", size=14), hoverinfo="skip"), row=2, col=4)
    if saved > 0:
        rf.add_trace(go.Bar(y=["TPE"], x=[100 * saved], orientation="h", marker=dict(color=T_STOP), showlegend=False, text=[f"{100 * saved:.0f}% saved"],
                            textposition="inside", textfont=dict(color="white", size=13), hoverinfo="skip"), row=2, col=4)
    theme.style_fig(rf, height=560, legend=False, barmode="stack")
    for i in (1, 2):
        for j in (1, 2):
            rf.update_xaxes(tickvals=list(range(len(va))), ticktext=[_std(v) if ka in RULE_SETTINGS else v for v in va], range=[-0.6, len(va) - 0.4], showgrid=False, zeroline=False,
                            showticklabels=(i == 2), tickfont=dict(size=11), row=i, col=j + 1)
            rf.update_yaxes(tickvals=list(range(len(vb))), ticktext=[_std(v) if kb in RULE_SETTINGS else v for v in vb], range=[-0.6, len(vb) - 0.4], showgrid=False, zeroline=False,
                            showticklabels=(j == 1), tickfont=dict(size=11), row=i, col=j + 1)
    rf.update_xaxes(range=[0, n_t - 1], tickvals=[0, n_t - 1], showgrid=False, row=1, col=4)
    rf.update_yaxes(range=[min(v for _, v in ends_.values()) - 0.02, 1.01], showticklabels=False, row=1, col=4)
    rf.update_xaxes(range=[0, 104], showticklabels=False, showgrid=False, row=2, col=4)
    rf.update_yaxes(showgrid=False, tickfont=dict(size=13), automargin=True, row=2, col=4)
    for i in (1, 2):                                                                  # the label column has no axes
        rf.update_xaxes(visible=False, row=i, col=1)
        rf.update_yaxes(visible=False, row=i, col=1)
    rf.update_layout(margin=dict(t=126, b=70, l=10, r=10))
    rf.layout.shapes = tuple(sh for sh in rf.layout.shapes if sh.xref not in ("x domain", "x5 domain"))          # no axis line in the label column
    dom = {(1, 1): "x domain", (1, 2): "x2 domain", (1, 3): "x3 domain", (1, 4): "x4 domain", (2, 1): "x5 domain", (2, 2): "x6 domain", (2, 3): "x7 domain", (2, 4): "x8 domain"}
    # row labels and column titles
    for (i, j), title in (((1, 2), "first 10 trials"), ((1, 3), f"the other {n_t - 1 - n_early} trials"), ((1, 4), "result: best score so far")):
        rf.add_annotation(x=0.5, y=1.0, xref=dom[(i, j)], yref="paper", yshift=8, yanchor="bottom", showarrow=False, text=f"<b>{title}</b>", font=dict(size=13, color=theme.INK_PRIMARY))
    rf.add_annotation(x=0.5, y=0.4, xref=dom[(2, 4)], yref="paper", yshift=12, yanchor="bottom", showarrow=False, text="<b>work done</b>", font=dict(size=13, color=theme.INK_PRIMARY))
    for (i, _), lab, ypos in (((1, 1), "<b style='font-size:14px'>RANDOM<br>CONTROL</b><br><br>picks cells blindly,<br>never stops", 0.8), ((2, 1), "<b style='font-size:14px'>TPE</b><br><br>guesses 10 times,<br>then learns from<br>the scores and<br>stops weak trials", 0.2)):
        rf.add_annotation(x=0.059, y=ypos, xref="paper", yref="paper", showarrow=False, font=dict(size=11, color=theme.INK_SECONDARY), align="center", text=lab)
    # end labels of the two lines
    rf.add_annotation(x=ends_["tpe"][0], y=ends_["tpe"][1], xref="x4", yref="y4", xanchor="right", yanchor="top", yshift=-8, showarrow=False, font=dict(size=12, color=theme.INK_PRIMARY),
                      text=f"TPE <b>{ends_['tpe'][1]:.3f}</b>")
    rf.add_annotation(x=ends_["random"][0], y=ends_["random"][1], xref="x4", yref="y4", xanchor="right", yanchor="bottom", yshift=8, showarrow=False, font=dict(size=12, color=theme.INK_MUTED),
                      text=f"random <b>{ends_['random'][1]:.3f}</b>")
    rf.add_annotation(x=0, y=1.0, xref="x4", yref="y4", xanchor="left", yanchor="bottom", yshift=3, showarrow=False, font=dict(size=11, color=theme.INK_MUTED), text="defaults = 1.000")
    rf.add_annotation(x=0, y=1.0, yshift=84, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=12, color=theme.INK_SECONDARY), align="left",
                      text=f"<span style='color:{GOLD}'>■</span> gold = a good combination &nbsp;&nbsp; <span style='color:{T_DONE}'>●</span> a trial &nbsp;&nbsp; <span style='color:{T_STOP}'>✕</span> stopped early &nbsp;&nbsp; "
                           f"<span style='color:{T_WIN}'>★</span> the best one found<br><span style='color:{theme.INK_MUTED}'>every cell is one combination of {ka.replace('_', ' ')} and {kb.replace('_', ' ')}</span>")
    st.plotly_chart(rf, width="stretch", config={"displayModeBar": False})
    st.caption(f"The same {n_t} trials for each search, latest tuning round on the maps; the result and the work are averaged over all rounds. Day-ahead price only: the control was not run for the other targets."
               + (" This algorithm's grid is small, so both searches try every cell." if rc_model == "decision_tree" else ""))

st.subheader("What the search chose, round by round")
lg = tlog[(tlog["target"] == target) & (tlog["search"] == "tpe") & (tlog["model"] == model)].copy()
if len(lg):
    lg["gain_%"] = (100 * (1 - lg["score"] / lg["default_score"])).round(1)
    lg["params"] = lg["params"].map(lambda p: _std_text(ml_data.plain_params(p)))
    ml_ui.table(lg[["tuned_at", "score", "default_score", "gain_%", "n_trials", "n_pruned", "params"]].rename(columns={
        "tuned_at": "Round", "score": "Chosen (inner MAE)", "default_score": "Defaults (inner MAE)", "gain_%": "Inner gain (%)", "n_trials": "Trials",
        "n_pruned": "Pruned", "params": "Chosen setting"}).round(3))
    st.caption("The chosen setting changes from round to round: do not read one tuned setting as 'the' best one.")

if target == "da_price":
    io = ml_data.evidence_table("inner_vs_outer_tpe")
    imp = ml_data.evidence_table("importance_random_search")
    l, r_ = st.columns(2)
    with l:
        st.markdown("**The gain seen when choosing against the gain afterwards**")
        if not io.empty:
            g = io.groupby("model")[["inner_gain", "outer_gain"]].mean().reindex(ml_data.ordered_models(io["model"].unique())).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[ml_data.model_label(x) for x in g["model"]], y=100 * g["inner_gain"], name="on the validation blocks", marker_color=theme.BLUE))
            fig.add_trace(go.Bar(x=[ml_data.model_label(x) for x in g["model"]], y=100 * g["outer_gain"], name="on the days that followed", marker_color=theme.ORANGE))
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(theme.style_fig(fig, height=340, yaxis_title="MAE gain over defaults (%)", barmode="group"), width="stretch")
            st.caption("Choosing the best of 40 settings on the same blocks flatters it: the gain afterwards is smaller. This is selection optimism, measured.")
    with r_:
        st.markdown("**How much of the score variance each setting explains (random-search trials)**")
        if not imp.empty:
            top = imp.sort_values("eta2", ascending=False).groupby("model").head(3)
            fig = go.Figure()
            for mm in ml_data.ordered_models(top["model"].unique()):
                x = top[top["model"] == mm]
                fig.add_trace(go.Bar(y=[f"{ml_data.model_label(mm)}: {str(s_).replace('_', ' ')}" for s_ in x["setting"]], x=x["eta2"], orientation="h",
                                     marker_color=theme.CATEGORICAL[ml_data.model_index(mm) % 8], showlegend=False))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(theme.style_fig(fig, height=420, xaxis_title="share of variance explained", legend=False), width="stretch")
            st.caption("No single setting dominates: at most 36% (tree depth) and typically 10% to 22%.")

st.caption("The plant inputs (solar irradiance, temperature, natural inflow) are not part of this lab: they are not tuned here. Their algorithms are compared with simple forecasts (and, for irradiance and temperature, with the weather forecast) on the Plant inputs page.")
