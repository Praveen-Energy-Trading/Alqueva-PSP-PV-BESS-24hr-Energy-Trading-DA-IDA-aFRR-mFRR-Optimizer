"""ML Lifecycle 6 - Tuning lab: nested time-aware tuning, every trial, Optuna against random search, what the search found."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Tuning lab", "", step=5)
ml_ui.tour_bar(6)

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
_note(ft, 0, yr, "<b>The route of one setting</b>", size=13, anchor="left")
cy, nh = yr + 1.35, 1.0
def _node(x0, x1, color, label, badges=()):
    _pill(ft, x0, x1, cy - nh / 2, cy + nh / 2, color, 0.14, r_px=6)
    _pill(ft, x0, x0 + 4, cy - nh / 2, cy + nh / 2, color, 1.0, r_px=2)
    _note(ft, x0 + 14, cy, label, size=12, anchor="left")
    for i, b in enumerate(badges):                                                   # the blocks it is scored on
        bx = x1 - 12 - (len(badges) - 1 - i) * 26
        _pill(ft, bx - 11, bx + 11, cy - 0.28, cy + 0.28, VIOLET, 0.9, r_px=3)
        _note(ft, bx, cy, f"<b>{b}</b>", size=11, color="white")
_node(0, 150, VIOLET, "a setting")
_node(205, 385, VIOLET, "latest block", badges=(nb,))
_arrow(ft, 152, cy, 203, cy)
_arrow(ft, 387, cy, 425, cy)
dx, dh = 545, 0.85
_shape(type="path", fillcolor=ORANGE, opacity=0.16, line_width=0,
             path=f"M {dx - 120},{cy} L {dx},{cy - dh} L {dx + 120},{cy} L {dx},{cy + dh} Z")
_shape(type="path", fillcolor="rgba(0,0,0,0)", line=dict(color=ORANGE, width=1.5),
             path=f"M {dx - 120},{cy} L {dx},{cy - dh} L {dx + 120},{cy} L {dx},{cy + dh} Z")
_note(ft, dx, cy, "worse than<br>the median?", size=11)
_arrow(ft, dx + 122, cy, 700, cy)
_note(ft, dx + 150, cy - 0.38, "no", size=11, color=MUTED)
_node(702, 900, VIOLET, "older blocks", badges=tuple(range(nb - 1, 0, -1)))
_arrow(ft, 902, cy, 925, cy)
oy = cy + 1.95
_arrow(ft, dx, cy + dh + 0.02, dx, oy - 0.45, color=ORANGE)
_note(ft, dx + 22, cy + dh + 0.35, "yes", size=11, color=MUTED, anchor="left")
_pill(ft, dx - 90, dx + 90, oy - 0.45, oy + 0.45, ORANGE, 0.9, r_px=6)
_note(ft, dx, oy, "<b>✕ stopped</b>", size=12, color="white")
_pill(ft, 925, X_END, cy - nh / 2, cy + nh / 2, VIOLET, 0.9, r_px=6)
_note(ft, (925 + X_END) / 2, cy, "<b>score</b>", size=12, color="white")
_arrow(ft, (925 + X_END) / 2, cy + nh / 2 + 0.02, (925 + X_END) / 2, oy - 0.45, color=AQUA)
_pill(ft, 880, X_END, oy - 0.45, oy + 0.45, AQUA, 0.9, r_px=6)
_note(ft, (880 + X_END) / 2, oy, "<b>best score</b>", size=12, color="white")
_pill(ft, 0, 150, oy - 0.45, oy + 0.45, BLUE, 0.9, r_px=6)
_note(ft, 75, oy, "<b>trial 0</b>", size=12, color="white")
_note(ft, 165, oy, "defaults, never stopped", size=11, color=MUTED, anchor="left")

# ---- 5. the same budget for every algorithm: the trials of one real round, TPE and the random control
ydt = oy + 1.3
_note(ft, 0, ydt + 0.3, f"<b>The same budget for every algorithm: up to {n_t} trials each</b><br><span style='color:{MUTED}'>{ml_data.label(target)}, its latest tuning round &nbsp;·&nbsp; every square is a trial that was kept ({len(trials):,} in all)</span>", size=13, anchor="left", align="left")
_leg = " &nbsp;&nbsp; ".join(f"<span style='color:{c}'>■</span> {x}" for c, x in ((BLUE, "defaults (trial 0)"), (VIOLET, "scored on all blocks"), (ORANGE, "stopped early"), (AQUA, "the winner"), ("#c9c7be", "trial not needed")))
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
                col, mark = BLUE, ""
            elif bool(r.selected):
                col, mark = AQUA, "★"
            elif bool(r.pruned):
                col, mark = ORANGE, "×"
            else:
                col, mark = VIOLET, ""
            _pill(ft, x0, x0 + cw - 3, y, y + bh, col, 0.9 if col != VIOLET else 0.75, r_px=2)
            if mark:
                _note(ft, x0 + (cw - 3) / 2, y + bh / 2, f"<b>{mark}</b>", size=10, color="white")
        if mdl == "decision_tree" and method == "tpe" and len(used) < n_t:
            _note(ft, xs0 + (len(used) + n_t) / 2 * cw, y + bh / 2, f"whole grid ({combos_dt} settings) already tried", size=10, color=MUTED)
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
ml_ui.table(pd.DataFrame([{"Setting": k.replace("_", " "), "Values": ", ".join(map(str, v)), "Choices": len(v)} for k, v in space.items()]))
import math
st.caption(f"{math.prod(len(v) for v in space.values()):,} combinations. With 40 trials the large spaces are searched thinly (Optuna steers toward good "
           "regions); the decision tree's 25 combinations are searched exhaustively.")

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
st.caption(f"{len(r)} trials in this round; {npruned} were pruned after the first block (not shown), and {int(r['duplicate'].astype(bool).sum())} repeated an earlier setting. "
           "The star is the setting chosen: the best mean score over the three blocks.")

st.subheader("What moves the score")
exp = tt[(tt["model"] == model) & ~tt["is_default"].astype(bool) & tt["rel"].notna()]
if len(exp):
    keys = sorted({k for p in exp["params_d"] for k in p})
    k = st.selectbox("Setting", keys, key="ml6_setting", format_func=lambda x: x.replace("_", " "))
    vals = exp["params_d"].map(lambda p: str(p.get(k, "")))
    fig = go.Figure()
    for i, v in enumerate(sorted(vals.unique())):
        fig.add_trace(go.Box(y=exp.loc[vals == v, "rel"], name=v, marker_color=theme.CATEGORICAL[i % 8], boxpoints="all", jitter=0.4, pointpos=0))
    fig.add_hline(y=1.0, line_dash="dot", line_color=theme.INK_MUTED)
    st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title="score / defaults", legend=False), width="stretch")
    st.caption("All non-default trials of all rounds for this algorithm, by the value of the chosen setting. Differences between boxes show what the "
               "setting does; overlapping boxes show it hardly matters.")

st.subheader("Optuna against random search (day-ahead price, same budget)")
da = ml_ui.cached(ml_data.prepared_trials)(trials, "da_price")
if da["search"].nunique() < 2:
    ml_ui.not_yet_run("the random-search control")
else:
    curves = {s_: ml_data.best_so_far(da[da["search"] == s_]) for s_ in ("tpe", "random")}
    ms = ml_data.ordered_models(da["model"].unique())
    cols = st.columns(len(ms))
    for col, mm in zip(cols, ms):
        with col:
            fig = go.Figure()
            for name, c, colr in (("Optuna (TPE)", curves["tpe"], theme.BLUE), ("random", curves["random"], theme.ORANGE)):
                cc = c[c["model"] == mm]
                fig.add_trace(go.Scatter(x=cc["trial"], y=cc["best_rel"], mode="lines", name=name, line=dict(color=colr, width=2)))
            st.markdown(f"**{ml_data.model_label(mm)}**")
            st.plotly_chart(theme.style_fig(fig, height=230, legend=False), width="stretch")
    st.caption("Best score so far (relative to the defaults) as trials are used, averaged over the six tuning rounds. After 40 trials the two searches "
               "are about equal (LightGBM 0.916 with Optuna against 0.919 with random search); Optuna's pruner saved 34% to 40% of the fits.")

st.subheader("What the search chose, round by round")
lg = tlog[(tlog["target"] == target) & (tlog["search"] == "tpe") & (tlog["model"] == model)].copy()
if len(lg):
    lg["gain_%"] = (100 * (1 - lg["score"] / lg["default_score"])).round(1)
    lg["params"] = lg["params"].map(ml_data.plain_params)
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
