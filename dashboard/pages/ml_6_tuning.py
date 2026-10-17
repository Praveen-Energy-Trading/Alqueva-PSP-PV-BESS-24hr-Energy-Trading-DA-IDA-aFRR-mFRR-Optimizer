"""ML Lifecycle 6 - Tuning lab: nested time-aware tuning, every trial, Optuna against random search, what the search found."""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Tuning lab", "Hyperparameters are searched on validation blocks inside each training window, never on the locked test year, with the same budget for "
             "every algorithm. Every trial is kept, so the search itself can be inspected.", step=5)
ml_ui.tour_bar(6)

cfg = ml_ui.cached(lambda: ml_data.config("tuning"))()
trials = ml_ui.cached(lambda: ml_data.pack("trials"))()
tlog = ml_ui.cached(lambda: ml_data.pack("tuning_log"))()
if trials.empty or not cfg:
    ml_ui.not_yet_run("the tuning trials (data pack)")
    st.stop()

s = cfg["search"]
ml_ui.cards([("Trials per algorithm and round", s["n_trials"]), ("Inner validation", f"{s['n_blocks']} blocks of {s['validation_days']} days"),
                    ("Re-tuned every", f"{s['retune_every_days']} days"), ("Search method", f"{s['method'].upper()} + random control"),
                    ("Trials logged", f"{len(trials):,}")], ncols=5, pre_formatted=True)
ev = ml_ui.cached(lambda: ml_data.config("evaluation"))()
win, L, nb, gap = int(ev["schedule"]["train_window_days"]), int(s["validation_days"]), int(s["n_blocks"]), int(s.get("gap_days", 1))
BLUE, VIOLET, ORANGE, RED, MUTED, INK = theme.BLUE, theme.VIOLET, theme.ORANGE, theme.STATUS_CRITICAL, theme.INK_MUTED, theme.INK_PRIMARY
GREY = "#d9d7cf"
X_END = win + 280                                                                    # room on the right for the labels and the days tuning never looks at
px_day = 1150.0 / X_END


def _pill(fig, x0, x1, y0, y1, color, opacity=1.0, r_px=4.0, row_px=40.0):
    """A rounded bar as a path; x in days of the window, y in rows."""
    rx, ry = min(r_px / px_day, (x1 - x0) / 2), min(r_px / row_px, (y1 - y0) / 2)
    fig.add_shape(type="path", fillcolor=color, opacity=opacity, line_width=0,
                  path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"))


body = win - nb * L                                                                  # the days before the tuning blocks
ft = go.Figure()
ft.add_trace(go.Scatter(x=[0, X_END], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
legend = (f"<span style='color:{BLUE}'>■</span> days a model learns from &nbsp;&nbsp; <span style='color:{VIOLET}'>■</span> days it is scored on &nbsp;&nbsp; "
          f"<span style='color:{ORANGE}'>■</span> the refit's own forecast days &nbsp;&nbsp; <span style='color:#c9c7be'>■</span> later days, not used by that round")
ft.add_annotation(x=0, y=-1.55, xanchor="left", yanchor="middle", showarrow=False, text=legend, font=dict(size=12, color=theme.INK_SECONDARY))
# ---- the window of one refit: training, then the tuning days cut into blocks
rh = 0.8
_pill(ft, 0, body, 0, rh, BLUE, 0.62)
_pill(ft, body + 0.6, win, 0, rh, VIOLET, 0.9)
for k in range(1, nb):                                                               # the block boundaries inside the tuning days
    ft.add_shape(type="line", x0=body + k * L, x1=body + k * L, y0=0.08, y1=rh - 0.08, line=dict(color="white", width=2))
_pill(ft, win + gap, win + gap + 30, 0, rh, ORANGE, 1.0)
ft.add_annotation(x=8, y=rh / 2, xanchor="left", showarrow=False, font=dict(size=12, color=INK), align="left", text=f"<b>One refit: its own {win}-day window</b> &nbsp;·&nbsp; {body} days of training, then {nb * L} tuning days in {nb} blocks of {L}")
for k in range(nb):
    ft.add_annotation(x=body + k * L + L / 2, y=rh / 2, showarrow=False, font=dict(size=12, color="white"), text=f"<b>{k + 1}</b>")
ft.add_annotation(x=win + gap + 36, y=rh / 2, xanchor="left", showarrow=False, font=dict(size=12, color=ORANGE), text="the refit forecasts these days")
# ---- the three tuning rounds: each block is predicted by a model that has seen only the days before it
n_rows = 3
for k in range(1, nb + 1):
    y0 = 1.55 + (k - 1) * 1.0
    b0 = body + (k - 1) * L                                                          # first day of block k
    tr_days = b0 - gap
    _pill(ft, 0, tr_days, y0, y0 + rh, BLUE, 0.62)
    _pill(ft, b0, b0 + L, y0, y0 + rh, VIOLET, 0.9)
    if b0 + L < win:
        _pill(ft, b0 + L + 0.6, win, y0, y0 + rh, GREY, 0.8)
    ft.add_annotation(x=8, y=y0 + rh / 2, xanchor="left", showarrow=False, font=dict(size=12, color=INK), align="left",
                      text=f"<b>Round {k}</b> &nbsp;·&nbsp; a model trains on the {tr_days} days before block {k}")
    ft.add_annotation(x=b0 + L / 2, y=y0 + rh / 2, showarrow=False, font=dict(size=12, color="white"), text=f"<b>{k}</b>")
    last = k == nb
    ft.add_annotation(x=win + gap + 36, y=y0 + rh / 2, xanchor="left", showarrow=False, font=dict(size=12, color=VIOLET), align="left",
                      text=f"predicts block {k}: <b>error {k}</b>" + ("<br>scored first: the pruner looks here" if last else ""))
# ---- the score of a setting
ys = 1.55 + nb * 1.0 + 0.25
ft.add_annotation(x=win + gap + 36, y=ys + 0.25, xanchor="left", showarrow=False, font=dict(size=13, color=VIOLET), bgcolor="white", bordercolor=VIOLET, borderwidth=1, borderpad=6,
                  align="left", text="<b>score of a setting</b> =<br>the average of " + ", ".join(f"error {k}" for k in range(1, nb + 1)))
# ---- the days tuning never looks at, and where they start
ysn = ys + 1.25
_pill(ft, win + gap, win + 140, ysn, ysn + rh, ORANGE, 0.35)
_pill(ft, win + 144, X_END, ysn, ysn + rh, RED, 0.3)
ft.add_annotation(x=win + 70, y=ysn + rh / 2, showarrow=False, font=dict(size=11, color=INK), text="the backtest's validation<br>days")
ft.add_annotation(x=win + 144 + (X_END - win - 144) / 2, y=ysn + rh / 2, showarrow=False, font=dict(size=11, color=RED), text="the locked test year")
ft.add_annotation(x=win - 8, y=ysn + rh / 2, xanchor="right", showarrow=False, font=dict(size=12, color=RED), align="right", text="<b>never looked at by the tuning</b> →")
ft.add_shape(type="line", x0=win + 0.5, x1=win + 0.5, y0=-0.55, y1=ysn + rh + 0.1, line=dict(color="#a8a69d", width=1.2, dash="dot"))
ft.add_annotation(x=win - 6, y=-0.55, xanchor="right", showarrow=False, font=dict(size=11, color=MUTED), text="the only days the tuning can use ←")
theme.style_fig(ft, height=int((ysn + rh + 2.2) * 40 + 30), xaxis_title=None, legend=False)
ft.update_yaxes(autorange=False, range=[ysn + rh + 0.35, -2.0], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
ft.update_xaxes(range=[-4, X_END], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
ft.update_layout(margin=dict(t=2, b=2, l=10, r=10))
ft.layout.shapes = tuple(sh for sh in ft.layout.shapes if not (sh.line.color == theme.AXIS_LINE and sh.x0 == sh.x1 and sh.y0 != -0.55))
st.markdown("##### One tuning round")
st.plotly_chart(ft, width="stretch", config={"displayModeBar": False})

# ---- the trials of a round: what happens to one setting
n_t, startup = int(s["n_trials"]), int((s.get("pruning") or {}).get("startup_trials", 5))
steps = [(BLUE, "Trial 0", "The defaults", "Always scored first and never stopped, so a tuned model can never be worse than its default on these blocks."),
         (VIOLET, f"Trials 1 to {n_t - 1}", "A new setting is proposed", f"Optuna's TPE proposes it, steered by the scores so far. Every algorithm gets the same {n_t} trials."),
         (ORANGE, "Score on the latest block", "Worse than the median? Stop", f"After the first {startup} trials, a setting worse than the median of the earlier ones is stopped here (pruned). Otherwise the older blocks are scored too."),
         (theme.AQUA, "End of the round", "The lowest average wins", f"Its settings are used by every refit until the next tuning round, {s['retune_every_days']} days later.")]
_esc = html.escape
tiles = "".join(
    f'<div style="flex:1 1 0; min-width:0; background:{c}12; border-left:4px solid {c}; border-radius:10px; padding:0.6rem 0.9rem;">'
    f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600;">{_esc(a)}</div>'
    f'<div style="font-size:0.95rem; font-weight:700; color:{theme.INK_PRIMARY}; margin:0.1rem 0 0.25rem;">{_esc(b)}</div>'
    f'<div style="font-size:0.84rem; color:{theme.INK_SECONDARY}; line-height:1.4;">{_esc(d)}</div></div>'
    + ('<div style="align-self:center; color:%s; font-size:1.3rem;">→</div>' % theme.INK_MUTED if i < len(steps) - 1 else "")
    for i, (c, a, b, d) in enumerate(steps))
st.markdown("##### What happens to one setting")
st.markdown(f'<div style="display:flex; gap:0.6rem; align-items:stretch;">{tiles}</div>', unsafe_allow_html=True)

target = ml_ui.target_picker(targets=sorted(set(trials["target"]), key=lambda t: ml_data.TARGETS.index(t)))
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
