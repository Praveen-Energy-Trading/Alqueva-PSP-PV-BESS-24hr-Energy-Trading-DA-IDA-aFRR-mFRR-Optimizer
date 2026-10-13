"""ML Lifecycle 6 - Tuning lab: nested time-aware tuning, every trial, Optuna against random search, what the search found."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Tuning lab", "Hyperparameters are searched inside the rolling backtest, never on the test period, with the same budget for "
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
st.markdown("""
**How the tuning is protected from leakage.** For every refit of the backtest only that refit's training window is used. Its latest 90 days form
three inner validation blocks; each block is predicted by a model trained only on the days before it. A setting's score is the mean over the
three blocks. The defaults are always trial 0, so a tuned model can never be worse than its default on the inner validation. The test days are
never looked at. A median **pruner** stops a trial that is worse than the median after the first block.
""")

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

st.subheader("A numerical failure, found and guarded")
st.markdown("""
In the random-search control, one tuned XGBoost with the **pseudo-Huber objective** diverged in one fold (April 2023) and predicted prices up to
5e21: a boosted model can leave the range of its training outcomes, and the transform is inverted with `sinh`. The fix limits every forecast, in
transformed space, to the range of the training outcomes plus a margin of one unit, in training, in the tuning validation and in the served
champion. Rerun with the guard, the Optuna experiment is **bit-identical** to the first run (no forecast was ever limited), and the 720 limited
forecasts of the control are exactly that incident.
""")

