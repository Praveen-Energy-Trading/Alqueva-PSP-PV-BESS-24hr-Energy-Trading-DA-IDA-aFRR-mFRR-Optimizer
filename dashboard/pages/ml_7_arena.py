"""ML Lifecycle 7 - Model arena: which algorithm is better, where, and whether the difference is real."""
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
ml_ui.cards([("Best algorithm", ml_data.model_label(best["model"])), ("Its MAE", f"{best['mae']:.3f} {ml_data.UNITS[target]}"),
                    ("Days compared", f"{round(best['n'] / 24):,}"), ("Pairs that differ (Holm, 5%)", f"{int(pw['significant'].sum())} of {len(pw)}" if not pw.empty else "-")],
                   ncols=4, pre_formatted=True)

st.subheader("Scoreboard")
l, r = st.columns([3, 2])
with l:
    fig = go.Figure(go.Bar(x=[ml_data.model_label(m) for m in board["model"]], y=board["mae"], marker_color=[theme.BLUE if i == 0 else theme.INK_MUTED for i in range(len(board))],
                           text=[f"{v:.2f}" for v in board["mae"]], textposition="outside"))
    st.plotly_chart(theme.style_fig(fig, height=300, yaxis_title=f"MAE ({ml_data.UNITS[target]})", legend=False), width="stretch")
with r:
    sb = board.assign(model=board["model"].map(ml_data.model_label), days_best=(100 * board["days_best"]).round(0))
    ml_ui.table(sb[["model", "mae", "rmse", "days_best"]].rename(columns={"model": "Algorithm", "mae": "MAE", "rmse": "RMSE", "days_best": "Best on (% of days)"}),
                formats={"MAE": "%.2f", "RMSE": "%.2f", "Best on (% of days)": "%d"})
st.caption("Tuned algorithms on the development window. 'Best on' is the share of days on which an algorithm had the smallest error of the day: "
           "the average error and the winning share can disagree.")

st.subheader("How the test works, on real days")
st.markdown("The champion has a smaller average error than the decision tree. Is that a real advantage, or luck on these particular days? Four pictures answer it, "
            "with the real days of the development window and the target chosen above. The grid below runs the same test on every pair.")
if daily.empty:
    ml_ui.not_yet_run("the daily losses (data pack)")
else:
    res = ml_data.dm_example(daily, target)
    if res is None:
        st.info("Not enough days for this target.")
    else:
        BLUE, ORANGE, MUTED, INK = theme.BLUE, theme.ORANGE, theme.INK_MUTED, theme.INK_PRIMARY            # blue = the champion, orange = the decision tree, on every picture below
        diff, los = res["differences"].copy(), res["losses"].copy()
        diff["date"] = diff["date"].astype(str)
        mean, stat, p, n_d = float(res["mean_diff"]), float(res["statistic"]), float(res["p_value"]), int(res["n_days"])
        wins = int((diff["difference"] < 0).sum())

        st.markdown("##### 1. Every day, both models are scored")
        st.caption("Each bar is the average error of that delivery day (all 24 hours together, so a day counts once). The shorter bar was closer to the real price. The last 21 days are shown.")
        last = los.tail(21).reset_index(drop=True)
        lab = [f"{pd.Timestamp(x):%d-%m}" for x in last["date"]]
        f1 = go.Figure()
        f1.add_trace(go.Bar(x=lab, y=last["a"], name="champion", marker_color=BLUE))
        f1.add_trace(go.Bar(x=lab, y=last["b"], name="decision tree", marker_color=ORANGE))
        k = int((last["a"] - last["b"] - (last["a"] - last["b"]).mean()).abs().idxmin())                    # a typical day, to read the gap on
        f1.add_annotation(x=lab[k], y=max(last["a"][k], last["b"][k]), ax=0, ay=-48, showarrow=True, arrowhead=3, arrowwidth=1, arrowcolor=INK, font=dict(size=12, color=INK),
                          text=f"<b>this day</b><br>champion {last['a'][k]:.1f}, tree {last['b'][k]:.1f}<br>gap = {last['a'][k] - last['b'][k]:+.1f}", align="center",
                          bgcolor="white", bordercolor=INK, borderwidth=1, borderpad=4, yanchor="bottom")
        theme.style_fig(f1, height=300, yaxis_title="average error of the day", barmode="group")
        f1.update_xaxes(type="category", tickangle=0, tickfont=dict(size=10, color=MUTED))
        f1.update_yaxes(range=[0, float(max(last["a"].max(), last["b"].max())) * 1.55])
        st.plotly_chart(f1, width="stretch", config={"displayModeBar": False})

        st.markdown("##### 2. One gap per day, over the whole window")
        st.caption(f"The gap of a day is the champion's error minus the tree's. Blue: the champion was closer. Orange: the tree was closer. "
                   f"The champion was closer on **{wins:,} of {n_d:,} days ({wins / n_d:.0%})**; the line is the running average of the gaps.")
        neg, pos = diff["difference"].clip(upper=0), diff["difference"].clip(lower=0)
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=diff["date"], y=neg, name="champion closer", marker_color=BLUE))
        f2.add_trace(go.Bar(x=diff["date"], y=pos, name="tree closer", marker_color=ORANGE))
        f2.add_trace(go.Scatter(x=diff["date"], y=diff["difference"].expanding().mean(), mode="lines", line=dict(color=INK, width=2), name="running average"))
        f2.add_hline(y=0, line_color=MUTED)
        theme.style_fig(f2, height=300, yaxis_title="champion − tree (error of the day)", barmode="relative")
        st.plotly_chart(f2, width="stretch", config={"displayModeBar": False})

        st.markdown("##### 3. The average gap, against how far luck alone could push it")
        se = abs(mean / stat) if stat else float(diff["difference"].std() / n_d ** 0.5)
        band = 1.96 * se
        m = max(abs(mean), band) * 1.3
        c_obs = BLUE if mean < 0 else ORANGE
        f3 = go.Figure()
        f3.add_shape(type="rect", x0=-band, x1=band, y0=0.5, y1=1.0, fillcolor="#c4c2b9", opacity=0.45, line_width=0)
        f3.add_shape(type="line", x0=0, x1=0, y0=0.42, y1=1.08, line=dict(color=MUTED, width=1.5, dash="dot"))
        f3.add_annotation(x=0, y=1.1, yanchor="bottom", showarrow=False, font=dict(size=12, color=MUTED), align="center",
                          text=f"<b>the luck zone</b> (±{band:.2f})<br>if the two models were equal, the average would land here")
        f3.add_annotation(x=mean, y=0.75, ax=0, ay=0.75, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=c_obs, text="")
        f3.add_annotation(x=mean / 2, y=0.75, yshift=15, showarrow=False, font=dict(size=13, color=c_obs),
                          text=f"<b>{abs(stat):.1f} luck-widths</b>")
        f3.add_trace(go.Scatter(x=[mean], y=[0.75], mode="markers", marker=dict(symbol="diamond", size=17, color=c_obs, line=dict(color="white", width=1.5)), showlegend=False,
                                hovertemplate=f"observed average gap {mean:+.3f}<extra></extra>"))
        f3.add_annotation(x=mean, y=0.75, yshift=-30, showarrow=False, font=dict(size=12, color=c_obs), align="center",
                          text=f"<b>observed average gap {mean:+.2f}</b><br>{'champion' if mean < 0 else 'tree'} lower by {abs(mean):.2f}")
        kmax = int(abs(stat))
        step = 1 if kmax <= 12 else 5
        ks = list(range(step, kmax + 1, step))
        sgn = -1 if mean < 0 else 1
        if ks:
            f3.add_trace(go.Scatter(x=[sgn * k * se for k in ks], y=[0.22] * len(ks), mode="markers+text", text=[str(k) for k in ks], textposition="top center",
                                    textfont=dict(size=10, color=MUTED), marker=dict(symbol="line-ns", size=11, color=MUTED, line=dict(width=1.5, color=MUTED)),
                                    showlegend=False, hoverinfo="skip"))
        f3.add_annotation(x=m, y=0.06, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=11, color=MUTED), align="right",
                          text=f"one luck-width = {se:.2f}: how much the average gap moves by chance<br>(allowing for hours and neighbouring days that move together)")
        theme.style_fig(f3, height=250, xaxis_title="champion − tree, average over all days", legend=False)
        f3.update_xaxes(range=[-m, m], zeroline=False, showgrid=False)
        f3.update_yaxes(range=[0, 1.55], visible=False, fixedrange=True)
        f3.layout.shapes = tuple(sh for sh in f3.layout.shapes if not (sh.line.color == theme.AXIS_LINE and sh.x0 == sh.x1 and sh.y0 != 0.42))
        st.plotly_chart(f3, width="stretch", config={"displayModeBar": False})

        st.markdown("##### 4. The verdict")
        ml_ui.cards([("Days compared", n_d), ("Average gap", f"{mean:+.3f}"), ("DM statistic = gap in luck-widths", f"{stat:.2f}"), ("p-value", f"{p:.2g}")], ncols=4, pre_formatted=True)
        if p < 0.05:
            st.success(f"Real, not luck: if the models were truly equal, a gap this big would be seen with probability {p:.2g}. The DM statistic is just the picture above in one number: how many luck-widths the average is from zero.")
        else:
            st.info(f"Not distinguishable from luck: with probability {p:.2g} equal models would show a gap this big. The average is only {abs(stat):.1f} luck-widths from zero, inside the luck zone.")
        st.caption("A rule of thumb: beyond about 2 luck-widths the gap is called real (the edge of the luck zone). With many pairs, some would look real by chance, so the grid below corrects for it with the Holm method.")

st.subheader("Is the difference real? Every pair, with the Holm correction")
if pw.empty:
    ml_ui.not_yet_run("the pairwise tests")
else:
    models = list(board["model"])
    stat = pd.DataFrame(np.nan, index=models, columns=models)
    star = pd.DataFrame("", index=models, columns=models)
    for _, x in pw.iterrows():
        stat.loc[x["a"], x["b"]], stat.loc[x["b"], x["a"]] = x["statistic"], -x["statistic"]
        mark = "*" if x["p_holm"] < 0.05 else ""
        star.loc[x["a"], x["b"]] = star.loc[x["b"], x["a"]] = mark
    lim = max(3.0, float(np.nanmax(np.abs(stat.values))))
    txt = [[("" if i == j or np.isnan(stat.iloc[i, j]) else f"{stat.iloc[i, j]:.1f}{star.iloc[i, j]}") for j in range(len(models))] for i in range(len(models))]
    fig = go.Figure(go.Heatmap(z=stat.values, x=[ml_data.model_label(m) for m in models], y=[ml_data.model_label(m) for m in models], colorscale="RdBu",
                               zmin=-lim, zmax=lim, text=txt, texttemplate="%{text}", colorbar=dict(title="DM")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(theme.style_fig(fig, height=380, legend=False), width="stretch")
    st.caption("Diebold-Mariano statistic of the row algorithm against the column algorithm. Negative (blue): the row algorithm has the smaller error. "
               "A star marks a difference that is still significant after correcting for the number of pairs tested (Holm).")
    vb = tabs["versus_best"]
    if not vb.empty:
        tied = [m for m, d in zip(vb["model"], vb["distinguishable_from_best"]) if not d]
        st.markdown(f"**Best by mean loss:** {ml_data.model_label(vb['best'].iloc[0])}. **Cannot be told apart from it:** "
                    + (", ".join(ml_data.model_label(m) for m in tied) if tied else "none, every other algorithm is significantly worse."))
        with st.expander("Each algorithm against the best (Holm over these comparisons)"):
            t2 = vb[["model", "mean_loss", "best_loss", "mean_diff", "statistic", "p_value", "p_holm", "distinguishable_from_best"]].copy()
            t2["model"] = t2["model"].map(ml_data.model_label)
            ml_ui.table(t2.rename(columns={"model": "Algorithm", "mean_loss": "Mean loss", "best_loss": "Loss of the best", "mean_diff": "Difference", "statistic": "Statistic",
                                            "p_value": "p-value", "p_holm": "p-value (Holm)", "distinguishable_from_best": "Significantly worse"}).round(4))

st.subheader("Where each algorithm wins")
by = st.radio("Group by", ["hour", "month", "regime"], horizontal=True, key="ml7_by", format_func=lambda x: {"hour": "Hour of day", "month": "Month of the year", "regime": "Market regime"}[x])
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
    st.plotly_chart(theme.style_fig(fig, height=340, yaxis_title="MAE", barmode="group"), width="stretch")
    win = t[["group", "units", "lowest_mae", "winner", "winner_share"]].copy()
    win["lowest_mae"] = win["lowest_mae"].map(ml_data.model_label)
    win["winner"] = win["winner"].map(ml_data.model_label)
    with st.expander("Lowest average error and most frequent winner per group"):
        ml_ui.table(win.rename(columns={"group": by.capitalize(), "units": "Units", "lowest_mae": "Lowest MAE", "winner": "Wins most units", "winner_share": "Share of units won"}).round(3))

st.subheader("Error through time")
if daily.empty:
    ml_ui.not_yet_run("the daily losses (data pack)")
else:
    d = daily[daily["target"] == target]
    opts = ml_data.ordered_models(m for m in d["model"].unique() if m not in ("champion", "reference"))
    pick = st.multiselect("Algorithms", opts, default=[o for o in ["lightgbm", "decision_tree"] if o in opts] or opts[:2], format_func=ml_data.model_label, key="ml7_pick")
    win_d = st.slider("Smoothing (days)", 1, 60, 28, key="ml7_smooth")
    fig = go.Figure()
    for m in ml_data.ordered_models(pick):
        x = d[d["model"] == m].sort_values("date")
        fig.add_trace(go.Scatter(x=x["date"], y=x["mae"].rolling(win_d, min_periods=1).mean(), mode="lines", name=ml_data.model_label(m),
                                 line=dict(color=theme.CATEGORICAL[ml_data.model_index(m) % 8], width=1.6)))
    st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title="daily MAE, smoothed"), width="stretch")
    st.caption("The errors of all algorithms rise and fall together over time, which is why the tests compare them on paired daily losses.")

st.caption("The plant inputs (solar irradiance, temperature, natural inflow) are compared on their own page, against simple forecasts and, for irradiance and temperature, the weather forecast.")
