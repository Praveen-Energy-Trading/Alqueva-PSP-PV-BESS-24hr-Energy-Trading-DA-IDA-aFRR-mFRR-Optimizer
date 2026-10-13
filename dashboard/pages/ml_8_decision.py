"""ML Lifecycle 8 - Decision and locked test: a written selection rule, applied by code, and one honest test on locked data."""
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

ml_ui.header("Decision and locked test", "The champion is chosen by a rule written down before the results existed, applied by code, and then "
             "scored once on a year that nothing was tuned or chosen on.", step=7)
ml_ui.tour_bar(8)

policy = ml_ui.cached(lambda: ml_data.config("model_selection").get("policy", {}))()
decisions = ml_ui.cached(ml_data.decisions)()
locked = ml_ui.cached(ml_data.locked_results)()
lf = ml_ui.cached(lambda: ml_data.pack("locked_forecasts"))()
bench = ml_ui.cached(ml_data.benchmarks)()

st.subheader("The rule (fixed before the results)")
if policy:
    st.markdown(f"""
0. **Evidence:** refuse to choose with fewer than the required development days ({policy['min_days']} as standard) or poor coverage.
1. **Rank** the algorithms by mean absolute error on the same hours.
2. **Parsimony:** the best must beat the simplest ({ml_data.model_label(policy['reference'])}) significantly at {policy['alpha']:.0%}; otherwise the simplest wins.
3. **Ties:** every algorithm that cannot be told apart from the best (Holm correction) is tied. With one left, it wins; with several, the champion is their
   **{policy['average_method']}** (ties are broken by the {policy['tie_break'].replace('_', ' ')}), unless that average is significantly worse than the best single algorithm.
""")
else:
    ml_ui.not_yet_run("the selection policy")

targets = [t for t in ml_data.TARGETS if t != "mfrr_price"]
target = ml_ui.target_picker(targets=targets)
d, lk = decisions.get(target), locked.get(target)
if d is None:
    ml_ui.not_yet_run(f"the champion decision for {ml_data.label(target)}")
    st.stop()

st.subheader(f"Decision for {ml_data.label(target)}")
st.markdown(f"**Champion: {ml_data.champion_text(d['champion'])}**  &nbsp; ({d['n_days']} development days; treatment: {d.get('treatment', {}).get('name', 'standard').replace('_', ' ')})")
if d.get("treatment", {}).get("name") == "limited_evidence":
    st.warning("Limited evidence: " + d["treatment"]["label"] + ". With about 300 days only larger differences can be established, so ties are common.")
for i, reason in enumerate(d["reasons"], 1):
    st.markdown(f"{i}. {ml_data.humanize(reason)}")
l, r = st.columns(2)
with l:
    rk = pd.DataFrame(d["ranking"])
    rk["model"] = rk["model"].map(ml_data.model_label)
    st.markdown("**Ranking on the development window**")
    ml_ui.table(rk[["model", "mean_loss", "mae", "rmse", "sse"]].rename(columns={"model": "Algorithm", "mean_loss": "Mean daily loss"}).round(3))
with r:
    if d.get("versus_best"):
        vb = pd.DataFrame(d["versus_best"])
        vb["model"] = vb["model"].map(ml_data.model_label)
        st.markdown(f"**Each algorithm against the best ({ml_data.model_label(d['best'])})**")
        ml_ui.table(vb[["model", "statistic", "p_value", "p_holm", "distinguishable_from_best"]].rename(columns={"model": "Algorithm", "statistic": "Statistic", "p_value": "p-value", "p_holm": "p-value (Holm)", "distinguishable_from_best": "Significantly worse"}).round(4))
if d.get("combination"):
    c = d["combination"]
    st.caption(f"Average of the tied algorithms against the best single one: mean loss difference {c['mean_diff']:+.3f}, p = {c['p_value']:.3g}: "
               + ("not worse, so the average is the champion." if c["accepted"] else "significantly worse, so the best single algorithm is the champion."))

st.subheader("The one look at the locked year")
if lk is None:
    ml_ui.not_yet_run(f"the locked test of {ml_data.label(target)}")
else:
    s, ref = lk["locked_scores"], lk.get("reference_scores")
    gain = 1 - s["mae"] / ref["mae"] if ref else None
    ml_ui.cards([("Locked-year MAE", f"{s['mae']:.3f}"), ("Reference MAE", f"{ref['mae']:.3f}" if ref else "-"),
                        ("Gain over the reference", f"{100 * gain:.1f}%" if gain is not None else "-"),
                        ("Test against the reference (p)", f"{lk['dm_vs_reference']['p_value']:.2g}" if lk.get("dm_vs_reference") else "-")], ncols=4, pre_formatted=True)
    st.markdown(f"Window {lk['window'][0]} to {lk['window'][1]}; outcomes frozen by digest (lock status: {lk.get('lock_status')}); "
                f"scored once, with code revision {lk['code']}.")
    if lk["loss_ratio_locked_over_dev"] < 0.8 or lk["loss_ratio_locked_over_dev"] > 1.25:
        st.info(f"The locked-year loss is {lk['loss_ratio_locked_over_dev']:.2f} times the development loss. That compares two different price regimes, "
                "so it is not a measure of skill: the gain over the reference on the same hours is.")
    else:
        st.caption(f"Locked-year loss against development loss: {lk['loss_ratio_locked_over_dev']:.2f}.")
    if lk.get("synthetic_fill"):
        sf, ro = lk["synthetic_fill"], lk["real_only"]
        st.error(f"{len(sf['days'])} of the locked days have ARTIFICIAL outcomes ({sf['method']}); they are not real prices. On real days only: champion MAE "
                 f"{ro['champion']['mae']:.3f}" + (f", reference {ro['reference']['mae']:.3f}." if "reference" in ro else "."))
    if lk.get("members"):
        mm = pd.DataFrame([{"Forecaster": k, **v} for k, v in lk["members"].items()])
        with st.expander("The other algorithms scored in the same locked run (reported, not used for any choice)"):
            ml_ui.table(mm.round(3))

    if not lf.empty and target in set(lf["target"]):
        x = lf[lf["target"] == target].copy()
        x["ts"] = pd.to_datetime(x["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
        days = sorted(x["ts"].dt.date.unique())
        pick = st.select_slider("Week to inspect", options=days[::7], value=days[::7][len(days[::7]) // 2], key="ml8_week")
        w = x[(x["ts"].dt.date >= pick) & (x["ts"].dt.date < pick + pd.Timedelta(days=7))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["ts"], y=w["y_true"], mode="lines", name="outcome", line=dict(color="black", width=1.6)))
        fig.add_trace(go.Scatter(x=w["ts"], y=w["champion"], mode="lines", name="champion", line=dict(color=theme.BLUE, width=1.6)))
        fig.add_trace(go.Scatter(x=w["ts"], y=w["reference"], mode="lines", name="reference", line=dict(color=theme.ORANGE, width=1.2, dash="dot")))
        st.plotly_chart(theme.style_fig(fig, height=320, yaxis_title=ml_data.UNITS[target]), width="stretch")
        st.caption("A week of the locked year: outcome, champion forecast and reference forecast. These forecasts were produced by models refitted "
                   "every 30 days on past data only.")

bm = bench.get(target)
st.subheader("Against a naive forecast")
if not bm:
    ml_ui.not_yet_run(f"the naive benchmark of {ml_data.label(target)}")
else:
    st.markdown("A decision tree shows what the extra algorithms buy. A **naive forecast** shows whether machine learning is worth having at all: "
                "it needs no model. *Seasonal naive* is yesterday's price for delivery days Tuesday to Friday and last week's for Saturday to Monday; "
                "*yesterday* and *last week* are the two simple versions of it. For the intraday targets, which are modelled as a spread to the day-ahead "
                "price, the naive forecasts add the earlier spread to the day-ahead price, and *day-ahead price, no spread* simply says the intraday price "
                "equals the day-ahead price. All are scored on exactly the same hours as the champion.")
    names = {"champion": "Champion", "reference": "Decision tree", "base_price": "Day-ahead price, no spread", "naive_seasonal": "Seasonal naive", "naive_day": "Naive: yesterday", "naive_week": "Naive: last week"}
    for stage, title in (("development", "Development window"), ("locked", "Locked year")):
        b = bm.get(stage)
        if not b:
            continue
        rows = []
        for n, sc in b["scores"].items():
            gain = b["gain_of_champion_over"].get(n)
            p = b["dm_champion_vs"].get(n, {}).get("p_value")
            rows.append({"Forecast": names.get(n, n), "MAE": sc["mae"], "RMSE": sc["rmse"],
                         "Champion's gain (%)": None if gain is None else round(100 * gain, 1), "p-value": ml_ui.fmt_p(p)})
        st.markdown(f"**{title}** ({b['window'][0]} to {b['window'][1]}, {b['hours']:,} hours)")
        ml_ui.table(pd.DataFrame(rows), formats={"MAE": "%.2f", "RMSE": "%.2f", "Champion's gain (%)": "%.1f"})
    if bm.get("locked") and bm["locked"].get("synthetic_days"):
        st.caption(f"The locked year includes {bm['locked']['synthetic_days']} artificial days; they are left out here for every forecast, because a forecast built on the day-ahead price would be flattered by them.")
    st.caption("Reading it: a gain of 15% means the champion's mean absolute error is 15% lower. The naive forecasts are not tuned and not chosen, "
               "so they cannot be flattered by the selection; they cannot change the champion either.")

