"""ML Lifecycle 9 - Run it: registry, champion promotion, monitoring, retraining schedule and the live switch."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Run it: registry, monitoring, live", "A model in production needs a record of how it was made, a rule for replacing it, a watch on "
             "its errors and a safe way to switch it on. This page shows each.", step=9)
ml_ui.tour_bar(9)

snap = ml_ui.cached(lambda: ml_data.pack_json("registry_snapshot"))() or {}
daily = ml_ui.cached(lambda: ml_data.pack("daily_loss"))()
locked = ml_ui.cached(lambda: ml_data.pack("locked_forecasts"))()
facts = ml_ui.cached(lambda: ml_data.pack_json("facts"))() or {}
pol = ml_ui.cached(lambda: ml_data.config("champion_policy").get("promotion", {}))()
mon = ml_ui.cached(lambda: ml_data.config("monitoring"))()
live = ml_ui.cached(lambda: ml_data.config("live_models").get("targets", {}))()

st.subheader("The experiment registry")
st.markdown("Every run is recorded with its scores, the digests of its configuration and data, and the code version. An **experiment** is "
            "identified by what defines it; running it again is a new attempt, and the registry checks that all attempts give the same scores.")
runs = pd.DataFrame(snap.get("runs", []))
if runs.empty:
    ml_ui.not_yet_run("the registry snapshot")
else:
    ml_ui.cards([("Experiments", len(runs)), ("Repeated", int((runs["attempts"] > 1).sum())),
                        ("Reproduced exactly", int((runs["reproduced"] == True).sum())), ("Not reproduced", int((runs["reproduced"] == False).sum()))], ncols=4)  # noqa: E712
    show = runs[["kind", "target", "stage", "window_start", "window_end", "attempts", "reproduced", "source"]].copy()
    show["target"] = show["target"].map(ml_data.label)
    for c in ("kind", "stage", "source"):
        show[c] = show[c].map(ml_data.run_label)
    ml_ui.table(show.rename(columns={"kind": "Kind", "target": "Target", "stage": "Stage", "window_start": "From", "window_end": "To",
                                      "attempts": "Attempts", "reproduced": "Reproduced", "source": "Recorded by"}))
    st.caption("The Optuna day-ahead experiment, run twice, gave identical predictions. The random-search control is shown as not reproduced on "
               "purpose: its second attempt ran with the numerical guard, which changed one unstable row.")

st.subheader("Champions and the promotion rule")
champs = snap.get("champions", {})
if champs:
    rows = [{"Target": ml_data.label(t), "Version": v["version"], "Status": v["status"], "Promoted": v["promoted_utc"]} for t, vs in champs.items() for v in vs]
    ml_ui.table(pd.DataFrame(rows))
else:
    st.info("No champion is registered yet: the decisions exist, and registering a champion is a deliberate step. The registry holds the versions, "
            "the fitted models with their checksums and an append-only event log.")
if pol:
    st.markdown(f"**A challenger replaces the champion only if all of these hold** (fixed before any challenge): "
                f"over the last **{pol['window_days']} days** the Diebold-Mariano test says it is better at **{pol['alpha']:.0%}**; its loss is at least "
                f"**{pol['min_improvement']:.0%}** lower (no churn for tiny gains); and at least **{pol['min_days_between_promotions']} days** have passed "
                "since the last promotion. Every challenge, accepted or rejected, is logged with its numbers.")

st.subheader("Monitoring: replay over the locked year")
targets = sorted(set(locked["target"]), key=lambda t: ml_data.TARGETS.index(t)) if not locked.empty else []
if not targets:
    ml_ui.not_yet_run("the locked-year forecasts")
else:
    tgt = ml_ui.target_picker(targets=targets)
    window = int(mon.get("tracking", {}).get("window_days", 28))
    q = float(mon.get("tracking", {}).get("quantile", 0.95))
    st.markdown(f"The rule: the rolling **{window}-day mean absolute error** of the champion is compared with the **{q:.0%} quantile of the same "
                "statistic on its development window**, i.e. what normal looked like before going live. This replay uses the locked year, a genuine "
                "out-of-sample year, as if it were live.")
    sim = st.toggle("Simulate a market change (not real data)", key="ml9_sim")
    shock, factor = None, 1.0
    if sim:
        lo, hi = dt.date(2025, 10, 12), dt.date(2026, 10, 11)
        shock = str(st.slider("From day", lo, hi, dt.date(2026, 4, 1), key="ml9_shock"))
        factor = st.slider("Errors grow by a factor of", 1.0, 3.0, 2.0, 0.1, key="ml9_factor")
    rep = ml_data.monitoring_replay(locked, daily, tgt, window, q, shock, factor)
    if rep is None:
        st.info("Not enough development days to set a threshold for this target.")
    else:
        dd = rep["daily"].dropna(subset=["rolling_mae"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd["date"], y=dd["rolling_mae"], mode="lines", name=f"rolling {window}-day MAE", line=dict(color=theme.BLUE, width=2)))
        fig.add_hline(y=rep["level"], line_dash="dot", line_color=theme.INK_MUTED, annotation_text="development level")
        fig.add_hline(y=rep["threshold"], line_dash="dash", line_color=theme.STATUS_CRITICAL, annotation_text=f"threshold ({q:.0%} of development windows)")
        flagged = dd[dd["flag"]]
        if len(flagged):
            fig.add_trace(go.Scatter(x=flagged["date"], y=flagged["rolling_mae"], mode="markers", name="flag", marker=dict(color=theme.STATUS_CRITICAL, size=6)))
        if sim and shock:
            fig.add_shape(type="line", x0=pd.Timestamp(shock), x1=pd.Timestamp(shock), y0=0, y1=1, yref="paper", line=dict(color=theme.ORANGE, width=2))
            fig.add_annotation(x=pd.Timestamp(shock), y=1, yref="paper", text="simulated change", showarrow=False, yanchor="bottom", font=dict(color=theme.ORANGE))
        st.plotly_chart(theme.style_fig(fig, height=340, yaxis_title=ml_data.UNITS[tgt]), width="stretch")
        status = "level flag raised" if rep["flagged_days"] else "no flag"
        ml_ui.cards([("Development level", f"{rep['level']:.2f}"), ("Threshold", f"{rep['threshold']:.2f}"), ("Days flagged", rep["flagged_days"]),
                            ("First flag", str(rep["first_flag"].date()) if rep["first_flag"] is not None else "none")], ncols=4, pre_formatted=True)
        if sim:
            st.warning("SIMULATION: the errors after the chosen day were multiplied. This shows how the flag reacts; it is not a result.")
        elif rep["flagged_days"]:
            st.info(f"In the real locked year the rolling error crosses the threshold on {rep['flagged_days']} day(s), first on {rep['first_flag'].date()}. That is the "
                    "monitor doing its job: it raises a flag and recommends a review and, if the cause is a change in the market, a re-tuning. It never "
                    "changes the champion by itself.")
        st.caption("The full rule also uses a bias flag, a coverage flag and a comparison with the reference; a 'degraded' status needs the reference to be "
                   "better or both level and bias to be off. Nothing changes the champion by itself: a report recommends a review.")

st.subheader("Retraining schedule")
sch = mon.get("schedule", {})
if sch:
    ml_ui.cards([("Refit every", f"{sch['refit_every_days']} days (settings reused)"), ("Re-tune every", f"{sch['retune_every_days']} days"),
                        ("Checked against", "the evaluation and tuning settings")], ncols=3, pre_formatted=True)
    start = dt.date(2025, 10, 12)
    refits = [start + dt.timedelta(days=i * sch["refit_every_days"]) for i in range(13)]
    tunes = [start + dt.timedelta(days=i * sch["retune_every_days"]) for i in range(3)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=refits, y=["refit"] * len(refits), mode="markers", marker=dict(size=11, color=theme.BLUE), name="refit"))
    fig.add_trace(go.Scatter(x=tunes, y=["re-tune"] * len(tunes), mode="markers", marker=dict(size=15, color=theme.ORANGE, symbol="diamond"), name="re-tune"))
    st.plotly_chart(theme.style_fig(fig, height=210), width="stretch")
    st.caption("A refit retrains on the latest data and reuses the tuned settings; every 180 days the search runs again. The live schedule equals the one "
               "used in the evaluation (checked when the configuration is loaded), so live behaviour is the behaviour that was scored.")

st.subheader("Switching a forecast on, safely")
if live:
    mode_names = {"legacy": "Original forecaster", "auto": "Champion, original as fallback", "registry": "Champion only"}
    rows = [{"Forecast": ml_data.label(t), "Currently uses": mode_names.get(v["mode"], v["mode"])} for t, v in live.items()]
    ml_ui.table(pd.DataFrame(rows))
    st.markdown("""
- **Original forecaster:** the model the pipeline used before; **champion, original as fallback:** the champion if it is registered and verified, otherwise the original with a warning; **champion only:** the champion, or an error.
- The champion is loaded only if the file matches its registered checksum, belongs to the active version and the feature definitions have not changed.
- A target is switched by editing one setting, after the champion has been scored against the legacy forecaster on the same days.
- aFRR is switched one side at a time: the up and the down price are separate targets, and a side without a champion keeps the original forecaster. mFRR cannot be switched: the original mFRR forecast is the capacity price, while the framework's target is the activation price.
""")

st.subheader("Engineering")
ml_ui.cards([("Automated tests", f"{facts.get('test_functions', '-')} in {facts.get('test_files', '-')} files"), ("Framework modules", facts.get("framework_modules", "-")),
                    ("Command-line tools", facts.get("tools", "-")), ("Configuration files", facts.get("config_files", "-"))], ncols=4, pre_formatted=True)

