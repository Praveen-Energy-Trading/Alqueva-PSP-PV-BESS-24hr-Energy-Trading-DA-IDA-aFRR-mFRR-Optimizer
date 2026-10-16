"""ML Lifecycle 4 - Features: a catalogue with written availability, feature sets."""
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

ml_ui.header("Features", "", step=3)
ml_ui.tour_bar(4)

cat = ml_ui.cached(lambda: ml_data.config("feature_catalogue"))()
avail = ml_ui.cached(lambda: ml_data.pack("feature_availability"))()
feats = pd.DataFrame(cat.get("features", []))
if not avail.empty and not feats.empty:
    avail = avail[avail["feature"].isin(feats["id"])]                 # the pack may still list features that are no longer in the catalogue
if feats.empty:
    ml_ui.not_yet_run("the feature catalogue")
    st.stop()

pick_col, info_col = st.columns([1, 2])                           # the target picker and the summary share one row
with pick_col:
    target = ml_ui.target_picker()
_desc = dict(zip(feats["id"], feats["description"]))


def fname(fid: str) -> str:
    """A feature by what it is (its description), not by its identifier."""
    d = str(_desc.get(fid, fid))
    return d[:1].upper() + d[1:]


n_target = int(avail[(avail["target"] == target) & (avail["available"])].shape[0]) if not avail.empty else None
with info_col:
    st.markdown('<div style="height:1.7rem"></div>', unsafe_allow_html=True)           # lines the table up with the selector box
    ml_ui.table(pd.DataFrame([{"Features in the catalogue": len(feats),
                               f"Available for {ml_data.label(target)}": n_target if n_target is not None else "–"}]), stretch=True)

st.subheader("The catalogue")
g = feats.groupby("group").size().sort_values(ascending=False)
fig = go.Figure(go.Bar(x=g.values, y=g.index.map(ml_data.pretty), orientation="h", marker_color=theme.BLUE))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(theme.style_fig(fig, height=260, xaxis_title="features", legend=False), width="stretch")
grp = st.multiselect("Groups", list(g.index), default=list(g.index), key="ml4_groups", format_func=ml_data.pretty)
show = feats[feats["group"].isin(grp)]
usable = set(avail[(avail["target"] == target) & (avail["available"])]["feature"]) if not avail.empty else set(show["id"])
show = show.assign(ok=show["id"].isin(usable)).sort_values("ok", ascending=False, kind="stable")          # the features this target may use first, numbered
show = show.assign(no=[str(i) if o else "–" for i, o in zip(range(1, len(show) + 1), show["ok"])], feature=show["id"].map(fname),
                   group=show["group"].map(ml_data.pretty), kind=show["kind"].map(ml_data.pretty))
ml_ui.table(show[["no", "feature", "group", "kind"]].rename(columns={"no": "No.", "feature": "Feature", "group": "Group", "kind": "Kind"}), height=340, stretch=True)
st.subheader("Availability: what each target may use")
if avail.empty:
    ml_ui.not_yet_run("the availability matrix")
else:
    a = avail[avail["group"].isin(grp)]
    order = [t for t in ml_data.TARGETS if t in set(a["target"])]
    cnt = a.groupby(["group", "target"])["available"].agg(["sum", "count"]).reset_index()
    ok = cnt.pivot(index="group", columns="target", values="sum")[order]
    tot = cnt.pivot(index="group", columns="target", values="count")[order]
    text = ok.astype(int).astype(str) + " of " + tot.astype(int).astype(str)
    fig = go.Figure(go.Heatmap(z=(ok / tot).values, x=[ml_data.short(t) for t in order], y=[ml_data.pretty(g) for g in ok.index], text=text.values,
                               texttemplate="%{text}", colorscale=[[0, theme.STATUS_CRITICAL], [0.5, "#f2e6a0"], [1, theme.STATUS_GOOD]], showscale=False, zmin=0, zmax=1,
                               hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    st.plotly_chart(theme.style_fig(fig, height=60 + 42 * len(ok), legend=False), width="stretch")
    bad = avail[(avail["target"] == target) & (~avail["available"])][["feature", "reason"]].assign(
        reason=lambda x: x["reason"].map(ml_data.plain_text))
    with st.expander(f"Why {len(bad)} features are unavailable for {ml_data.label(target)}"):
        ml_ui.table(bad.assign(feature=bad["feature"].map(fname)).rename(columns={"feature": "Feature", "reason": "Reason"}))

st.subheader("Plant inputs")
_rows = []
for _key, _name, _ in ml_data.PLANT_INPUTS:
    _r = ml_data.plant_input(_key)
    if _r:
        _rows.append({"Forecast": f"{_name} ({_r['unit']})", "Features": _r["features"].replace("_", " "), "Number": len(_r["features"].split(", "))})
if _rows:
    ml_ui.table(pd.DataFrame(_rows), stretch=True)
    st.caption("Irradiance and temperature are taken from the weather forecast; these are the features of the fallback model used without it. Its lags and rolling means "
               "end at the same hour yesterday, which is later than what is observed at 11:00 on the day before (see the Plant inputs page). It adds a clear-sky value and "
               "the calendar; the inflow forecast adds the calendar only.")
else:
    ml_ui.not_yet_run("the plant input evaluation (tools/evaluate_plant_inputs.py)")
