"""ML Lifecycle 4 - Features: a catalogue with written availability, feature sets, stability selection, the point-in-time check."""
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

ml_ui.header("Features", "Every feature is defined once, in a catalogue, and built by one function for training, evaluation and live "
             "prediction. Whether a feature may be used for a target is derived from what it reads, not typed by hand.", step=3)
ml_ui.tour_bar(4)

cat = ml_ui.cached(lambda: ml_data.config("feature_catalogue"))()
sets = ml_ui.cached(lambda: ml_data.config("feature_sets").get("feature_sets", {}))()
sel = ml_ui.cached(lambda: ml_data.config("feature_selection_results").get("targets", {}))()
avail = ml_ui.cached(lambda: ml_data.pack("feature_availability"))()
feats = pd.DataFrame(cat.get("features", []))
if feats.empty:
    ml_ui.not_yet_run("the feature catalogue")
    st.stop()

target = ml_ui.target_picker()
_desc = dict(zip(feats["id"], feats["description"]))


def fname(fid: str) -> str:
    """A feature by what it is (its description), not by its identifier."""
    d = str(_desc.get(fid, fid))
    return d[:1].upper() + d[1:]


n_target = int(avail[(avail["target"] == target) & (avail["available"])].shape[0]) if not avail.empty else None
blocked = int(feats[feats.get("status", pd.Series(dtype=object)).eq("not_collected")].shape[0]) if "status" in feats else 0
ml_ui.cards([("Features in the catalogue", len(feats)), ("Feature kinds", feats["kind"].nunique()),
                    (f"Available for {ml_data.label(target)}", n_target if n_target is not None else "-"), ("Defined, data not collected", blocked)], ncols=4)

st.subheader("The catalogue")
g = feats.groupby("group").size().sort_values(ascending=False)
c1, c2 = st.columns([2, 3])
with c1:
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=theme.BLUE))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(theme.style_fig(fig, height=280, xaxis_title="features", legend=False), width="stretch")
with c2:
    grp = st.multiselect("Groups", list(g.index), default=list(g.index), key="ml4_groups", format_func=ml_data.pretty)
    show = feats[feats["group"].isin(grp)]
    show = show.assign(feature=show["id"].map(fname), group=show["group"].map(ml_data.pretty), kind=show["kind"].map(ml_data.pretty))
    ml_ui.table(show[["feature", "group", "kind"]].rename(columns={"feature": "Feature", "group": "Group", "kind": "Kind"}), height=250)
st.caption("A feature is a *kind* (a lag, a rolling statistic, a calendar field, the day-ahead price...) with parameters and a version. "
           "What it reads and when that is known follows from the kind and the parameters.")

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
    st.caption("Each cell counts the features of a group that the target may use at its forecast time, out of all features of the group. A feature that "
               "reads yesterday's price is unavailable for XBID, which may use its own price only up to two days back, and for mFRR, three days back.")
    bad = avail[(avail["target"] == target) & (~avail["available"])][["feature", "reason"]].assign(
        reason=lambda x: x["reason"].map(ml_data.plain_text))
    with st.expander(f"Why {len(bad)} features are unavailable for {ml_data.label(target)}"):
        ml_ui.table(bad.assign(feature=bad["feature"].map(fname)).rename(columns={"feature": "Feature", "reason": "Reason"}))

st.subheader("Feature sets")
rows = []
for name, spec in sets.items():
    ids = spec["features"]
    if isinstance(ids, list):
        n = None if avail.empty else int(avail[(avail["target"] == target) & (avail["available"]) & (avail["feature"].isin(ids))].shape[0])
        size = f"{len(ids)} listed" + (f", {n} available for this target" if n is not None else "")
    else:
        size = n_target and f"{n_target} available for this target" or ids
    rows.append({"Set": ml_data.pretty(name), "What it is": ml_data.plain_text(spec["description"]), "Size": size})
ml_ui.table(pd.DataFrame(rows), wrap=True)
st.caption("A set's version is a hash of its features' definitions, so a changed definition is a new version, never a silent edit. "
           "The champions use the baseline set, or its variants for the targets that cannot use yesterday's price.")

st.subheader("Stability selection")
r = sel.get(target)
if not r:
    ml_ui.not_yet_run(f"stability selection for {ml_data.label(target)}")
else:
    ml_ui.cards([("Candidates", r["candidates"]), ("Kept", len(r["selected"])), ("Periods", len(r["periods"])),
                        ("Mean rank correlation across periods", r["mean_rank_correlation"])], ncols=4)
    st.markdown("Kept: " + "; ".join(fname(x).lower() for x in r["selected"]))
    st.caption(f"Rule: a feature is kept if it carries at least {100 * r['rule']['min_share']:.0f}% of the permutation importance in at least "
               f"{100 * r['rule']['min_period_fraction']:.0f}% of the periods ({', '.join(r['periods'])}), using development data only. "
               "On the day-ahead price the full set improved the squared error for LightGBM (p = 0.003) but not the absolute error significantly (p = 0.08), and not for XGBoost, so the small set was kept.")

st.subheader("No leakage by construction")
st.markdown("""
The **point-in-time check** builds a feature twice for the same delivery day: once on the as-of view and once with every value that
should not be known replaced by noise. The two results must be identical. If a feature read a value from the future, they would differ and
the test fails. The same single builder serves training, evaluation and live prediction, so a feature cannot differ between them.
""")

