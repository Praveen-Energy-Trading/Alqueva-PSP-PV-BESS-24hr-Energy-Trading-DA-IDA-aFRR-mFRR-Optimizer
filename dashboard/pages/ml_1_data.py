"""ML Lifecycle 1 - Data and quality: the sources, the audited raw zone, the quality gate, missing and empty days."""
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

ml_ui.header("Data and quality", "Every model starts from data that is downloaded and stored untouched, checked every day, "
             "cleaned and completed. Every value that was filled is flagged.", step=0)
ml_ui.tour_bar(1)

manifest = ml_ui.cached(lambda: ml_data.pack("raw_manifest_summary"))()
gate = ml_ui.cached(lambda: ml_data.pack_json("quality_gate"))()
qd = ml_ui.cached(lambda: ml_data.pack("quality_daily"))()
status = ml_ui.cached(lambda: ml_data.pack("load_status_monthly"))()

st.subheader("Data sources")
if manifest.empty:
    ml_ui.not_yet_run("the data pack")
else:
    m = manifest[["source", "days", "first_key", "last_key"]].copy()
    m["source"] = m["source"].map(ml_data.source_label)
    m = m.rename(columns={"source": "Source", "days": "Days", "first_key": "First day", "last_key": "Last day"}).reset_index(drop=True)
    half = (len(m) + 1) // 2                              # two tables side by side: less height, the width is used
    left, right = st.columns(2)
    with left:
        ml_ui.table(m.iloc[:half], compact=True)
    with right:
        ml_ui.table(m.iloc[half:], compact=True)

st.subheader("Checking, cleaning and filling")
if gate is None:
    ml_ui.not_yet_run("the quality check")
else:
    passed = bool(gate.get("passed"))
    st.markdown(f"Latest decision: {ml_ui.badge('PASS' if passed else 'FAIL', 'done' if passed else 'not run')} "
                f"&nbsp; window {gate['window'][0]} to {gate['window'][1]}", unsafe_allow_html=True)
    for r in gate.get("reasons", []):
        st.markdown(f"- {ml_data.plain_text(r)}")
    st.caption("The check stops a run only when there is too little real data to repair from, a feed has been silent for too long, or a stored file "
               "no longer matches its checksum. Missing days are repaired, not blocked.")
cur = ml_ui.cached(lambda: ml_data.pack("curation_summary"))()
if not cur.empty:
    c = cur.copy()
    c["filled_share"] = (100 * c["filled_share"]).round(1)
    # days with a gap that were filled, from the quality gate's lines such as 'afrr_price: 1/2842 days missing (0.0%)' and 'ida_price session 2: ...'
    import re
    series_of = {"afrr_price": ["afrr_up", "afrr_dn"], "mfrr_price": ["mfrr_ap"], "imbalance_price": ["imb_short", "imb_long"], "xbid_price": ["xbid_pt"], "da_price": ["da_pt"]}
    days_filled = {}
    for line in (gate or {}).get("repairs", []):
        hit = re.match(r"^(\w+?)(?: session (\d))?: (\d+)/(\d+) days missing \(([\d.]+)%\)", str(line))
        if hit:
            name, session, miss, _, share = hit.groups()
            for key in ([f"ida{session}_pt"] if name == "ida_price" and session else series_of.get(name, [])):
                days_filled[key] = f"{int(miss):,} ({float(share):.1f}%)"
    c["days_filled"] = [days_filled.get(k, "0") for k in c["series"]]
    c["series"] = c["series"].map(ml_data.series_label)
    st.markdown("**What was filled**")
    ml_ui.table(c[["series", "hours", "errors_removed", "days_filled", "filled", "filled_share", "straight_line", "carried_forward", "day_ahead_plus_spread", "same_hour_earlier"]].rename(columns={
        "series": "Series", "hours": "Hours", "errors_removed": "Data errors removed", "days_filled": "Days with a gap, filled", "filled": "Hours filled", "filled_share": "Filled (%)",
        "straight_line": "Straight line", "carried_forward": "Carried forward", "day_ahead_plus_spread": "Day-ahead price + spread",
        "same_hour_earlier": "Same hour earlier"}), formats={"Filled (%)": "%.1f"})
    st.caption("A gap of up to three hours is bridged by a straight line; a longer one repeats the same hour a week earlier (or yesterday). "
               "Band prices are carried forward first. An intraday price that is missing, for example in an empty auction, becomes the day-ahead price "
               "plus the typical spread of that hour. Genuine price spikes and negative prices are kept.")

st.subheader("Plant input series")
_plant = [(key, name, ml_data.plant_input(key)) for key, name, _ in ml_data.PLANT_INPUTS]
if not any(r for _, _, r in _plant):
    ml_ui.not_yet_run("the plant input description (tools/evaluate_plant_inputs.py)")
else:
    _day = lambda iso: f"{iso[8:10]}‑{iso[5:7]}‑{iso[:4]}"          # dd-mm-yyyy with non-breaking hyphens, so a date never splits over two lines
    _rows = []
    for key, name, r in _plant:
        d = (r or {}).get("data")
        if not d:
            continue
        _rows.append({"Series": f"{name} ({d['unit']}, {r['resolution']})",
                      "Source": "Plant sensor" if key in ("ghi", "temperature") else "Guadiana inflow research dataset",
                      "Measured from - to": f"{_day(d['first'])} to {_day(d['last_measured'])}", "File ends": _day(d["last"]),
                      "Measured": d["measured"], "Filled": d["filled"], "Filled (%)": 100.0 * d["filled"] / max(d["rows"], 1)})
    ml_ui.table(pd.DataFrame(_rows), formats={"Filled (%)": "%.1f", "Measured": "%d", "Filled": "%d"}, stretch=True)
    st.caption("These series come from files, not from a daily feed. After the last measured day the file holds filled values (a synthetic curve for the sensor series, "
               "the monthly mean for inflow). Only measured values are used as a test target.")

st.subheader("Quality score over time")
if qd.empty:
    ml_ui.not_yet_run("the quality database")
else:
    ds = sorted(qd["dataset"].unique())
    pick = st.multiselect("Datasets", ds, default=ds[:4], key="ml1_ds", format_func=ml_data.dataset_label)
    fig = go.Figure()
    for i, d in enumerate(pick):
        x = qd[qd["dataset"] == d].groupby("delivery_date")["score"].mean().sort_index()
        fig.add_trace(go.Scatter(x=x.index, y=x.rolling(14, min_periods=3).mean(), mode="lines", name=ml_data.dataset_label(d),
                                 line=dict(color=theme.CATEGORICAL[i % len(theme.CATEGORICAL)], width=1.6)))
    st.plotly_chart(theme.style_fig(fig, height=330, yaxis_title="score (14-day mean)"), width="stretch")
    st.caption("The score describes the data as the sources published it, before cleaning and filling: a day with no data scores 0.")

st.subheader("Missing and empty days")
if status.empty:
    ml_ui.not_yet_run("the load log summary")
else:
    src = st.selectbox("Source", sorted(status["source"].unique()), index=sorted(status["source"].unique()).index("omie_ida1")
                       if "omie_ida1" in set(status["source"]) else 0, key="ml1_src", format_func=ml_data.source_label)
    s = status[status["source"] == src].pivot_table(index="month", columns="status", values="days", aggfunc="sum").fillna(0)
    palette = {"loaded": theme.BLUE, "empty_publication": theme.ORANGE, "not_published": theme.RED, "rejected": theme.VIOLET}
    fig = go.Figure()
    for c in s.columns:
        fig.add_trace(go.Bar(x=s.index, y=s[c], name=c.replace("_", " "), marker_color=palette.get(c, theme.INK_MUTED)))
    st.plotly_chart(theme.style_fig(fig, height=330, yaxis_title="days", barmode="stack"), width="stretch")
