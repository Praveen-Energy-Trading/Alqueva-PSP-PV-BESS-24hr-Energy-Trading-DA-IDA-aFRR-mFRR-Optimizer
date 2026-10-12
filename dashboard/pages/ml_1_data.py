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

ml_ui.header("Data and quality", "Every model starts from data that is downloaded, kept untouched, checked every day and allowed "
             "through a quality gate. Nothing is modelled until the data is trusted.", step=1)
ml_ui.tour_bar(1)

manifest = ml_ui.cached(lambda: ml_data.pack("raw_manifest_summary"))()
gate = ml_ui.cached(lambda: ml_data.pack_json("quality_gate"))()
qd = ml_ui.cached(lambda: ml_data.pack("quality_daily"))()
status = ml_ui.cached(lambda: ml_data.pack("load_status_monthly"))()

st.subheader("Sources and the raw zone")
st.markdown("Eight sources (OMIE and REN) are downloaded into an **immutable raw zone**: every file is stored once with its SHA-256 checksum "
            "and a version number; a changed file is a new version, never an overwrite. The clean zone is rebuilt from the raw zone, so any "
            "number can be traced back to a file.")
if manifest.empty:
    ml_ui.not_yet_run("the data pack (python dashboard/ml_pack.py)")
else:
    m = manifest.copy()
    m["size_mb"] = (m["bytes"] / 1048576).round(2)
    st.dataframe(m[["source", "files", "days", "versions_above_1", "size_mb", "first_key", "last_key"]].rename(columns={
        "source": "Source", "files": "Files stored", "days": "Days", "versions_above_1": "Files with a later version", "size_mb": "Size (MB)",
        "first_key": "First day", "last_key": "Last day"}), hide_index=True, width="stretch")

st.subheader("The quality gate")
if gate is None:
    ml_ui.not_yet_run("the quality gate")
else:
    passed = bool(gate.get("passed"))
    st.markdown(f"Latest decision: {ml_ui.badge('PASS' if passed else 'FAIL', 'done' if passed else 'not run')} "
                f"&nbsp; window {gate['window'][0]} to {gate['window'][1]}", unsafe_allow_html=True)
    for r in gate.get("reasons", []):
        st.markdown(f"- {r}")
    st.caption("The gate stops a model run when the data is not good enough. It currently fails on purpose: IDA1 has more empty auctions than "
               "its 6% limit allows, and one feed is older than allowed. Whether a day on which the source published an empty file should count as "
               "missing is an open design question, and the right fix is a rule that tells confirmed-empty days from ingestion gaps.")

st.subheader("Quality score over time")
if qd.empty:
    ml_ui.not_yet_run("the quality database")
else:
    ds = sorted(qd["dataset"].unique())
    pick = st.multiselect("Datasets", ds, default=ds[:4], key="ml1_ds")
    fig = go.Figure()
    for i, d in enumerate(pick):
        x = qd[qd["dataset"] == d].groupby("delivery_date")["score"].mean().sort_index()
        fig.add_trace(go.Scatter(x=x.index, y=x.rolling(14, min_periods=3).mean(), mode="lines", name=d,
                                 line=dict(color=theme.CATEGORICAL[i % len(theme.CATEGORICAL)], width=1.6)))
    st.plotly_chart(theme.style_fig(fig, height=330, yaxis_title="score (14-day mean)"), width="stretch")

st.subheader("Missing and empty days")
if status.empty:
    ml_ui.not_yet_run("the load log summary")
else:
    src = st.selectbox("Source", sorted(status["source"].unique()), index=sorted(status["source"].unique()).index("omie_ida1")
                       if "omie_ida1" in set(status["source"]) else 0, key="ml1_src")
    s = status[status["source"] == src].pivot_table(index="month", columns="status", values="days", aggfunc="sum").fillna(0)
    palette = {"loaded": theme.BLUE, "empty_publication": theme.ORANGE, "not_published": theme.RED, "rejected": theme.VIOLET}
    fig = go.Figure()
    for c in s.columns:
        fig.add_trace(go.Bar(x=s.index, y=s[c], name=c.replace("_", " "), marker_color=palette.get(c, theme.INK_MUTED)))
    st.plotly_chart(theme.style_fig(fig, height=330, yaxis_title="days", barmode="stack"), width="stretch")
    st.caption("`empty publication`: the source published a file with no data (the auction cleared nothing). `not published`: no file at all "
               "(HTTP 404). Empty IDA1 auctions have become more frequent since October 2025, which is why the IDA1 gate threshold trips.")

st.subheader("Time is part of the data")
st.markdown("Delivery days are local days, so a day has **23 hours** when the clocks go forward and **25** when they go back; the day-ahead "
            "market has 96 periods of 15 minutes (92 or 100 on those days) since 2025-10-01. The pipeline keeps UTC start times, "
            "parses every day against the calendar and rejects a file whose number of periods does not match. Three of these days are saved "
            "as real test files.")

