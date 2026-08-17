"""Decision Rationale — WHY each gate re-bid or held, straight from the
append-only audit trail. Previously this only existed as console scrollback:
ida_reoptimiser.py/xbid_optimiser.py compute improvement_eur and
dynamic_threshold_eur for every decision and log them via AuditLogger, but no
dashboard or report ever surfaced them until now."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme

st.title("🧭 Decision Rationale")
st.caption(
    "For IDA1-3 and XBID, a gate only re-bids if the expected improvement "
    "clears a dynamic threshold (0.15% of the day's DA position value, per "
    "`config/market.yaml`). This page shows that comparison directly from "
    "the audit trail — the same numbers printed in the console at decision "
    "time, not recomputed here."
)

selected_date = st.session_state.get("selected_date")
if not selected_date:
    st.warning("No runs found yet — visit Run & Monitor to start one.")
    st.stop()

GATES = ["IDA1", "IDA2", "IDA3", "XBID"]
DECISION_EVENTS = {"SUBMITTED", "NO_CHANGE", "ORDERS_PLACED", "NO_ORDER", "RISK_BLOCKED", "BIDCHECK_FAILED"}

rows = []
for gate in GATES:
    events = data.load_audit_events(selected_date, event_prefix=gate)
    for e in events:
        suffix = e["event"][len(gate) + 1:]  # e.g. "IDA1_SUBMITTED" -> "SUBMITTED"
        if suffix not in DECISION_EVENTS:
            continue
        rows.append({
            "Gate": gate,
            "Decision": suffix,
            "Time (CET)": e.get("timestamp_cet", "")[:19],
            "Improvement EUR": e.get("improvement_eur"),
            "Threshold EUR": e.get("dynamic_threshold_eur"),
            "DA value tradable EUR": e.get("da_value_tradable_eur"),
            "One-way vol MWh": e.get("one_way_vol_mwh"),
            "Reason": e.get("reason") or (str(e.get("violations")) if e.get("violations") else ""),
        })

if not rows:
    st.info(f"No gate re-optimisation decisions found in the audit trail for **{selected_date}** yet.")
    st.stop()

df = pd.DataFrame(rows).sort_values("Time (CET)", ascending=False)

st.subheader("All decisions this delivery date")
st.dataframe(df, width="stretch", height=min(400, 60 + 35 * len(df)))

st.markdown("---")

st.subheader("Improvement vs. threshold — the 'why' behind each re-bid")
priced = df.dropna(subset=["Improvement EUR", "Threshold EUR"])
if priced.empty:
    st.info("No priced (SUBMITTED/NO_CHANGE) decisions yet — only START/blocked events so far.")
else:
    # Latest decision per gate, so the chart isn't cluttered by repeated test runs.
    latest = priced.sort_values("Time (CET)").groupby("Gate", as_index=False).last()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=latest["Gate"], y=latest["Improvement EUR"], name="Improvement EUR",
                          marker_color=[theme.STATUS_GOOD if v >= t else theme.STATUS_CRITICAL
                                        for v, t in zip(latest["Improvement EUR"], latest["Threshold EUR"])]))
    fig.add_trace(go.Scatter(x=latest["Gate"], y=latest["Threshold EUR"], name="Re-bid threshold EUR",
                              mode="markers", marker=dict(size=14, symbol="line-ew", color=theme.INK_SECONDARY)))
    theme.style_fig(fig, height=380, yaxis_title="EUR")
    st.plotly_chart(fig, width="stretch")
    st.caption("Bar = expected improvement from re-bidding. Black tick = the bar it had to clear. "
               "Green = cleared it (re-bid); red = didn't (held position). Note: XBID's threshold "
               "(spread x volume) isn't logged as a separate field, so XBID never appears in this "
               "chart — see the full table above for its improvement_eur instead.")

st.markdown("---")
st.subheader("Blocked or failed decisions")
blocked = df[df["Decision"].isin(["RISK_BLOCKED", "BIDCHECK_FAILED"])]
if blocked.empty:
    st.success("No decisions were blocked by the physical bid checker or pre-trade risk checker.")
else:
    st.warning(f"{len(blocked)} decision(s) were blocked even though a re-bid may have been economically justified:")
    st.dataframe(blocked[["Gate", "Decision", "Time (CET)", "Reason"]], width="stretch")
