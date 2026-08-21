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
st.caption("Did the expected gain clear the re-bid threshold? Green cleared it, red didn't.")

@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    selected_date = st.session_state.get("selected_date")
    if not selected_date:
        st.warning("No runs found yet — visit Run & Monitor to start one.")
        return

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
                "Window": e.get("window") or "-",
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
        return

    df_all = pd.DataFrame(rows).sort_values("Time (CET)")
    # A pipeline re-run (retry, resumed run) writes the same decision again
    # with a new timestamp -- the audit trail is append-only, so without
    # this the table shows the same IDA1/IDA2/IDA3 decision 3x and looks
    # like 3x more re-bid activity happened than actually did. Keep only
    # each (Gate, Window)'s most recent record -- XBID genuinely has up to
    # 6 real windows (W1-W6), which "Window" keeps distinct from reruns.
    df = df_all.groupby(["Gate", "Window"], as_index=False).last().sort_values("Time (CET)", ascending=False)

    st.subheader("All decisions this delivery date")
    st.dataframe(
        df.style.format({"Improvement EUR": "{:.2f}", "Threshold EUR": "{:.2f}",
                          "DA value tradable EUR": "{:.0f}", "One-way vol MWh": "{:.2f}"}),
        width="stretch", height=min(400, 60 + 35 * len(df)),
    )

    st.markdown("---")

    st.subheader("Improvement vs. threshold — the 'why' behind each re-bid")
    priced = df.dropna(subset=["Improvement EUR", "Threshold EUR"])
    if priced.empty:
        st.info("No priced (SUBMITTED/NO_CHANGE) decisions yet — only START/blocked events so far.")
    else:
        # IDA1-3 re-bid the whole gate at once (one threshold each); XBID
        # re-evaluates per rolling window (up to 6/day), so it needs its
        # own small multiple rather than being squeezed onto the same axis.
        ida = priced[priced["Gate"] != "XBID"].sort_values("Time (CET)").groupby("Gate", as_index=False).last()
        xbid = priced[priced["Gate"] == "XBID"].sort_values("Window")

        chart_cols = st.columns([2, 1]) if not xbid.empty else [st.container()]
        with chart_cols[0]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ida["Gate"], y=ida["Improvement EUR"], name="Improvement EUR",
                                  marker_color=[theme.STATUS_GOOD if v >= t else theme.STATUS_CRITICAL
                                                for v, t in zip(ida["Improvement EUR"], ida["Threshold EUR"])]))
            fig.add_trace(go.Scatter(x=ida["Gate"], y=ida["Threshold EUR"], name="Re-bid threshold EUR",
                                      mode="markers", marker=dict(size=14, symbol="line-ew", color=theme.INK_SECONDARY)))
            theme.style_fig(fig, height=340, yaxis_title="EUR", legend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("Green cleared the threshold (re-bid) · red didn't (held).")

        if not xbid.empty:
            with chart_cols[1]:
                fig_x = go.Figure(go.Bar(
                    x=xbid["Window"], y=xbid["Improvement EUR"],
                    marker_color=[theme.STATUS_GOOD if v >= t else theme.STATUS_CRITICAL
                                  for v, t in zip(xbid["Improvement EUR"], xbid["Threshold EUR"])],
                ))
                theme.style_fig(fig_x, height=340, yaxis_title="EUR", legend=False)
                st.plotly_chart(fig_x, width="stretch")
                st.caption("XBID re-evaluates every rolling window separately.")

    st.markdown("---")
    st.subheader("Blocked or failed decisions")
    blocked = df[df["Decision"].isin(["RISK_BLOCKED", "BIDCHECK_FAILED"])]
    if blocked.empty:
        st.success("No decisions were blocked by the physical bid checker or pre-trade risk checker.")
    else:
        st.warning(f"{len(blocked)} decision(s) were blocked even though a re-bid may have been economically justified:")
        st.dataframe(blocked[["Gate", "Decision", "Time (CET)", "Reason"]], width="stretch")


_render()
