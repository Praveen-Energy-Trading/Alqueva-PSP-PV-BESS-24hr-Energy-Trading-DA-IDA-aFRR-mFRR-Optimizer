"""Pipeline Flowchart — a real, credits-scroll trace of run_production.py's
own 19-phase execution for the selected delivery date. Every phase shown
mirrors run_production.py's _PHASES table exactly; every status/detail/
elapsed comes from that day's real run_status_<date>.json (the same file
Run & Monitor's health banner reads) -- not a simplification and not
re-simulated, just the pipeline's own real structure and real outcome made
visible and scrollable. See dashboard/pipeline_flow.py for the render
function."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components

import data
import pipeline_flow
import theme

st.title("🔀 Pipeline Flowchart")


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    selected_date = st.session_state.get("selected_date")
    if not selected_date:
        st.warning("No runs found yet — visit Run & Monitor to start one.")
        return

    st.caption(
        "A real trace of run_production.py's own 19 phases for this delivery date -- "
        "grouped into the same 4 stages the phase numbering already implies (1-3, 4, 5, 6), "
        "not a simplification. Every status, detail, and elapsed time below is real, from "
        "that day's run_status file."
    )
    run_status = data.load_run_status(selected_date)
    components.html(
        pipeline_flow.render_pipeline_flow_card(run_status, selected_date),
        height=620,
    )

    st.markdown("---")
    st.caption(
        "Every line below is a real line from that day's actual pipeline log "
        "(runtime/logs/pipeline_<date>.log) -- same file Console Log shows in full. "
        "Only structured milestone lines are animated here; raw data output (bid "
        "tables etc.) stays on Console Log rather than being replayed one row at a time."
    )
    log_text = data.load_log(selected_date)
    components.html(
        pipeline_flow.render_log_trace_card(log_text),
        height=460,
    )


_render()
