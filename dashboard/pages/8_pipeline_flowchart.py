"""Pipeline Flowchart - the real MILP formulation, explained.

Static reference material: every equation is copied verbatim from
common_layer/optimisation_model/core_milp_builder.py and core_milp_solver.py
(72 equations, none omitted). No ComponentStore/AuditStore data is read here
-- this page explains the model's structure, which doesn't change day to
day, not a particular day's solved result.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit.components.v1 as components
import streamlit as st

import pipeline_flow

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] * { font-family: 'Times New Roman', Times, serif !important; }
h1, h2, h3, [data-testid="stHeading"] { font-weight: 700 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔀 Pipeline Flowchart")

st.subheader("The Repetition Loop")
components.html(pipeline_flow.gate_repetition_loop_html(), height=226, scrolling=True)

st.subheader("Input Data")
components.html(pipeline_flow.pipeline_and_model_html(), height=4735, scrolling=True)

st.subheader("What Actually Happens, Gate By Gate")
components.html(pipeline_flow.pipeline_steps_plain_html(), height=905, scrolling=True)
