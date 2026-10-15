"""Figures - the eight figures of the pipeline (figures/output/*.svg), one under the other, each as wide as the page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import data
import ml_ui
import theme

st.title("🖼️ Figures")


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    svgs = sorted(data.FIGURES_DIR.glob("*.svg"))
    if not svgs:
        st.info("No figures found - run the pipeline to generate them.")
        return
    for svg in svgs:
        ml_ui.figure(svg)


_render()
