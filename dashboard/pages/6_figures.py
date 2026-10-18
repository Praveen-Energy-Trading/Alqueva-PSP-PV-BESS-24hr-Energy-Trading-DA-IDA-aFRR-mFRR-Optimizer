"""Figures - the eight figures of the pipeline (figures/output/*.svg), one under the other, each as wide as the page, in four groups with a heading and a line on what the group shows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import data
import ml_ui
import theme

st.title("🖼️ Figures")

# the figures carry their own title and subtitle, so the page only groups them (drawn by figures/pipeline_figures.py, day_figures.py and backtest_figures.py)
GROUPS = [
    ("The trading day", "When each market decides, in the order of the day", ("01_trading_day",)),
    ("One solved day", "The plan, the capacity stack, the water and the battery, and the efficiency of the machine, all from one solved day",
     ("02_day_plan", "03_capacity_stack", "04_water_and_battery", "05_efficiency_surface")),
    ("Forecast quality", "How good the price forecasts are on a period they never saw during tuning", ("06_forecast_skill",)),
    ("A year of results", "Where the money comes from, and how bad a bad day can be, from the rolling backtest", ("07_revenue_sources", "08_risk")),
]
NUMBERS = "①②③④⑤⑥"


def _head(num: str, title: str, hint: str = "") -> None:
    """A group heading: a number, a bold title and a muted line on what the group shows."""
    st.markdown(f"<div style='margin:1.6rem 0 0.4rem;'><div style='font-size:1.3rem; font-weight:700; color:{theme.INK_PRIMARY}; line-height:1.25;'>"
                f"<span style='font-weight:400; color:#6b6a64;'>{num}</span>&nbsp; {title}</div>"
                + (f"<div style='font-size:0.86rem; color:{theme.INK_MUTED}; margin-top:0.15rem;'>{hint}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    svgs = {p.stem: p for p in sorted(data.FIGURES_DIR.glob("*.svg"))}
    if not svgs:
        st.markdown('<div style="background:#eef1f8; color:#36415c; border-radius:12px; padding:8px 16px; font-size:14px; line-height:1.45; margin:10px 0;">'
                    "No figures found. Run the pipeline to generate them.</div>", unsafe_allow_html=True)
        return
    shown = set()
    n = 0
    for title, hint, stems in GROUPS:
        group = [svgs[s_] for s_ in stems if s_ in svgs]
        if not group:
            continue
        _head(NUMBERS[n], title, hint)
        n += 1
        for svg in group:
            ml_ui.figure(svg)
            shown.add(svg.stem)
    extra = [p for s_, p in svgs.items() if s_ not in shown]                                   # a figure that is not in a group is still shown, after the groups
    if extra:
        _head(NUMBERS[n] if n < len(NUMBERS) else str(n + 1), "More figures")
        for svg in extra:
            ml_ui.figure(svg)


_render()
