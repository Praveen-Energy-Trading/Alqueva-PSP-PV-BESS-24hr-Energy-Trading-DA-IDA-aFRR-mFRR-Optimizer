"""Console Log — raw log tail for the selected delivery date. Kept simple
and unfiltered on purpose: everything else on this dashboard is a curated
view of the same underlying run; this page is the ground truth to fall
back on when something needs double-checking."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import data
import theme

st.title("🖥️ Console Log")

@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    selected_date = st.session_state.get("selected_date")
    if not selected_date:
        st.warning("No runs found yet — visit Run & Monitor to start one.")
        return

    st.subheader(f"Pipeline console log — {selected_date}")
    log_text = data.load_log(selected_date)
    if not log_text:
        st.info(f"No log file found at `runtime/logs/pipeline_{selected_date}.log`. "
                f"This page mirrors your terminal — run the pipeline, then Refresh.")
    else:
        st.caption("The exact log your terminal/VS Code run wrote to disk — always in sync "
                   "regardless of whether you ran in Trader mode or --auto mode.")

        all_lines = log_text.splitlines()
        n_total = len(all_lines)

        # run_production.py truncates this file at the start of every run, so it
        # always holds exactly the current/most-recent run for this delivery
        # date — no stitched-together history from earlier attempts to wade
        # through. A single verbose run can still run long (bake-off logging,
        # trader-mode bid tables), so the tail default just keeps the most
        # recent output in view without scrolling.
        show_all = st.toggle("Show full log", value=False,
                              help=f"{n_total:,} lines in this run's log.")
        if show_all:
            st.code(log_text, language="text", line_numbers=True)
        else:
            tail_n = 300
            tail_lines = all_lines[-tail_n:]
            if n_total > tail_n:
                st.caption(f"Showing the last {tail_n} of {n_total:,} lines — turn on "
                           f"'Show full log' above to see the whole run.")
            st.code("\n".join(tail_lines), language="text", line_numbers=True)


_render()
