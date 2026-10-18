"""Run & Monitor - trigger a pipeline run from here and watch it live, with a
machine-readable health banner instead of guessing status from log text."""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import data
import runner
import theme

st.title("🚀 Run & Monitor")

CARD_LINE = "#e3e1d9"
_BANNER = {"good": ("#e6f4ea", "#1e6b33"), "warn": ("#fdf0d0", "#7a5200"), "bad": ("#fdecea", "#9b2c2c"), "info": ("#eef1f8", "#36415c")}
_PILL = {"PASS": ("#006100", "#e3f1e3"), "SKIP": ("#5a5953", "#ecebe6"), "WARN": ("#7a5200", "#fdf0d0"), "FAIL": ("#8a1f1f", "#f7e3e3")}


def _head(title: str, hint: str = "") -> None:
    """A section heading: bold, sentence case, with a muted line under it on how to read the block."""
    st.markdown(f"<div style='margin:1.4rem 0 0.4rem;'><div style='font-size:1.3rem; font-weight:700; color:{theme.INK_PRIMARY}; line-height:1.25;'>{title}</div>"
                + (f"<div style='font-size:0.86rem; color:{theme.INK_MUTED}; margin-top:0.15rem;'>{hint}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _banner(kind: str, html_text: str) -> None:
    """A run status as one slim, coloured line: green is good, amber needs a look, red failed, blue is for information."""
    bg, fg = _BANNER[kind]
    st.markdown(f'<div style="background:{bg}; color:{fg}; border-radius:12px; padding:8px 16px; font-size:14px; line-height:1.45; margin:10px 0;">{html_text}</div>', unsafe_allow_html=True)


def _pill(status: str) -> str:
    fg, bg = _PILL.get(status, ("#5a5953", "#ecebe6"))
    return f"<span style='background:{bg}; color:{fg}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{html.escape(status.capitalize())}</span>"


def _phase_table(results: list) -> None:
    """The phases of a run as a table: muted header, thin row lines, a status pill per phase."""
    grid = "display:grid; grid-template-columns:minmax(12rem, 2fr) minmax(5.5rem, 0.7fr) minmax(14rem, 3.2fr); align-items:center; column-gap:0.6rem;"
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{theme.INK_MUTED}; font-weight:600;'>"
            "<span>Phase</span><span style='text-align:center;'>Status</span><span>Detail</span></div>")
    body = "".join(f"<div style='{grid} min-height:2.4rem; font-size:0.86rem; border-bottom:1px solid #eeede8;'>"
                   f"<span style='color:{theme.INK_PRIMARY};'>{html.escape(str(r['key']))}</span><span style='text-align:center;'>{_pill(r['status'])}</span>"
                   f"<span style='color:{theme.INK_SECONDARY};'>{html.escape(str(r['detail'])) if r['detail'] else '–'}</span></div>" for r in results)
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{head}{body}</div>", unsafe_allow_html=True)


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    # runner.start() (an on_click callback) can't force app.py's sidebar to
    # redraw itself via st.rerun() - that call is a no-op inside a callback.
    # It leaves this flag instead; from here (regular fragment body, not a
    # callback) st.rerun(scope="app") actually works, forcing the one extra
    # full-page rerun the sidebar's auto-refresh toggle needs to pick up the
    # True that start() just wrote to session_state.
    if st.session_state.pop("_just_started_run", False):
        st.rerun(scope="app")
    runner_state = st.session_state.runner
    selected_date = st.session_state.get("selected_date")

    # ---------------------------------------------------------------------------
    # Health banner - reads runtime/logs/run_status_<date>.json (written by
    # run_production.py at the end of every run, including dry-run). This is the
    # machine-readable twin of the console's PASS/SKIP/WARN/FAIL table, so the
    # banner reflects a real outcome instead of parsing scrollback text.
    # ---------------------------------------------------------------------------

    if selected_date:
        _head("Status of the run", f"Delivery {selected_date}")
        run_status = data.load_run_status(selected_date)
        state = data.run_phase_state(selected_date)
        if state == "running":
            _banner("good", f"🟢 The pipeline is running now for delivery <b>{selected_date}</b>. The log is updating live: see the live console below, or the Console Log "
                            "page for the full tail. The status here fills in once the run finishes.")
        elif state == "idle_running":
            if run_status and run_status.get("mode") == "trader":
                _banner("warn", f"🟡 A run is in progress for <b>{selected_date}</b> but the log has gone quiet. Most likely it is paused on an Approve/Reject or ENTER prompt, "
                                "waiting for you in trader mode. It was not started from this dashboard, so it cannot be answered here: check the terminal it was started from, "
                                "or the last line in the Console Log.")
            else:
                _banner("warn", f"🟡 A run is in progress for <b>{selected_date}</b> but the log has gone quiet. Most likely it is still computing (for example a slow model fit) "
                                "rather than stuck. Check the Console Log for the last line printed.")
        elif state == "stopped":
            _banner("bad", f"⚫ A run for <b>{selected_date}</b> was started but the process is no longer running. Most likely it was stopped with Ctrl+C, "
                           "or it crashed while paused waiting for input (Approve/Reject or ENTER). Start a fresh run when ready.")
        elif state == "none":
            _banner("info", f"There is no run-status record for <b>{selected_date}</b> yet: either it has not run, or it predates the status tracking of this dashboard "
                            "(older runs only have the raw console log).")
        else:
            results = run_status["results"]
            counts = {s: sum(1 for r in results if r["status"] == s) for s in ("PASS", "SKIP", "WARN", "FAIL")}
            n_fail = counts["FAIL"]
            n_total = len(results)

            if n_fail:
                failed = [r for r in results if r["status"] == "FAIL"]
                _banner("bad", f"🔴 <b>{selected_date}</b>: {n_fail} phase{'' if n_fail == 1 else 's'} failed (finished {run_status['finished_at']}, mode: {run_status['mode']})")
                for r in failed:
                    st.caption(f"✗ **{r['key']}**: {r['detail']}")
            elif counts["WARN"]:
                _banner("warn", f"🟡 <b>{selected_date}</b>: {counts['PASS']} of {n_total} phases passed, {counts['WARN']} warned (finished {run_status['finished_at']})")
            else:
                _banner("good", f"🟢 <b>{selected_date}</b>: {counts['PASS']} of {n_total} phases passed cleanly (finished {run_status['finished_at']}, mode: {run_status['mode']})")

            with st.expander(f"Per-phase breakdown: {counts['PASS']} passed, {counts['SKIP']} skipped, {counts['WARN']} warned, {counts['FAIL']} failed"):
                _phase_table(results)

    # ---------------------------------------------------------------------------
    # Run controls
    # ---------------------------------------------------------------------------

    _head("Run the pipeline from here", "Starts a run and streams its console output live below")
    st.caption(
        "This launches `run_production.py --auto` as a subprocess and shows its console output live, the same log a terminal run would print. "
        "`--auto` is required: the interactive Approve/Reject and Enter-to-continue prompts cannot be answered through a subprocess pipe."
    )

    col1, col2 = st.columns([2, 2], gap="medium")
    date_arg = col1.text_input(
        "Delivery date (DD-MM-YYYY, or 'auto' for tomorrow)", value="auto",
        disabled=runner_state["running"],
    )
    col2.write("")
    synthetic = col2.toggle("Offline demo (synthetic data)", value=False, disabled=runner_state["running"],
                             help="Off = downloads the latest OMIE and REN data; an input whose download fails falls back to synthetic values, "
                                  "and a warning is logged. On = no downloads at all: missing history is filled with synthetic values (fast demo mode).")

    b1, b2 = st.columns([1, 1], gap="medium")
    # on_click callbacks run *before* the script reruns and widgets are
    # re-instantiated, so it's safe for runner.start to touch
    # st.session_state["auto_refresh_toggle"] there. Doing the same thing inline
    # in the script body raises StreamlitAPIException because the sidebar
    # toggle widget (same key) was already instantiated in app.py this run.
    b1.button("▶️ Run pipeline", disabled=runner_state["running"], type="primary",
              width="stretch",
              on_click=runner.start, args=(date_arg.strip(), synthetic))
    b2.button("⏹️ Stop", disabled=not runner_state["running"], width="stretch",
              on_click=runner.stop)

    if runner_state["proc"] is not None:
        # A run started from THIS page - stream its captured stdout directly,
        # the most immediate view possible (no file round-trip).
        if runner_state["running"]:
            _banner("good", f"🟢 Running for delivery date <b>{runner_state['date']}</b> (PID {runner_state['proc'].pid}).")
        elif runner_state["return_code"] == 0:
            _banner("good", f"✅ Finished for delivery date <b>{runner_state['date']}</b> with exit code 0.")
        else:
            _banner("bad", f"❌ Finished for delivery date <b>{runner_state['date']}</b> with exit code {runner_state['return_code']}. Check the log below for what failed.")

        _head("Live console")
        console_text = "\n".join(runner_state["lines"][-2000:])
        st.code(console_text or "(waiting for output...)", language="text", line_numbers=True)
    elif selected_date and data.is_pipeline_active(selected_date):
        # No run started from this dashboard session, but SOMETHING (VS Code,
        # terminal, Spyder) is actively writing to this date's log right now  -
        # mirror it here too, not just on the separate Console Log page, so
        # this page reflects reality regardless of where the run was started.
        _banner("good", f"🟢 An external run is writing to the log of <b>{selected_date}</b> right now. It was not started from this dashboard, but it is mirrored live below.")
        _head(f"Live console", f"Tailing pipeline_{selected_date}.log")
        log_text = data.load_log(selected_date)
        tail = "\n".join(log_text.splitlines()[-300:])
        st.code(tail or "(waiting for output...)", language="text", line_numbers=True)
    elif selected_date:
        _banner("info", f"No run has been started in this session, and nothing is writing to the log of <b>{selected_date}</b>. Set the options above and click Run pipeline, "
                        "or run it externally (VS Code or a terminal) and this page will pick it up live.")
    else:
        _banner("info", "No run has been started in this session. Set the options above and click Run pipeline.")


_render()
