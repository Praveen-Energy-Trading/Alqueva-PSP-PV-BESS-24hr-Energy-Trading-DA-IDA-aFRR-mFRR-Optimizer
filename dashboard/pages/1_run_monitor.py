"""Run & Monitor - trigger a pipeline run from here and watch it live, with a
machine-readable health banner instead of guessing status from log text."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import data
import runner
import theme

st.title("🚀 Run & Monitor")

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

    STATUS_ICON = theme.STATUS_ICON

    if selected_date:
        run_status = data.load_run_status(selected_date)
        state = data.run_phase_state(selected_date)
        if state == "running":
            st.success(f"🟢 Pipeline is actively running right now for delivery **{selected_date}** "
                       f" -  log is updating live. See the live console below, or the Console Log "
                       f"page for the full tail. Status here fills in once it finishes.")
        elif state == "idle_running":
            if run_status and run_status.get("mode") == "trader":
                st.warning(f"🟡 A run is in progress for **{selected_date}** but the log has "
                           f"gone quiet - most likely paused on an Approve/Reject or ENTER "
                           f"prompt waiting on you in trader mode (not started from this "
                           f"dashboard, so it can't be answered here). Check the terminal "
                           f"it was started from, or the last line in Console Log.")
            else:
                st.warning(f"🟡 A run is in progress for **{selected_date}** but the log has "
                           f"gone quiet - most likely still computing (e.g. a slow model "
                           f"fit) rather than stuck. Check Console Log for the last line printed.")
        elif state == "stopped":
            st.error(f"⚫ A run for **{selected_date}** was started but the process is no longer "
                     f"running - most likely Ctrl+C or a crash while it was paused waiting for "
                     f"input (Approve/Reject or ENTER). Start a fresh run when ready.")
        elif state == "none":
            st.info(f"No run-status record for **{selected_date}** yet - either it hasn't "
                    f"run yet, or it predates this dashboard's status-tracking (older runs "
                    f"only have the raw console log).")
        else:
            results = run_status["results"]
            counts = {s: sum(1 for r in results if r["status"] == s) for s in ("PASS", "SKIP", "WARN", "FAIL")}
            n_fail = counts["FAIL"]
            n_total = len(results)

            if n_fail:
                failed = [r for r in results if r["status"] == "FAIL"]
                st.error(f"🔴 **{selected_date}** - {n_fail} phase(s) FAILED "
                         f"(finished {run_status['finished_at']}, mode={run_status['mode']})")
                for r in failed:
                    st.caption(f"  ✗ **{r['key']}**: {r['detail']}")
            elif counts["WARN"]:
                st.warning(f"🟡 **{selected_date}** - {counts['PASS']}/{n_total} passed, "
                           f"{counts['WARN']} warned (finished {run_status['finished_at']})")
            else:
                st.success(f"🟢 **{selected_date}** - {counts['PASS']}/{n_total} phases passed cleanly "
                           f"(finished {run_status['finished_at']}, mode={run_status['mode']})")

            with st.expander("Per-phase breakdown"):
                for r in results:
                    icon = STATUS_ICON.get(r["status"], "❓")
                    st.markdown(f"{icon} **{r['key']}** - {r['status']}"
                                + (f" - {r['detail']}" if r["detail"] else ""))

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Run controls
    # ---------------------------------------------------------------------------

    st.subheader("Run the pipeline from here")
    st.caption(
        "Launches `run_production.py --auto` as a subprocess and streams its "
        "console output live below - the same log a terminal run would show. "
        "`--auto` is required: interactive Approve/Reject and Enter-to-continue "
        "prompts can't be answered through a subprocess pipe."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    date_arg = col1.text_input(
        "Delivery date (DD-MM-YYYY, or 'auto' for tomorrow)", value="auto",
        disabled=runner_state["running"],
    )
    synthetic = col2.toggle("Synthetic data", value=False, disabled=runner_state["running"],
                             help="Off = live OMIE/REN fetch with synthetic fallback (production mode). "
                                  "On = fully synthetic, no network calls (fast backtest mode).")
    col3.write("")
    col3.write("")

    b1, b2 = st.columns([1, 1])
    # on_click callbacks run *before* the script reruns and widgets are
    # re-instantiated, so it's safe for runner.start to touch
    # st.session_state["auto_refresh_toggle"] there. Doing the same thing inline
    # in the script body raises StreamlitAPIException because the sidebar
    # toggle widget (same key) was already instantiated in app.py this run.
    b1.button("▶️ Run Pipeline", disabled=runner_state["running"], type="primary",
              width="stretch",
              on_click=runner.start, args=(date_arg.strip(), synthetic))
    b2.button("⏹️ Stop", disabled=not runner_state["running"], width="stretch",
              on_click=runner.stop)

    st.markdown("---")

    if runner_state["proc"] is not None:
        # A run started from THIS page - stream its captured stdout directly,
        # the most immediate view possible (no file round-trip).
        if runner_state["running"]:
            st.success(f"🟢 Running for delivery date **{runner_state['date']}** "
                       f"(PID {runner_state['proc'].pid})...")
        elif runner_state["return_code"] == 0:
            st.success(f"✅ Finished for delivery date **{runner_state['date']}** - exit code 0.")
        else:
            st.error(f"❌ Finished for delivery date **{runner_state['date']}** - exit code "
                     f"{runner_state['return_code']}. Check the log below for what failed.")

        st.subheader("Live console")
        console_text = "\n".join(runner_state["lines"][-2000:])
        st.code(console_text or "(waiting for output...)", language="text", line_numbers=True)
    elif selected_date and data.is_pipeline_active(selected_date):
        # No run started from this dashboard session, but SOMETHING (VS Code,
        # terminal, Spyder) is actively writing to this date's log right now  - 
        # mirror it here too, not just on the separate Console Log page, so
        # this page reflects reality regardless of where the run was started.
        st.success(f"🟢 An external run is actively writing to **{selected_date}**'s log "
                   f"right now (not started from this dashboard, but mirrored live below).")
        st.subheader("Live console (tailing pipeline_{}.log)".format(selected_date))
        log_text = data.load_log(selected_date)
        tail = "\n".join(log_text.splitlines()[-300:])
        st.code(tail or "(waiting for output...)", language="text", line_numbers=True)
    elif selected_date:
        st.info("No run started yet this session, and nothing is actively writing to "
                f"**{selected_date}**'s log. Set the options above and click Run Pipeline, "
                f"or run it externally (VS Code/terminal) and this page will pick it up live.")
    else:
        st.info("No run started yet this session. Set the options above and click Run Pipeline.")


_render()
