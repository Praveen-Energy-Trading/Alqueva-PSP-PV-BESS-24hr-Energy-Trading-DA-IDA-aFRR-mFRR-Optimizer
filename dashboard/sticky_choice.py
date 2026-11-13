"""A segmented control that always keeps one option selected.

st.segmented_control lets the user click the active option again, which
deselects it and sets the value to None. Pages that then look the value up in
their list of labels either stop drawing the control (the section vanishes) or
show nothing selected. This wraps the control so that a click on the active
option puts the same option back, in the on_change callback that runs before
the rerun, so the control never renders without a selection.
"""
from __future__ import annotations

import streamlit as st


def _keep_selection(key: str) -> None:
    last_key = f"{key}__last"
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(last_key)
    else:
        st.session_state[last_key] = st.session_state[key]


def segmented_control(label: str, options: list, key: str, default=None, **kwargs):
    """Drop-in for st.segmented_control(selection_mode="single") that cannot be deselected."""
    options = list(options)
    last_key = f"{key}__last"
    if key not in st.session_state:
        first = default if default in options else options[0]
        st.session_state[key] = first
        st.session_state[last_key] = first
    elif st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(last_key, options[0])
    # the page may have seeded `key` itself, so the remembered value can be missing
    st.session_state.setdefault(last_key, st.session_state[key] if st.session_state[key] in options else options[0])
    value = st.segmented_control(label, options, key=key, on_change=_keep_selection, args=(key,), **kwargs)
    if value is None:  # defensive: callback could not run (e.g. widget just re-created)
        value = st.session_state.get(last_key, options[0])
    else:
        st.session_state[last_key] = value
    return value
