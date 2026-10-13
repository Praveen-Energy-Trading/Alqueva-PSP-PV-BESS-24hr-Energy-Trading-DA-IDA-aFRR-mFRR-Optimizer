"""
date_format.py - every date the dashboard shows is written day-month-year (dd-mm-yyyy).

The data, the files and the code keep ISO dates (2026-10-14); only what is displayed changes. `install()` wraps the display functions of Streamlit once
per process, so no page has to call a formatter:

    text elements (markdown, caption, write, header ... metric, expander) and the HTML of components.html
        every ISO date inside the text, also 2026-10-13T09:26:24 (the 'T' becomes a space), is rewritten;
        a date inside a file name or identifier (backtest_2025-10-01_365d.xlsx) is left alone
    selection widgets (selectbox, radio, segmented_control, multiselect, pills)
        their labels are rewritten the same way
    plotly_chart
        the axes that hold dates get tick labels and hover values in dd-mm-yyyy

    fmt(text) -> str      the rewrite of one string
"""
from __future__ import annotations

import functools
import inspect
import re
from typing import Any

# yyyy-mm-dd, optionally followed by 'T' and a time; not part of a longer word, file name or number
_DATE = re.compile(r"(?<![\w/.\-])(\d{4})-(\d{2})-(\d{2})(?:(T)(?=\d)|(?![\w\-]))")

TEXT_FUNCTIONS = ("markdown", "caption", "text", "header", "subheader", "title", "success", "info", "warning", "error", "write", "metric", "expander", "code")
CHOICE_FUNCTIONS = ("selectbox", "radio", "multiselect", "segmented_control", "pills", "select_slider")
TEXT_KEYWORDS = ("body", "label", "value", "delta")


def fmt(text: Any) -> Any:
    """Every ISO date in `text` as dd-mm-yyyy; a value that is not a string is returned as it is."""
    if not isinstance(text, str) or "-" not in text:
        return text
    return _DATE.sub(lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}" + (" " if m.group(4) else ""), text)


def _is_date_like(values) -> bool:
    try:
        import numpy as np
        if np.issubdtype(np.asarray(values[:3]).dtype, np.datetime64):
            return True
    except Exception:
        pass
    try:
        first = next((v for v in list(values)[:3] if v is not None), None)
    except TypeError:
        return False
    if first is None:
        return False
    if hasattr(first, "year") and hasattr(first, "month"):
        return True
    return isinstance(first, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?", first))


def format_plotly_dates(fig) -> None:
    """dd-mm-yyyy on the ticks and in the hover box of every x axis that holds dates."""
    try:
        axes = set()
        for trace in fig.data:
            xs = getattr(trace, "x", None)
            if xs is not None and _is_date_like(xs):
                axes.add(getattr(trace, "xaxis", None) or "x")
        for name in axes:
            fig.update_layout({"xaxis" + name[1:]: dict(tickformat="%d-%m-%Y", hoverformat="%d-%m-%Y")})
    except Exception:
        pass


def _text_wrapper(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        args = [fmt(a) for a in args]
        for key in TEXT_KEYWORDS:
            if key in kwargs:
                kwargs[key] = fmt(kwargs[key])
        return original(self, *args, **kwargs)
    return wrapper


def _choice_wrapper(original):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError:
            return original(self, *args, **kwargs)
        given = bound.arguments.get("format_func")
        if "format_func" in signature.parameters:
            bound.arguments["format_func"] = (lambda x: fmt(given(x))) if given else (lambda x: fmt(str(x)))
        return original(*bound.args, **bound.kwargs)
    return wrapper


def _slider_wrapper(original):
    """A slider over dates shows its values as DD-MM-YYYY."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError:
            return original(self, *args, **kwargs)
        values = [bound.arguments.get(k) for k in ("min_value", "max_value", "value")]
        if any(hasattr(v, "year") for v in values) and bound.arguments.get("format") is None:
            bound.arguments["format"] = "DD-MM-YYYY"
        return original(*bound.args, **bound.kwargs)
    return wrapper


def install() -> None:
    """Wrap the display functions of Streamlit (once per process)."""
    import streamlit as st
    import streamlit.components.v1 as components
    from streamlit.delta_generator import DeltaGenerator

    if getattr(st, "_dates_dd_mm_yyyy", False):
        return
    st._dates_dd_mm_yyyy = True
    for name in TEXT_FUNCTIONS:
        if hasattr(DeltaGenerator, name):
            setattr(DeltaGenerator, name, _text_wrapper(getattr(DeltaGenerator, name)))
    for name in CHOICE_FUNCTIONS:
        if hasattr(DeltaGenerator, name):
            setattr(DeltaGenerator, name, _choice_wrapper(getattr(DeltaGenerator, name)))
    DeltaGenerator.slider = _slider_wrapper(DeltaGenerator.slider)
    for name in TEXT_FUNCTIONS + CHOICE_FUNCTIONS + ("slider",):                      # st.markdown and the others were bound to the main container when streamlit was imported
        if hasattr(st, name) and hasattr(st, "_main"):
            setattr(st, name, getattr(st._main, name))

    original_html = components.html

    @functools.wraps(original_html)
    def html(source, *args, **kwargs):
        return original_html(fmt(source), *args, **kwargs)
    components.html = html

    original_plotly = DeltaGenerator.plotly_chart

    @functools.wraps(original_plotly)
    def plotly_chart(self, figure_or_data, *args, **kwargs):
        format_plotly_dates(figure_or_data)
        return original_plotly(self, figure_or_data, *args, **kwargs)
    DeltaGenerator.plotly_chart = plotly_chart
    st.plotly_chart = st._main.plotly_chart
