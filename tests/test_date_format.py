"""Every date the dashboard shows is dd-mm-yyyy; identifiers and file names that contain a date are left alone."""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))

from date_format import fmt, format_plotly_dates  # noqa: E402


def test_iso_dates_in_text_become_day_month_year():
    assert fmt("2026-10-14") == "14-10-2026"
    assert fmt("window 2019-01-01 to 2026-10-12") == "window 01-01-2019 to 12-10-2026"
    assert fmt("decided 2026-10-13T09:26:24") == "decided 13-10-2026 09:26:24"
    assert fmt("(2026-10-14).") == "(14-10-2026)."


def test_identifiers_file_names_and_other_numbers_are_not_touched():
    for text in ("backtest_2025-10-01_365d.xlsx", "gatebtn_2026-10-14_DA", "a/2026-10-14/b", "2026-10", "x2026-10-14", "2026-10-14abc"):
        assert fmt(text) == text
    assert fmt(42) == 42


def test_only_the_axes_with_dates_get_the_format():
    dated = go.Figure(go.Bar(x=pd.date_range("2026-01-01", periods=3), y=[1, 2, 3]))
    format_plotly_dates(dated)
    assert dated.layout.xaxis.tickformat == "%d-%m-%Y" and dated.layout.xaxis.hoverformat == "%d-%m-%Y"
    numeric = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    format_plotly_dates(numeric)
    assert numeric.layout.xaxis.tickformat is None
