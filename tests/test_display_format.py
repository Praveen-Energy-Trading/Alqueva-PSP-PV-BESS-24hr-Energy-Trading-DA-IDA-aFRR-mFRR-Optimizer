"""test_display_format.py - a real value never shows as zero: percentages ("<0.1%") and table numbers ("<0.01")."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))

import html_table
import theme


def test_a_small_share_is_not_shown_as_zero_or_a_large_one_as_100():
    assert theme.pct(0.035) == "<0.1%" and theme.pct(0.0) == "0.0%" and theme.pct(0.2) == "0.2%"
    assert theme.pct(99.97) == ">99.9%" and theme.pct(100.0) == "100.0%" and theme.pct(57.14) == "57.1%"
    assert theme.pct(-0.01) == ">-0.1%" and theme.pct(-3.0, signed=True) == "-3.0%" and theme.pct(None) == "-"


def test_a_small_table_number_is_not_shown_as_zero():
    assert html_table._fmt_number(0.004, "%.2f") == "<0.01" and html_table._fmt_number(-0.004, "%.2f") == ">-0.01"
    assert html_table._fmt_number(0.0, "%.2f") == "0.00" and html_table._fmt_number(1.236, "%.2f") == "1.24"
    assert html_table._fmt_number(0.004, None) == "<0.01"
