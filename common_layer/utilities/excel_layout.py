"""
excel_layout.py — make generated Excel sheets look tidy: column widths that follow the CONTENT, not the header.

A column of small numbers under a long header such as `Reservoir_inflow_m3h` used to be as wide as the header. Here the width is that of
the widest value (as it is displayed), the header wraps onto several lines inside the narrow column, floats get a sensible number format and
the header row is tall enough for its wrapped text.

    fit_columns(ws, header_row=3)           one table sheet (rows above the header, such as a title, do not count)
    fit_key_value_sheet(ws)                 a sheet of label / value rows with no header
    format_workbook(wb)                     every sheet of a workbook: tables by their first row, small label sheets as key/value

Only layout is changed (widths, wrapping, row height, number format of floats, centring of the cells); no value is changed.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Iterable, Optional

from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

FLOAT_FORMAT = "#,##0.00"


def shown_length(value, number_format: str = "General") -> int:
    """How many characters a cell value takes when displayed."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 4 if value else 5                                  # TRUE or FALSE
    if isinstance(value, int):
        return len(f"{value:,}") if "," in number_format else len(str(value))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 3
        decimals = 2
        if "." in number_format and number_format != "General":
            decimals = len(number_format.split(".")[1].rstrip("%"))
        return len(f"{value:,.{decimals}f}")
    if isinstance(value, (dt.datetime, dt.date)):
        return 19 if isinstance(value, dt.datetime) else 10
    return max((len(line) for line in str(value).splitlines()), default=0)


def _set_float_formats(ws, first_row: int) -> None:
    """Floats written with the General format show up to 15 digits; show two decimals instead."""
    for row in ws.iter_rows(min_row=first_row):
        for c in row:
            if isinstance(c.value, float) and c.number_format == "General":
                c.number_format = FLOAT_FORMAT


LONG_TEXT = 40          # a text cell longer than this stays left-aligned (a description is not centred)


def _centre_cells(ws, first_row: int, first_col: int = 1) -> None:
    """Centre the cells of the table body horizontally (Ctrl+E in Excel); long text stays left-aligned. Wrapping and vertical alignment are kept."""
    for row in ws.iter_rows(min_row=first_row, min_col=first_col):
        for c in row:
            if c.value is None:
                continue
            long_text = isinstance(c.value, str) and len(c.value) > LONG_TEXT
            a = c.alignment
            c.alignment = Alignment(horizontal="left" if long_text else "center", vertical=a.vertical or "center", wrap_text=a.wrap_text or long_text)


def fit_columns(ws, header_row: int = 1, first_data_row: Optional[int] = None, min_w: float = 8, max_w: float = 40, header_cap: int = 8,
                max_header_lines: int = 5, format_floats: bool = True) -> None:
    """Column widths from the data cells; the header contributes at most `header_cap` characters and wraps onto several lines."""
    first = first_data_row or header_row + 1
    if format_floats:
        _set_float_formats(ws, first)
    lines_needed = 1
    for col in ws.iter_cols(min_col=1, max_col=ws.max_column):
        letter = get_column_letter(col[0].column)
        head = ws.cell(header_row, col[0].column)
        head_text = "" if head.value is None else str(head.value)
        body = max((shown_length(c.value, c.number_format) for c in col if c.row >= first), default=0)
        width = min(max_w, max(min_w, body + 2, min(len(head_text), header_cap) + 2))
        ws.column_dimensions[letter].width = width
        if head_text:
            lines_needed = max(lines_needed, math.ceil(len(head_text) / max(1.0, width - 1)))
        head.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[header_row].height = 15.0 * min(max_header_lines, lines_needed) + (3 if lines_needed > 1 else 0)
    _centre_cells(ws, first)


def fit_key_value_sheet(ws, min_w: float = 10, max_w: float = 70) -> None:
    """A sheet of labels and values (no header): each column as wide as its longest cell, with a limit."""
    _set_float_formats(ws, 1)
    for col in ws.iter_cols(min_col=1, max_col=ws.max_column):
        widest = 0
        for c in col:
            if c.row == 1 and c.column == 1:
                continue                                     # a title in A1 may spill over to the right; it does not set the width
            widest = max(widest, shown_length(c.value, c.number_format))
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_w, max(min_w, widest + 2))
    _centre_cells(ws, 1, first_col=2)                     # the labels (first column) stay on the left; the values are centred


def _looks_like_table(ws) -> bool:
    if ws.max_column < 3 or ws.max_row < 2:
        return False
    first = [c.value for c in ws[1]]
    return all(isinstance(v, str) for v in first if v is not None) and sum(v is not None for v in first) >= 3


def format_workbook(wb, key_value_sheets: Iterable[str] = ("Summary", "Risk")) -> None:
    """Tidy every sheet: a table (first row of three or more labels) by its content, label/value sheets by their longest cell, and
    the header row of a table frozen. Sheets that already had their layout set (a frozen header) keep their freeze."""
    kv = set(key_value_sheets)
    for ws in wb.worksheets:
        if ws.title in kv or not _looks_like_table(ws):
            fit_key_value_sheet(ws)
        else:
            fit_columns(ws, header_row=1)
            if ws.freeze_panes is None:
                ws.freeze_panes = "A2"
