"""
html_table.py — a table whose columns are as wide as their CONTENT and whose headers wrap onto two or more lines.

st.dataframe cannot wrap a header, so a long header ("Data errors removed") over a column of small numbers forces the whole column wide.
Cells and headers are centred (long text stays left-aligned). Here the table is plain HTML: the browser sizes every column to its widest value, and the header wraps at its spaces (and after the
underscores of a name such as Reservoir_inflow_m3h), so a header never widens a column beyond its longest word.

    html_table(df, formats=None, height=None) -> str      the HTML (styles included), to be passed to st.markdown(..., unsafe_allow_html=True)
"""
from __future__ import annotations

import html
from typing import Optional

import pandas as pd

GRID = "#e1e0d9"
HEAD_BG = "#f4f4f1"
INK = "#0b0b0b"
INK_MUTED = "#6d6c68"

CSS = f"""
<style>
.htab-wrap {{ overflow: auto; border: 1px solid {GRID}; border-radius: 8px; background: #fcfcfb; margin: 0.2rem 0 0.8rem 0; max-width: 100%; }}
table.htab {{ border-collapse: separate; border-spacing: 0; width: auto !important; min-width: 0 !important; font-size: 0.86rem; color: {INK}; }}
table.htab th {{ position: sticky; top: 0; z-index: 1; background: {HEAD_BG}; color: {INK_MUTED}; font-weight: 500; text-align: center;
                 vertical-align: bottom; padding: 7px 10px; border-bottom: 1px solid {GRID}; border-right: 1px solid {GRID};
                 white-space: normal; line-height: 1.25; }}
table.htab td {{ padding: 6px 10px; border-bottom: 1px solid {GRID}; border-right: 1px solid {GRID}; white-space: nowrap; vertical-align: middle; text-align: center; }}
table.htab td.txt {{ white-space: nowrap; }}
table.htab td.long {{ white-space: normal; min-width: 260px; max-width: 460px; text-align: left; }}
table.htab td.num {{ font-variant-numeric: tabular-nums; }}
table.htab tr:last-child td {{ border-bottom: none; }}
table.htab th:last-child, table.htab td:last-child {{ border-right: none; }}
</style>
"""


def _fmt_number(v, fmt: Optional[str]) -> str:
    if pd.isna(v):
        return "–"
    if fmt in (None, "%d"):
        return f"{v:.0f}" if fmt == "%d" or float(v).is_integer() else f"{v:.2f}"
    return fmt % v


def _auto_format(col: pd.Series) -> str:
    vals = col.dropna()
    if not len(vals):
        return "%d"
    if (vals % 1 == 0).all() and vals.abs().max() < 1e9:
        return "%d"
    return "%.3g" if vals.abs().max() < 1e-2 else "%.2f"


def _header(text) -> str:
    """Header text, escaped; a break is allowed after every underscore so that snake_case names wrap."""
    return html.escape(str(text)).replace("_", "_<wbr>")


def html_table(df: pd.DataFrame, formats: Optional[dict] = None, height: Optional[int] = None) -> str:
    formats = formats or {}
    heads, numeric, long_text = [], [], []
    for c in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        numeric.append(is_num)
        long_text.append((not is_num) and df[c].astype(str).str.len().max() > 45 if len(df) else False)
        heads.append(f'<th class="{"num" if is_num else ""}">{_header(c)}</th>')
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c, is_num, is_long in zip(df.columns, numeric, long_text):
            v = r[c]
            if is_num:
                cells.append(f'<td class="num">{_fmt_number(v, formats.get(c) or _auto_format(df[c]))}</td>')
            else:
                if isinstance(v, (bool,)):
                    v = "yes" if v else "no"
                text = "–" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
                cells.append(f'<td class="{"long" if is_long else "txt"}">{html.escape(text)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    style = f' style="max-height:{int(height)}px"' if height else ""
    return (CSS + f'<div class="htab-wrap"{style}><table class="htab"><thead><tr>' + "".join(heads) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")
