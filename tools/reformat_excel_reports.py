"""
reformat_excel_reports.py — give the Excel reports that already exist the content-based layout of common_layer/utilities/excel_layout.py.

    python tools/reformat_excel_reports.py                 # every report in runtime/reports/
    python tools/reformat_excel_reports.py --dir PATH      # another folder

New reports get this layout when they are written; reports written earlier keep their old widths until they are passed through here.
Only layout changes (column widths, wrapped headers, row height, number format of floats); no value changes.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from openpyxl import load_workbook  # noqa: E402

from common_layer.utilities import excel_layout  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# sheet -> (header row, minimum width, maximum width) of the daily report, as written by daily_report_exporter
DAILY = {"Dispatch_Hourly": (3, 7, 22), "ISP_Activation": (2, 8, 30), "Gate_Decisions": (2, 8, 30),
         "Summary_KPIs": (2, 10, 45), "Glossary": (2, 10, 60)}


def reformat(path: str) -> str:
    wb = load_workbook(path)
    name = os.path.basename(path)
    if name.startswith("daily_report_"):
        for ws in wb.worksheets:
            hr, lo, hi = DAILY.get(ws.title, (2, 8, 30))
            excel_layout.fit_columns(ws, header_row=hr, min_w=lo, max_w=hi)
    else:
        excel_layout.format_workbook(wb)
    wb.save(path)
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(_REPO, "runtime", "reports"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.xlsx")))
    for f in files:
        try:
            print("reformatted", reformat(f))
        except PermissionError:
            print("skipped (open in Excel?):", os.path.basename(f))
    print(f"{len(files)} workbook(s) in {a.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
