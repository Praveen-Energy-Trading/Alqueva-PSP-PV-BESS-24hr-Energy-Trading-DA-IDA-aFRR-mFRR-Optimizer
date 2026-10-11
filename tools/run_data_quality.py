"""
run_data_quality.py — run the Phase 3 quality checks on the clean zone and write the report.

    python tools/run_data_quality.py                 # whole history
    python tools/run_data_quality.py --since 2024-01-01
    python tools/run_data_quality.py --require-green # exit code 1 if the stored result is red/stale

Outputs (under runtime/): data/quality/dq.db, data/quality/gate.json, reports/data_quality_<date>.md
Exit code 0 = gate green, 1 = red.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.data_layer.ingest import LoadLog  # noqa: E402
from common_layer.data_layer.raw_store import RawStore  # noqa: E402
from common_layer.data_quality import runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=dt.date.fromisoformat, help="first delivery day to check")
    ap.add_argument("--until", type=dt.date.fromisoformat, help="last delivery day to check (default: yesterday)")
    ap.add_argument("--require-green", action="store_true", help="only check the stored result")
    a = ap.parse_args()
    if a.require_green:
        try:
            runner.require_green()
        except runner.DataQualityGateError as exc:
            print("RED:", exc)
            return 1
        print("GREEN")
        return 0
    run = runner.run_quality(CleanStore(), RawStore(), LoadLog(), start=a.since, end=a.until)
    paths = runner.store_results(run)
    print("GATE:", "GREEN" if run.gate.passed else "RED")
    for r in run.gate.reasons[:15]:
        print(" -", r)
    print("report:", paths["report"])
    return 0 if run.gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
