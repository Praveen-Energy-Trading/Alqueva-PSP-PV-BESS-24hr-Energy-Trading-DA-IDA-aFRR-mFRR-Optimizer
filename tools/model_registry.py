"""
model_registry.py — look at, check and fill the experiment registry (stage 1 of docs/Model_Development_Plan.pdf).

    python tools/model_registry.py list [--target da_price] [--kind evaluation]
    python tools/model_registry.py show <run_key>             scores, comparisons and tuning choices of a run
    python tools/model_registry.py check                      experiments run more than once: same scores each time?
    python tools/model_registry.py backfill <folder> ...      record earlier runs from their output folders

Evaluation runs are recorded automatically by tools/evaluate_forecaster.py (unless --no-register) and
tools/run_schedule_comparison.py. File: runtime/model_registry/registry.db.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.model_development import registry  # noqa: E402
from common_layer.forecast_evaluation.tuning import load_tuning_config  # noqa: E402


def _show(df: pd.DataFrame) -> None:
    print(df.round(4).to_string(index=False) if len(df) else "(none)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--target")
    p.add_argument("--kind")
    p = sub.add_parser("show")
    p.add_argument("run_key")
    p.add_argument("--attempt", type=int)
    sub.add_parser("check")
    p = sub.add_parser("backfill")
    p.add_argument("folders", nargs="+")
    a = ap.parse_args()

    if a.cmd == "list":
        _show(registry.list_runs(a.target, a.kind))
    elif a.cmd == "show":
        print("scores"); _show(registry.models_of(a.run_key, a.attempt))
        print("\ncomparisons (negative statistic = smaller loss than the reference)"); _show(registry.comparisons_of(a.run_key, a.attempt))
        tc = registry.tuning_choices_of(a.run_key, a.attempt)
        if len(tc):
            print("\ntuning choices"); _show(tc.drop(columns=["params_json"]))
    elif a.cmd == "check":
        r = registry.reproducibility()
        _show(r)
        if len(r) and not r["reproduced"].all():
            print("\nSOME EXPERIMENTS GAVE DIFFERENT SCORES ON REPEAT")
            return 1
    elif a.cmd == "backfill":
        tcfg = load_tuning_config()
        for folder in a.folders:
            r = registry.register_run_dir(folder, tuning_cfg=tcfg)
            print(f"{folder}: run {r['run_key']} attempt {r['attempt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
