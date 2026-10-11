"""
run_market_analysis.py — Phase 4: run the exploratory market analysis on the clean zone.

    python tools/run_market_analysis.py                    # tables to runtime/analysis/phase4, figures to docs/forecasting/figures
    python tools/run_market_analysis.py --no-figures
    python tools/run_market_analysis.py --out DIR --fig-dir DIR

Needs the quality gate to be green for the history you analyse (tools/run_data_quality.py); a warning is
printed, not enforced, because the analysis is read-only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.market_analysis.config import load_analysis_config  # noqa: E402
from common_layer.market_analysis.run import run_analysis, write_outputs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--fig-dir")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    cfg = load_analysis_config()
    t0 = time.time()
    tables, panel, scalars = run_analysis(CleanStore(), cfg)
    print(f"analysis done in {time.time() - t0:.0f} s: {len(tables)} tables, "
          f"{scalars['n_days_primary']} days of day-ahead prices ({scalars['first_day']} .. {scalars['last_day']})")
    paths = write_outputs(tables, panel, scalars, cfg, a.out, a.fig_dir, figures=not a.no_figures)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
