"""ML Lifecycle 5 - Evaluation design and tuning lab: the calendar of the forecast (what is fixed, what moves), then how the settings are chosen.

One graph carries the evaluation design (the data, the rolling windows, the gap, the backtest, the locked year, the live forecast) and the tuning rounds;
the numbered steps below it are the tuning lab (dashboard/parts/ml_6_tuning.py)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ml_ui

ml_ui.header("Evaluation design and tuning lab", "", step=4)
ml_ui.tour_bar(5)

runpy.run_path(str(Path(__file__).resolve().parent.parent / "parts" / "ml_6_tuning.py"), run_name="part_tuning")
