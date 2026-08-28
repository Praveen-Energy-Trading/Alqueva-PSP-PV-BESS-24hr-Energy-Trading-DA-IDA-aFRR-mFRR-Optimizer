"""
scenario_generator.py — DA price scenario fan for two-stage stochastic MILP.

Scope (v1, DA gate only): price uncertainty only. PV and inflow are still
passed as single point values into build_core_model_stochastic — this is a
documented, deliberate scope limit, not an oversight. Price is the dominant
uncertainty driver for a day-ahead bid decision, and it's the one both
reference job postings (Statkraft, Milan; Axpo, Genova) named explicitly.
Extending to PV/inflow scenario fans is a natural next iteration once this
is validated end-to-end.

Method — grounded in real, already-computed model error, not an invented
assumption. The DA price forecaster (da_price_forecaster.py) already runs
walk-forward cross-validation and records each candidate model's real MAE
in da_selected_model_isp.json (e.g. {"selected": "XGBoost", "cv_mae":
{"XGBoost": 16.98, ...}}). This module reads that number and builds a
small discrete fan of symmetric offsets around the point forecast, scaled
by the selected model's own validated MAE:

    scenario prices = point_forecast[h] + k * mae   for k in OFFSET_SIGMAS

This is standard scenario-fan construction (see e.g. Rockafellar-Uryasev-
style discrete approximations of a continuous error distribution) and is
explicitly NOT a fabricated or synthetic-labeled-as-real number — the MAE
it's built from is a real, reproducible walk-forward-CV result already
sitting in the repo. Contrast with the (rejected) idea of putting the
pipeline's synthetic-mode P&L on a resume: that number is entirely
fictional data run through the model. This number is a real, measured
property (average absolute forecast error) of a real, already-selected
production model, used for its documented purpose (expressing forecast
uncertainty), not presented as an actual outcome.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

# Discrete fan of offsets, in units of the model's own MAE (a coarse proxy
# for one std-dev of forecast error). Five points: keeps scenario count (and
# hence solve-time growth, since every dispatch variable is replicated per
# scenario — see core_milp_builder.build_core_model_stochastic) small while
# still spanning a meaningful spread either side of the point forecast.
OFFSET_SIGMAS: List[float] = [-1.5, -0.75, 0.0, 0.75, 1.5]


def load_selected_model_mae(selected_model_json_path: str) -> float:
    """Read the walk-forward-CV MAE of the currently selected DA price model.

    Raises FileNotFoundError / KeyError loudly rather than silently falling
    back to a guessed number — per project standard (PR-13-style: never
    silently substitute an unproven number), a missing/malformed
    da_selected_model*.json should stop scenario generation, not quietly
    invent an error magnitude.
    """
    with open(selected_model_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    selected = data["selected"]
    return float(data["cv_mae"][selected])


def generate_price_scenarios(
    point_forecast: Dict[int, float],
    mae: float,
    n_scenarios: Optional[int] = None,
    offsets: Optional[List[float]] = None,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float]]:
    """Build a discrete DA price scenario fan around a point forecast.

    Args:
        point_forecast: {h: EUR/MWh} — the existing single-point forecast
            (e.g. from forecast_da_prices_isp()).
        mae: real walk-forward-CV mean absolute error of the model that
            produced point_forecast (from load_selected_model_mae()).
        n_scenarios: if given, must equal len(offsets) once resolved — a
            sanity check against accidentally passing a mismatched fan size.
        offsets: sigma-multipliers to use; defaults to OFFSET_SIGMAS.

    Returns:
        (scenarios, probabilities):
          scenarios     = {scenario_id: {h: EUR/MWh}}, scenario_id = 0..N-1
          probabilities = {scenario_id: float}, equal-weighted, sums to 1.0

    The same offset is applied to every hour within one scenario (a
    correlated, whole-day price shift) rather than an independent per-hour
    draw — day-ahead price forecast error is strongly autocorrelated across
    a delivery day (systematic over/under-prediction of the whole demand/
    supply picture), so a whole-day shift is the more defensible modeling
    choice than 96 independent per-ISP coin flips, and it keeps the
    resulting MILP's scenario count meaningful rather than combinatorial.
    """
    fan = list(offsets) if offsets is not None else list(OFFSET_SIGMAS)
    if n_scenarios is not None and n_scenarios != len(fan):
        raise ValueError(
            f"n_scenarios={n_scenarios} does not match len(offsets)={len(fan)}; "
            "pass a matching `offsets` list or leave n_scenarios=None")
    if mae < 0:
        raise ValueError(f"mae must be >= 0, got {mae!r}")

    scenarios: Dict[int, Dict[int, float]] = {}
    for sid, k in enumerate(fan):
        scenarios[sid] = {h: max(0.0, price + k * mae) for h, price in point_forecast.items()}

    n = len(fan)
    probabilities: Dict[int, float] = {sid: 1.0 / n for sid in scenarios}
    return scenarios, probabilities


def default_scenarios_for_da(
    point_forecast: Dict[int, float],
    repo_root: str,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float]]:
    """Convenience wrapper: reads the real selected-model MAE for the DA
    gate from its standard location and builds the default 5-scenario fan.
    """
    mae_path = os.path.join(
        repo_root, "phase_1_da_day_ahead_bidding",
        "da_price_pv_inflow_forecasting", "da_selected_model_isp.json")
    mae = load_selected_model_mae(mae_path)
    return generate_price_scenarios(point_forecast, mae)
