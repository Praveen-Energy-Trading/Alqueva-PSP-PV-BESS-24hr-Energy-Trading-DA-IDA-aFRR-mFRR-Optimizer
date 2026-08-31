"""
test_model_selection_metrics.py — RMSE/MAPE/directional-accuracy/
significance-test additions to ml_train_val_test_common.py, and the
DA forecaster's extended model-selection JSON output.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "phase_1_da_day_ahead_bidding", "da_price_pv_inflow_forecasting"))

from ml_train_val_test_common import (
    mape, directional_accuracy, paired_significance_test,
    walk_forward_cv, walk_forward_cv_extended, MODEL_NAMES,
)


# ── mape ─────────────────────────────────────────────────────────────────

def test_mape_perfect_prediction_is_zero():
    y_true = np.array([100.0, 50.0, 200.0])
    assert mape(y_true, y_true) == pytest.approx(0.0)


def test_mape_excludes_near_zero_true_values():
    """A near-zero y_true would otherwise blow up %-error for that one
    period and poison the mean -- confirm it's excluded, not counted."""
    y_true = np.array([0.1, 100.0])   # 0.1 is below _MAPE_EPS (1.0)
    y_pred = np.array([50.0, 105.0])  # huge %-error on the excluded point
    result = mape(y_true, y_pred)
    # Only the second period should count: |105-100|/100 = 5%
    assert result == pytest.approx(5.0, abs=0.01)


def test_mape_all_excluded_returns_nan():
    y_true = np.array([0.1, 0.2])
    assert np.isnan(mape(y_true, y_true))


# ── directional_accuracy ─────────────────────────────────────────────────

def test_directional_accuracy_perfect_calls():
    y_prev = np.array([50.0, 50.0, 50.0])
    y_true = np.array([60.0, 40.0, 55.0])   # up, down, up
    y_pred = np.array([55.0, 45.0, 52.0])   # same direction each time
    assert directional_accuracy(y_true, y_pred, y_prev) == pytest.approx(1.0)


def test_directional_accuracy_wrong_calls():
    y_prev = np.array([50.0, 50.0])
    y_true = np.array([60.0, 40.0])   # up, down
    y_pred = np.array([45.0, 55.0])   # called down, up -- both wrong
    assert directional_accuracy(y_true, y_pred, y_prev) == pytest.approx(0.0)


def test_directional_accuracy_excludes_no_move_periods():
    y_prev = np.array([50.0, 50.0])
    y_true = np.array([50.0, 60.0])   # first period: no real move
    y_pred = np.array([999.0, 55.0])  # first prediction irrelevant, excluded
    # Only the second period counts: up predicted correctly -> 1.0
    assert directional_accuracy(y_true, y_pred, y_prev) == pytest.approx(1.0)


def test_directional_accuracy_all_flat_returns_nan():
    y_prev = np.array([50.0, 50.0])
    y_true = np.array([50.0, 50.0])
    assert np.isnan(directional_accuracy(y_true, y_true, y_prev))


# ── paired_significance_test ─────────────────────────────────────────────

def test_significance_identical_errors_not_significant():
    errors = [1.0, 2.0, 1.5, 3.0, 2.5]
    result = paired_significance_test(errors, errors)
    assert result["significant_at_0.05"] is False


def test_significance_clearly_different_errors():
    errors_a = [1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.3, 1.1]
    errors_b = [10.0, 11.0, 9.0, 12.0, 10.0, 8.0, 13.0, 11.0]
    result = paired_significance_test(errors_a, errors_b)
    assert result["significant_at_0.05"] is True
    assert result["p_value"] < 0.05


def test_significance_too_few_folds_returns_nan_not_significant():
    result = paired_significance_test([1.0], [2.0])
    assert np.isnan(result["p_value"])
    assert result["significant_at_0.05"] is False


# ── walk_forward_cv_extended matches walk_forward_cv's MAE exactly ──────

def test_extended_cv_matches_plain_cv_mae():
    """Regression check: the shared _walk_forward_folds helper must not
    have silently changed which data each model saw."""
    rng = np.random.RandomState(0)
    n = 400
    feat_df = __import__("pandas").DataFrame({
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
    })
    y = 50 + feat_df["f1"].values * 5 + rng.normal(scale=2, size=n)
    lag = np.roll(y, 1)
    lag[0] = y[0]

    plain = walk_forward_cv(feat_df, y, lag, ["f1", "f2"], n_folds=3,
                            model_names=["RandomForest"])
    extended = walk_forward_cv_extended(feat_df, y, lag, ["f1", "f2"], n_folds=3,
                                        model_names=["RandomForest"])
    assert plain["RandomForest"] == pytest.approx(extended["RandomForest"]["MAE"], rel=1e-9)


# ── real end-to-end: DA ISP forecaster writes the new keys ──────────────

@pytest.mark.integration
def test_da_isp_selection_json_has_extended_metrics():
    """Force a real re-selection on the real DA ISP training data and
    confirm the written JSON has all four new keys with real, finite
    values, restoring the original file afterward (this is a real
    production artifact, not a throwaway test fixture)."""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "phase_1_da_day_ahead_bidding", "da_price_pv_inflow_forecasting",
        "da_selected_model_isp.json")
    backup_path = json_path + ".test_backup"
    assert os.path.isfile(json_path), "real da_selected_model_isp.json must exist"
    shutil.copy(json_path, backup_path)
    try:
        with open(json_path) as f:
            original = json.load(f)
        # Force a re-run by rolling data_end_date back.
        forced = dict(original)
        forced["data_end_date"] = "2000-01-01"
        with open(json_path, "w") as f:
            json.dump(forced, f)

        import da_price_forecaster as dpf
        import importlib
        importlib.reload(dpf)
        # Drive the real code path (builds ISP features, filters, dropna)
        # rather than reconstructing dpf's internal train_df shape here.
        dpf.forecast_da_prices_isp(list(range(1, 97)), "2026-09-05")

        with open(json_path) as f:
            new_info = json.load(f)

        for key in ("cv_rmse", "cv_mape", "cv_directional_accuracy", "significance_top2"):
            assert key in new_info, f"missing key: {key}"
        assert new_info["selected"] in new_info["cv_rmse"]
        for v in new_info["cv_rmse"].values():
            assert v == float(v) and v >= 0
        sig = new_info["significance_top2"]
        assert "p_value" in sig and "significant_at_0.05" in sig
    finally:
        shutil.copy(backup_path, json_path)
        os.remove(backup_path)


@pytest.mark.integration
@pytest.mark.parametrize("gate,module_name,json_name,forecast_fn_name", [
    ("IDA1", "ida1_price_forecaster", "ida1_selected_model.json", "forecast_ida1_prices"),
    ("IDA2", "ida2_price_forecaster", "ida2_selected_model.json", "forecast_ida2_prices"),
    ("IDA3", "ida3_price_forecaster", "ida3_selected_model.json", "forecast_ida3_prices"),
])
def test_ida_selection_json_has_extended_metrics(gate, module_name, json_name, forecast_fn_name):
    """Same real end-to-end check as the DA test, extended to IDA1/2/3:
    force a re-selection on real training data, confirm the written JSON
    has the new keys with real finite values, restore the original file."""
    _gate_dir = {
        "IDA1": ("phase_2a_ida1_intraday_auction_1", "ida1_price_forecasting"),
        "IDA2": ("phase_2b_ida2_intraday_auction_2", "ida2_price_forecasting"),
        "IDA3": ("phase_2c_ida3_intraday_auction_3", "ida3_price_forecasting"),
    }[gate]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_dir = os.path.join(repo_root, *_gate_dir)
    json_path = os.path.join(module_dir, json_name)
    backup_path = json_path + ".test_backup"
    assert os.path.isfile(json_path), f"real {json_name} must exist"
    shutil.copy(json_path, backup_path)
    try:
        with open(json_path) as f:
            original = json.load(f)
        forced = dict(original)
        forced["data_end_date"] = "2000-01-01"
        with open(json_path, "w") as f:
            json.dump(forced, f)

        sys.path.insert(0, module_dir)
        import importlib
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        forecast_fn = getattr(mod, forecast_fn_name)
        # da_prices dict is only consumed to compute IDA = DA + spread;
        # any real-shaped point forecast drives the same selection path.
        da_prices = {h: 55.0 for h in range(1, 25)}
        forecast_fn(list(range(1, 25)), "2026-09-05", da_prices)

        with open(json_path) as f:
            new_info = json.load(f)

        for key in ("cv_rmse", "cv_mape", "cv_directional_accuracy", "significance_top2"):
            assert key in new_info, f"missing key: {key}"
        assert new_info["selected"] in new_info["cv_rmse"]
        for v in new_info["cv_rmse"].values():
            assert v == float(v) and v >= 0
        sig = new_info["significance_top2"]
        assert "p_value" in sig and "significant_at_0.05" in sig
    finally:
        shutil.copy(backup_path, json_path)
        os.remove(backup_path)
        if module_dir in sys.path:
            sys.path.remove(module_dir)
