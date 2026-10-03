"""
ml_train_val_test_common.py — shared ML utilities for Phase 1 forecasters.

Used by:
    da_price_forecaster.py        (serving)
    pv_power_forecaster.py        (serving)
    da_price_train_val_test.py    (offline evaluation)
    pv_train_val_test.py          (offline evaluation — future)

Single source of truth: fix here, all consumers benefit.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_xgb(X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """XGBoost regressor optimised for energy time-series (MAE objective)."""
    import xgboost as xgb
    model = xgb.XGBRegressor(
        objective         = "reg:absoluteerror",
        eval_metric       = "mae",
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 20,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        n_estimators      = 500,
        verbosity         = 0,
    )
    model.fit(X, y)
    return model


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """LightGBM regressor optimised for energy time-series (MAE objective)."""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective         = "regression",
        metric            = "mae",
        num_leaves        = 64,
        learning_rate     = 0.05,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 5,
        min_child_samples = 20,
        lambda_l1         = 0.1,
        lambda_l2         = 0.1,
        n_estimators      = 500,
        verbose           = -1,
    )
    model.fit(X, y, feature_name=feature_names)
    return model


def fit_rf(X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """Random Forest regressor — bagging ensemble, different family than boosting."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators      = 300,
        max_depth         = 12,
        min_samples_leaf  = 5,
        max_features      = 0.8,
        n_jobs            = -1,
        random_state      = 42,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def paired_significance_test(errors_a: List[float], errors_b: List[float]) -> Dict[str, object]:
    """Wilcoxon signed-rank test on two models' per-fold absolute errors
    (paired -- same folds for both models, the correct test shape here;
    non-parametric -- walk-forward CV produces few folds, too few to
    assume a normal distribution of fold errors). Answers "is model a's
    edge over model b real, or could it be noise across these folds?"

    Returns {"p_value": float, "significant_at_0.05": bool}. If there are
    fewer than 2 paired folds (or all differences are exactly zero, which
    scipy's wilcoxon cannot test), returns p_value=NaN and
    significant_at_0.05=False -- explicitly "not enough evidence to call
    it significant," never silently claims significance from an
    untestable input.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    n = min(len(a), len(b))
    if n < 2 or np.allclose(a[:n], b[:n]):
        return {"p_value": float("nan"), "significant_at_0.05": False}
    try:
        _, p_value = wilcoxon(a[:n], b[:n])
    except ValueError:
        # e.g. all differences zero after all -- scipy raises rather than
        # returning a degenerate p-value.
        return {"p_value": float("nan"), "significant_at_0.05": False}
    return {"p_value": float(p_value), "significant_at_0.05": bool(p_value < 0.05)}


def metrics(y_true: np.ndarray, y_pred: np.ndarray,
            mae_naive: float) -> Dict[str, float]:
    """MAE, RMSE, Bias (ME), Skill vs naive persistence."""
    err   = y_pred - y_true
    _mae  = float(np.mean(np.abs(err)))
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    bias  = float(np.mean(err))
    skill = float(1.0 - _mae / mae_naive) if mae_naive > 0 else float("nan")
    return {"MAE": _mae, "RMSE": rmse, "Bias": bias, "Skill": skill}


# ---------------------------------------------------------------------------
# Walk-forward cross-validation (model selection / validation role)
# ---------------------------------------------------------------------------

MODEL_NAMES = ["LightGBM", "XGBoost", "RandomForest"]

_FITTERS = {
    "LightGBM"    : fit_lgbm,
    "XGBoost"     : fit_xgb,
    "RandomForest": fit_rf,
}


def fit_selected(name: str, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """Fit whichever of the 3 competing models `name` refers to."""
    return _FITTERS[name](X, y, feature_names)


def _walk_forward_folds(feat_df: pd.DataFrame, y: np.ndarray, lag: np.ndarray,
                        fcols: List[str], n_folds: int, names: List[str]):
    """Shared fold-splitting/fit loop for walk_forward_cv and
    walk_forward_cv_extended -- single source of truth for the train/val
    slicing and the < 48 / == 0 skip guard, so the two functions can never
    silently diverge on which folds/data each model actually saw.

    Yields (name, y_val, y_pred, y_val_prev) per fold per model. y_val_prev
    is the lag-array slice aligned to y_val; both walk_forward_cv and
    walk_forward_cv_extended currently ignore it, kept in the yield shape
    for callers that pass a real lag/previous-value array anyway.
    """
    n       = len(feat_df)
    fold_sz = n // (n_folds + 1)

    for fold in range(n_folds):
        tr_end  = fold_sz * (fold + 1)
        val_end = tr_end + fold_sz
        X_tr    = feat_df.iloc[:tr_end]
        X_val   = feat_df.iloc[tr_end:val_end]
        y_tr    = y[:tr_end]
        y_val   = y[tr_end:val_end]
        lag_val = lag[tr_end:val_end]

        if len(X_tr) < 48 or len(X_val) == 0:
            continue

        for name in names:
            model = fit_selected(name, X_tr, y_tr, fcols)
            y_pred = model.predict(X_val)
            yield name, y_val, y_pred, lag_val


def walk_forward_cv(feat_df: pd.DataFrame, y: np.ndarray, lag: np.ndarray,
                    fcols: List[str], n_folds: int,
                    model_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Compare candidate models via walk-forward CV.

    Each fold trains on all prior data, validates on the next block —
    no future leakage. Returns mean MAE per model across folds.

    model_names: which models to compare. Defaults to MODEL_NAMES (the 3
        boosting/ensemble models) -- every caller keeps that behavior
        unchanged unless it passes its own subset in.
    """
    names = model_names if model_names is not None else MODEL_NAMES
    fold_mae: Dict[str, list] = {name: [] for name in names}

    for name, y_val, y_pred, _ in _walk_forward_folds(feat_df, y, lag, fcols, n_folds, names):
        fold_mae[name].append(mae(y_val, y_pred))

    return {k: float(np.mean(v)) if v else float("inf")
            for k, v in fold_mae.items()}


def walk_forward_cv_extended(feat_df: pd.DataFrame, y: np.ndarray, lag: np.ndarray,
                             fcols: List[str], n_folds: int,
                             model_names: Optional[List[str]] = None) -> Dict[str, dict]:
    """Same walk-forward CV as walk_forward_cv (identical fold-splitting,
    via the shared _walk_forward_folds helper -- proven identical MAE by
    test_model_selection_metrics.py's regression check), but also returns
    each model's raw per-fold MAE list (for paired_significance_test
    between the top 2).

    Returns {model_name: {"MAE", "fold_mae"}}. A model with zero valid
    folds gets MAE=inf, fold_mae=[] -- same "never silently invent a
    number" standard as the rest of this module.
    """
    names = model_names if model_names is not None else MODEL_NAMES
    fold_mae: Dict[str, list] = {name: [] for name in names}

    for name, y_val, y_pred, y_prev in _walk_forward_folds(feat_df, y, lag, fcols, n_folds, names):
        fold_mae[name].append(mae(y_val, y_pred))

    return {
        name: {
            "MAE": float(np.mean(fold_mae[name])) if fold_mae[name] else float("inf"),
            "fold_mae": fold_mae[name],
        }
        for name in names
    }
