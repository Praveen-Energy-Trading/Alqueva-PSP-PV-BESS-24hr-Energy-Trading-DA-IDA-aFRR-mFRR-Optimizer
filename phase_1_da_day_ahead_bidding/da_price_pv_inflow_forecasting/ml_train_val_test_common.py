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


def fit_catboost(X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """CatBoost regressor — gradient boosting with native categorical handling."""
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        loss_function     = "MAE",
        depth             = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        l2_leaf_reg       = 3.0,
        iterations        = 500,
        random_seed       = 42,
        verbose           = False,
        allow_writing_files = False,  # skip catboost_info/ training-log folder
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Classical econometric candidates — DA price forecaster only (see
# DA_MODEL_NAMES / da_price_forecaster.py). Not added to the other 9
# forecasters' MODEL_NAMES: ARIMA/GARCH are univariate-price models, a
# genuinely different fit than the tabular feature-based boosting models,
# and this project scopes the claim to where the evidence (real job
# postings) actually pointed — day-ahead price forecasting.
# ---------------------------------------------------------------------------

class _ARIMAPointForecast:
    """Wraps a fitted statsmodels ARIMAResults so it satisfies the same
    `.predict(X)` contract every other fit_* model here exposes (X is a
    features-only DataFrame; ARIMA ignores it and forecasts len(X) steps
    ahead from where it was fit — genuinely univariate, not a limitation
    hidden from the caller, it's the correct behavior for this model."""

    def __init__(self, fitted_result):
        self._fitted = fitted_result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._fitted.forecast(steps=len(X)))


def fit_arima(X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """Univariate ARIMA(p,d,q) on the price level, order chosen by AIC over
    a small bounded grid (p in {1,2}, d=1, q in {1,2} -- 4 fits). Bounded
    deliberately to keep walk-forward-CV runtime sane; pmdarima's unbounded
    auto-search isn't installed and isn't needed for a genuine, defensible
    ARIMA implementation. X/feature_names are accepted (for interface
    parity with the other fit_* functions / fit_selected) but not used --
    ARIMA is univariate by definition."""
    import statsmodels.api as sm

    best_aic = float("inf")
    best_result = None
    for p in (1, 2):
        for q in (1, 2):
            try:
                result = sm.tsa.ARIMA(y, order=(p, 1, q)).fit()
            except Exception:
                continue
            if result.aic < best_aic:
                best_aic = result.aic
                best_result = result
    if best_result is None:
        # Every candidate order failed to converge (can happen on short or
        # degenerate series) -- fall back to the simplest possible order
        # rather than silently return a broken model with no result at all.
        best_result = sm.tsa.ARIMA(y, order=(1, 1, 1)).fit()
    return _ARIMAPointForecast(best_result)


class _GARCHPointForecast:
    """Wraps a fitted arch AR-GARCH result. Exposes the MEAN-equation point
    forecast (the component comparable to the other models' price-level
    predictions via MAE) reconstructed back to price levels via cumulative
    sum from the last known price. The model's conditional-volatility
    forecast is also available on `self._fitted` for a future risk-overlay
    use, but is not what `.predict()` returns here -- this class is a
    point-forecast adapter, not a full exposure of the fitted GARCH model."""

    def __init__(self, fitted_result, last_price: float):
        self._fitted = fitted_result
        self._last_price = last_price

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        horizon = len(X)
        fc = self._fitted.forecast(horizon=horizon, reindex=False)
        mean_diffs = np.asarray(fc.mean.iloc[-1].values)   # length == horizon
        return self._last_price + np.cumsum(mean_diffs)


def fit_garch(X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """AR(1)-mean / GARCH(1,1)-volatility model (the `arch` package) fit on
    first-differenced price (standard practice -- GARCH assumes a
    stationary series, raw price levels usually aren't). The point forecast
    used for MAE comparison is the mean equation's forecast, reconstructed
    to price levels; see _GARCHPointForecast docstring. X/feature_names
    accepted for interface parity, not used -- same univariate reasoning as
    fit_arima."""
    from arch import arch_model

    y = np.asarray(y, dtype=float)
    y_diff = np.diff(y, prepend=y[0])
    am = arch_model(y_diff, mean="AR", lags=1, vol="GARCH", p=1, q=1, rescale=False)
    result = am.fit(disp="off")
    return _GARCHPointForecast(result, last_price=float(y[-1]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# Periods with |y_true| below this are excluded from MAPE's denominator --
# division by a near-zero price would otherwise produce a huge or infinite
# %-error for a single hour and poison the mean. Real DA/IDA data genuinely
# has near-zero and negative price hours (confirmed this session), so this
# is a documented exclusion, not a hidden one.
_MAPE_EPS = 1.0  # EUR/MWh


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, excluding periods where |y_true| <
    _MAPE_EPS (division by a near-zero price is meaningless, not a
    genuine %-error). Returns NaN if every period is excluded."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= _MAPE_EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """Fraction of periods where the model correctly called the direction
    of the move from y_prev (the same lag value every caller already
    passes walk_forward_cv for the naive-persistence baseline) -- did
    sign(y_pred - y_prev) match sign(y_true - y_prev). Periods where
    y_true == y_prev (no real move happened) are excluded from the
    denominator entirely, not counted as a miss -- there is no direction
    to call correctly or incorrectly. Returns NaN if every period is
    excluded (a fully flat series)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    actual_move = np.sign(y_true - y_prev)
    mask = actual_move != 0
    if not np.any(mask):
        return float("nan")
    pred_move = np.sign(y_pred[mask] - y_prev[mask])
    return float(np.mean(pred_move == actual_move[mask]))


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

MODEL_NAMES = ["LightGBM", "XGBoost", "RandomForest", "CatBoost"]

# DA price forecaster only -- see the module-level comment above fit_arima.
# Other 9 forecasters (PV, inflow, IDA1/2/3, XBID, aFRR/mFRR up/dn) keep
# using plain MODEL_NAMES via walk_forward_cv's default; only
# da_price_forecaster.py explicitly passes this list in.
DA_MODEL_NAMES = MODEL_NAMES + ["ARIMA", "GARCH"]

_FITTERS = {
    "LightGBM"    : fit_lgbm,
    "XGBoost"     : fit_xgb,
    "RandomForest": fit_rf,
    "CatBoost"    : fit_catboost,
    "ARIMA"       : fit_arima,
    "GARCH"       : fit_garch,
}


def fit_selected(name: str, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
    """Fit whichever of the 4 competing models `name` refers to."""
    return _FITTERS[name](X, y, feature_names)


def _walk_forward_folds(feat_df: pd.DataFrame, y: np.ndarray, lag: np.ndarray,
                        fcols: List[str], n_folds: int, names: List[str]):
    """Shared fold-splitting/fit loop for walk_forward_cv and
    walk_forward_cv_extended -- single source of truth for the train/val
    slicing and the < 48 / == 0 skip guard, so the two functions can never
    silently diverge on which folds/data each model actually saw.

    Yields (name, y_val, y_pred, y_val_prev) per fold per model, where
    y_val_prev is the lag-array slice aligned to y_val (the "previous
    actual value" each period's move is measured against -- used by
    directional_accuracy; the plain walk_forward_cv caller ignores it).
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

    model_names: which models to compare. Defaults to MODEL_NAMES (the
        original 4 boosting/ensemble models) -- every existing caller keeps
        that behavior unchanged. da_price_forecaster.py passes
        DA_MODEL_NAMES explicitly to also compare ARIMA/GARCH; no other
        forecaster does, by design (see module-level comment above
        fit_arima).
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
    test_model_selection_metrics.py's regression check), but reports MAE,
    RMSE, MAPE, and directional accuracy per model, plus each model's raw
    per-fold MAE list (for paired_significance_test between the top 2).

    Returns {model_name: {"MAE", "RMSE", "MAPE", "DirAcc", "fold_mae"}}.
    A model with zero valid folds gets MAE/RMSE=inf, MAPE/DirAcc=NaN,
    fold_mae=[] -- same "never silently invent a number" standard as the
    rest of this module.
    """
    names = model_names if model_names is not None else MODEL_NAMES
    fold_mae: Dict[str, list] = {name: [] for name in names}
    fold_rmse: Dict[str, list] = {name: [] for name in names}
    fold_mape: Dict[str, list] = {name: [] for name in names}
    fold_diracc: Dict[str, list] = {name: [] for name in names}

    for name, y_val, y_pred, y_prev in _walk_forward_folds(feat_df, y, lag, fcols, n_folds, names):
        fold_mae[name].append(mae(y_val, y_pred))
        fold_rmse[name].append(rmse(y_val, y_pred))
        fold_mape[name].append(mape(y_val, y_pred))
        fold_diracc[name].append(directional_accuracy(y_val, y_pred, y_prev))

    def _mean_or(values: list, default: float) -> float:
        clean = [v for v in values if not np.isnan(v)]
        return float(np.mean(clean)) if clean else default

    return {
        name: {
            "MAE": float(np.mean(fold_mae[name])) if fold_mae[name] else float("inf"),
            "RMSE": float(np.mean(fold_rmse[name])) if fold_rmse[name] else float("inf"),
            "MAPE": _mean_or(fold_mape[name], float("nan")),
            "DirAcc": _mean_or(fold_diracc[name], float("nan")),
            "fold_mae": fold_mae[name],
        }
        for name in names
    }
