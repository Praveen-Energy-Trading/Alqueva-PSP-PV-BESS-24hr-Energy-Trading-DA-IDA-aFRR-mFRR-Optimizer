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
    n       = len(feat_df)
    fold_sz = n // (n_folds + 1)
    fold_mae: Dict[str, list] = {name: [] for name in names}

    for fold in range(n_folds):
        tr_end  = fold_sz * (fold + 1)
        val_end = tr_end + fold_sz
        X_tr    = feat_df.iloc[:tr_end]
        X_val   = feat_df.iloc[tr_end:val_end]
        y_tr    = y[:tr_end]
        y_val   = y[tr_end:val_end]

        if len(X_tr) < 48 or len(X_val) == 0:
            continue

        for name in names:
            model = fit_selected(name, X_tr, y_tr, fcols)
            fold_mae[name].append(mae(y_val, model.predict(X_val)))

    return {k: float(np.mean(v)) if v else float("inf")
            for k, v in fold_mae.items()}
