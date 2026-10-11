"""
tree_models.py — decision tree, random forest, gradient boosting, XGBoost and LightGBM: the model family used
for every target.

All five share the features of one feature set (forecasters.FeatureModel), the same transformed target and the same folds, so the
only thing that differs is the learner. `DEFAULTS` are sensible fixed settings (the untuned models); the tuned models
(tuning.py) replace them by the best of a time-aware search inside each fold.

    candidates = tree_candidates()                      # untuned
    candidates = tree_candidates(tuned=True)            # nested, time-aware tuning inside the backtest
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional

from common_layer.feature_engineering.builder import FeatureBuilder
from common_layer.feature_engineering.catalogue import (feature_set_version, load_feature_catalogue, load_feature_sets,
                                                        resolve_feature_set)
from common_layer.forecast_evaluation.config import TREE_MODELS
from common_layer.forecast_evaluation.forecasters import FeatureModel

# the learners are fitted and applied on plain arrays in the same column order; their feature-name notice is noise
warnings.filterwarnings("ignore", message="X does not have valid feature names")

DEFAULTS: Dict[str, dict] = {
    "decision_tree": {"max_depth": 8, "min_samples_leaf": 20, "random_state": 0},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 5, "max_features": 0.5, "n_jobs": 4, "random_state": 0},
    "gradient_boosting": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "random_state": 0},
    "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8,
                "n_jobs": 4, "random_state": 0, "verbosity": 0},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "subsample_freq": 1,
                 "colsample_bytree": 0.8, "random_state": 0, "verbose": -1, "n_jobs": 4},
}


def build_learner(name: str, params: Optional[dict] = None):
    """A fresh learner: the defaults of `name` with `params` laid over them."""
    kw = {**DEFAULTS[name], **(params or {})}
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(**kw)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**kw)
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**kw)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**kw)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**kw)
    raise KeyError(name)


class TreeForecaster(FeatureModel):
    def __init__(self, view, learner_name: str, builder: FeatureBuilder, set_version: str, cache: Optional[Dict] = None):
        super().__init__(view, lambda: build_learner(learner_name), builder, set_version, cache)
        self.name = learner_name


def tree_candidates(names: Optional[list] = None, feature_set: str = "baseline_v1", tuned: bool = False,
                    tuning_cfg: Optional[dict] = None) -> Dict[str, Callable]:
    """One factory per tree model; each builds the features of `feature_set` available for its target.
    With tuned=True each model tunes its settings inside every fold (see tuning.py)."""
    chosen = list(names) if names else list(TREE_MODELS)
    unknown = [n for n in chosen if n not in TREE_MODELS]
    if unknown:
        raise KeyError(f"unknown tree model(s) {unknown}; available: {list(TREE_MODELS)}")
    cat, sets = load_feature_catalogue(), load_feature_sets()
    if tuned:
        from common_layer.forecast_evaluation.tuning import TunedForecaster, load_tuning_config
        tcfg = tuning_cfg or load_tuning_config()

    def make(n: str):
        def factory(iface, cache):
            ids = resolve_feature_set(feature_set, sets, cat, iface.spec, iface.offsets, iface.target_id)
            builder, ver = FeatureBuilder(iface, cat, ids), feature_set_version(ids, cat)
            if tuned:
                return TunedForecaster(iface, n, builder, ver, tcfg, cache)
            return TreeForecaster(iface, n, builder, ver, cache)
        return factory
    return {(f"{n}_tuned" if tuned else n): make(n) for n in chosen}
