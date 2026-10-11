"""Loads and validates config/evaluation.yaml."""
from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional

import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "evaluation.yaml")
POINT_METRICS = ("sse", "mae", "rmse")
TREE_MODELS = ("decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm")


def _d(v) -> dt.date:
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(str(v))


def load_evaluation_config(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    d, lk = cfg["development"], cfg["locked_test"]
    for k in ("first_origin", "test_start", "test_end"):
        d[k] = _d(d[k])
    for k in ("start", "end"):
        lk[k] = _d(lk[k])
    problems = validate_evaluation_config(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    return cfg


def validate_evaluation_config(cfg: dict) -> List[str]:
    out: List[str] = []
    d, lk, s = cfg["development"], cfg["locked_test"], cfg["schedule"]
    if not d["first_origin"] < d["test_start"] <= d["test_end"]:
        out.append("development dates must satisfy first_origin < test_start <= test_end")
    if not d["test_end"] < lk["start"] <= lk["end"]:
        out.append("the locked test period must start after the development period ends and must not overlap it")
    if (lk["start"] - d["test_end"]).days != 1:
        out.append("locked test must begin the day after the development period ends (no unused gap, no overlap)")
    for k in ("retrain_every_days", "gap_days", "min_train_days"):
        if not isinstance(s.get(k), int) or s[k] < 0:
            out.append(f"schedule.{k} must be a non-negative integer")
    if s["retrain_every_days"] < 1:
        out.append("schedule.retrain_every_days must be at least 1")
    if sorted(cfg["metrics"]["point"]) != sorted(POINT_METRICS):
        out.append(f"metrics.point must be exactly {POINT_METRICS}")
    m = cfg.get("models", {})
    if not m.get("candidates") or any(c not in TREE_MODELS for c in m["candidates"]):
        out.append(f"models.candidates must be a non-empty subset of {TREE_MODELS}")
    elif m.get("reference") not in m["candidates"]:
        out.append("models.reference must be one of models.candidates")
    if not m.get("feature_set"):
        out.append("models.feature_set must name a set in config/feature_sets.yaml")
    if cfg["comparison"]["loss"] not in ("absolute", "squared"):
        out.append("comparison.loss must be absolute or squared")
    return out
