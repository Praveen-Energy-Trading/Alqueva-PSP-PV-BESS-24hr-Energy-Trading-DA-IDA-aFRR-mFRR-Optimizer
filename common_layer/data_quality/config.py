"""Loads and validates config/data_quality.yaml."""
from __future__ import annotations

import os
from typing import Optional

import yaml

from common_layer.data_layer.contracts import DATASETS

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "data_quality.yaml")
_METHODS = {"none", "ffill", "prev_day"}


class DataQualityConfigError(ValueError):
    pass


def load_quality_config(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    problems = validate_quality_config(cfg)
    if problems:
        raise DataQualityConfigError("; ".join(problems))
    return cfg


def validate_quality_config(cfg: dict) -> list:
    out = []
    ds = cfg.get("datasets", {})
    for name in DATASETS:
        if name not in ds:
            out.append(f"dataset {name} missing from data_quality.yaml")
            continue
        spec = DATASETS[name]
        for col in ds[name].get("price_cols", []):
            if col not in spec.value_cols:
                out.append(f"{name}: unknown price column {col}")
        if not ds[name].get("sources") or not ds[name].get("expected_start"):
            out.append(f"{name}: sources and expected_start are required")
    for key, rule in cfg.get("imputation", {}).items():
        if key == "default":
            continue
        dsname, _, col = key.partition(".")
        if dsname not in DATASETS or col not in DATASETS[dsname].value_cols:
            out.append(f"imputation rule for unknown variable {key}")
        if rule.get("method") not in _METHODS:
            out.append(f"{key}: method must be one of {sorted(_METHODS)}")
        if rule.get("method") == "ffill" and int(rule.get("max_periods", 0)) < 1:
            out.append(f"{key}: ffill needs max_periods >= 1")
    # every price column used in checks must have an explicit imputation rule (decision on record)
    for name, d in ds.items():
        for col in d.get("price_cols", []):
            if f"{name}.{col}" not in cfg.get("imputation", {}):
                out.append(f"{name}.{col}: no imputation rule written (use method: none if none)")
    for k in ("spikes", "cross_source", "level_jump", "penalties", "gate", "provenance"):
        if k not in cfg:
            out.append(f"section {k} missing")
    return out
