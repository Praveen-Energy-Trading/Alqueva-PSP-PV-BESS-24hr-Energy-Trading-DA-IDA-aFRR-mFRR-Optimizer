"""
serving.py — fit the champion, store it, and load it for live use (stage 7 of docs/Model_Development_Plan.pdf).

`fit_artifact` fits the active champion exactly as the evaluation framework would before the first forecast day (same
training window, same gap, same features, same tuned settings procedure) and stores the fitted learners, the fill values
and the feature-set version in one file, whose SHA-256 is written to the registry. `load_champion` is what live code calls:
it takes the active version of a target, checks that the file is the one registered, that the artifact belongs to the
active version and that the feature definitions have not changed since the fit, and returns a forecaster that builds its
features with the SAME FeatureBuilder as training and evaluation. Anything that does not match is refused, not repaired.

The artifact is a pickle. It is only ever loaded after its hash matches the one the registry wrote when the file was made
on this machine; do not point the registry at files from elsewhere.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from common_layer.feature_engineering.builder import FeatureBuilder
from common_layer.feature_engineering.catalogue import (feature_set_version, load_feature_catalogue, load_feature_sets,
                                                         resolve_feature_set)
from common_layer.forecast_evaluation.asof import AsOfView, ModelInterface
from common_layer.forecast_evaluation.folds import make_folds
from common_layer.forecast_evaluation.forecasters import FitContext, clip_to_training_range
from common_layer.forecast_evaluation.tree_models import tree_candidates
from common_layer.forecast_formulation.transforms import make_transform
from common_layer.model_development import champions, registry


class ChampionError(RuntimeError):
    """The registered champion cannot be served safely."""


def models_root(db_path: Optional[str] = None) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_path or registry.default_db_path())), "models")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _builder(iface: ModelInterface, feature_set: str):
    cat = load_feature_catalogue()
    ids = resolve_feature_set(feature_set, load_feature_sets(), cat, iface.spec, iface.offsets, iface.target_id)
    return FeatureBuilder(iface, cat, ids), feature_set_version(ids, cat)


def _read_artifact(reg: dict) -> dict:
    """The pickled artifact of a registered file, after checking that it exists and matches its registered hash."""
    if not os.path.exists(reg["path"]):
        raise ChampionError(f"artifact file missing: {reg['path']}")
    if _sha256(reg["path"]) != reg["sha256"]:
        raise ChampionError(f"artifact {reg['path']} does not match the hash in the registry; refusing to load it")
    with open(reg["path"], "rb") as fh:
        return pickle.load(fh)


def role_spec(spec: dict, role: str) -> dict:
    """The specification to fit: the champion itself, or its reference (the simplest algorithm, same settings policy)."""
    if role == "champion":
        return spec
    if role == "reference":
        return {**spec, "type": "single", "members": [spec["reference"]], "method": None}
    raise ValueError("role must be 'champion' or 'reference'")


def fit_artifact(target: str, panel: pd.DataFrame, form: dict, cfg: dict, tuning_cfg: dict, first_forecast_day: dt.date,
                 db_path: Optional[str] = None, history_days: int = 400, now: Optional[dt.datetime] = None,
                 role: str = "champion", force_retune: bool = False) -> dict:
    """Fit the active champion of `target` (or its reference) to be used from `first_forecast_day` and register the artifact.

    Tuned settings of the previous artifact of the same version and role are reused until `retune_every_days` of history
    have passed since they were chosen (the rule of the evaluation framework); `force_retune` searches again."""
    champ = champions.active(target, db_path)
    if champ is None:
        raise ChampionError(f"{target} has no active champion")
    spec = role_spec(champ["spec"], role)
    view = AsOfView(panel, form, target)
    sched = {**cfg["schedule"], **{k: v for k, v in spec["schedule"].items() if v is not None or k == "train_window_days"}}
    fold = make_folds(first_forecast_day, first_forecast_day, view.first_day(), {**sched, "retrain_every_days": 1})[0]
    ctx = FitContext(view, fold.train_days, fold.train_end, history_days)
    iface, cache, members = view.interface(), {}, {}
    prior_reg = None if (force_retune or not spec["tuned"]) else champions.latest_artifact(target, champ["version"], db_path, role)
    prior = _read_artifact(prior_reg)["members"] if prior_reg else {}
    for name, factory in tree_candidates(spec["members"], spec["feature_set"], spec["tuned"], tuning_cfg).items():
        base = name[:-len("_tuned")] if name.endswith("_tuned") else name
        if base in prior and prior[base].get("tuned_at"):
            _, ver = _builder(iface, spec["feature_set"])
            cache[("tuned", target, base, ver)] = {"tuned_at": dt.date.fromisoformat(prior[base]["tuned_at"]), "params": prior[base]["params"]}
        fc = factory(iface, cache)
        fc.fit(ctx)
        state = cache.get(("tuned", target, base, fc.set_version))
        members[base] = {"model": fc.model, "cols": list(fc.cols), "fill": np.asarray(fc.fill, dtype=float), "z_range": fc.z_range,
                         "params": dict(getattr(fc, "params", {}) or {}), "set_version": fc.set_version,
                         "tuned_at": str(state["tuned_at"]) if state else None}
    versions = {m["set_version"] for m in members.values()}
    artifact = {"target": target, "version": champ["version"], "role": role, "spec": spec, "train_end": str(fold.train_end),
                "first_forecast_day": str(first_forecast_day), "members": members, "set_version": versions.pop(),
                "transform": dict(view.spec["transform"]), "history_days": history_days, "train_days": len(fold.train_days)}
    folder = os.path.join(models_root(db_path), target, f"v{champ['version']}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{role}_{fold.train_end}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tuned = [m["tuned_at"] for m in members.values() if m["tuned_at"]]
    tuned_at = dt.date.fromisoformat(min(tuned)) if tuned else None                      # the oldest tuning among the members
    champions.register_artifact(target, champ["version"], fold.train_end, first_forecast_day, path, _sha256(path), artifact["set_version"],
                                list(members), db_path, now, role, tuned_at)
    return {"path": path, "train_end": fold.train_end, "version": champ["version"], "members": list(members), "role": role,
            "tuned_at": tuned_at}


class ChampionForecaster:
    """Forecasts the modelled quantity of one target from a fitted artifact; same contract as a Forecaster."""

    def __init__(self, artifact: dict, iface: ModelInterface):
        self.artifact, self.iface = artifact, iface
        self.spec = artifact["spec"]
        self.tr = make_transform(artifact["transform"]["name"], artifact["transform"].get("scale"))
        self.builder, _ = _builder(iface, self.spec["feature_set"])
        self.name = f"champion_v{artifact['version']}"
        self.n_clipped = 0                                       # forecasts limited to the training range (a failure signal)

    def _member(self, name: str, view: pd.DataFrame, D: dt.date) -> np.ndarray:
        m = self.artifact["members"][name]
        f = self.builder.build(view, D)[m["cols"]]
        A = np.where(np.isnan(f.to_numpy()), m["fill"], f.to_numpy())
        z, n = clip_to_training_range(m["model"].predict(A), m["z_range"])      # the same guard as in training and evaluation
        self.n_clipped += n
        return self.tr.inverse(z)

    def predict(self, view: pd.DataFrame, D: dt.date) -> np.ndarray:
        preds = np.vstack([self._member(n, view, D) for n in self.artifact["members"]])
        if len(preds) == 1:
            return preds[0]
        return np.median(preds, axis=0) if self.spec.get("method") == "median" else preds.mean(axis=0)

    def forecast_prices(self, view: pd.DataFrame, D: dt.date) -> pd.Series:
        """Prices of delivery day D (a spread forecast plus the base price known at the gate), indexed like the day's rows."""
        rows = view[view["local_date"] == D.toordinal()]
        yhat = np.asarray(self.predict(view, D), dtype=float)
        base = self.iface.spec.get("base")
        return pd.Series(yhat + rows[base].to_numpy() if self.iface.spec["kind"] == "spread" else yhat, index=rows.index)

    def days_since_fit(self, day: dt.date) -> int:
        return (day - dt.date.fromisoformat(self.artifact["train_end"])).days


def load_champion(target: str, iface: ModelInterface, db_path: Optional[str] = None, role: str = "champion") -> ChampionForecaster:
    """The active champion of a target (or its reference), verified: registered file, matching hash, same version, unchanged
    feature definitions."""
    champ = champions.active(target, db_path)
    if champ is None:
        raise ChampionError(f"{target} has no active champion")
    reg = champions.latest_artifact(target, champ["version"], db_path, role)
    if reg is None:
        raise ChampionError(f"champion version {champ['version']} of {target} has no fitted {role} artifact; fit one first")
    art = _read_artifact(reg)
    if art["version"] != champ["version"] or art["spec"] != role_spec(champ["spec"], role):
        raise ChampionError("the artifact does not belong to the active champion version")
    _, current = _builder(iface, champ["spec"]["feature_set"])
    if current != art["set_version"]:
        raise ChampionError(f"feature definitions changed since the fit (artifact {art['set_version']}, now {current}); refit the champion")
    fc = ChampionForecaster(art, iface)
    fc.name = f"{role}_v{art['version']}"
    return fc
