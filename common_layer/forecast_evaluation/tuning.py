"""
tuning.py — nested, time-aware hyper-parameter tuning of the tree models.

For one outer fold only that fold's training window is used (the framework refuses anything later). Its latest
`validation_days` days are the inner validation set; the inner training days end `gap_days` + 1 days before it. The
first candidate is always the model's default settings, so a tuned model is never worse than the default on the inner
validation set; further candidates are drawn at random from config/tuning.yaml. Candidates are scored in price units
(MAE or RMSE) on the inner validation days and the best is refitted on the whole training window. The search is
repeated once `retune_every_days` of history have passed (the chosen settings are shared between folds through the
run's cache), so every choice uses only data known when it was made.
"""
from __future__ import annotations

import datetime as dt
import itertools
import os
import zlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from common_layer.forecast_evaluation.config import TREE_MODELS
from common_layer.forecast_evaluation.forecasters import FeatureModel, FitContext
from common_layer.forecast_evaluation.tree_models import build_learner

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "tuning.yaml")
METRICS = ("mae", "rmse")
ENSEMBLE_METHODS = ("mean", "median", "trimmed_mean", "inverse_error", "top3_mean")


def load_tuning_config(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    problems = validate_tuning_config(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    return cfg


def validate_tuning_config(cfg: dict) -> List[str]:
    out: List[str] = []
    s = cfg["search"]
    if s["metric"] not in METRICS:
        out.append(f"search.metric must be one of {METRICS}")
    if s["n_iter"] < 1 or s["validation_days"] < 30 or s["retune_every_days"] < 1 or s.get("gap_days", 1) < 0:
        out.append("search: n_iter >= 1, validation_days >= 30, retune_every_days >= 1, gap_days >= 0")
    if set(cfg["space"]) != set(TREE_MODELS):
        out.append(f"space must define exactly the tree models {TREE_MODELS}")
    for m, params in cfg["space"].items():
        for p, vals in params.items():
            if not isinstance(vals, list) or not vals:
                out.append(f"space.{m}.{p} must be a non-empty list")
    if any(m not in ENSEMBLE_METHODS for m in cfg["ensembles"]["methods"]):
        out.append(f"ensembles.methods must be among {ENSEMBLE_METHODS}")
    return out


def sample_configs(space: Dict[str, list], n_iter: int, rng: np.random.Generator) -> List[dict]:
    """`n_iter` distinct settings: the defaults (an empty override) first, then random draws from the grid
    (all of it if it is smaller than n_iter - 1)."""
    keys = list(space)
    sizes = [len(space[k]) for k in keys]
    total = int(np.prod(sizes))
    want = min(n_iter - 1, total)
    picks = rng.choice(total, size=want, replace=False) if want < total else np.arange(total)
    out: List[dict] = [{}]
    for idx in picks:
        combo, i = {}, int(idx)
        for k, n in zip(keys, sizes):
            combo[k] = space[k][i % n]
            i //= n
        out.append(combo)
    return out


class TunedForecaster(FeatureModel):
    def __init__(self, view, base_name: str, builder, set_version: str, tcfg: dict, cache: Optional[Dict] = None):
        super().__init__(view, lambda: build_learner(base_name, self.params), builder, set_version, cache)
        self.base_name, self.tcfg, self.params = base_name, tcfg, {}
        self.name = f"{base_name}_tuned"

    def _search(self, T: pd.DataFrame, ctx: FitContext) -> dict:
        s = self.tcfg["search"]
        days = np.sort(T["local_date"].unique())
        val_start = days[-int(s["validation_days"])]
        gap = int(s.get("gap_days", 1))
        tr, va = T[T["local_date"] < val_start - gap], T[T["local_date"] >= val_start]
        feats = [c for c in self.builder.ids if tr[c].notna().any()]
        fill = tr[feats].mean()
        A_tr, A_va = tr[feats].fillna(fill).to_numpy(), va[feats].fillna(fill).to_numpy()
        y_tr, y_va = self.tr.forward(tr["y_raw"].to_numpy()), va["y_raw"].to_numpy()
        seed = zlib.crc32(f"{s['seed']}|{self.view.target_id}|{self.base_name}|{ctx.train_end.toordinal()}".encode())
        cands = sample_configs(self.tcfg["space"][self.base_name], int(s["n_iter"]), np.random.default_rng(seed))
        trials = []
        for params in cands:
            model = build_learner(self.base_name, params).fit(A_tr, y_tr)
            err = self.tr.inverse(model.predict(A_va)) - y_va
            score = float(np.mean(np.abs(err))) if s["metric"] == "mae" else float(np.sqrt(np.mean(err ** 2)))
            trials.append({"params": params, "score": score})
        best = min(trials, key=lambda t: t["score"])
        return {"tuned_at": ctx.train_end, "params": best["params"], "score": best["score"],
                "default_score": trials[0]["score"], "n_trials": len(trials),
                "inner_train_days": int(len(np.unique(tr["local_date"]))), "validation_days": int(s["validation_days"])}

    def fit(self, ctx: FitContext) -> None:
        key = ("tuned", self.view.target_id, self.base_name, self.set_version)
        state = self.cache.get(key)
        if state is None or (ctx.train_end - state["tuned_at"]).days >= int(self.tcfg["search"]["retune_every_days"]):
            state = self._search(self._collect(ctx), ctx)
            self.cache[key] = state
            self.cache.setdefault(("tuning_log", self.view.target_id, self.base_name), []).append(state)
        self.params = state["params"]
        super().fit(ctx)
