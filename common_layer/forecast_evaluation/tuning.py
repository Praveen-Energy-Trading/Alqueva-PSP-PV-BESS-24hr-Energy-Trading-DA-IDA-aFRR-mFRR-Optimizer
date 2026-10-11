"""
tuning.py — nested, time-aware hyper-parameter tuning of the tree models (model development stage 2).

For one outer fold only that fold's training window is used (the framework refuses anything later). Its latest
`n_blocks` blocks of `validation_days` days are the inner validation sets. For each block the inner training days end
`gap_days` + 1 days before it, so a block is always predicted by a model that saw only earlier days; the score of a
setting is the mean over the blocks (MAE or RMSE in price units). The best setting is refitted on the whole training
window. The search is repeated once `retune_every_days` of history have passed (the chosen settings are shared between
folds through the run's cache), so every choice uses only data known when it was made.

Every algorithm gets the same number of trials (`n_trials`, the defaults being trial 0, never pruned, so a tuned model
is never worse than the default on the inner validation). Search methods:
    tpe         Optuna's tree-structured Parzen estimator, seeded; a median pruner stops a trial that is worse than
                the median after the latest block (the other blocks are then not fitted)
    random      random draws from the grid, no pruning (the control for tpe)
    exhaustive  every combination, used automatically when the grid is not larger than n_trials - 1
Every trial is kept (settings, score per block, pruned or not) so that the search itself can be shown.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import warnings
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
METHODS = ("tpe", "random")
ENSEMBLE_METHODS = ("mean", "median", "trimmed_mean", "inverse_error", "top3_mean")
MIN_INNER_TRAIN_DAYS = 60                     # a validation block is dropped if its inner training set would be smaller


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
    if s["method"] not in METHODS:
        out.append(f"search.method must be one of {METHODS}")
    if (s["n_trials"] < 2 or s["validation_days"] < 14 or s.get("n_blocks", 1) < 1 or s["retune_every_days"] < 1
            or s.get("gap_days", 1) < 0 or s.get("block_jobs", 1) < 1):
        out.append("search: n_trials >= 2, validation_days >= 14, n_blocks >= 1, retune_every_days >= 1, gap_days >= 0, "
                   "block_jobs >= 1")
    if set(cfg["space"]) != set(TREE_MODELS):
        out.append(f"space must define exactly the tree models {TREE_MODELS}")
    for m, params in cfg["space"].items():
        for p, vals in params.items():
            if not isinstance(vals, list) or not vals:
                out.append(f"space.{m}.{p} must be a non-empty list")
    if any(m not in ENSEMBLE_METHODS for m in cfg["ensembles"]["methods"]):
        out.append(f"ensembles.methods must be among {ENSEMBLE_METHODS}")
    return out


def grid_size(space: Dict[str, list]) -> int:
    return int(np.prod([len(v) for v in space.values()])) if space else 1


def sample_configs(space: Dict[str, list], n_trials: int, rng: np.random.Generator) -> List[dict]:
    """`n_trials` distinct settings: the defaults (an empty override) first, then random draws from the grid
    (all of it if it is smaller than n_trials - 1)."""
    keys = list(space)
    sizes = [len(space[k]) for k in keys]
    total = grid_size(space)
    want = min(n_trials - 1, total)
    picks = rng.choice(total, size=want, replace=False) if want < total else np.arange(total)
    out: List[dict] = [{}]
    for idx in picks:
        combo, i = {}, int(idx)
        for k, n in zip(keys, sizes):
            combo[k] = space[k][i % n]
            i //= n
        out.append(combo)
    return out


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


class TunedForecaster(FeatureModel):
    def __init__(self, view, base_name: str, builder, set_version: str, tcfg: dict, cache: Optional[Dict] = None):
        super().__init__(view, lambda: build_learner(base_name, self.params), builder, set_version, cache)
        self.base_name, self.tcfg, self.params = base_name, tcfg, {}
        self.name = f"{base_name}_tuned"

    def _blocks(self, T: pd.DataFrame) -> List[dict]:
        """Inner validation blocks, latest first: arrays to fit on the days before each block and predict the block."""
        s = self.tcfg["search"]
        L, gap, nb = int(s["validation_days"]), int(s.get("gap_days", 1)), int(s.get("n_blocks", 1))
        days = np.sort(T["local_date"].unique())
        out = []
        for b in range(nb):
            end = len(days) - b * L
            if end - L < 0:
                break
            first, last = days[end - L], days[end - 1]
            tr = T[T["local_date"] < first - gap]
            if b > 0 and tr["local_date"].nunique() < MIN_INNER_TRAIN_DAYS:
                break
            va = T[(T["local_date"] >= first) & (T["local_date"] <= last)]
            feats = [c for c in self.builder.ids if tr[c].notna().any()]
            fill = tr[feats].mean()
            out.append({"A_tr": tr[feats].fillna(fill).to_numpy(), "y_tr": self.tr.forward(tr["y_raw"].to_numpy()),
                        "A_va": va[feats].fillna(fill).to_numpy(), "y_va": va["y_raw"].to_numpy(),
                        "train_days": int(tr["local_date"].nunique())})
        if not out:
            raise ValueError("training window too short for one inner validation block")
        return out

    def _score_block(self, params: dict, blk: dict) -> float:
        model = build_learner(self.base_name, params).fit(blk["A_tr"], blk["y_tr"])
        err = self.tr.inverse(model.predict(blk["A_va"])) - blk["y_va"]
        return float(np.mean(np.abs(err))) if self.tcfg["search"]["metric"] == "mae" else float(np.sqrt(np.mean(err ** 2)))

    def _search(self, T: pd.DataFrame, ctx: FitContext) -> dict:
        s = self.tcfg["search"]
        space = self.tcfg["space"][self.base_name]
        n_trials, jobs = int(s["n_trials"]), int(s.get("block_jobs", 1))
        blocks = self._blocks(T)
        seed = zlib.crc32(f"{s['seed']}|{self.view.target_id}|{self.base_name}|{ctx.train_end.toordinal()}".encode())
        method = "exhaustive" if grid_size(space) <= n_trials - 1 else s["method"]
        pool = cf.ThreadPoolExecutor(max_workers=jobs) if jobs > 1 else None
        trials: List[dict] = []
        seen: Dict[str, float] = {}

        def score_all(params: dict, first: Optional[float] = None) -> List[Optional[float]]:
            """Block scores, latest block first; `first` is the already known score of the latest block."""
            head = [first if first is not None else self._score_block(params, blocks[0])]
            rest = blocks[1:]
            if pool and rest:
                tail = list(pool.map(lambda b: self._score_block(params, b), rest))
            else:
                tail = [self._score_block(params, b) for b in rest]
            return head + tail

        def record(params: dict, scores: List[Optional[float]], pruned: bool, duplicate: bool = False) -> Optional[float]:
            mean = None if pruned or any(x is None for x in scores) else float(np.mean(scores))
            trials.append({"trial": len(trials), "params": params, "block_scores": scores, "score": mean, "pruned": pruned,
                           "default": len(trials) == 0, "duplicate": duplicate})
            return mean

        try:
            record({}, score_all({}), False)                                    # trial 0: the defaults, never pruned
            if method in ("exhaustive", "random"):
                for params in sample_configs(space, n_trials, np.random.default_rng(seed))[1:]:
                    record(params, score_all(params), False)
            else:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                pcfg = s.get("pruning", {})
                prune = bool(pcfg.get("enabled", True)) and len(blocks) > 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, n_trials - 1), multivariate=True)
                    pruner = optuna.pruners.MedianPruner(n_startup_trials=int(pcfg.get("startup_trials", 5)))
                    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

                    def objective(trial) -> float:
                        params = {k: trial.suggest_categorical(k, v) for k, v in space.items()}
                        key = _key(params)
                        if key in seen:                                          # the same setting suggested again: reuse
                            record(params, [None] * len(blocks), False, duplicate=True)
                            trials[-1]["score"] = seen[key]
                            return seen[key]
                        head = self._score_block(params, blocks[0])
                        trial.report(head, 0)
                        if prune and trial.should_prune():
                            record(params, [head] + [None] * (len(blocks) - 1), True)
                            raise optuna.TrialPruned()
                        mean = record(params, score_all(params, head), False)
                        seen[key] = mean
                        return mean

                    study.optimize(objective, n_trials=n_trials - 1, gc_after_trial=True)
        finally:
            if pool:
                pool.shutdown()
        done = [t for t in trials if t["score"] is not None]
        best = min(done, key=lambda t: t["score"])                               # first of equal scores: the defaults win ties
        for t in trials:
            t["selected"] = t is best
        return {"tuned_at": ctx.train_end, "params": best["params"], "score": best["score"],
                "default_score": trials[0]["score"], "n_trials": len(trials),
                "n_unique": len({_key(t["params"]) for t in trials}), "n_pruned": sum(t["pruned"] for t in trials),
                "inner_train_days": blocks[0]["train_days"], "validation_days": int(s["validation_days"]),
                "n_blocks": len(blocks), "method": method, "trials": trials}

    def fit(self, ctx: FitContext) -> None:
        key = ("tuned", self.view.target_id, self.base_name, self.set_version)
        state = self.cache.get(key)
        if state is None or (ctx.train_end - state["tuned_at"]).days >= int(self.tcfg["search"]["retune_every_days"]):
            state = self._search(self._collect(ctx), ctx)
            self.cache[key] = state
            self.cache.setdefault(("tuning_log", self.view.target_id, self.base_name), []).append(state)
        self.params = state["params"]
        super().fit(ctx)
