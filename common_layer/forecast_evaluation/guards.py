"""
guards.py — the locked final test period, and the automated leakage tests.

Locked period: development runs must end before it starts; the only run allowed to score it is a `final` run over
exactly that window, which is logged, and each forecaster may have at most `max_final_runs_per_forecaster` of them.
Someone who looks at the locked score, tunes, and looks again is the failure this prevents.

Leakage test (`perturbation_check`): replace everything a forecaster is NOT allowed to know at origin D (series
outside the information set, and values of allowed series beyond their latest permitted day) with random numbers,
refit and predict again. A forecaster that uses only what the as-of view gives is unaffected; one that reaches
into the raw panel changes its answer, and the largest change is returned. Forecasters get a data-free interface and masked views only. Labels of days after the end of the
training window cannot be read at all (FitContext raises PermissionError).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation.asof import AsOfView, PANEL_TO_INPUT
from common_layer.forecast_evaluation.forecasters import FitContext, Forecaster, _rows_of
from common_layer.forecast_evaluation.folds import make_folds

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


class LockedPeriodError(RuntimeError):
    """A run would score the locked test period outside the rules."""


def check_stage(stage: str, test_start: dt.date, test_end: dt.date, cfg: dict) -> None:
    lk = cfg["locked_test"]
    if stage == "dev":
        if test_end >= lk["start"]:
            raise LockedPeriodError(f"development scoring must end before the locked period starts ({lk['start']}); got {test_end}")
    elif stage == "final":
        if (test_start, test_end) != (lk["start"], lk["end"]):
            raise LockedPeriodError(f"a final run must cover exactly the locked window {lk['start']}..{lk['end']}")
    else:
        raise ValueError("stage must be 'dev' or 'final'")


def _log_path(cfg: dict, root: Optional[str] = None) -> str:
    return os.path.join(root or _REPO, cfg["guards"]["access_log"])


def final_runs(cfg: dict, target: str, forecaster: str, root: Optional[str] = None) -> int:
    path = _log_path(cfg, root)
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return sum(1 for r in rows if r["target"] == target and r["forecaster"] == forecaster)


def register_final_run(cfg: dict, target: str, forecaster: str, extra: Optional[dict] = None,
                       root: Optional[str] = None) -> None:
    """Refuses a repeat final run, otherwise appends the access to the log."""
    limit = cfg["guards"]["max_final_runs_per_forecaster"]
    if final_runs(cfg, target, forecaster, root) >= limit:
        raise LockedPeriodError(f"{forecaster} has already been scored on the locked period for {target} "
                                f"(limit {limit}); a second look would let results influence tuning")
    path = _log_path(cfg, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "target": target,
                             "forecaster": forecaster, **(extra or {})}, sort_keys=True) + "\n")


def perturbation_check(factory: Callable[[object], Forecaster], panel: pd.DataFrame, formulation: dict, target_id: str,
                       origin: dt.date, schedule: dict, history_days: int = 400, seed: int = 0) -> float:
    """Largest change in a forecaster's prediction for `origin` when forbidden information is randomised."""
    view1 = AsOfView(panel, formulation, target_id)
    fold = make_folds(origin, origin, view1.first_day(), {**schedule, "retrain_every_days": 1})[0]

    def predict_with(p: pd.DataFrame) -> np.ndarray:
        view = AsOfView(p, formulation, target_id)
        f = factory(view.interface())
        f.fit(FitContext(view, fold.train_days, fold.train_end, history_days))
        v = view.at(origin, history_days)
        out = np.asarray(f.predict(v, origin), dtype=float)
        return out

    p1 = predict_with(panel)
    rng = np.random.default_rng(seed)
    noisy = panel.copy()
    local = noisy.index.tz_convert("Europe/Madrid")
    ordd = np.array([d.toordinal() for d in local.date])
    for col in noisy.columns:
        if col not in PANEL_TO_INPUT:
            continue
        off = view1.offsets.get(col)
        mask = np.ones(len(noisy), dtype=bool) if off is None else ordd > origin.toordinal() + off
        vals = noisy[col].to_numpy(dtype=float).copy()
        vals[mask] = rng.normal(50.0, 100.0, mask.sum())
        noisy[col] = vals
    p2 = predict_with(noisy)
    if len(p1) != len(p2):
        return float("inf")
    both = np.isfinite(p1) & np.isfinite(p2)
    if (np.isfinite(p1) != np.isfinite(p2)).any():
        return float("inf")
    return float(np.max(np.abs(p1[both] - p2[both]))) if both.any() else 0.0


def _lock_file(root: Optional[str] = None) -> str:
    return os.path.join(root or _REPO, "config", "evaluation_lock.json")


def read_locks(root: Optional[str] = None) -> Dict[str, dict]:
    path = _lock_file(root)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_lock(target: str, digest: str, window: str, root: Optional[str] = None) -> None:
    """Freeze the outcomes of the locked window for one target (their digest) in config/evaluation_lock.json."""
    locks = read_locks(root)
    locks[target] = {"data_digest": digest, "window": window}
    with open(_lock_file(root), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(locks, fh, indent=2, sort_keys=True)


def verify_lock(target: str, digest: str, root: Optional[str] = None) -> str:
    """'verified' if the locked outcomes still match their frozen digest, 'not locked' if none was frozen.
    A mismatch (the data of the locked period was revised or reloaded) raises: the locked test is no longer the same test."""
    entry = read_locks(root).get(target)
    if entry is None:
        return "not locked"
    if entry["data_digest"] != digest:
        raise LockedPeriodError(f"the outcomes of the locked period for {target} changed since they were frozen "
                                f"({entry['data_digest'][:12]} != {digest[:12]})")
    return "verified"
