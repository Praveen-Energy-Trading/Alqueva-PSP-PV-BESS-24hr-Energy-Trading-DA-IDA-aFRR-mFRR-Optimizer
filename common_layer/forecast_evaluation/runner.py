"""
runner.py — score the tree models (or any forecasters) for one target in one call.

    evaluate(target_id, panel, formulation, eval_cfg, periods, stage="dev", candidates=None, out_dir=...)

`candidates` maps a name to a factory `f(interface, cache) -> Forecaster`; with none, the tree models listed in
config/evaluation.yaml are scored.
Every forecaster is scored on the same folds, the same days and the same as-of information. The report also gives
the scores on the hours ALL forecasters predicted, so a forecaster that skips hard hours cannot look better.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation import guards, metrics, report
from common_layer.forecast_evaluation.asof import AsOfView, ModelInterface, target_spec
from common_layer.forecast_evaluation.backtest import BacktestResult, run_backtest
from common_layer.feature_engineering.catalogue import (feature_set_version, load_feature_catalogue, load_feature_sets,
                                                         resolve_feature_set)
from common_layer.forecast_evaluation.tree_models import tree_candidates
from common_layer.forecast_evaluation.dm import diebold_mariano
from common_layer.forecast_evaluation.ensemble import combine

Candidate = Callable[[ModelInterface, dict], object]


def _common(results: Dict[str, BacktestResult]) -> Dict[str, pd.DataFrame]:
    idx = None
    for r in results.values():
        ok = r.predictions.dropna(subset=["y_true", "y_pred"]).index
        idx = ok if idx is None else idx.intersection(ok)
    return {n: r.predictions.loc[idx] for n, r in results.items()}


def evaluate(target_id: str, panel: pd.DataFrame, formulation: dict, eval_cfg: dict, periods: List[dict],
             stage: str = "dev", candidates: Optional[Dict[str, Candidate]] = None,
             out_dir: Optional[str] = None, reference: Optional[str] = None, history_days: int = 400,
             test_window: Optional[tuple] = None, log_root: Optional[str] = None,
             ensembles: Optional[List[str]] = None, ensemble_cfg: Optional[dict] = None) -> dict:
    d, lk = eval_cfg["development"], eval_cfg["locked_test"]
    t0, t1 = test_window or ((d["test_start"], d["test_end"]) if stage == "dev" else (lk["start"], lk["end"]))
    if test_window is None:
        guards.check_stage(stage, t0, t1, eval_cfg)
    view = AsOfView(panel, formulation, target_id)
    kind = view.spec["kind"]
    iface = view.interface()
    cache: dict = {}
    factories: Dict[str, Callable[[], object]] = {}
    for cname, cand in (candidates or tree_candidates(eval_cfg["models"]["candidates"], eval_cfg["models"]["feature_set"])).items():
        factories[cname] = (lambda c=cand: c(iface, cache))
    if stage == "final":
        for name in list(factories) + [f"ensemble_{m}" for m in (ensembles or [])]:
            guards.register_final_run(eval_cfg, target_id, name, {"window": f"{t0}..{t1}"}, log_root)

    if candidates:
        feature_set_note = "custom candidates (their own features)"
    else:
        fs, fcat = eval_cfg["models"]["feature_set"], load_feature_catalogue()
        fids = resolve_feature_set(fs, load_feature_sets(), fcat, iface.spec, iface.offsets, target_id)
        feature_set_note = f"{fs}, version {feature_set_version(fids, fcat)}, {len(fids)} features"

    sched = eval_cfg["schedule"]
    results: Dict[str, BacktestResult] = {}
    for name, fac in factories.items():
        results[name] = run_backtest(fac, view, t0, t1, sched, periods, history_days, name)
    base_names = list(results)
    if ensembles and len(base_names) >= 2:
        ecfg = ensemble_cfg or {}
        for m in ensembles:
            opts = ecfg.get(m, {})
            pred = combine({n: results[n].predictions for n in base_names}, m, opts.get("window_days", 28), opts.get("lag_days", 2))
            first_res = results[base_names[0]]
            results[f"ensemble_{m}"] = BacktestResult(target_id, f"ensemble_{m}", pred, first_res.folds,
                                                      {**first_res.meta, "coverage": float(pred["y_pred"].notna().mean())})
    groupings = eval_cfg["metrics"]["groupings"]
    overall_rows, by_group = [], {g: {} for g in groupings if g != "overall"}
    for name, r in results.items():
        sc = r.scores(groupings)
        row = {"forecaster": name, **sc["overall"].iloc[0].to_dict(), "coverage": r.meta["coverage"]}
        overall_rows.append(row)
        for g in by_group:
            by_group[g][name] = sc[g]
    overall = pd.DataFrame(overall_rows)
    common = _common(results)
    common_scores = pd.DataFrame([{"forecaster": n, **metrics.score(p)} for n, p in common.items()])
    note = None
    if len(results) > 1:
        note = (f"Scores on the {len(next(iter(common.values())))} hours all forecasters predicted: "
                + "; ".join(f"{r.forecaster} MAE {r.mae:.3f}, RMSE {r.rmse:.3f}, SSE {r.sse:,.0f}" for r in common_scores.itertuples()))

    ref = reference or eval_cfg["models"]["reference"]
    if ref not in results and f"{ref}_tuned" in results:
        ref = f"{ref}_tuned"
    if ref not in results:
        ref = next(iter(results))
    cmp_rows = []
    for name, r in results.items():
        if name == ref or name not in base_names:                 # combinations are compared with the best single model below
            continue
        res = diebold_mariano(r.predictions, results[ref].predictions, eval_cfg["comparison"]["loss"],
                              eval_cfg["comparison"]["hac_lags"], eval_cfg["comparison"]["small_sample"])
        cmp_rows.append({"forecaster": name, "reference": ref, **res})
    if len(results) > len(base_names):                           # each combination against the best single model
        best = overall[overall["forecaster"].isin(base_names)].sort_values("mae").iloc[0]["forecaster"]
        for name in [n for n in results if n.startswith("ensemble_")]:
            res = diebold_mariano(results[name].predictions, results[best].predictions, eval_cfg["comparison"]["loss"],
                                  eval_cfg["comparison"]["hac_lags"], eval_cfg["comparison"]["small_sample"])
            cmp_rows.append({"forecaster": name, "reference": best, **res})
    comparisons = pd.DataFrame(cmp_rows)

    spec = target_spec(formulation, target_id)
    truth_view = view.truth_matrix(t0, t1)
    first = next(iter(results.values()))
    meta = {"target": target_id, "kind": kind, "stage": stage, "test_start": str(t0), "test_end": str(t1),
            "n_folds": first.meta["n_folds"], "retrain_every_days": sched["retrain_every_days"],
            "gap_days": sched["gap_days"], "train_window_days": sched.get("train_window_days"),
            "config_digest": report.config_digest({k: eval_cfg[k] for k in ("schedule", "metrics", "comparison", "models")}, spec),
            "data_digest": report.sha256_text(truth_view.round(6).to_csv()), "code": report.git_revision(),
            "reference": ref, "note_common": note, "feature_set": feature_set_note}
    if stage == "final":
        meta["lock_status"] = guards.verify_lock(target_id, meta["data_digest"], log_root)
    md = report.render_report(meta, overall, by_group, comparisons, first.folds)
    paths = {}
    if out_dir:
        paths = report.write_outputs(out_dir, md, overall, {n: r.predictions for n, r in results.items()},
                                     {**meta, "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    log_rows = [{"model": k[2], **{kk: v for kk, v in e.items() if kk != "params"}, "params": str(e["params"])}
                for k, entries in cache.items() if isinstance(k, tuple) and k[0] == "tuning_log" for e in entries]
    return {"results": results, "overall": overall, "common": common_scores, "comparisons": comparisons,
            "report": md, "meta": meta, "paths": paths,
            "tuning_log": pd.DataFrame(log_rows) if log_rows else None}
