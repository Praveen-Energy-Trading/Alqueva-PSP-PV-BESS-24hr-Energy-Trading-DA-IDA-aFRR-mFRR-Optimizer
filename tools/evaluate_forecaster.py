"""
evaluate_forecaster.py — score any forecaster of any target in the same way (Phase 6).

    python tools/evaluate_forecaster.py --target da_price                       # the five tree models, development window
    python tools/evaluate_forecaster.py --target da_price --models lightgbm xgboost
    python tools/evaluate_forecaster.py --target da_price --tuned --ensembles       # nested tuning, then combinations
    python tools/evaluate_forecaster.py --target da_price --candidate my_pkg.my_module:factory --name my_model
    python tools/evaluate_forecaster.py --target ida1_price --stage final --candidate ... --name my_model
    python tools/evaluate_forecaster.py --target da_price --lock                 # freeze the locked-window outcomes

A candidate is `factory(interface, cache) -> forecaster` (see common_layer/forecast_evaluation/forecasters.py).
Metrics are SSE, MAE and RMSE in price units. `--stage final` scores the locked period once per forecaster.
Output: runtime/evaluation/<target>/<stage>/ (report.md, metrics_overall.csv, predictions_*.csv, run.json).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import yaml  # noqa: E402

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_evaluation import guards, report  # noqa: E402
from common_layer.forecast_evaluation.asof import AsOfView  # noqa: E402
from common_layer.forecast_evaluation.config import load_evaluation_config  # noqa: E402
from common_layer.forecast_evaluation.runner import evaluate  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.market_analysis.config import load_analysis_config  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--stage", choices=["dev", "final"], default="dev")
    ap.add_argument("--models", nargs="+", help="subset of the tree models in config/evaluation.yaml")
    ap.add_argument("--feature-set", help="feature set from config/feature_sets.yaml (default: models.feature_set)")
    ap.add_argument("--no-register", action="store_true", help="do not record the run in the experiment registry")
    ap.add_argument("--tuned", action="store_true", help="tune each model inside every fold (config/tuning.yaml; nested, time-aware)")
    ap.add_argument("--search-method", choices=["tpe", "random"], help="override search.method of config/tuning.yaml")
    ap.add_argument("--n-trials", type=int, help="override search.n_trials (the same for every algorithm)")
    ap.add_argument("--ensembles", nargs="*", help="forecast combinations to add (default none; no names = all in config/tuning.yaml)")
    ap.add_argument("--candidate", help="module:function returning a forecaster factory(interface, cache)")
    ap.add_argument("--name", default="candidate")
    ap.add_argument("--out")
    ap.add_argument("--lock", action="store_true", help="freeze the locked-window outcomes of this target and exit")
    a = ap.parse_args()

    form = load_forecast_formulation()
    periods = load_analysis_config()["periods"]
    panel = pnl.build_panel(CleanStore())
    from common_layer.model_development import rollout
    eff = rollout.effective(a.target, view=AsOfView(panel, form, a.target))
    if eff["deferred"] or eff["blocked"]:
        print(f"{a.target} cannot be evaluated ({eff['treatment']}): " + "; ".join(eff["reasons"]))
        return 2
    eval_cfg = eff["eval_cfg"]
    if eff["treatment"] != "standard":
        print(f"{a.target}: {eff['treatment']} ({eff['label']}); first test day {eff['dev_start']}")
    if a.feature_set:
        eval_cfg["models"]["feature_set"] = a.feature_set
    if a.lock:
        view = AsOfView(panel, form, a.target)
        lk = eval_cfg["locked_test"]
        digest = report.sha256_text(view.truth_matrix(lk["start"], lk["end"]).round(6).to_csv())
        guards.write_lock(a.target, digest, f"{lk['start']}..{lk['end']}")
        print(f"locked {a.target}: {digest[:16]} for {lk['start']}..{lk['end']}")
        return 0
    candidate = None
    if a.candidate:
        mod, _, fn = a.candidate.partition(":")
        candidate = getattr(importlib.import_module(mod), fn)
    out = a.out or os.path.join(_REPO, "runtime", "evaluation", a.target, a.stage)
    t0 = time.time()
    cands = {a.name: candidate} if candidate else None
    from common_layer.forecast_evaluation.tree_models import tree_candidates
    from common_layer.forecast_evaluation.tuning import load_tuning_config
    tcfg = load_tuning_config()
    if a.search_method:
        tcfg["search"]["method"] = a.search_method
    if a.n_trials:
        tcfg["search"]["n_trials"] = a.n_trials
    if cands is None and (a.models or a.tuned):
        cands = tree_candidates(a.models or eval_cfg["models"]["candidates"], eval_cfg["models"]["feature_set"], a.tuned, tcfg)
    ens = None
    if a.ensembles is not None:
        ens = a.ensembles or tcfg["ensembles"]["methods"]
    res = evaluate(a.target, panel, form, eval_cfg, periods, a.stage, cands, out, ensembles=ens, ensemble_cfg=tcfg["ensembles"])
    print(f"evaluated {a.target} ({res['meta']['kind']}, {a.stage}) in {time.time() - t0:.0f} s -> {res['paths'].get('report')}")
    print(res["overall"].round(3).to_string(index=False))
    print(res["comparisons"].round(4).to_string(index=False))
    clipped = {n: r.meta.get("clipped", 0) for n, r in res["results"].items() if r.meta.get("clipped", 0)}
    if clipped:
        print(f"WARNING: forecasts limited to the training range (a numerical failure of the model): {clipped}")
    if res.get("tuning_log") is not None:
        res["tuning_log"].to_csv(os.path.join(out, "tuning_log.csv"), index=False, float_format="%.4f")
        print(res["tuning_log"].round(3).to_string(index=False))
    if res.get("tuning_trials") is not None:
        res["tuning_trials"].to_csv(os.path.join(out, "tuning_trials.csv"), index=False, float_format="%.4f")
    if not a.no_register:
        from common_layer.model_development import registry
        r = registry.register_evaluation(res, tuning_cfg=tcfg, artifacts_dir=out)
        print(f"registered run {r['run_key']} attempt {r['attempt']}"
              + ("" if r["reproduced"] is None else f" (same scores as earlier attempts: {r['reproduced']})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
