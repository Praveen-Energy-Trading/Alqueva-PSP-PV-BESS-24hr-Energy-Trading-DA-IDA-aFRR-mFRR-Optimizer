"""
explore_mfrr.py — what the short mFRR history can show: an EXPLORATORY comparison, not a decision.

    python tools/explore_mfrr.py

mFRR is deferred in config/rollout.yaml: its prices start on 2024-11-27, so the development window (to 2025-10-11) gives far fewer scored days
than the 250 the evidence rule requires. This script does NOT change that rule and does not choose a champion. It scores the five tree
algorithms with their default settings (nothing is tuned) and the naive forecasts on the days the history allows, with the same
rolling-origin design as the other short-history targets (180 training days before the first test day, expanding window), on the
development window only. The locked test year is never read.

Output: model_development/exploratory/mfrr_price.json (scores on the common hours, the gain of the best algorithm over the best naive
forecast, the Diebold-Mariano test, the days and the rule that is not met). The dashboard shows it, labelled as exploratory.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd  # noqa: E402

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_evaluation.asof import AsOfView  # noqa: E402
from common_layer.forecast_evaluation.runner import evaluate  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.market_analysis.config import load_analysis_config  # noqa: E402
from common_layer.model_development import benchmark as bm, rollout  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TARGET = "mfrr_price"
NAIVE = {"naive_week": 7, "naive_3days": 3}              # mFRR prices are published three days late: the latest usable day is D-3


def main() -> int:
    t0 = time.time()
    form = load_forecast_formulation()
    periods = load_analysis_config()["periods"]
    panel = pnl.build_panel(CleanStore())
    view = AsOfView(panel, form, TARGET)

    roll = copy.deepcopy(rollout.load_rollout())
    roll["targets"][TARGET]["treatment"] = "limited_evidence"          # the schedule of the other short-history targets, for this exploration only
    eff = rollout.effective(TARGET, rollout=roll, view=view)
    required = int(rollout.load_rollout()["treatments"]["limited_evidence"]["selection"]["min_days"])
    out_dir = os.path.join(_REPO, "runtime", "analysis", "exploratory", TARGET)
    res = evaluate(TARGET, panel, form, eff["eval_cfg"], periods, "dev", None, out_dir)
    preds = {n: r.predictions for n, r in res["results"].items()}
    names = list(preds)

    # every algorithm and every naive forecast scored on the same hours
    first = preds[names[0]]
    tables = dict(preds)
    for name, lag in NAIVE.items():
        t = first.copy()
        t["y_pred"] = bm.naive_prediction(view, first, lag)
        tables[name] = t
    common = bm.comparison.prepare(tables)
    scores = {n: {k: float(v) for k, v in bm.metrics.score(p).items()} for n, p in common.items()}
    algos = [n for n in names]
    best = min(algos, key=lambda n: scores[n]["mae"])
    best_naive = min(NAIVE, key=lambda n: scores[n]["mae"])
    dm = bm.diebold_mariano(common[best], common[best_naive], "absolute", 7, "hln")
    days = sorted(set(int(x) for x in common[best]["local_date"]))
    result = {
        "target": TARGET, "status": "exploratory",
        "note": "Not a decision: fewer scored days than the evidence rule requires; nothing tuned; no champion chosen; the locked year was not read.",
        "first_test_day": str(pd.Timestamp.fromordinal(days[0]).date()), "last_test_day": str(pd.Timestamp.fromordinal(days[-1]).date()),
        "scored_days": len(days), "required_days": required, "scored_hours": int(scores[best]["n"]),
        "training": {"min_train_days": int(eff["eval_cfg"]["schedule"]["min_train_days"]), "window": "expanding", "refit_every_days": int(eff["eval_cfg"]["schedule"]["retrain_every_days"])},
        "feature_set": eff["eval_cfg"]["models"]["feature_set"], "naive_lags_days": NAIVE,
        "scores": scores, "best_algorithm": best, "best_naive": best_naive,
        "gain_of_best_over_best_naive": float(1 - scores[best]["mae"] / scores[best_naive]["mae"]),
        "dm_best_vs_best_naive": {k: float(v) for k, v in dm.items()},
    }
    out = os.path.join(_REPO, "model_development", "exploratory")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{TARGET}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
    print(f"{result['scored_days']} scored days ({result['first_test_day']} to {result['last_test_day']}), {result['required_days']} required")
    print(pd.DataFrame(scores).T[["mae", "rmse", "sse"]].round(3).to_string())
    print(f"best algorithm: {best}; best naive: {best_naive}; gain {100 * result['gain_of_best_over_best_naive']:.1f}%; DM p = {dm.get('p_value')}")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
