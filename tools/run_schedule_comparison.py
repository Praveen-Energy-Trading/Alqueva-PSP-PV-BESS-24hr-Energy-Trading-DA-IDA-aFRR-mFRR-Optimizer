"""
run_schedule_comparison.py — calibration-window length and retraining frequency (Phase 9), development window only.

    python tools/run_schedule_comparison.py --target da_price

Scores the same model (config/tuning.yaml: schedule_comparison.model) with each combination in
`schedule_comparison.variants` on the same delivery days, then compares every variant with the reference variant by the
Diebold-Mariano test. Output: runtime/analysis/phase9/schedule_<target>.csv and a printed table.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_evaluation import metrics  # noqa: E402
from common_layer.forecast_evaluation.config import load_evaluation_config  # noqa: E402
from common_layer.forecast_evaluation.dm import diebold_mariano  # noqa: E402
from common_layer.forecast_evaluation.runner import evaluate  # noqa: E402
from common_layer.forecast_evaluation.tree_models import tree_candidates  # noqa: E402
from common_layer.forecast_evaluation.tuning import load_tuning_config  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.market_analysis.config import load_analysis_config  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _key(v: dict) -> tuple:
    return (v["train_window_days"], v["retrain_every_days"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default=os.path.join(_REPO, "runtime", "analysis", "phase9"))
    a = ap.parse_args()
    tcfg = load_tuning_config()
    sc = tcfg["schedule_comparison"]
    base_cfg, form = load_evaluation_config(), load_forecast_formulation()
    periods = load_analysis_config()["periods"]
    panel = pnl.build_panel(CleanStore())
    variants = list(sc["variants"])
    if _key(sc["reference"]) not in [_key(v) for v in variants]:
        variants.append(sc["reference"])

    preds, rows = {}, []
    for v in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["schedule"]["train_window_days"] = v["train_window_days"]
        cfg["schedule"]["retrain_every_days"] = v["retrain_every_days"]
        t0 = time.time()
        res = evaluate(a.target, panel, form, cfg, periods, "dev", tree_candidates([sc["model"]], base_cfg["models"]["feature_set"]))
        p = res["results"][sc["model"]].predictions
        preds[_key(v)] = p
        s = metrics.score(p)
        rows.append({"train_window_days": v["train_window_days"] or "expanding", "retrain_every_days": v["retrain_every_days"],
                     "folds": res["meta"]["n_folds"], **{k: s[k] for k in ("n", "sse", "mae", "rmse")}})
        print(f"{v}: MAE {s['mae']:.3f} RMSE {s['rmse']:.3f} ({time.time() - t0:.0f} s)")
    ref = preds[_key(sc["reference"])]
    out = pd.DataFrame(rows)
    stats = []
    for v in variants:
        if _key(v) == _key(sc["reference"]):
            stats.append({"dm_vs_reference": float("nan"), "p_value": float("nan")})
            continue
        r = diebold_mariano(preds[_key(v)], ref, base_cfg["comparison"]["loss"], base_cfg["comparison"]["hac_lags"],
                            base_cfg["comparison"]["small_sample"])
        stats.append({"dm_vs_reference": r["statistic"], "p_value": r["p_value"]})
    out = pd.concat([out, pd.DataFrame(stats)], axis=1)
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, f"schedule_{a.target}.csv"), index=False, float_format="%.4f")
    print(f"\nreference: window {sc['reference']['train_window_days']} days, refit every {sc['reference']['retrain_every_days']} days "
          f"(negative statistic = smaller loss than the reference)")
    print(out.round(3).to_string(index=False))
    from common_layer.model_development import registry
    label = lambda v: f"{sc['model']}|window={v['train_window_days'] or 'expanding'}|refit={v['retrain_every_days']}"  # noqa: E731
    r = registry.register(
        "schedule_comparison", a.target,
        [{"forecaster": label(v), "n": row["n"], "sse": row["sse"], "mae": row["mae"], "rmse": row["rmse"], "role": "schedule"}
         for v, row in zip(variants, rows)],
        stage="dev", window=(res["meta"]["test_start"], res["meta"]["test_end"]), feature_set=str(base_cfg["models"]["feature_set"]),
        schedule={"reference": sc["reference"]}, config_digest=registry.digest_of({"schedule_comparison": sc, "evaluation": base_cfg["comparison"]}),
        data_digest=res["meta"]["data_digest"], code_revision=res["meta"]["code"],
        comparisons=[{"forecaster": label(v), "reference": label(sc["reference"]), "n_days": None, "mean_diff": None,
                      "statistic": st["dm_vs_reference"], "p_value": st["p_value"]}
                     for v, st in zip(variants, stats) if _key(v) != _key(sc["reference"])],
        source="run_schedule_comparison", artifacts_dir=a.out)
    print(f"registered run {r['run_key']} attempt {r['attempt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
