"""
make_target_evidence.py — the two tuning-evidence tables of every price target (stage 3 of docs/Model_Development_Plan.pdf, rolled out).

    python tools/make_target_evidence.py

`tools/make_tuning_evidence.py` writes the evidence of the day-ahead price. The same two tables are needed for every target, so that the
dashboard can follow its target dropdown:

    importance_by_target.csv        share of the variance of the relative score that each setting explains, from the uniformly drawn trials of the
                                    random-search control (marginal, not an interaction analysis)
    inner_vs_outer_by_target.csv    per tuning round: the gain of the chosen setting on the validation blocks it was chosen on (inner) against the gain
                                    on the days that then used it, compared with the untuned model on the same days (outer)

Run folders per target (the day-ahead price keeps its stage 2 runs):
    Optuna search   runtime/analysis/rollout/<target>/tuned          (tools/run_rollout.py)
    random control  runtime/analysis/rollout/<target>/tuned_random   (evaluate_forecaster.py --tuned --search-method random)
    untuned models  runtime/analysis/rollout/<target>/untuned        (evaluate_forecaster.py, default settings, the same folds)
A target is skipped, and said so, when one of its folders is missing. The output depends only on the folders, so rerunning gives the same files.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.model_development import evidence as ev  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TARGETS = ("da_price", "afrr_cap_price", "afrr_dn_price", "ida1_price", "ida2_price", "ida3_price", "xbid_price", "mfrr_price")


def run_folders(target: str) -> dict:
    ph9 = os.path.join(_REPO, "runtime", "analysis", "phase9")
    if target == "da_price":
        return {"tpe": os.path.join(ph9, "da_stage2_tpe_v2"), "random": os.path.join(ph9, "da_stage2_random_v2"), "untuned": os.path.join(ph9, "da_untuned")}
    base = os.path.join(_REPO, "runtime", "analysis", "rollout", target)
    return {"tpe": os.path.join(base, "tuned"), "random": os.path.join(base, "tuned_random"), "untuned": os.path.join(base, "untuned")}


def main() -> int:
    tabs = os.path.join(_REPO, "model_development", "tables")
    os.makedirs(tabs, exist_ok=True)
    imp_all, io_all = [], []
    for t in TARGETS:
        f = run_folders(t)
        missing = [k for k, p in f.items() if not os.path.exists(os.path.join(p, "tuning_trials.csv" if k != "untuned" else "run.json"))]
        if missing:
            print(f"{t}: skipped, missing {', '.join(missing)}")
            continue
        rnd = ev.load_trials(f["random"])
        imp = pd.concat([ev.importance(rnd, m) for m in sorted(rnd["model"].unique())], ignore_index=True)
        imp_all.append(imp.assign(target=t))
        log = pd.read_csv(os.path.join(f["tpe"], "tuning_log.csv"))
        tuned = {k.replace("_tuned", ""): v for k, v in ev.read_predictions(f["tpe"], "_tuned").items()}
        io = ev.inner_vs_outer(log, tuned, ev.read_predictions(f["untuned"]))
        io_all.append(io.assign(target=t))
        print(f"{t}: {len(imp)} importance rows, {len(io)} tuning rounds")
    if not imp_all:
        print("nothing to write")
        return 1
    cols = lambda d, first: d[["target"] + [c for c in d.columns if c != "target"]]                # noqa: E731
    cols(pd.concat(imp_all, ignore_index=True), 1).to_csv(os.path.join(tabs, "importance_by_target.csv"), index=False, float_format="%.4f")
    cols(pd.concat(io_all, ignore_index=True), 1).to_csv(os.path.join(tabs, "inner_vs_outer_by_target.csv"), index=False, float_format="%.4f")
    print(f"wrote {os.path.join(tabs, 'importance_by_target.csv')} and inner_vs_outer_by_target.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
