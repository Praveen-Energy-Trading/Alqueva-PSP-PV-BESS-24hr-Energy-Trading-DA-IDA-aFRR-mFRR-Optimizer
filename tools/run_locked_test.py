"""
run_locked_test.py — the one look at the locked test period (stage 6 of docs/Model_Development_Plan.pdf).

    python tools/run_locked_test.py --target da_price                      # DRY RUN: only the preflight checks
    python tools/run_locked_test.py --target da_price --confirm-locked     # the real run, once, if every check passes

Reads the champion decision (model_development/selection/<target>.json, made by tools/select_champion.py). A real run
fits and scores the champion and the reference on the locked window, logs the access (an algorithm can be scored once),
writes model_development/locked_test/<target>.json and .md, and records the runs in the experiment registry.
Nothing here can change the decision; a poor locked score is reported as it is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_evaluation.config import load_evaluation_config  # noqa: E402
from common_layer.forecast_evaluation.tuning import load_tuning_config  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.market_analysis.config import load_analysis_config  # noqa: E402
from common_layer.model_development import locked_test as lt, registry  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--decision", help="decision file (default model_development/selection/<target>.json)")
    ap.add_argument("--confirm-locked", action="store_true", help="really score the locked period (without it only the checks run)")
    ap.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes in code, configuration and tools")
    ap.add_argument("--no-reference", action="store_true", help="do not score the reference algorithm")
    ap.add_argument("--out", default=os.path.join(_REPO, "model_development", "locked_test"))
    a = ap.parse_args()
    path = a.decision or os.path.join(_REPO, "model_development", "selection", f"{a.target}.json")
    decision = json.load(open(path, encoding="utf-8"))
    form = load_forecast_formulation()
    panel = pnl.build_panel(CleanStore())
    from common_layer.forecast_evaluation.asof import AsOfView
    from common_layer.model_development import rollout
    eff = rollout.effective(a.target, view=AsOfView(panel, form, a.target))
    cfg = eff["eval_cfg"]
    pre = lt.preflight(decision, panel, form, cfg, allow_dirty=a.allow_dirty, policy=eff["policy"], min_truth_share=eff["min_truth_share"])
    print(f"preflight for {a.target} (champion {decision.get('champion')}):")
    for c in pre["checks"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    if not pre["ok"]:
        print("NOT READY: the locked period has not been touched.")
        return 1
    if not a.confirm_locked:
        names = ", ".join(lt.forecaster_names(decision, not a.no_reference))
        print(f"READY. A real run would score once: {names}. Add --confirm-locked to do it.")
        return 0
    tcfg = load_tuning_config()
    rep = lt.run_final(decision, panel, form, cfg, load_analysis_config()["periods"], tcfg,
                       os.path.join(_REPO, "runtime", "evaluation", a.target, "final"), with_reference=not a.no_reference)
    for r in rep["_results"]:
        registry.register_evaluation(r, tuning_cfg=tcfg, source="run_locked_test")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.target}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(lt.to_json(rep), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(a.out, f"{a.target}.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(lt.render(rep, decision))
    s = rep["locked_scores"]
    print(f"locked test {a.target}: MAE {s['mae']:.3f}, RMSE {s['rmse']:.3f}, SSE {s['sse']:,.0f}; "
          f"loss ratio locked/dev {rep['loss_ratio_locked_over_dev']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
