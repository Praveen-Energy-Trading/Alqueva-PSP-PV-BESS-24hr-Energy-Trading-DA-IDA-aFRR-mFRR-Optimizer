"""
benchmark_naive.py — score the naive benchmark next to the champion, on the same hours (see common_layer/model_development/benchmark.py).

    python tools/benchmark_naive.py --all                 # every target that has a champion decision
    python tools/benchmark_naive.py --target da_price

For each target it reads the champion decision, the champion's development-window predictions and (if the locked test has been run)
its locked-window predictions, builds the naive forecasts from the prices known at the time, and writes
model_development/benchmark/<target>.json: the scores of the champion, the reference (decision tree) and each naive forecast, the gain
of the champion over each, and the Diebold-Mariano test. A naive forecast has nothing to fit, so this never touches the locked-test
lock and changes no decision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd  # noqa: E402

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_evaluation.asof import AsOfView  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.model_development import benchmark as bm, locked_test as lt, monitoring, rollout, selection, synthetic  # noqa: E402
from common_layer.model_development import evidence as ev  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _locked_tables(decision: dict, target: str):
    base = os.path.join(_REPO, "runtime", "evaluation", target, "final")
    champ = lt.champion_name(decision)
    ref = decision["policy"]["reference"] + ("_tuned" if decision["source"]["tuned"] else "")
    cf = os.path.join(base, "champion", f"predictions_{champ}.csv")
    rf = next((p for p in (os.path.join(base, "reference", f"predictions_{ref}.csv"), os.path.join(base, "champion", f"predictions_{ref}.csv"))
               if os.path.exists(p)), None)
    if not os.path.exists(cf) or rf is None:
        return None, None
    read = lambda p: pd.read_csv(p, index_col=0, parse_dates=True)       # noqa: E731
    return read(cf), read(rf)


def one(target: str, panel: pd.DataFrame, form: dict) -> dict | None:
    path = os.path.join(_REPO, "model_development", "selection", f"{target}.json")
    if not os.path.exists(path):
        return None
    decision = json.load(open(path, encoding="utf-8"))
    if decision.get("champion") is None:
        return None
    eff = rollout.effective(target, view=AsOfView(panel, form, target))
    cfg = eff["eval_cfg"]
    loss = selection.LOSS_OF_METRIC[decision["policy"]["metric"]]
    comp = cfg["comparison"]
    out = {"target": target, "loss": loss, "lags": {k: v for k, v in bm.NAIVE_LAGS.items()}, "headline": bm.SEASONAL}

    preds = ev.read_predictions(os.path.join(_REPO, decision["source"]["run"]), decision["source"]["suffix"])
    spec = decision["champion"]
    spec = {"type": spec["type"], "members": spec.get("members") or [spec["model"]], "method": spec.get("method")}
    champ_dev = monitoring.dev_predictions(spec, preds)
    ref_dev = preds.get(decision["policy"]["reference"])
    view = AsOfView(panel, form, target)
    out["development"] = bm.assess(bm.with_naive(view, champ_dev, ref_dev), loss, comp["hac_lags"], comp["small_sample"])
    out["development"]["window"] = [str(champ_dev.index.min().date()), str(champ_dev.index.max().date())]

    cl, rl = _locked_tables(decision, target)
    if cl is not None:
        filled, fill = synthetic.fill_locked(panel, form, target, cfg)           # the same (labelled) artificial outcomes as the locked test
        lv = AsOfView(filled, form, target)
        tables = bm.with_naive(lv, cl, rl)
        if fill.get("method"):                                                  # artificial days would flatter a forecast that is built from the base price
            tables = {n: synthetic.real_only(t, fill) for n, t in tables.items()}
        out["locked"] = bm.assess(tables, loss, comp["hac_lags"], comp["small_sample"])
        out["locked"]["window"] = [str(cl.index.min().date()), str(cl.index.max().date())]
        if fill.get("method"):
            out["locked"]["synthetic_days"] = len(fill["days"])
            out["locked"]["note"] = "scored on real days only: the artificial days are left out for every forecast"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", help="target id (repeatable)")
    ap.add_argument("--all", action="store_true", help="every price target that has a champion decision")
    ap.add_argument("--out", default=os.path.join(_REPO, "model_development", "benchmark"))
    a = ap.parse_args()
    targets = list(rollout.PRICE_TARGETS) if a.all else (a.target or [])
    if not targets:
        ap.error("give --target or --all")
    form = load_forecast_formulation()
    panel = pnl.build_panel(CleanStore())
    os.makedirs(a.out, exist_ok=True)
    for t in targets:
        r = one(t, panel, form)
        if r is None:
            print(f"{t}: no champion decision, skipped")
            continue
        with open(os.path.join(a.out, f"{t}.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(r, fh, indent=2, sort_keys=True)
            fh.write("\n")
        for stage in ("development", "locked"):
            if stage not in r:
                continue
            s, g = r[stage]["scores"], r[stage]["gain_of_champion_over"]
            names = " | ".join(f"{n} {s[n]['mae']:.2f}" for n in s)
            gains = ", ".join(f"{n} {100 * v:+.1f}%" for n, v in g.items() if v is not None)
            print(f"{t} {stage}: MAE {names}; champion gain over: {gains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
