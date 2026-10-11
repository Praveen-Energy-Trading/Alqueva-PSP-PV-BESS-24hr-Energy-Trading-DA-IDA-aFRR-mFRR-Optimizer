"""
run_feature_selection.py — stability-checked feature selection for one target (Phase 7), development data only.

    python tools/run_feature_selection.py --target da_price

Builds every available catalogue feature for each period of config/feature_selection.yaml, measures out-of-sample
permutation importance (LightGBM, in price units) and keeps the features that matter in most periods. Writes the
ranking to runtime/analysis/phase7/ and the selection to config/feature_selection_results.yaml (generated file,
used by the feature set `selected_v1`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.feature_engineering import selection  # noqa: E402
from common_layer.feature_engineering.builder import FeatureBuilder  # noqa: E402
from common_layer.feature_engineering.catalogue import (RESULTS_PATH, feature_set_version, load_feature_catalogue,  # noqa: E402
                                                        load_feature_sets, load_selection_results, resolve_feature_set)
from common_layer.forecast_evaluation.asof import AsOfView  # noqa: E402
from common_layer.forecast_evaluation.config import load_evaluation_config  # noqa: E402
from common_layer.forecast_formulation.transforms import make_transform  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default=os.path.join(_REPO, "runtime", "analysis", "phase7"))
    ap.add_argument("--no-write", action="store_true", help="do not update config/feature_selection_results.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(_REPO, "config", "feature_selection.yaml"), encoding="utf-8"))
    eval_cfg, form = load_evaluation_config(), load_forecast_formulation()
    locked_start = eval_cfg["locked_test"]["start"]
    panel = pnl.build_panel(CleanStore())
    view = AsOfView(panel, form, a.target)
    iface = view.interface()
    dev_end = locked_start - dt.timedelta(days=1)
    spec = iface.spec
    ycol = panel[spec["series"]] - panel[spec["base"]] if spec["kind"] == "spread" else panel[spec["series"]]
    first_label = ycol.dropna().index.min().tz_convert("Europe/Madrid").date()
    periods = []
    configured = [(p["id"], dt.date.fromisoformat(str(p["start"])), dt.date.fromisoformat(str(p["end"]))) for p in cfg["periods"]]
    if first_label > configured[0][1] + dt.timedelta(days=30):
        # history starts late: consecutive blocks over the development part of the available history
        start, k = first_label + dt.timedelta(days=30), 1          # 30 days of warm-up for lags
        while start <= dev_end:
            end = min(start + dt.timedelta(days=cfg["auto_block_days"] - 1), dev_end)
            if (end - start).days + 1 >= cfg["min_block_days"]:
                periods.append((f"block{k}", start, end))
                k += 1
            start = end + dt.timedelta(days=1)
        print(f"labelled history starts {first_label}: using {len(periods)} blocks of about {cfg['auto_block_days']} days")
    else:
        periods = configured
    for pid, s, e in periods:
        if e >= locked_start:
            raise SystemExit(f"period {pid} reaches the locked test period; selection must use development data only")
    if len(periods) < 2:
        raise SystemExit("fewer than two periods available: no stability check is possible")
    cat = load_feature_catalogue()
    ids = resolve_feature_set("full_v1", load_feature_sets(), cat, iface.spec, iface.offsets, a.target)
    builder = FeatureBuilder(iface, cat, ids)
    tr = make_transform(iface.spec["transform"]["name"], iface.spec["transform"].get("scale"))

    from lightgbm import LGBMRegressor
    lp = cfg["learner"]
    make = lambda: LGBMRegressor(random_state=cfg["seed"], verbose=-1, n_jobs=4, **lp)  # noqa: E731

    t0, imps = time.time(), {}
    for pid, s, e in periods:
        days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
        table = selection.training_table(view, builder, days, tr)
        imps[pid] = selection.permutation_importance(table, make, ids, cfg["train_fraction"], cfg["permutation_repeats"],
                                                     cfg["seed"], tr)
        print(f"{pid}: {len(table)} hours, holdout MAE {imps[pid].attrs['base_mae']:.2f} EUR, {time.time() - t0:.0f} s")
    rank = selection.stability(imps, cfg["min_share"], cfg["min_period_fraction"])
    os.makedirs(a.out, exist_ok=True)
    rank.to_csv(os.path.join(a.out, f"selection_{a.target}.csv"), float_format="%.5f")
    chosen = [f for f in ids if f in set(rank.index[rank["selected"]])]
    print(f"\n{len(chosen)} of {len(ids)} features selected; mean rank correlation between periods "
          f"{rank.attrs['mean_rank_correlation']:.2f}")
    print(rank.head(25)[["mean_share", "periods_passed", "selected"]].round(4).to_string())
    if not a.no_write:
        res = load_selection_results() or {"version": 1, "generated_by": "tools/run_feature_selection.py", "targets": {}}
        res["targets"][a.target] = {
            "selected": chosen, "feature_set_version": feature_set_version(chosen, cat), "candidates": len(ids),
            "periods": [p[0] for p in periods], "rule": {"min_share": cfg["min_share"], "min_period_fraction": cfg["min_period_fraction"]},
            "mean_rank_correlation": round(rank.attrs["mean_rank_correlation"], 3)}
        with open(RESULTS_PATH, "w", encoding="utf-8", newline="\n") as fh:
            yaml.safe_dump(res, fh, sort_keys=True, width=110)
        print("selection written to", RESULTS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
