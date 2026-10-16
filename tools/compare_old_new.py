"""
compare_old_new.py — the champion against the legacy forecaster on the same days (stage 10 of docs/Model_Development_Plan.pdf).

    python tools/compare_old_new.py --target da_price --run runtime/analysis/phase9/da_stage2_tpe --suffix _tuned --days 60

The champion side is the out-of-sample prediction table of a development run (the champion's own algorithm, or the average of
its algorithms, from the active champion or from --decision). The legacy side is the forecaster of the phase folder, called day by
day for the last --days days of the development window, with the live-model mode forced to legacy. For the spread targets the
legacy forecaster is given the actual day-ahead prices of the day as base, the same base the champion uses. Output:
model_development/old_vs_new/<target>.md and .json. A record only; nothing is switched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import sys

os.environ["LIVE_MODELS_MODE"] = "legacy"                                   # the legacy side must be the legacy model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_evaluation.asof import AsOfView  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.model_development import champions, evidence, monitoring, old_new, rollout  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LEGACY = {"da_price": ("phase_2_da_day_ahead_bidding.da_price_pv_inflow_forecasting.da_price_forecaster", "forecast_da_prices"),
          "ida1_price": ("phase_3a_ida1_intraday_auction_1.ida1_price_forecasting.ida1_price_forecaster", "forecast_ida1_prices"),
          "ida2_price": ("phase_3b_ida2_intraday_auction_2.ida2_price_forecasting.ida2_price_forecaster", "forecast_ida2_prices"),
          "ida3_price": ("phase_3c_ida3_intraday_auction_3.ida3_price_forecasting.ida3_price_forecaster", "forecast_ida3_prices"),
          "xbid_price": ("phase_3d_xbid_continuous_intraday.xbid_price_forecasting.xbid_price_forecaster", "forecast_xbid_prices")}
AFRR = ("phase_4a_afrr_automatic_frequency_reserve.afrr_price_forecasting.afrr_price_forecaster", "forecast_afrr_cap_prices")
AFRR_SIDE = {"afrr_cap_price": 0, "afrr_dn_price": 1}                         # the legacy forecaster returns (up, down)
CAP_MAX = 250.0                                                               # the legacy forecaster limit (afrr_price_forecaster._CAP_MAX)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=sorted(set(LEGACY) | set(AFRR_SIDE)))
    ap.add_argument("--run", required=True, help="run folder with the champion's development predictions")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--decision", help="champion decision file (default: the active champion of the registry)")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(_REPO, "model_development", "old_vs_new"))
    a = ap.parse_args()
    if a.decision:
        d = json.load(open(a.decision, encoding="utf-8"))
        spec = champions.spec_from_decision(d, rollout.effective(a.target)["eval_cfg"]["models"]["feature_set"])
    else:
        champ = champions.active(a.target)
        if champ is None:
            raise SystemExit(f"{a.target} has no active champion; pass --decision")
        spec = champ["spec"]
    new = monitoring.dev_predictions(spec, evidence.read_predictions(a.run, a.suffix))
    last_day = int(new["local_date"].max())
    days = sorted({int(x) for x in new["local_date"]})[-a.days:]
    new = new[new["local_date"].isin(days)]
    form = load_forecast_formulation()
    view = AsOfView(pnl.build_panel(CleanStore()), form, a.target)
    if a.target in AFRR_SIDE:
        mod, fn = AFRR
        both = getattr(importlib.import_module(mod), fn)
        legacy = lambda hours, day: both(hours, day, CAP_MAX)[AFRR_SIDE[a.target]]      # noqa: E731
    else:
        mod, fn = LEGACY[a.target]
        legacy = getattr(importlib.import_module(mod), fn)
    spread = view.spec["kind"] == "spread"
    old = {}
    for day in days:
        D = dt.date.fromordinal(day)
        hours = sorted(int(h) + 1 for h in new.loc[new["local_date"] == day, "local_hour"])          # the hours the champion forecasts that day (IDA3: 12-24)
        if spread:
            base = view.truth_day(D)[view.spec["base"]].to_numpy()
            old[day] = legacy(hours, str(D), {h: float(base[h - 1]) for h in hours})
        else:
            old[day] = legacy(hours, str(D))
        print(f"{D}: legacy forecast done", flush=True)
    from common_layer.forecast_evaluation.config import load_evaluation_config
    cmpcfg = load_evaluation_config()["comparison"]
    rep = old_new.compare(new, old, cmpcfg["loss"], cmpcfg["hac_lags"], cmpcfg["small_sample"])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.target}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rep, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(a.out, f"{a.target}.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(old_new.render(a.target, rep, os.path.relpath(a.run, _REPO).replace(os.sep, "/") if a.run.startswith(_REPO) else a.run))
    print(f"{a.target}: champion MAE {rep['new']['mae']:.3f}, legacy MAE {rep['old']['mae']:.3f} ({rep['mae_change']:+.1%}), "
          f"DM p = {rep['dm']['p_value']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
