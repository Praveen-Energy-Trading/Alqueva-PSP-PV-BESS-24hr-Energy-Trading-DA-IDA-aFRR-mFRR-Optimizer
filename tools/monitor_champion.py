"""
monitor_champion.py — log, track and report on the champion (stage 8 of docs/Model_Development_Plan.pdf).

    python tools/monitor_champion.py calibrate --target da_price --run runtime/analysis/phase9/da_stage2_tpe --suffix _tuned
    python tools/monitor_champion.py log --target da_price --day 2026-10-12        append one day's forecasts (champion and reference)
    python tools/monitor_champion.py plan --target da_price --as-of 2026-11-15     is a refit or a re-tuning due?
    python tools/monitor_champion.py report --target da_price --as-of 2026-11-15   status, flags, recommendations

Rule and thresholds: config/monitoring.yaml. Nothing here changes the champion; a report only recommends.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.forecast_evaluation.config import load_evaluation_config  # noqa: E402
from common_layer.forecast_evaluation.tuning import load_tuning_config  # noqa: E402
from common_layer.model_development import champions, evidence, monitoring  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _view(target: str):
    from common_layer.configuration.forecast_formulation import load_forecast_formulation
    from common_layer.data_layer.clean_store import CleanStore
    from common_layer.forecast_evaluation.asof import AsOfView
    from common_layer.market_analysis import panel as pnl
    return AsOfView(pnl.build_panel(CleanStore()), load_forecast_formulation(), target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--target", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--suffix", default="")
    p = sub.add_parser("log")
    p.add_argument("--target", required=True)
    p.add_argument("--day", required=True, type=_date)
    p.add_argument("--no-reference", action="store_true")
    for name in ("plan", "report"):
        p = sub.add_parser(name)
        p.add_argument("--target", required=True)
        p.add_argument("--as-of", required=True, type=_date)
    p.add_argument("--out", default=os.path.join(_REPO, "model_development", "monitoring"))
    a = ap.parse_args()
    ecfg = load_evaluation_config()
    cfg = monitoring.load_monitoring_config(tuning_cfg=load_tuning_config(), eval_cfg=ecfg)

    if a.cmd == "calibrate":
        champ = champions.active(a.target)
        if champ is None:
            raise SystemExit(f"{a.target} has no active champion")
        base = monitoring.calibrate(monitoring.dev_predictions(champ["spec"], evidence.read_predictions(a.run, a.suffix)), cfg)
        monitoring.save_baseline(a.target, champ["version"], base)
        print(f"{a.target} v{champ['version']}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in base.items() if k != "config_digest"))
    elif a.cmd == "log":
        from common_layer.model_development import serving
        view = _view(a.target)
        champ = champions.active(a.target)
        for role in ("champion",) + (() if a.no_reference else ("reference",)):
            fc = serving.load_champion(a.target, view.interface(), role=role)
            n = monitoring.log_day(a.target, role, fc, view, a.day, champ["version"], fc.artifact["train_end"])
            print(f"{role}: logged {n} hours for {a.day}")
    elif a.cmd == "plan":
        print(json.dumps(monitoring.plan(a.target, a.as_of, cfg, ecfg), indent=2, default=str))
    elif a.cmd == "report":
        view = _view(a.target)
        rep = monitoring.build_report(a.target, a.as_of, view, cfg, ecfg)
        base = monitoring.load_baseline(a.target, rep["version"])
        os.makedirs(a.out, exist_ok=True)
        fig = f"{a.target}_rolling_error.png"
        monitoring.plot_rolling(rep, base, os.path.join(a.out, fig))
        with open(os.path.join(a.out, f"{a.target}.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(monitoring.render(rep, fig))
        with open(os.path.join(a.out, f"{a.target}.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(monitoring.to_json(rep), fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        print(f"{a.target}: status {rep['status']['status']}; schedule action {rep['schedule']['action']}")
        for r in rep["recommendations"]:
            print(" -", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
