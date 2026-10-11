"""
champion.py — the champion registry from the command line (stage 7 of docs/Model_Development_Plan.pdf).

    python tools/champion.py init --decision model_development/selection/da_price.json     first champion of a target
    python tools/champion.py show --target da_price                                         active champion and versions
    python tools/champion.py history --target da_price                                      every change, with its numbers
    python tools/champion.py fit --target da_price --first-day 2026-10-12                   fit and store the artifact
        [--role reference] [--force-retune]       (the reference is fitted for the monitoring comparison)
    python tools/champion.py challenge --target da_price --challenger-decision FILE \\
        --champion-preds champion.csv --challenger-preds challenger.csv --as-of 2026-12-31   apply the promotion rule
    python tools/champion.py rollback --target da_price --to 1 --reason "why"                reactivate an earlier version

Rule for promotion: config/champion_policy.yaml. File: runtime/model_registry/registry.db.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.forecast_evaluation.config import load_evaluation_config  # noqa: E402
from common_layer.model_development import champions  # noqa: E402


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _show(df: pd.DataFrame) -> None:
    print(df.to_string(index=False) if len(df) else "(none)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--decision", required=True)
    p = sub.add_parser("show")
    p.add_argument("--target", required=True)
    p = sub.add_parser("history")
    p.add_argument("--target")
    p = sub.add_parser("fit")
    p.add_argument("--target", required=True)
    p.add_argument("--first-day", required=True, type=_date)
    p.add_argument("--role", choices=["champion", "reference"], default="champion", help="fit the champion or its reference")
    p.add_argument("--force-retune", action="store_true", help="search the settings again instead of reusing the earlier ones")
    p = sub.add_parser("challenge")
    p.add_argument("--target", required=True)
    p.add_argument("--challenger-decision", required=True)
    p.add_argument("--champion-preds", required=True)
    p.add_argument("--challenger-preds", required=True)
    p.add_argument("--as-of", required=True, type=_date)
    p = sub.add_parser("rollback")
    p.add_argument("--target", required=True)
    p.add_argument("--to", required=True, type=int)
    p.add_argument("--reason", required=True)
    a = ap.parse_args()
    cfg = load_evaluation_config()

    if a.cmd == "init":
        d = json.load(open(a.decision, encoding="utf-8"))
        r = champions.promote_initial(d, cfg["models"]["feature_set"], source=os.path.basename(a.decision))
        print(f"{r['target']}: version {r['version']} is the active champion: {r['spec']['type']} of {', '.join(r['spec']['members'])}")
    elif a.cmd == "show":
        c = champions.active(a.target)
        print("no active champion" if c is None else f"{a.target}: version {c['version']} since {c['promoted_utc']}\n{json.dumps(c['spec'], indent=2)}")
        _show(champions.versions(a.target).drop(columns=["spec_json"]))
    elif a.cmd == "history":
        ev = champions.events(a.target)
        _show(ev.assign(detail=ev["detail_json"].str.slice(0, 110)).drop(columns=["detail_json"]))
    elif a.cmd == "fit":
        from common_layer.configuration.forecast_formulation import load_forecast_formulation
        from common_layer.data_layer.clean_store import CleanStore
        from common_layer.forecast_evaluation.tuning import load_tuning_config
        from common_layer.market_analysis import panel as pnl
        from common_layer.model_development import serving
        r = serving.fit_artifact(a.target, pnl.build_panel(CleanStore()), load_forecast_formulation(), cfg, load_tuning_config(), a.first_day,
                                 role=a.role, force_retune=a.force_retune)
        print(f"fitted {r['role']} of version {r['version']} ({', '.join(r['members'])}), training ends {r['train_end']}, "
              f"settings tuned {r['tuned_at']}: {r['path']}")
    elif a.cmd == "challenge":
        pol = champions.load_champion_policy()
        read = lambda f: pd.read_csv(f, index_col=0, parse_dates=True)       # noqa: E731
        ev = champions.evaluate_challenge(read(a.champion_preds), read(a.challenger_preds), pol, cfg["comparison"], a.as_of,
                                          champions.days_since_last_promotion(a.target, a.as_of))
        d = json.load(open(a.challenger_decision, encoding="utf-8"))
        r = champions.apply_challenge(a.target, d, cfg["models"]["feature_set"], ev, source=os.path.basename(a.challenger_decision))
        print(("PROMOTED to version %d" % r["version"]) if r["promoted"] else "REJECTED; champion unchanged")
        for x in r["reasons"]:
            print(" -", x)
    elif a.cmd == "rollback":
        r = champions.rollback(a.target, a.to, a.reason)
        print(f"{a.target}: version {r['version']} is active again (was {r['from_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
