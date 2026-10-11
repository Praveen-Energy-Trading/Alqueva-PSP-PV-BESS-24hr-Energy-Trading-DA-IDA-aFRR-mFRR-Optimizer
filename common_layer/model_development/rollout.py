"""
rollout.py — what each price target's data allows (stage 9 of docs/Model_Development_Plan.pdf).

Reads config/rollout.yaml and returns, for one target, the evaluation configuration and the selection policy to use:

    standard           the base configuration, only the target's feature set applied
    limited_evidence   a shorter minimum training history, an expanding window and a lower minimum of scored development
                       days; the first test day is the first day with enough history, computed from the data
    deferred           no honest run is possible; the reason is returned and the tools refuse to run

`status` summarises every target for the roll-out table. Nothing here touches the locked period.
"""
from __future__ import annotations

import copy
import datetime as dt
import os
from typing import Dict, List, Optional

import pandas as pd
import yaml

from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.model_development import selection

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "rollout.yaml")
PRICE_TARGETS = ("da_price", "ida1_price", "ida2_price", "ida3_price", "xbid_price", "afrr_cap_price", "mfrr_price")


def load_rollout(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        r = yaml.safe_load(fh)
    problems = []
    for t, v in r["targets"].items():
        if v["treatment"] not in r["treatments"]:
            problems.append(f"{t}: unknown treatment {v['treatment']!r}")
        if v["treatment"] == "deferred" and not v.get("reason"):
            problems.append(f"{t}: a deferred target needs a reason")
        if not v.get("feature_set"):
            problems.append(f"{t}: feature_set is required")
    missing = [t for t in PRICE_TARGETS if t not in r["targets"]]
    if missing:
        problems.append(f"price targets without a treatment: {missing}")
    a = r["availability"]
    if not 0 <= a["availability_margin"] < 0.5 or not 0 < a["max_truth_share"] <= 1:
        problems.append("availability: margin in [0, 0.5), max_truth_share in (0, 1]")
    if problems:
        raise ValueError("; ".join(problems))
    return r


def truth_days(view: AsOfView, first: dt.date, last: dt.date) -> List[int]:
    """Ordinals of the days in [first, last] on which the target has any outcome."""
    tm = view.truth_matrix(first, last)
    return [int(d) for d in tm.dropna(how="all").index]


def effective(target: str, base_eval: Optional[dict] = None, base_policy: Optional[dict] = None, rollout: Optional[dict] = None,
              view: Optional[AsOfView] = None) -> dict:
    """The configuration to use for `target`. With a `view` the data-dependent parts are computed: the first test day of a
    limited-evidence target, the days possible for development, and the locked-test availability floor."""
    base_eval = base_eval or load_evaluation_config()
    base_policy = base_policy or selection.load_policy()
    r = rollout or load_rollout()
    spec = r["targets"].get(target, {"treatment": "standard", "feature_set": base_eval["models"]["feature_set"]})
    tr = r["treatments"][spec["treatment"]]
    cfg, policy = copy.deepcopy(base_eval), copy.deepcopy(base_policy)
    cfg["models"]["feature_set"] = spec["feature_set"]
    out: Dict = {"target": target, "treatment": spec["treatment"], "label": tr["description"], "deferred": spec["treatment"] == "deferred",
                 "blocked": False, "reasons": [], "min_truth_share": float(r["availability"]["max_truth_share"])}
    if out["deferred"]:
        out["reasons"].append(spec["reason"])
    cfg["schedule"].update(tr.get("schedule", {}))
    policy.update(tr.get("selection", {}))
    out["eval_cfg"], out["policy"] = cfg, policy
    if view is None:
        return out
    d, lk, s = cfg["development"], cfg["locked_test"], cfg["schedule"]
    all_days = truth_days(view, dt.date(2019, 1, 1), lk["end"])
    first = dt.date.fromordinal(min(all_days))
    out["first_truth"] = str(first)
    start = max(d["test_start"], first + dt.timedelta(days=s["min_train_days"] + s["gap_days"] + 1))
    d["test_start"] = start
    out["dev_start"] = str(start)
    out["dev_days_possible"] = max(0, (d["test_end"] - start).days + 1)
    dev_have = [x for x in all_days if start.toordinal() <= x <= d["test_end"].toordinal()]
    lk_have = [x for x in all_days if lk["start"].toordinal() <= x <= lk["end"].toordinal()]
    out["dev_days_with_outcomes"], out["locked_days_with_outcomes"] = len(dev_have), len(lk_have)
    out["locked_days"] = (lk["end"] - lk["start"]).days + 1
    share = len(dev_have) / out["dev_days_possible"] if out["dev_days_possible"] else 0.0
    out["dev_outcome_share"] = share
    out["min_truth_share"] = float(min(r["availability"]["max_truth_share"], max(0.0, share - r["availability"]["availability_margin"])))
    if len(dev_have) < policy["min_days"] and not out["deferred"]:
        out["blocked"] = True
        out["reasons"].append(f"{len(dev_have)} development days with outcomes from {start}, fewer than the {policy['min_days']} required")
    return out


def status(targets=PRICE_TARGETS, view_for=None) -> pd.DataFrame:
    """One row per target: treatment, feature set, and (with `view_for(target) -> AsOfView`) data coverage and readiness."""
    rows = []
    for t in targets:
        eff = effective(t, view=view_for(t) if view_for else None)
        row = {"target": t, "treatment": eff["treatment"], "feature_set": eff["eval_cfg"]["models"]["feature_set"]}
        if "first_truth" in eff:
            row.update(first_truth=eff["first_truth"], dev_start=eff["dev_start"], dev_days_with_outcomes=eff["dev_days_with_outcomes"],
                       min_dev_days=eff["policy"]["min_days"], locked_days=f"{eff['locked_days_with_outcomes']}/{eff['locked_days']}",
                       min_locked_share=round(eff["min_truth_share"], 3))
        row["ready"] = "deferred" if eff["deferred"] else ("blocked" if eff["blocked"] else "yes")
        rows.append(row)
    return pd.DataFrame(rows)
