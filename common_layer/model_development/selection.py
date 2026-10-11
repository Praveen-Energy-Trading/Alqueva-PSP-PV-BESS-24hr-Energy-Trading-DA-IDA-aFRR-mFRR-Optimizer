"""
selection.py — the champion selection policy, applied by code (stage 5 of docs/Model_Development_Plan.pdf).

The rule is written in config/model_selection.yaml (read its header for the steps); this module applies it to the
out-of-sample predictions of the tuned algorithms on the development window and returns a decision that records every test
and every reason. The decision is a plain dictionary with no time stamps, so the same predictions and the same policy
always give the same decision; the policy digest ties a decision to the exact rule that produced it.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from common_layer.forecast_evaluation.dm import daily_losses, diebold_mariano
from common_layer.forecast_evaluation.ensemble import combine
from common_layer.model_development import comparison as cmp
from common_layer.model_development.registry import digest_of

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "model_selection.yaml")
METRICS = ("mae", "rmse", "sse")
TIE_BREAKS = ("average", "simplest", "best_mae")
AVERAGES = ("mean", "median")
LOSS_OF_METRIC = {"mae": "absolute", "rmse": "squared", "sse": "squared"}


def load_policy(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        policy = yaml.safe_load(fh)["policy"]
    problems = validate_policy(policy)
    if problems:
        raise ValueError("; ".join(problems))
    return policy


def validate_policy(p: dict) -> List[str]:
    out: List[str] = []
    if p["metric"] not in METRICS:
        out.append(f"metric must be one of {METRICS}")
    if p["tie_break"] not in TIE_BREAKS:
        out.append(f"tie_break must be one of {TIE_BREAKS}")
    if p["average_method"] not in AVERAGES:
        out.append(f"average_method must be one of {AVERAGES}")
    if not 0 < p["alpha"] < 1 or p["min_days"] < 1 or not 0 < p["min_coverage"] <= 1:
        out.append("alpha in (0, 1), min_days >= 1, min_coverage in (0, 1]")
    if len(p["candidates"]) < 2 or len(set(p["candidates"])) != len(p["candidates"]):
        out.append("candidates must be at least two distinct algorithms")
    if p["reference"] not in p["candidates"]:
        out.append("reference must be one of the candidates")
    if sorted(p["complexity_order"]) != sorted(p["candidates"]):
        out.append("complexity_order must list exactly the candidates")
    return out


def policy_digest(policy: dict) -> str:
    return digest_of(policy)


def _with_keep_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The ensemble code expects month and period columns; files of a run do not carry them."""
    out = df.copy()
    if "month" not in out:
        out["month"] = [dt.date.fromordinal(int(d)).month for d in out["local_date"]]
    if "period" not in out:
        out["period"] = "all"
    return out


def _f(x) -> float:
    return float(x) if x == x else float("nan")


def select(preds: Dict[str, pd.DataFrame], policy: dict, comparison: dict, target: str = "") -> dict:
    """Apply the policy to the predictions of the candidates (`comparison`: hac_lags and small_sample of the tests)."""
    cands = list(policy["candidates"])
    missing = [m for m in cands if m not in preds]
    if missing:
        raise KeyError(f"no predictions for candidates {missing}")
    loss = LOSS_OF_METRIC[policy["metric"]]
    alpha, lags, ss = float(policy["alpha"]), int(comparison["hac_lags"]), comparison["small_sample"]
    raw = {m: preds[m] for m in cands}
    total = max(len(p.dropna(subset=["y_true"])) for p in raw.values())
    coverage = min(float(p.dropna(subset=["y_true"])["y_pred"].notna().sum() / max(1, total)) for p in raw.values())
    prep = {m: _with_keep_columns(p) for m, p in cmp.prepare(raw).items()}
    first = prep[cands[0]]
    n_days, n_hours = int(first["local_date"].nunique()), int(len(first))
    d = {"target": target, "policy_digest": policy_digest(policy), "policy": policy, "n_days": n_days, "n_hours": n_hours,
         "coverage": coverage, "champion": None, "reasons": []}

    # 0. evidence
    if n_days < int(policy["min_days"]) or coverage < float(policy["min_coverage"]):
        d["status"] = "insufficient_evidence"
        d["reasons"].append(f"{n_days} delivery days (need {policy['min_days']}) and coverage {coverage:.3f} "
                            f"(need {policy['min_coverage']}): no champion is chosen")
        return d
    d["status"] = "chosen"

    # 1. rank
    board = cmp.scoreboard(prep)
    mean_loss = {m: _f(daily_losses(p, loss).mean()) for m, p in prep.items()}
    order = sorted(cands, key=lambda m: (mean_loss[m], policy["complexity_order"].index(m)))
    d["ranking"] = [{"model": m, "mean_loss": mean_loss[m], **{k: _f(board.set_index("model").loc[m, k]) for k in ("mae", "rmse", "sse")}}
                    for m in order]
    best = order[0]
    d["best"] = best
    d["reasons"].append(f"ranked by mean {loss} loss on {n_days} days: {', '.join(order)}; best is {best}")

    # 2. parsimony
    ref = policy["reference"]
    if best == ref:
        d["parsimony"] = {"reference": ref, "tested": False, "passed": True}
        d["reasons"].append(f"the best algorithm is the reference ({ref}); nothing simpler to prefer")
    else:
        r = diebold_mariano(prep[best], prep[ref], loss, lags, ss)
        passed = bool(r["p_value"] < alpha and r["mean_diff"] < 0)
        d["parsimony"] = {"reference": ref, "tested": True, "statistic": _f(r["statistic"]), "p_value": _f(r["p_value"]),
                          "mean_diff": _f(r["mean_diff"]), "passed": passed}
        if not passed:
            d["champion"] = {"type": "single", "model": ref}
            d["champion_dev_loss"] = mean_loss[ref]
            d["reasons"].append(f"{best} is not significantly better than the reference {ref} (p = {r['p_value']:.3g}, "
                                f"alpha {alpha}); the reference is the champion")
            return d
        d["reasons"].append(f"{best} beats the reference {ref} (DM statistic {r['statistic']:.2f}, p = {r['p_value']:.3g})")

    # 3. ties
    vb = cmp.versus_best(prep, loss, lags, ss)
    d["versus_best"] = [{k: (v if isinstance(v, str) else bool(v) if isinstance(v, (bool, np.bool_)) else _f(v)) for k, v in row.items()}
                        for row in vb.to_dict("records")]
    tied = cmp.top_set(vb)
    d["tied"] = sorted(tied, key=lambda m: policy["complexity_order"].index(m))
    d["champion_dev_loss"] = mean_loss[best]
    if len(tied) == 1:
        d["champion"] = {"type": "single", "model": best}
        d["reasons"].append(f"every other algorithm is significantly worse than {best} after the Holm correction; {best} is the champion")
        return d
    d["reasons"].append(f"{', '.join(d['tied'])} cannot be told apart from {best} (Holm over the comparisons with the best, alpha {alpha})")
    tb = policy["tie_break"]
    if tb == "best_mae":
        d["champion"] = {"type": "single", "model": best}
        d["reasons"].append(f"tie_break best_mae: {best}")
    elif tb == "simplest":
        simplest = d["tied"][0]
        d["champion"] = {"type": "single", "model": simplest}
        d["champion_dev_loss"] = mean_loss[simplest]
        d["reasons"].append(f"tie_break simplest: {simplest} is the simplest of the tied algorithms")
    else:
        combo = combine({m: prep[m] for m in d["tied"]}, policy["average_method"])
        r = diebold_mariano(combo, prep[best], loss, lags, ss)
        worse = bool(r["p_value"] < alpha and r["mean_diff"] > 0)
        d["combination"] = {"members": d["tied"], "method": policy["average_method"], "mean_diff": _f(r["mean_diff"]),
                            "statistic": _f(r["statistic"]), "p_value": _f(r["p_value"]), "accepted": not worse}
        if worse:
            d["champion"] = {"type": "single", "model": best}
            d["reasons"].append(f"the {policy['average_method']} of the tied algorithms is significantly worse than {best} "
                                f"(p = {r['p_value']:.3g}); {best} is the champion")
        else:
            d["champion"] = {"type": "ensemble", "members": d["tied"], "method": policy["average_method"]}
            d["champion_dev_loss"] = _f(daily_losses(combo, loss).mean())
            d["reasons"].append(f"tie_break average: the {policy['average_method']} of {', '.join(d['tied'])} is not worse than "
                                f"{best} (mean loss difference {r['mean_diff']:+.3f}, p = {r['p_value']:.3g}); it is the champion")
    return d
