"""
locked_test.py — score the champion once on the locked test period (stage 6 of docs/Model_Development_Plan.pdf).

The locked period (config/evaluation.yaml) is the one stretch of data that took no part in tuning, comparing or choosing the
champion. It may be scored once per algorithm; every access is logged (forecast_evaluation.guards). This module adds the
checks that must hold BEFORE that one look, and the report after it:

    preflight   the decision exists and was made by the current policy; the outcomes of the period are all there and frozen
                (digest in config/evaluation_lock.json); no algorithm to be scored has been scored before; the schedule
                matches the one the decision was made with; the code is committed
    run_final   fits and scores the champion (a single tuned or default algorithm, or the average of the tied algorithms)
                exactly as in the development runs, and the reference (the simplest algorithm) for the same days
    report      locked scores (SSE, MAE, RMSE), the champion against the reference by the Diebold-Mariano test, and the loss
                on the locked period against the loss on the development window

Nothing here changes a decision. If the locked score is disappointing, the number is reported as it is.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation import guards, report as eval_report
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.dm import daily_losses, diebold_mariano
from common_layer.forecast_evaluation.runner import evaluate
from common_layer.forecast_evaluation.tree_models import tree_candidates
from common_layer.model_development import registry, selection

MIN_TRUTH_SHARE = 0.99                 # share of locked days whose outcomes must be present


def champion_members(decision: dict) -> List[str]:
    c = decision["champion"]
    return list(c["members"]) if c["type"] == "ensemble" else [c["model"]]


def champion_name(decision: dict) -> str:
    c = decision["champion"]
    if c["type"] == "ensemble":
        return f"ensemble_{c['method']}"
    return c["model"] + ("_tuned" if decision["source"]["tuned"] else "")


def forecaster_names(decision: dict, with_reference: bool = True) -> List[str]:
    """Every forecaster the locked run will score (each is one logged access)."""
    tuned = decision["source"]["tuned"]
    names = [m + ("_tuned" if tuned else "") for m in champion_members(decision)]
    if decision["champion"]["type"] == "ensemble":
        names.append(champion_name(decision))
    ref = decision["policy"]["reference"]
    if with_reference and ref not in champion_members(decision):
        names.append(ref + ("_tuned" if tuned else ""))
    return names


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def truth_digest(view: AsOfView, cfg: dict) -> tuple:
    lk = cfg["locked_test"]
    tm = view.truth_matrix(lk["start"], lk["end"])
    expected = (lk["end"] - lk["start"]).days + 1                      # calendar days of the window, whatever the data holds
    return eval_report.sha256_text(tm.round(6).to_csv()), expected, int(tm.dropna(how="all").shape[0])


def preflight(decision: dict, panel: pd.DataFrame, form: dict, cfg: dict, *, policy_path: Optional[str] = None,
              log_root: Optional[str] = None, lock_root: Optional[str] = None, allow_dirty: bool = False,
              code_revision: Optional[str] = None, policy: Optional[dict] = None, min_truth_share: float = MIN_TRUTH_SHARE) -> dict:
    """The checks before the one look at the locked period. `ok` is True only if every check passes."""
    target = decision["target"]
    checks: List[dict] = []
    chosen = decision.get("status") == "chosen" and decision.get("champion") is not None
    checks.append(_check("decision_chosen", chosen, f"status {decision.get('status')}, champion {decision.get('champion')}"))
    if not chosen:
        return {"ok": False, "checks": checks}
    try:
        current = selection.policy_digest(policy or selection.load_policy(policy_path))
    except Exception as e:                                                      # pragma: no cover - a broken policy file
        current = f"unreadable: {e}"
    checks.append(_check("policy_unchanged", current == decision["policy_digest"],
                         f"decision made with policy {decision['policy_digest'][:12]}, current policy {str(current)[:12]}"))
    view = AsOfView(panel, form, target)
    digest, expected, present = truth_digest(view, cfg)
    checks.append(_check("outcomes_present", expected > 0 and present / expected >= min_truth_share,
                         f"{present} of {expected} locked days have outcomes (need {min_truth_share:.1%})"))
    try:
        status = guards.verify_lock(target, digest, lock_root)
        checks.append(_check("outcomes_frozen", status == "verified", f"lock status: {status} (freeze with "
                             f"tools/evaluate_forecaster.py --target {target} --lock once the data is final)"))
    except guards.LockedPeriodError as e:
        checks.append(_check("outcomes_frozen", False, str(e)))
    prior = {n: guards.final_runs(cfg, target, n, log_root) for n in forecaster_names(decision)}
    used = [n for n, k in prior.items() if k >= cfg["guards"]["max_final_runs_per_forecaster"]]
    checks.append(_check("not_scored_before", not used, "already scored on the locked period: " + (", ".join(used) or "none")))
    src, sch = decision.get("source", {}), decision.get("source", {}).get("schedule", {})
    want = {k: cfg["schedule"].get(k) for k in ("retrain_every_days", "gap_days", "train_window_days")}
    same = bool(sch) and all(sch.get(k) == v for k, v in want.items())
    checks.append(_check("schedule_matches_development", same, f"decision made with {sch or 'no schedule recorded'}, locked run uses {want}"))
    rev = code_revision or registry.code_state(eval_report.git_revision())
    checks.append(_check("code_committed", allow_dirty or not rev.endswith("+dirty"), f"code revision {rev}"
                         + (" (uncommitted changes allowed by flag)" if allow_dirty and rev.endswith("+dirty") else "")))
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "truth_digest": digest, "code_revision": rev}


def run_final(decision: dict, panel: pd.DataFrame, form: dict, cfg: dict, periods: List[dict], tuning_cfg: dict,
              out_dir: Optional[str] = None, log_root: Optional[str] = None, with_reference: bool = True) -> dict:
    """Score the champion (and the reference) on the locked window: this is the one look. Returns the report dictionary."""
    target = decision["target"]
    tuned = bool(decision["source"]["tuned"])
    members, c = champion_members(decision), decision["champion"]
    fs = cfg["models"]["feature_set"]
    sub = (lambda n: os.path.join(out_dir, n)) if out_dir else (lambda n: None)
    res = evaluate(target, panel, form, cfg, periods, "final", tree_candidates(members, fs, tuned, tuning_cfg), sub("champion"),
                   ensembles=[c["method"]] if c["type"] == "ensemble" else None, ensemble_cfg=tuning_cfg["ensembles"], log_root=log_root)
    champ = champion_name(decision)
    ref = decision["policy"]["reference"]
    ref_name = ref + ("_tuned" if tuned else "")
    ref_res = None
    if with_reference and ref not in members:
        ref_res = evaluate(target, panel, form, cfg, periods, "final", tree_candidates([ref], fs, tuned, tuning_cfg), sub("reference"),
                           log_root=log_root)
    ref_pred = (ref_res or res)["results"][ref_name].predictions if with_reference else None
    cp = res["results"][champ].predictions
    loss = selection.LOSS_OF_METRIC[decision["policy"]["metric"]]
    comp = cfg["comparison"]
    out: Dict = {"target": target, "window": [res["meta"]["test_start"], res["meta"]["test_end"]], "champion": c,
                 "champion_forecaster": champ, "variant": "tuned" if tuned else "default", "lock_status": res["meta"].get("lock_status"),
                 "data_digest": res["meta"]["data_digest"], "code": res["meta"]["code"], "loss": loss}
    row = res["overall"].set_index("forecaster").loc[champ]
    out["locked_scores"] = {k: float(row[k]) for k in ("n", "sse", "mae", "rmse", "coverage")}
    lm = float(daily_losses(cp, loss).mean())
    out["locked_loss"], out["dev_loss"] = lm, float(decision["champion_dev_loss"])
    out["loss_ratio_locked_over_dev"] = lm / out["dev_loss"]
    if with_reference:
        rr = (ref_res or res)["overall"].set_index("forecaster").loc[ref_name]
        out["reference"] = ref_name
        out["reference_scores"] = {k: float(rr[k]) for k in ("n", "sse", "mae", "rmse", "coverage")}
        if champ != ref_name:
            d = diebold_mariano(cp, ref_pred, loss, comp["hac_lags"], comp["small_sample"])
            out["dm_vs_reference"] = {k: float(v) for k, v in d.items()}
    out["members"] = {n: {k: float(v) for k, v in res["overall"].set_index("forecaster").loc[n, ["sse", "mae", "rmse"]].items()}
                      for n in res["overall"]["forecaster"] if n != champ}
    out["_results"] = [res] + ([ref_res] if ref_res else [])
    return out


def to_json(rep: dict) -> dict:
    """The report without the in-memory results (what is written to disk)."""
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def render(rep: dict, decision: dict) -> str:
    c = rep["champion"]
    what = f"the {c['method']} of {', '.join(c['members'])}" if c["type"] == "ensemble" else c["model"]
    s = rep["locked_scores"]
    out = [f"# Locked test: {rep['target']}", "", "Generated by `tools/run_locked_test.py`; do not edit by hand.", "",
           f"Champion: **{what}** ({rep['variant']} settings), scored once on the locked window {rep['window'][0]} to {rep['window'][1]} "
           f"(lock status: {rep['lock_status']}, code `{rep['code']}`).", "", "| | n hours | SSE | MAE | RMSE |", "|---|---|---|---|---|",
           f"| champion | {s['n']:,.0f} | {s['sse']:,.0f} | {s['mae']:.3f} | {s['rmse']:.3f} |"]
    if "reference_scores" in rep:
        r = rep["reference_scores"]
        out.append(f"| reference ({rep['reference']}) | {r['n']:,.0f} | {r['sse']:,.0f} | {r['mae']:.3f} | {r['rmse']:.3f} |")
    if "dm_vs_reference" in rep:
        d = rep["dm_vs_reference"]
        out += ["", f"Champion against the reference, Diebold-Mariano on daily {rep['loss']} loss over {d['n_days']:.0f} days: statistic "
                f"{d['statistic']:.2f}, p = {d['p_value']:.3g} (negative = the champion has the smaller loss)."]
    out += ["", f"Mean daily {rep['loss']} loss: development window {rep['dev_loss']:.3f}, locked period {rep['locked_loss']:.3f} "
            f"(ratio {rep['loss_ratio_locked_over_dev']:.2f}; above 1 = worse on the locked period than in development).", ""]
    if rep.get("members"):
        out += ["Other forecasters scored in the same locked run (reported, not used for any choice):", "",
                "| forecaster | SSE | MAE | RMSE |", "|---|---|---|---|"]
        out += [f"| {n} | {v['sse']:,.0f} | {v['mae']:.3f} | {v['rmse']:.3f} |" for n, v in sorted(rep["members"].items())]
    out += ["", f"Decision: policy `{decision['policy_digest'][:12]}`, made from `{decision['source']['run']}` on the development window."]
    return "\n".join(out) + "\n"
