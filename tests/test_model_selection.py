"""
test_model_selection.py — stage 5: the champion selection policy, checked on constructed forecasts with known answers.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from common_layer.forecast_evaluation.config import load_evaluation_config
from common_layer.model_development import registry, selection

from tests.test_algorithm_comparison import make

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
POLICY = selection.load_policy()
CMP = load_evaluation_config()["comparison"]
NAMES = ["decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]


def run(errors: dict, days: int = 400, seed: int = 1, policy: dict = None) -> dict:
    return selection.select(make(errors, days=days, seed=seed), policy or POLICY, CMP, "da_price")


def sds(dt_=9.0, rf=6.0, gb=5.0, xg=5.0, lg=2.5) -> dict:
    return dict(zip(NAMES, (dt_, rf, gb, xg, lg)))


def tied(days: int = 400, seed: int = 3) -> dict:
    """Three algorithms with the same error up to a small independent part: statistically indistinguishable by construction."""
    rng = np.random.default_rng(seed)
    n = days * 24
    base = rng.normal(0, 3.0, n)
    errors = sds()
    for m in ("gradient_boosting", "xgboost", "lightgbm"):
        errors[m] = base + rng.normal(0, 0.4, n)
    return errors


def test_the_policy_in_the_repository_is_valid_and_bad_policies_are_rejected():
    assert selection.validate_policy(POLICY) == []
    bad = copy.deepcopy(POLICY)
    bad.update(metric="mape", tie_break="vote", alpha=2, reference="svm", complexity_order=["decision_tree"])
    assert len(selection.validate_policy(bad)) >= 4


def test_a_clear_winner_is_the_champion():
    d = run(sds())
    assert d["status"] == "chosen" and d["champion"] == {"type": "single", "model": "lightgbm"}
    assert d["best"] == "lightgbm" and d["tied"] == ["lightgbm"] and d["parsimony"]["passed"]
    assert d["ranking"][0]["model"] == "lightgbm" and d["n_days"] == 400
    assert any("significantly worse" in r for r in d["reasons"])


def test_statistically_tied_algorithms_are_averaged_when_the_average_is_not_worse():
    d = run(tied())
    assert d["champion"]["type"] == "ensemble" and d["champion"]["method"] == "mean"
    members = d["champion"]["members"]
    assert 2 <= len(members) <= 3 and set(members) <= {"gradient_boosting", "xgboost", "lightgbm"}
    assert members == sorted(members, key=NAMES.index)                       # listed in the order of the complexity list
    assert d["combination"]["accepted"] and "decision_tree" not in members and "random_forest" not in members


def test_the_average_is_dropped_when_it_is_significantly_worse_than_the_best(monkeypatch):
    def worse(preds, method, *a, **k):
        best = next(iter(preds.values()))
        out = best.copy()
        out["y_pred"] = out["y_pred"] + 25.0                                # a deliberately poor "average"
        out["err"] = out["y_pred"] - out["y_true"]
        out["abs_err"], out["sq_err"] = out["err"].abs(), out["err"] ** 2
        return out
    monkeypatch.setattr(selection, "combine", worse)
    d = run(tied())
    assert d["champion"]["type"] == "single" and d["champion"]["model"] == d["best"]
    assert not d["combination"]["accepted"]


def test_the_other_tie_breaks():
    errors = tied()
    simplest = run(errors, policy={**POLICY, "tie_break": "simplest"})
    assert simplest["champion"] == {"type": "single", "model": simplest["tied"][0]}
    assert simplest["tied"][0] == min(simplest["tied"], key=NAMES.index)
    best = run(errors, policy={**POLICY, "tie_break": "best_mae"})
    assert best["champion"] == {"type": "single", "model": best["best"]}
    median = run(errors, policy={**POLICY, "average_method": "median"})
    assert median["champion"]["method"] == "median" or median["champion"]["type"] == "single"


def test_no_algorithm_is_chosen_over_the_simplest_unless_it_is_significantly_better():
    d = run({m: 4.0 for m in NAMES}, seed=11)                                # all equally good
    assert d["champion"] == {"type": "single", "model": "decision_tree"}
    assert (d["best"] == "decision_tree") or (not d["parsimony"]["passed"])


def test_too_little_evidence_gives_no_champion():
    short = run(sds(), days=100)
    assert short["status"] == "insufficient_evidence" and short["champion"] is None and "need 365" in short["reasons"][0]
    data = make(sds(), days=400)
    p = data["lightgbm"]
    p.iloc[: int(0.1 * len(p)), p.columns.get_loc("y_pred")] = np.nan          # 10% of the hours not predicted
    thin = selection.select(data, POLICY, CMP, "da_price")
    assert thin["status"] == "insufficient_evidence" and thin["coverage"] < 0.99


def test_the_decision_is_reproducible_serialisable_and_tied_to_the_policy():
    a, b = run(sds()), run(sds())
    assert a == b and json.loads(json.dumps(a)) == a
    other = run(sds(), policy={**POLICY, "alpha": 0.01})
    assert other["policy_digest"] != a["policy_digest"]


def test_the_rule_can_use_squared_loss():
    d = run(sds(), policy={**POLICY, "metric": "rmse"})
    assert d["champion"]["model"] == "lightgbm" and any("squared" in r for r in d["reasons"])


def test_the_tool_writes_the_decision_and_records_it(tmp_path):
    runf = tmp_path / "run"
    runf.mkdir()
    for m, df in make(tied(), days=400).items():
        df.to_csv(runf / f"predictions_{m}_tuned.csv")
    env = {**os.environ, "MODEL_REGISTRY_DB": str(tmp_path / "reg.db")}
    outs = []
    for i in (1, 2):
        out = tmp_path / f"o{i}"
        r = subprocess.run([sys.executable, os.path.join(_REPO, "tools", "select_champion.py"), "--run", str(runf), "--suffix", "_tuned",
                            "--target", "da_price", "--out", str(out)], capture_output=True, text=True, cwd=_REPO, env=env,
                           stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stderr[-1500:]
        outs.append(out)
    j = json.loads((outs[0] / "da_price.json").read_text(encoding="utf-8"))
    assert j["status"] == "chosen" and j["champion"]["type"] == "ensemble"
    assert "## Reasons" in (outs[0] / "da_price_decision.md").read_text(encoding="utf-8")
    for f in ("da_price.json", "da_price_decision.md"):                       # same input, same files
        assert (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes()
    rec = registry.latest_selection("da_price", db_path=str(tmp_path / "reg.db"))
    assert rec["champion"] == j["champion"] and rec["policy_digest"] == j["policy_digest"]
    assert registry.latest_selection("ida1_price", db_path=str(tmp_path / "reg.db")) is None
