"""
test_results_summary.py — the one-table summary of every price target reads only the files of the earlier stages.
"""
from __future__ import annotations

import json

from common_layer.model_development import rollout, summary


def write(d, name, obj):
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


def test_the_summary_lists_every_price_target_and_reads_decisions_and_locked_tests(tmp_path):
    sel, lk = tmp_path / "sel", tmp_path / "lk"
    sel.mkdir()
    lk.mkdir()
    write(sel, "da_price.json", {"status": "chosen", "champion": {"type": "single", "model": "lightgbm"}, "n_days": 1015, "champion_dev_loss": 17.4,
                                 "tied": ["lightgbm"], "parsimony": {"tested": True, "p_value": 1e-24}})
    write(sel, "ida1_price.json", {"status": "chosen", "champion": {"type": "ensemble", "members": ["xgboost", "lightgbm"], "method": "mean"},
                                   "n_days": 293, "champion_dev_loss": 5.0, "tied": ["xgboost", "lightgbm"], "parsimony": {"tested": False}})
    write(sel, "xbid_price.json", {"status": "insufficient_evidence", "champion": None, "n_days": 100})
    write(lk, "da_price.json", {"locked_scores": {"mae": 18.1}, "loss_ratio_locked_over_dev": 1.04})
    df = summary.collect(str(sel), str(lk), active_version=lambda t: 1 if t == "da_price" else None)
    assert list(df["target"]) == list(rollout.PRICE_TARGETS)
    r = df.set_index("target")
    assert r.loc["da_price", "champion"] == "lightgbm" and r.loc["da_price", "locked_mae"] == 18.1 and r.loc["da_price", "registered"] == "v1"
    assert r.loc["ida1_price", "champion"] == "mean of xgboost, lightgbm" and r.loc["ida1_price", "tied"] == "xgboost, lightgbm"
    assert r.loc["xbid_price", "champion"] == "insufficient_evidence" and r.loc["afrr_cap_price", "champion"] == "-"
    assert r.loc["mfrr_price", "treatment"] == "very_limited_evidence" and r.loc["ida1_price", "treatment"] == "limited_evidence"
    text = summary.render(df)
    assert "| da_price | standard | 1015 | lightgbm | 17.400 | 1e-24 |" in text and "| mfrr_price | very_limited_evidence |" in text
    assert summary.render(df) == text                                                             # deterministic
