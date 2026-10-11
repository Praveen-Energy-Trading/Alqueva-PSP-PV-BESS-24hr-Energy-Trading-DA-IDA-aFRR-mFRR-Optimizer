"""
test_forecast_evaluation.py — Phase 6: the evaluation framework scores the tree models correctly and cannot be fooled.

Synthetic hourly panels with known structure: metrics against hand computation, folds against the schedule, the
as-of view against the information set, the Diebold-Mariano test against planted differences, the locked-period
rules, and the leakage check against a deliberately cheating forecaster (which it must catch). Tree models on
synthetic data use small settings so the tests stay fast; the model family itself is checked for completeness.
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation import guards, metrics
from common_layer.forecast_evaluation.asof import AsOfView, PANEL_TO_INPUT
from common_layer.forecast_evaluation.config import TREE_MODELS, load_evaluation_config, validate_evaluation_config
from common_layer.forecast_evaluation.dm import diebold_mariano
from common_layer.forecast_evaluation.folds import make_folds
from common_layer.forecast_evaluation.forecasters import FitContext, Forecaster
from common_layer.forecast_evaluation.runner import evaluate
from common_layer.forecast_evaluation.tree_models import tree_candidates

CFG = load_evaluation_config()
FORM = load_forecast_formulation()
D0 = dt.date(2022, 1, 1)
SCHED = {"retrain_every_days": 10, "gap_days": 1, "min_train_days": 60, "train_window_days": 100}
PERIODS = [{"id": "all", "start": dt.date(2000, 1, 1), "end": dt.date(2100, 1, 1)}]
FAST = ["decision_tree", "lightgbm"]


def synthetic_panel(days: int = 300, seed: int = 1) -> pd.DataFrame:
    """Daily shape + weekend dip + noise for the day-ahead price; intraday prices = day-ahead + a small spread."""
    r = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(D0), periods=24 * days, freq="h", tz="Europe/Madrid").tz_convert("UTC")
    local = idx.tz_convert("Europe/Madrid")
    hour, dow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    base = 60 + 25 * np.sin(2 * np.pi * hour / 24) - 10 * (dow >= 5) + r.normal(0, 3, len(idx))
    panel = pd.DataFrame(index=idx)
    panel["da_pt"] = base
    panel["da_es"] = base
    panel["ida1_pt"] = base + r.normal(0, 4, len(idx))
    panel["ida2_pt"] = base + r.normal(0, 4, len(idx))
    panel["ida3_pt"] = base + r.normal(0, 4, len(idx))
    panel["xbid_pt"] = base + r.normal(0, 5, len(idx))
    panel["afrr_up"] = 10 + r.normal(0, 1, len(idx))
    panel["afrr_dn"] = panel["afrr_up"]
    panel["mfrr_ap"] = base + r.normal(0, 8, len(idx))
    panel["imb_short"] = base + r.normal(0, 8, len(idx))
    panel["imb_long"] = base - 5 + r.normal(0, 8, len(idx))
    return panel


def cfg_small(reference: str = "decision_tree") -> dict:
    c = copy.deepcopy(CFG)
    c["schedule"] = dict(SCHED)
    c["models"] = {"candidates": list(FAST), "reference": reference}
    return c


# ----------------------------------------------------------------------- config and metrics
def test_evaluation_config_is_valid_and_lists_only_tree_models():
    assert validate_evaluation_config(CFG) == []
    assert CFG["metrics"]["point"] == ["sse", "mae", "rmse"]
    assert CFG["models"]["candidates"] == list(TREE_MODELS) == ["decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]
    bad = copy.deepcopy(CFG)
    bad["models"]["candidates"] = ["decision_tree", "naive_24h"]
    assert any("models.candidates" in p for p in validate_evaluation_config(bad))
    bad = copy.deepcopy(CFG)
    bad["models"]["reference"] = "lightgbm_tuned"
    assert any("models.reference" in p for p in validate_evaluation_config(bad))
    bad = copy.deepcopy(CFG)
    bad["development"]["test_end"] = dt.date(2025, 12, 31)                 # runs into the locked window
    assert any("locked" in p for p in validate_evaluation_config(bad))
    bad = copy.deepcopy(CFG)
    bad["metrics"]["point"] = ["mae", "rmse"]
    assert any("metrics.point" in p for p in validate_evaluation_config(bad))


def test_sse_mae_rmse_match_hand_computation():
    df = metrics.add_error_columns(pd.DataFrame({"y_true": [10.0, 20.0, 30.0, 40.0], "y_pred": [12.0, 18.0, 33.0, 40.0]}))
    s = metrics.score(df)
    assert s["n"] == 4 and s["sse"] == pytest.approx(4 + 4 + 9 + 0) and s["mae"] == pytest.approx(7 / 4)
    assert s["rmse"] == pytest.approx(np.sqrt(17 / 4)) and s["rmse"] == pytest.approx(np.sqrt(s["sse"] / s["n"]))
    nan = metrics.add_error_columns(pd.DataFrame({"y_true": [1.0, 2.0], "y_pred": [1.0, np.nan]}))
    assert metrics.score(nan)["n"] == 1                                    # missing predictions are excluded, not zero


# ------------------------------------------------------------------------------ folds
def test_folds_follow_the_retraining_schedule_with_gap_and_rolling_window():
    folds = make_folds(dt.date(2022, 6, 1), dt.date(2022, 6, 25), dt.date(2022, 1, 1), SCHED)
    assert [len(f.test_days) for f in folds] == [10, 10, 5]
    f0 = folds[0]
    assert f0.train_end == dt.date(2022, 6, 1) - dt.timedelta(days=2)       # gap_days + 1 before the first predicted day
    assert len(f0.train_days) == 100 and f0.train_days[-1] == f0.train_end   # rolling window
    with pytest.raises(ValueError):
        make_folds(dt.date(2022, 2, 1), dt.date(2022, 2, 10), dt.date(2022, 1, 1), SCHED)   # fewer than 60 training days


# ------------------------------------------------------------------------- as-of view
def test_asof_view_masks_everything_the_information_set_does_not_allow():
    panel = synthetic_panel(80)
    view = AsOfView(panel, FORM, "da_price")
    D = dt.date(2022, 2, 20)
    v = view.at(D)
    today = v[v["local_date"] == D.toordinal()]
    yesterday = v[v["local_date"] == D.toordinal() - 1]
    assert today["da_pt"].isna().all() and yesterday["da_pt"].notna().all()          # da_result: up to D-1
    assert v[v["local_date"] == D.toordinal() - 2]["xbid_pt"].notna().all() and yesterday["xbid_pt"].isna().all()   # xbid_summary: up to D-2
    assert v["imb_short"][v["local_date"] > D.toordinal() - 6].isna().all()          # imbalance: up to D-6 (published about four days late)
    assert v["da_es"].notna().any()                                                   # same information item as da_pt
    ida = AsOfView(panel, FORM, "ida1_price")
    day = ida.at(D)[ida.at(D)["local_date"] == D.toordinal()]
    assert day["da_pt"].notna().all() and day["ida1_pt"].isna().all()               # day-ahead of D known at IDA1, IDA1 itself not


def test_a_forecaster_cannot_read_labels_or_views_beyond_the_training_window():
    view = AsOfView(synthetic_panel(80), FORM, "da_price")
    end = dt.date(2022, 2, 10)
    ctx = FitContext(view, [end - dt.timedelta(days=i) for i in range(5)], end)
    assert len(ctx.label(end)) == 24
    with pytest.raises(PermissionError):
        ctx.label(end + dt.timedelta(days=1))
    with pytest.raises(PermissionError):
        ctx.view_at(end + dt.timedelta(days=1))


def test_a_spread_target_needs_its_base_price_known_for_the_delivery_day():
    bad = copy.deepcopy(FORM)
    t = next(t for t in bad["targets"] if t["id"] == "ida1_price")
    next(i for i in t["inputs"] if i["series"] == "da_result")["latest_day_offset"] = -1
    with pytest.raises(ValueError, match="base price"):
        AsOfView(synthetic_panel(30), bad, "ida1_price")


# --------------------------------------------------------------------- the tree models
def test_the_model_family_is_the_five_tree_algorithms_and_nothing_else():
    cands = tree_candidates()
    assert list(cands) == ["decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]
    with pytest.raises(KeyError):
        tree_candidates(["naive_24h"])


def test_tree_models_learn_a_daily_shape_and_the_report_carries_the_agreed_metrics():
    out = evaluate("da_price", synthetic_panel(220), FORM, cfg_small(), PERIODS, "dev",
                   tree_candidates(FAST), test_window=(dt.date(2022, 5, 1), dt.date(2022, 6, 20)))
    o = out["overall"].set_index("forecaster")
    assert set(o.index) == set(FAST) and (o["coverage"] > 0.99).all()
    assert o.loc["lightgbm", "rmse"] < 6.0                                          # planted noise sd 3 -> irreducible ~3
    assert o.loc["lightgbm", "rmse"] == pytest.approx(np.sqrt(o.loc["lightgbm", "sse"] / o.loc["lightgbm", "n"]))
    assert {"sse", "mae", "rmse"} <= set(o.columns) and "mase" not in o.columns
    rep = out["report"]
    assert "SSE" in rep and "MASE" not in rep and "naive" not in rep.lower() and "Diebold-Mariano" in rep


def test_spread_targets_are_reconstructed_to_prices_and_cannot_beat_the_planted_spread_noise():
    panel = synthetic_panel(150)
    out = evaluate("ida1_price", panel, FORM, cfg_small(), PERIODS, "dev", tree_candidates(["lightgbm"]),
                   test_window=(dt.date(2022, 4, 1), dt.date(2022, 4, 30)))
    p = out["results"]["lightgbm"].predictions
    assert np.allclose(p["y_true"], panel.loc[p.index, "ida1_pt"])                  # truth is the price, not the spread
    assert 3.4 < out["overall"].set_index("forecaster").loc["lightgbm", "rmse"] < 5.0   # planted spread noise sd = 4


def test_comparison_uses_a_tree_model_as_the_reference():
    out = evaluate("da_price", synthetic_panel(150), FORM, cfg_small("decision_tree"), PERIODS, "dev",
                   tree_candidates(FAST), test_window=(dt.date(2022, 4, 1), dt.date(2022, 4, 20)))
    c = out["comparisons"]
    assert list(c["forecaster"]) == ["lightgbm"] and (c["reference"] == "decision_tree").all()


# --------------------------------------------------------------------- Diebold-Mariano
def _pred(errors: np.ndarray, days: int) -> pd.DataFrame:
    return pd.DataFrame({"err": errors, "local_date": np.repeat(np.arange(days), len(errors) // days)})


def test_diebold_mariano_detects_a_planted_difference_and_not_a_null_one():
    r = np.random.default_rng(3)
    days, per = 300, 24
    a = _pred(r.normal(0, 10, days * per), days)
    b = _pred(r.normal(0, 5, days * per), days)
    res = diebold_mariano(a, b)
    assert res["statistic"] > 5 and res["p_value"] < 1e-6 and res["mean_diff"] > 0          # A is worse
    assert diebold_mariano(b, a)["statistic"] == pytest.approx(-res["statistic"], rel=1e-6)
    c = _pred(r.normal(0, 5, days * per), days)
    assert diebold_mariano(b, c)["p_value"] > 0.01
    short = diebold_mariano(_pred(r.normal(size=24 * 5), 5), _pred(r.normal(size=24 * 5), 5))
    assert np.isnan(short["statistic"])                                                     # too few days to test


# ----------------------------------------------------------------------- locked period
def test_locked_period_rules(tmp_path):
    lk = CFG["locked_test"]
    guards.check_stage("dev", dt.date(2023, 1, 1), CFG["development"]["test_end"], CFG)
    with pytest.raises(guards.LockedPeriodError):
        guards.check_stage("dev", dt.date(2025, 1, 1), lk["start"], CFG)                    # touches the locked period
    with pytest.raises(guards.LockedPeriodError):
        guards.check_stage("final", lk["start"], dt.date(2026, 6, 30), CFG)                 # not exactly the locked window
    guards.check_stage("final", lk["start"], lk["end"], CFG)
    root = str(tmp_path)
    guards.register_final_run(CFG, "da_price", "lightgbm", root=root)
    with pytest.raises(guards.LockedPeriodError, match="already been scored"):
        guards.register_final_run(CFG, "da_price", "lightgbm", root=root)
    guards.register_final_run(CFG, "da_price", "xgboost", root=root)                        # another model is fine
    guards.register_final_run(CFG, "ida1_price", "lightgbm", root=root)                     # another target is fine


def test_a_changed_locked_outcome_is_detected(tmp_path):
    root = str(tmp_path)
    (tmp_path / "config").mkdir()
    assert guards.verify_lock("da_price", "abc", root) == "not locked"
    guards.write_lock("da_price", "abc", "w", root)
    assert guards.verify_lock("da_price", "abc", root) == "verified"
    with pytest.raises(guards.LockedPeriodError, match="changed since"):
        guards.verify_lock("da_price", "xyz", root)


# --------------------------------------------------------------------------- leakage
class LeakyForecaster(Forecaster):
    """Cheats: keeps the fit context and, at predict time, reaches through it into the unmasked data for day D."""
    name = "leaky"

    def fit(self, ctx: FitContext) -> None:
        self.ctx = ctx

    def predict(self, view: pd.DataFrame, D: dt.date) -> np.ndarray:
        return self.ctx._view.truth_day(D)["da_pt"].to_numpy()


def test_the_leakage_check_passes_the_tree_models_and_catches_a_cheating_forecaster():
    panel = synthetic_panel(200)
    origin = dt.date(2022, 5, 10)
    cands = tree_candidates(FAST)
    for name in FAST:
        diff = guards.perturbation_check(lambda v, n=name: cands[n](v, {}), panel, FORM, "da_price", origin, SCHED)
        assert diff == pytest.approx(0.0, abs=1e-9), name
    leak = guards.perturbation_check(lambda v: LeakyForecaster(), panel, FORM, "da_price", origin, SCHED)
    assert leak > 10.0                                                                         # the cheat is exposed


# --------------------------------------------------------------------- reproducibility
def test_the_same_run_gives_a_byte_identical_report_and_a_data_change_changes_the_digest(tmp_path):
    panel = synthetic_panel(130)
    kw = dict(test_window=(dt.date(2022, 4, 1), dt.date(2022, 4, 15)))
    a = evaluate("da_price", panel, FORM, cfg_small(), PERIODS, "dev", tree_candidates(FAST), out_dir=str(tmp_path / "a"), **kw)
    b = evaluate("da_price", panel, FORM, cfg_small(), PERIODS, "dev", tree_candidates(FAST), out_dir=str(tmp_path / "b"), **kw)
    assert open(a["paths"]["report"], "rb").read() == open(b["paths"]["report"], "rb").read()
    assert open(a["paths"]["overall"], "rb").read() == open(b["paths"]["overall"], "rb").read()
    changed = panel.copy()
    changed.iloc[24 * 95, changed.columns.get_loc("da_pt")] += 1.0                              # one hour in the scored window
    c = evaluate("da_price", changed, FORM, cfg_small(), PERIODS, "dev", tree_candidates(FAST), **kw)
    assert c["meta"]["data_digest"] != a["meta"]["data_digest"] and c["meta"]["config_digest"] == a["meta"]["config_digest"]
